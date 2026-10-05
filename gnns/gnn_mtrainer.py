import torch
from time import time
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from dataset.data_utils import get_init_dataset_for_gnn
from common import GNNEncoder, compute_acc_and_f1
from common import load_graph_dataset_for_tape, set_seed
from train_llm.efficiency import ResourceMonitor
from yacs.config import CfgNode as CN
import json
from pathlib import Path

def get_datasets_path():
    """Get the absolute path to the datasets folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    datasets_path = os.path.join(repo_root, "datasets")
    return datasets_path

# GNN encoders accepted by common.gnn.build_conv. To add a new encoder, register it
# in build_conv and append its name here; tuning/training/lookup pick it up automatically.
SUPPORTED_GNN_MODELS = ("SAGE", "GCN", "GAT", "GIN", "TransformerConv")
DEFAULT_GNN_MODEL = "SAGE"


def get_gnn_hyperparameter_dir():
    return Path(__file__).resolve().parents[1] / 'gnn_hyperparameters'


def gnn_hyperparameter_filename(dataset_name, supervised, llm_name=None, seed=None, gnn_model_name=None):
    """
    Canonical hyperparameter JSON name:
        <dataset>[_<llm>][_<gnn>][_semi_supervised][_seed<seed>].json
    Without the gnn tag this reproduces the legacy names (e.g. cora_semi_supervised.json,
    cora_llama_3.2_1B_semi_supervised_seed42.json), so existing files keep working.
    """
    parts = [dataset_name]
    if llm_name:
        parts.append(llm_name)
    if gnn_model_name:
        parts.append(gnn_model_name)
    if not supervised:
        parts.append('semi_supervised')
    if seed is not None:
        parts.append(f'seed{seed}')
    return '_'.join(parts) + '.json'


def candidate_gnn_hyperparameter_paths(dataset_name, supervised, llm_name=None, seed=None, gnn_model_name=None):
    """
    Hyperparameter files to try, most specific first. Files tagged with the requested
    GNN encoder win over untagged (legacy, SAGE-tuned) files; within each group the
    order is llm+seed -> llm -> seed -> generic per-dataset file.
    """
    hp_dir = get_gnn_hyperparameter_dir()
    gnn = gnn_model_name or DEFAULT_GNN_MODEL
    names = []
    for g in (gnn, None):
        for l, s in ((llm_name, seed), (llm_name, None), (None, seed), (None, None)):
            names.append(gnn_hyperparameter_filename(dataset_name, supervised, l, s, g))
    # dict.fromkeys dedups (e.g. when llm_name/seed are None) while keeping order
    return [hp_dir / n for n in dict.fromkeys(names)]


def llm_split_seed(dataset_name):
    """Seed of the train/val/test split the LLM adapters are fine-tuned on by default
    (configs/dataset/<dataset>.json -> cfg.dataset.seed). None if the config is missing."""
    try:
        from config import load_dataset_config_json
        return int(load_dataset_config_json(dataset_name).seed)
    except Exception:
        return None


def set_mgnn_cfg(datset_name='cora', supervised=True, seed=None, llm_name=None, gnn_model_name=None,
                 split_seed=None):
    """
    Build a yacs CfgNode for GNN training from the most specific JSON in
    `gnn_hyperparameters/` (see candidate_gnn_hyperparameter_paths), falling back
    to defaults when none exists.

    The returned cfg contains fields consumed by `GNNTrainer`:
      - seed, device, dataset, gnn_model_name, llm_name
      - hidden_dim, num_layers, dropout, lr, epochs, early_stop
      - batch_norm, weight_decay, re_split
      - hp_source: path of the JSON actually loaded ("defaults" if none), so callers
        can verify which tuned hyperparameters a GNN run used.

    Args:
        seed: Optional override for the GNN training seed (and, via the caller,
              the seed used to select the matching LLM embedding cache). If None,
              the seed from the hyperparameter JSON / defaults is used unchanged.
        llm_name: Optional LLM whose tuned hyperparameters to load (embedding
              dimensionality/quality differs per LLM, so hyperparameters tuned via
              gnns/tune_gnn_hyperparameter.py are saved per-LLM/per-seed). If None,
              falls back to the generic per-dataset file for backwards compatibility.
        gnn_model_name: GNN encoder (one of SUPPORTED_GNN_MODELS). If None, the
              encoder stored in the loaded JSON is used (SAGE by default).
        split_seed: seed of the train/val/test node split (cfg.split_seed). It must be the
              seed the LLM adapter producing the embedding was fine-tuned with: with a
              different seed the GNN's test nodes contain nodes whose labels the LLM was
              trained on (test-label leakage). Defaults to `seed` when given, otherwise
              to the dataset config seed (configs/dataset/<dataset>.json), which is the
              split the LLM is fine-tuned on by default. cfg.seed then only seeds the GNN
              weights, so several GNN seeds can be run on the same (leak-free) split.
    """
    if gnn_model_name is not None and gnn_model_name not in SUPPORTED_GNN_MODELS:
        raise ValueError(f"Unsupported gnn_model_name '{gnn_model_name}'. Choose from {SUPPORTED_GNN_MODELS}")

    candidates = candidate_gnn_hyperparameter_paths(datset_name, supervised, llm_name=llm_name, seed=seed,
                                                    gnn_model_name=gnn_model_name)
    hp, hp_source = {}, "defaults"
    for c in candidates:
        if c.exists():
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    hp = json.load(f)
                hp_source = str(c)
                print(f"Loaded GNN hyperparameters from: {c}")
                break
            except Exception as e:
                print(f"Failed to load {c}: {e}")
    if hp_source == "defaults":
        print(f"[set_mgnn_cfg] No tuned hyperparameters found for dataset={datset_name} llm={llm_name} "
              f"seed={seed} gnn={gnn_model_name or DEFAULT_GNN_MODEL}; using defaults.")
    elif gnn_model_name is not None and hp.get('gnn_model_name', DEFAULT_GNN_MODEL) != gnn_model_name:
        print(f"[set_mgnn_cfg] WARNING: {Path(hp_source).name} was tuned for "
              f"{hp.get('gnn_model_name', DEFAULT_GNN_MODEL)}, but {gnn_model_name} was requested. "
              f"Run gnns/tune_gnn_hyperparameter.py --gnn_model_name {gnn_model_name} to tune it.")

    # Defaults
    defaults = {
        'seed': 42,
        'device': 0,
        'dataset': datset_name,
        'gnn_model_name': 'SAGE',
        'llm_name': 'llama_3.2_1B',
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'lr': 0.05,
        'epochs': 200,
        'early_stop': 20,
        'batch_norm': 1,
        'weight_decay': 0.1,
        're_split': 1 if supervised else 0,
    }

    # Merge loaded hyperparams with defaults
    for k, v in defaults.items():
        defaults[k] = hp.get(k, v)
    if seed is not None:
        defaults['seed'] = seed
    if llm_name is not None:
        defaults['llm_name'] = llm_name
    if gnn_model_name is not None:
        defaults['gnn_model_name'] = gnn_model_name

    cfg = CN()
    cfg.seed = int(defaults['seed'])
    cfg.device = int(defaults['device'])
    cfg.dataset = defaults['dataset']
    cfg.gnn_model_name = defaults['gnn_model_name']
    cfg.llm_name = defaults['llm_name']
    cfg.hidden_dim = int(defaults['hidden_dim'])
    cfg.num_layers = int(defaults['num_layers'])
    cfg.dropout = float(defaults['dropout'])
    cfg.lr = float(defaults['lr'])
    cfg.epochs = int(defaults['epochs'])
    cfg.early_stop = int(defaults['early_stop'])
    cfg.batch_norm = bool(int(defaults['batch_norm']))
    cfg.weight_decay = float(defaults['weight_decay'])
    cfg.re_split = bool(int(defaults.get('re_split', 1)))
    cfg.hp_source = hp_source
    if split_seed is None:
        split_seed = seed if seed is not None else llm_split_seed(datset_name)
    cfg.split_seed = int(split_seed) if split_seed is not None else cfg.seed

    return cfg


# Message passing on a sparse adjacency (spmm) instead of edge_index. With edge_index, PyG
# gathers one feature row per edge ([num_edges, feature_dim]; ~10 GB for computer/arxiv with
# 2048-dim LLM embeddings), with adj_t memory is O(num_nodes * feature_dim). Same model and
# weights (outputs equal up to float rounding); only valid for sum/mean/GCN-normalised
# aggregation, so attention convs (GAT, TransformerConv) keep edge_index.
SPARSE_ADJ_GNN_MODELS = ("SAGE", "GCN", "GIN")
# 'auto' switches to adj_t once the per-edge gather would hold >= 2^28 elements (1 GB in fp32)
SPARSE_ADJ_AUTO_ELEMENTS = 2 ** 28


def use_sparse_adj(mode, gnn_model_name, edge_index, feature_dim):
    """mode: 'auto' (default), 'on' or 'off'. False for non-dense edge_index (e.g. a CSR adj)."""
    if mode == 'off' or gnn_model_name not in SPARSE_ADJ_GNN_MODELS:
        return False
    if not isinstance(edge_index, torch.Tensor) or edge_index.layout != torch.strided:
        return False
    return mode == 'on' or edge_index.shape[1] * feature_dim >= SPARSE_ADJ_AUTO_ELEMENTS


def to_adj_t(edge_index, num_nodes):
    """Transposed sparse adjacency (adj_t[i, j] = 1 for every edge j -> i), as PyG convs expect."""
    from torch_sparse import SparseTensor
    return SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(num_nodes, num_nodes))


def resolve_path_prefix():
    """Relative path from the CWD to the repo root (what load_graph_dataset_for_tape expects)."""
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            return '.'
        return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'


class GNNTrainer():
    def __init__(self, cfg, feature, modified_dataset=None, does_print_training_process=False,
                 data=None, track_history=True, verbose=True, sparse_adj='auto'):
        """
        Args:
            sparse_adj: 'auto' / 'on' / 'off' - run message passing on a sparse adj_t instead
                  of edge_index (see use_sparse_adj). 'auto' only kicks in on large graphs
                  (computer, arxiv, ...); data.edge_index itself is left unchanged.
            data: Optional pre-loaded graph (from load_graph_dataset_for_tape). Pass it when
                  training many GNNs on the same graph (e.g. hyperparameter tuning) to skip
                  reloading/re-splitting the dataset on every run. Must have been loaded with
                  the same dataset/re_split/seed as `cfg`.
            track_history: If False, only val/test accuracy are computed per epoch (on GPU) and
                  full metrics (F1, per-class errors) only at improving epochs. Much faster,
                  used for tuning; the returned history then only has 'loss' and 'val_acc'.
            verbose: Print split sizes / parameter counts / final accuracy.
        """
        self.seed = cfg.seed
        # Node split seed (see set_mgnn_cfg): must match the LLM fine-tuning split, independent of the GNN init seed
        self.split_seed = cfg.get('split_seed', cfg.seed)
        set_seed(cfg.seed)
        self.device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn_model_name
        self.lm_model_name = cfg.llm_name
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.dropout = cfg.dropout
        self.lr = cfg.lr
        self.features = feature
        self.epochs = cfg.epochs
        self.patience = cfg.early_stop
        self.batch_norm = cfg.batch_norm
        self.weight_decay = cfg.weight_decay
        self.does_print_training_process = does_print_training_process
        self.track_history = track_history
        self.verbose = verbose
        self.hp_source = cfg.get('hp_source', 'unknown')
        self.error_rate = dict()
        self.error_number = dict()
        self.best_model = None
        # Load data
        set_seed(self.seed)
        if data is None:
            # (dataset loading only touches numpy's RNG, so torch-side model init below is
            # identical whether or not `data` is passed in)
            data, num_classes, _ = load_graph_dataset_for_tape(cfg.dataset, self.device, re_split=cfg.re_split, path_prefix=resolve_path_prefix(), seed=self.split_seed, modified_dataset=modified_dataset, )
            set_seed(self.seed)  # GNN weight init depends only on the GNN seed, not on the split seed
        else:
            num_classes = int(data.y.max().item() + 1)

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()
        if self.verbose:
            print(f"train_size: {data.train_mask.sum().item()}, val_size: {data.val_mask.sum().item()}, "
                  f"test_size: {data.test_mask.sum().item()}")
            print(f"GNN: {self.gnn_model_name} | hyperparameters from: {self.hp_source}")
        # Init gnn feature

        self.model = GNNEncoder(
            input_dim=self.features.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=int(data.y.max().item() + 1),
            n_layers=self.num_layers,
            gnn_type=self.gnn_model_name,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)


        self.features = self.features.to(self.device)
        self.data = data.to(self.device)
        # Graph passed to the model: edge_index, or a sparse adj_t on large graphs
        self.graph = self.data.edge_index
        if use_sparse_adj(sparse_adj, self.gnn_model_name, self.graph, self.features.shape[1]):
            self.graph = to_adj_t(self.data.edge_index, self.num_nodes)
            if self.verbose:
                print(f"Message passing on sparse adj_t ({self.data.edge_index.shape[1]} edges)")
        if self.verbose:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\nNumber of GNN parameters: {trainable_params}")
        self.ckpt = f"../../results/TAPE/{self.dataset_name}/{self.gnn_model_name}.pt"

    def _train(self):
        # the same in ../main.py where we train GNNs
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.features, self.graph)
        loss = F.cross_entropy(output[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _evaluate(self):
        # the same in ../main.py where we evaluate GNNs
        self.model.eval()
        logits = self.model(self.features, self.graph)
        pred = logits.argmax(dim=1)
        accuracy, macrof1_scores, weightf1_scores = [], [], []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            acc, macro_f1, weight_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), self.data.y[mask].cpu().numpy())

            accuracy.append(acc)
            macrof1_scores.append(macro_f1)
            weightf1_scores.append(weight_f1)

        # Per-class error stats on the **test** split
        test_mask = self.data.test_mask
        error_rate, error_number = self.compute_error_rate(
            pred[test_mask].cpu().numpy(),
            self.data.y[test_mask].cpu().numpy()
        )
        return accuracy, macrof1_scores, weightf1_scores, logits, error_rate, error_number

    @torch.no_grad()
    def _evaluate_fast(self):
        """Val/test accuracy computed on-device (no CPU sync per mask, no sklearn).
        Same scale/rounding as compute_acc_and_f1 (percent, 2 decimals) so early stopping
        behaves exactly like the track_history=True path."""
        self.model.eval()
        logits = self.model(self.features, self.graph)
        correct = (logits.argmax(dim=1) == self.data.y).float()
        val_acc = round(correct[self.data.val_mask].mean().item() * 100.0, 2)
        test_acc = round(correct[self.data.test_mask].mean().item() * 100.0, 2)
        return val_acc, test_acc, logits


    def compute_error_rate(self, pred, y):
        """
            Compute per-class error statistics on a given split.

            Args:
                pred (np.ndarray): predicted class indices for the split
                y (np.ndarray): ground-truth class indices for the split

            Returns:
                error_rate (dict):   {class_id: error_rate_percentage}
                error_number (dict): {class_id: number_of_misclassified_examples}
        """
        error_rate = {}
        error_number = {}

        # Ensure numpy arrays
        pred = np.asarray(pred)
        y = np.asarray(y)

        # Compute stats for every class present in the dataset
        for class_id in range(self.num_classes):
            class_mask = (y == class_id)
            total = int(class_mask.sum())

            if total == 0:
                # No examples of this class in the evaluated split
                error_number[class_id] = 0
                error_rate[class_id] = 0.0
                continue

            mistakes = int((pred[class_mask] != y[class_mask]).sum())
            error_number[class_id] = mistakes
            # Store percentage for easier interpretation
            error_rate[class_id] = float(mistakes) / float(total) * 100.0

        # Also store on the trainer for external inspection
        self.error_rate = error_rate
        self.error_number = error_number

        return error_rate, error_number

    def plot_learning_curves(self, history, save_dir="../../results/TAPE/learning_curves", show=True):
        """
        Plot and save learning curves for loss, accuracy, and F1 scores.

        Args:
            history: Dictionary containing training history with keys:
                - 'loss': list of training losses
                - 'train_acc', 'val_acc', 'test_acc': lists of accuracies
                - 'train_f1', 'val_f1', 'test_f1': lists of F1 scores
            save_dir: Directory to save the plots
            show: Whether to display the plot (default True)
        """
        os.makedirs(save_dir, exist_ok=True)

        epochs = range(1, len(history['loss']) + 1)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Loss
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[1].plot(epochs, history['train_acc'], 'g-', linewidth=2, label='Train Acc', alpha=0.7)
        axes[1].plot(epochs, history['val_acc'], 'b-', linewidth=2, label='Val Acc')
        axes[1].plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Test Acc')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: F1 Score
        axes[2].plot(epochs, history['train_f1'], 'g-', linewidth=2, label='Train F1', alpha=0.7)
        axes[2].plot(epochs, history['val_f1'], 'b-', linewidth=2, label='Val F1')
        axes[2].plot(epochs, history['test_f1'], 'r-', linewidth=2, label='Test F1')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('F1 Score', fontsize=12)
        axes[2].set_title(f'F1 Score', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(save_dir, f'{self.dataset_name}_seed{self.seed}.png')
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        print(f"Learning curves saved to: {save_path}")
        plt.close()

    def train(self, show_plots=False, epoch_callback=None):
        """
        Train the GNN model.

        Args:
            show_plots: Whether to display plots (default False to avoid interruption during hyperparameter tuning)
            epoch_callback: Optional fn(epoch, val_acc) -> bool; returning True stops training
                early (e.g. to prune a clearly bad hyperparameter trial).

        Returns (results, best_logits, history); results also holds the training cost:
            train_time_sec, peak_vram_mb, peak_vram_delta_mb (see ResourceMonitor),
            epochs_run and time_per_epoch_ms.
        """
        with ResourceMonitor() as mon:
            results, best_logits, history = self._fit(show_plots=show_plots, epoch_callback=epoch_callback)
        cost = mon.metrics()
        epochs_run = len(history['loss'])
        results.update({
            "train_time_sec": cost["time_sec"],
            "peak_vram_mb": cost["peak_vram_mb"],
            "peak_vram_delta_mb": cost["peak_vram_delta_mb"],
            "epochs_run": epochs_run,
            "time_per_epoch_ms": round(cost["time_sec"] / max(epochs_run, 1) * 1000.0, 3),
        })
        if self.verbose:
            print(f"GNN training cost: {results['train_time_sec']:.2f}s ({epochs_run} epochs, "
                  f"{results['time_per_epoch_ms']:.1f} ms/epoch), peak VRAM {results['peak_vram_mb']} MB "
                  f"(+{results['peak_vram_delta_mb']} MB during training)")
        return results, best_logits, history

    def _fit(self, show_plots=False, epoch_callback=None):
        # ! Training
        best_eval_acc = best_test_acc = 0.0
        best_eval_f1 = best_test_f1 = 0.0
        best_eval_weightf1 = best_test_weightf1 = 0.0
        best_error_rate, best_error_number = {}, {}
        counter, best_logits, best_state = 0, None, None

        # Initialize history tracking
        if self.track_history:
            history = {
                'loss': [],
                'train_acc': [], 'val_acc': [], 'test_acc': [],
                'train_f1': [], 'val_f1': [], 'test_f1': [],
                # Per-epoch test-set error statistics (list of dicts)
                'error_rate': [],
                'error_number': []
            }
        else:
            history = {'loss': [], 'val_acc': []}

        for epoch in range(1, 1 + self.epochs):
            loss = self._train()
            if self.track_history:
                accuracy, f1_scores, weightf1_scores, cur_logits, error_rate, error_number = self._evaluate()
                train_acc, val_acc, test_acc = accuracy
                train_f1, val_f1, test_f1 = f1_scores
                train_weightf1, val_weightf1, test_weightf1 = weightf1_scores

                # Track metrics
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
                history['train_f1'].append(train_f1)
                history['val_f1'].append(val_f1)
                history['test_f1'].append(test_f1)
                history['error_rate'].append(error_rate)
                history['error_number'].append(error_number)
            else:
                val_acc, test_acc, cur_logits = self._evaluate_fast()
            history['loss'].append(loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_eval_acc:
                if not self.track_history:
                    # full metrics only at improving epochs
                    accuracy, f1_scores, weightf1_scores, cur_logits, error_rate, error_number = self._evaluate()
                    _, val_f1, test_f1 = f1_scores
                    _, val_weightf1, test_weightf1 = weightf1_scores
                best_eval_acc = val_acc
                best_test_acc = test_acc
                counter = 0
                best_logits = cur_logits.detach().clone()
                best_eval_f1, best_test_f1 = val_f1, test_f1
                best_eval_weightf1, best_test_weightf1 = val_weightf1, test_weightf1
                best_error_rate, best_error_number = error_rate, error_number
                # snapshot weights (a plain reference would keep training and end up as the last epoch)
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1

            if epoch % 10 == 0 and self.does_print_training_process:
                if self.track_history:
                    print(
                        f"Epoch {epoch:03d} Loss {loss:.4f}  Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_f1:.3f} Val F1 {val_f1:.3f} Test F1 {test_f1:.4f}")
                else:
                    print(f"Epoch {epoch:03d} Loss {loss:.4f}  Val acc {val_acc:.3f} Test acc {test_acc:.3f}")

            # Early stopping
            if counter >= self.patience:
                break
            if epoch_callback is not None and epoch_callback(epoch, val_acc):
                break

        if self.verbose:
            print(f'\nTest Acc {best_test_acc:.3f}  Test F1 {best_test_f1:.3f}\n')
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_model = self.model
        if show_plots and self.track_history:
            self.plot_learning_curves(history, show=True)

        return {
            "test_acc": best_test_acc,
            "test_f1": best_test_f1,
            "val_acc": best_eval_acc,
            "val_f1": best_eval_f1,
            "val_weight_f1": best_eval_weightf1,
            "test_weight_f1": best_test_weightf1,
            "error_rate": best_error_rate,
            "error_number": best_error_number,
        }, best_logits, history

def append_gnn_result_csv(row: dict, csv_path: str) -> str:
    """
    Append a single GNN/ensemble result row to a CSV report, creating the file
    (and its parent directory) with a header on first write. Column order is
    stable across appends; unseen keys are appended at the end.
    """
    import csv

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    fieldnames = list(row.keys())
    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header:
            fieldnames = existing_header + [k for k in row.keys() if k not in existing_header]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return csv_path


def gnn_train_and_report(dataset_name, embedding, title="", does_print_training_process=False, reporter=None,
                          supervised=True, seed=None, llm_name=None, peft_type=None, init_weight_approach=None,
                          csv_path="results/gnn_training/gnn_training_results.csv", gnn_model_name=None,
                          return_best_logits=False):
    """
    Train a GNN model on the given dataset using provided embeddings and report results.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'cora', 'pubmed')
        embedding (torch.Tensor): Node embeddings to use as features
        title (str): Title for reporting (e.g., 'PISSA', 'ORTHOGONAL')
        does_print_training_process (bool): Whether to print training progress
        reporter: Reporter object for writing results to file
        supervised (bool): Supervised vs semi-supervised GNN training setup
        seed: Seed to use for this GNN run (and to record as the seed of the LLM
              adapter/embedding used to build `embedding`, for CSV provenance).
              If None, the seed from gnn_hyperparameters/<dataset>.json is used.
        llm_name, peft_type, init_weight_approach: Provenance info for the embedding
              used, recorded in the CSV report (init_weight_approach defaults to `title`).
        csv_path: Where to append the CSV result row ("" / None disables CSV logging).
        gnn_model_name: GNN encoder to train (see SUPPORTED_GNN_MODELS); None keeps the
              encoder from the tuned hyperparameter JSON (SAGE by default).
        return_best_logits: also return the full-graph logits of the best (val-selected)
              epoch, i.e. what the returned model outputs in eval mode.

    Returns:
        dict: Training results containing test_acc, test_f1, val_acc, val_f1
        model: Trained GNN model (best-epoch weights)
        best_logits: [num_nodes, num_classes] tensor, only if return_best_logits=True
    """
    print(f"\n{'='*80}")
    print(f"Training GNN on {dataset_name.upper()} dataset")
    print(f"Embedding shape: {embedding.shape}")
    print(f"{'='*80}\n")

    # Setup GNN configuration for the dataset (picks up per-LLM/per-seed tuned
    # hyperparameters from gnn_hyperparameters/ if available, see set_mgnn_cfg)
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name,
                       gnn_model_name=gnn_model_name)

    # Initialize GNN trainer with the embeddings
    trainer = GNNTrainer(cfg, embedding, does_print_training_process=does_print_training_process)

    # Train the model and get results
    results, best_logits, history = trainer.train()

    # Print final results
    print(f"\n{'='*80}")
    print(f"Final Results for {dataset_name.upper()}:")
    print(f"  Test Accuracy:       {results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {results['val_f1']:.4f}")
    print(f"  Val F1 (Weighted):   {results['val_weight_f1']:.4f}")
    print(f"  Training time:       {results['train_time_sec']:.2f}s ({results['epochs_run']} epochs, "
          f"{results['time_per_epoch_ms']:.1f} ms/epoch)")
    print(f"  Peak VRAM:           {results['peak_vram_mb']} MB (+{results['peak_vram_delta_mb']} MB during training)")
    print(f"{'='*80}\n")

    # Report results using reporter
    if reporter is not None:
        report_title = f"GNN Training Results - {title}" if title else f"GNN Training Results - {dataset_name.upper()}"
        report_text = f"""
                        Final Results:
                        • Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']:.2%})
                        • Test F1 (Macro): {results['test_f1']:.4f}
                        • Test F1 (Weighted): {results['test_weight_f1']:.4f}
                        • Validation Accuracy: {results['val_acc']:.4f} ({results['val_acc']:.2%})
                        • Validation F1 (Macro): {results['val_f1']:.4f}
                        • Validation F1 (Weighted): {results['val_weight_f1']:.4f}
                        Cost:
                        • Training time: {results['train_time_sec']:.2f}s ({results['epochs_run']} epochs, {results['time_per_epoch_ms']:.1f} ms/epoch)
                        • Peak VRAM: {results['peak_vram_mb']} MB (+{results['peak_vram_delta_mb']} MB during training)
                        """
        reporter.report(report_title, report_text)

    # Append complete-info CSV row (dataset, LLM/embedding provenance, GNN hyperparams, metrics)
    if csv_path:
        from datetime import datetime
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "llm_name": llm_name if llm_name is not None else cfg.llm_name,
            "peft_type": peft_type if peft_type is not None else "",
            "init_weight_approach": init_weight_approach if init_weight_approach is not None else title,
            "embedding_seed": seed if seed is not None else "",
            "gnn_seed": cfg.seed,
            "supervised": supervised,
            "embedding_shape": tuple(embedding.shape),
            "gnn_model_name": cfg.gnn_model_name,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_norm": cfg.batch_norm,
            "epochs": cfg.epochs,
            "early_stop": cfg.early_stop,
            "re_split": cfg.re_split,
            "hp_source": os.path.basename(cfg.hp_source),
            "test_accuracy": results["test_acc"],
            "test_macro_f1": results["test_f1"],
            "test_weighted_f1": results["test_weight_f1"],
            "val_accuracy": results["val_acc"],
            "val_macro_f1": results["val_f1"],
            "val_weighted_f1": results["val_weight_f1"],
            "train_time_sec": results["train_time_sec"],
            "epochs_run": results["epochs_run"],
            "time_per_epoch_ms": results["time_per_epoch_ms"],
            "peak_vram_mb": results["peak_vram_mb"],
            "peak_vram_delta_mb": results["peak_vram_delta_mb"],
        }
        append_gnn_result_csv(row, csv_path)
        print(f"[OK] GNN result appended to: {csv_path}")

    if return_best_logits:
        return results, trainer.model, best_logits
    return results, trainer.model

def gnn_train_and_report_with_modified_dataset(modified_dataset, embedding, dataset_name, supervised=True,
                                                seed=None, llm_name=None, gnn_model_name=None):
    """
    Train a GNN model on the given modified dataset using provided embeddings and report results.

    Args:
        modified_dataset: Modified dataset object with updated edge_index
        embedding (torch.Tensor): Node embeddings to use as features
        dataset_name (str): Name of the dataset (e.g., 'cora', 'pubmed')
        seed: Seed for the GNN run (see set_mgnn_cfg).
        llm_name: LLM whose tuned hyperparameters to load (see set_mgnn_cfg).
        gnn_model_name: GNN encoder to train (see set_mgnn_cfg).

    Returns:
        dict: Training results containing test_acc, test_f1, val_acc, val_f1
        model: Trained GNN model
    """
    print(f"\n{'='*80}")
    print(f"Training GNN on {dataset_name.upper()} dataset (MODIFIED)")
    print(f"Embedding shape: {embedding.shape}")
    print(f"{'='*80}\n")

    # Setup GNN configuration for the dataset
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name,
                       gnn_model_name=gnn_model_name)

    # Initialize GNN trainer with the embeddings and modified dataset
    trainer = GNNTrainer(cfg, embedding, modified_dataset=modified_dataset)

    # Train the model and get results
    results, best_logits, history = trainer.train()

    # Print final results
    print(f"\n{'='*80}")
    print(f"Final Results for {dataset_name.upper()} (MODIFIED):")
    print(f"  Test Accuracy:       {results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {results['val_f1']:.4f}")
    print(f"  Val F1 (Weighted):   {results['val_weight_f1']:.4f}")
    print(f"  Training time:       {results['train_time_sec']:.2f}s ({results['epochs_run']} epochs, "
          f"{results['time_per_epoch_ms']:.1f} ms/epoch)")
    print(f"  Peak VRAM:           {results['peak_vram_mb']} MB (+{results['peak_vram_delta_mb']} MB during training)")
    print(f"{'='*80}\n")

    return results, trainer.model

if __name__ == '__main__':
    from config import setup_finetuning_cfg

    print("*" * 80)
    print("Check Hyperparamters")
    gnn_cfg = set_mgnn_cfg('pubmed', supervised=False)
    ft_cfg = setup_finetuning_cfg(
        dataset_name=gnn_cfg.dataset,
        llm_name=gnn_cfg.llm_name,
        peft_type='lora',
    )
    init_caches = get_init_dataset_for_gnn(ft_cfg, supervised=False)
    print(f"Learning rate: {gnn_cfg.lr}")
    print(f"Epochs: {gnn_cfg.epochs}")
    print(f"Dropout: {gnn_cfg.dropout}")
    print(f"Num Layers: {gnn_cfg.num_layers}")
    print(f"Loaded {len(init_caches)} init-weight caches (pissa, orthogonal, gaussian, loftq, eva)")

