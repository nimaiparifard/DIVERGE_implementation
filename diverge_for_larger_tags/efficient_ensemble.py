"""
Memory-bounded drop-in for ensemble.ensemble_gnns_learning.tune_ensemble_hyperparameter when the
GNN logits are precomputed (diverge_from_logits / ablation_from_logits).

Same search grid, epochs, selection rule, results/cost dicts and report as the original, but:
  - labels/masks are loaded once per (dataset, split, seed) and only y + masks go to the GPU
    (the original torch.load()s the whole dataset onto the GPU for every config);
  - the training step is full-batch gradient descent computed in node chunks with gradient
    accumulation (same gradient up to dropout noise), so activation memory is bounded by
    max_chunk_mb instead of growing with num_nodes x hidden_dim x num_classes;
  - 'learnable_per_classes' runs its per-class MLPs as one batched op (initialised from the
    same PerClassMLPEnsemble weights) instead of a Python loop with in-place slice writes;
  - per-epoch selection evaluates only the selection-split nodes; the full-graph logits are
    computed once at the end from the best epoch's weights;
  - the CUDA cache is emptied after every config, so reserved memory does not creep up.

Numbers are not bit-identical to the original (different dropout RNG stream) but follow the
same protocol.
"""
import torch
from torch import nn
import torch.nn.functional as F

from common import load_graph_dataset_for_tape, compute_acc_and_f1, set_seed
from ensemble.ensemble_gnns_learning import _get_path_prefix, compute_error_rate_and_number
from ensemble.weight_learner_ensemble import WeightLearner, WeightLearnerPerClass, PerClassMLPEnsemble
from gnns.gnn_mtrainer import set_mgnn_cfg
from train_llm.efficiency import ResourceMonitor

DEFAULT_MAX_CHUNK_MB = 1024

# Same search space as tune_ensemble_hyperparameter
NUM_LAYERS_OPTIONS = [2, 3]
HIDDEN_DIM_OPTIONS = [64, 128, 256]
DROPOUT_OPTIONS = [0.3, 0.5]
LEARNING_RATE_OPTIONS = [0.001, 0.01, 0.0001]
EPOCHS = 200

_TARGETS_CACHE = {}


def load_targets(dataset_name, supervised, seed, llm_name, device):
    """Labels, split masks and num_classes on `device`; the graph itself stays on the CPU and is dropped.
    Cached so every config / ensemble approach / M value reuses one load."""
    key = (dataset_name, supervised, seed, llm_name, str(device))
    if key not in _TARGETS_CACHE:
        cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name)
        graph_data, num_classes, _ = load_graph_dataset_for_tape(
            dataset_name, 'cpu', re_split=1 if supervised else 0, path_prefix=_get_path_prefix(), seed=cfg.split_seed)
        _TARGETS_CACHE[key] = {
            "seed": cfg.seed,
            "num_classes": num_classes,
            "y": graph_data.y.to(device),
            "masks": {s: getattr(graph_data, f"{s}_mask").to(device) for s in ('train', 'val', 'test')},
        }
        del graph_data
    return _TARGETS_CACHE[key]


class BatchedPerClassMLP(nn.Module):
    """PerClassMLPEnsemble with the per-class MLPs stacked into batched weights [C, in, out]."""

    def __init__(self, ref: PerClassMLPEnsemble):
        super().__init__()
        self.use_mlp = ref.use_mlp
        if not self.use_mlp:
            self.weights = nn.Parameter(ref.weights.detach().clone())  # [C, M]
            return
        per_class = [[m for m in seq if isinstance(m, nn.Linear)] for seq in ref.class_mlps]
        n_lin = len(per_class[0])
        self.W = nn.ParameterList(
            nn.Parameter(torch.stack([lins[i].weight.detach().t() for lins in per_class]).contiguous())
            for i in range(n_lin))
        self.b = nn.ParameterList(
            nn.Parameter(torch.stack([lins[i].bias.detach() for lins in per_class]).unsqueeze(1).contiguous())
            for i in range(n_lin))
        p = next(m.p for m in ref.class_mlps[0] if isinstance(m, nn.Dropout))
        self.dropout = nn.Dropout(p)

    def forward(self, P):
        """P: [M, B, C] -> (ensemble [B, C], mean weights [C, M])"""
        x = P.permute(2, 1, 0)  # [C, B, M]
        if not self.use_mlp:
            w = F.softmax(self.weights, dim=1)  # [C, M]
            return (x * w.unsqueeze(1)).sum(-1).t(), w
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, W)
            if i < len(self.W) - 1:
                h = self.dropout(F.relu(h))
        w = F.softmax(h, dim=-1)  # [C, B, M]
        return (w * x).sum(-1).t(), w.mean(dim=1)


class StackedInput(nn.Module):
    """Adapts WeightLearner / WeightLearnerPerClass (forward(*predictions)) to a stacked [M, B, C] input."""

    def __init__(self, learner):
        super().__init__()
        self.learner = learner

    def forward(self, P):
        return self.learner(*P.unbind(0))


def build_head(approach, num_models, num_classes, hidden_dim, num_layers, dropout):
    if approach == 'learnable':
        return StackedInput(WeightLearner(num_models=num_models, num_classes=num_classes if num_layers > 1 else None,
                                          hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout))
    if approach == 'learnable_classes':
        return StackedInput(WeightLearnerPerClass(num_models=num_models, num_classes=num_classes,
                                                  hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout))
    if approach == 'learnable_per_classes':
        return BatchedPerClassMLP(PerClassMLPEnsemble(num_models=num_models, num_classes=num_classes,
                                                      hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout))
    raise ValueError(f"Unknown weight_approach: {approach}")


def rows_per_chunk(approach, num_models, num_classes, hidden_dim, num_layers, max_chunk_mb):
    """Nodes per training chunk so the autograd activations stay around max_chunk_mb."""
    g = num_classes if approach == 'learnable_per_classes' else 1
    floats = 4 * num_models * num_classes + g * (4 * hidden_dim * max(num_layers - 1, 0) + 4 * num_models)
    return max(256, int(max_chunk_mb * 2 ** 20 // (4 * floats)))


def _chunked_forward(head, P, chunk):
    """Eval-mode forward over [M, N, C] in node chunks -> (logits [N, C], node-weighted mean weights)."""
    outs, w_sum, n = [], None, P.shape[1]
    for s in range(0, n, chunk):
        out, w = head(P[:, s:s + chunk])
        outs.append(out)
        w = w * (out.shape[0] / n)
        w_sum = w if w_sum is None else w_sum + w
    return torch.cat(outs), w_sum


def ensemble_fit(P, targets, approach, num_layers, hidden_dim, dropout, learning_rate, epochs=EPOCHS,
                 select_split='val', max_chunk_mb=DEFAULT_MAX_CHUNK_MB, does_report_training_process=False,
                 split_cache=None):
    """
    Memory-bounded ensemble_gnn_learning for stacked logits P [M, N, C].
    Returns (ensemble_predictions [N], results, weights) like ensemble_gnn_learning.
    split_cache: optional dict reused across configs for the train/selection slices of P.
    """
    if select_split not in ('val', 'test'):
        raise ValueError(f"select_split must be 'val' or 'test', got {select_split!r}")
    set_seed(targets["seed"])
    y, masks, num_classes = targets["y"], targets["masks"], targets["num_classes"]
    num_models = P.shape[0]

    if approach == 'uniform':
        ensemble_logits = P.mean(dim=0)
        weights = [1.0 / num_models] * num_models
        print(f"Using uniform weights: {weights}")
    else:
        sc = split_cache if split_cache is not None else {}
        if "train_P" not in sc:
            sc["train_P"], sc["train_y"] = P[:, masks['train']], y[masks['train']]
            sc["sel_P"], sc["sel_y"] = P[:, masks[select_split]], y[masks[select_split]]
        train_P, train_y, sel_P, sel_y = sc["train_P"], sc["train_y"], sc["sel_P"], sc["sel_y"]
        n_train = train_y.shape[0]

        head = build_head(approach, num_models, num_classes, hidden_dim, num_layers, dropout).to(P.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        chunk = rows_per_chunk(approach, num_models, num_classes, hidden_dim, num_layers, max_chunk_mb)
        eval_chunk = 4 * chunk

        best_sel_acc, best_state, best_epoch = -1.0, None, 0
        for epoch in range(epochs):
            # One full-batch step, accumulated over node chunks
            head.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for s in range(0, n_train, chunk):
                out, _ = head(train_P[:, s:s + chunk])
                loss = F.cross_entropy(out, train_y[s:s + chunk], reduction='sum') / n_train
                loss.backward()
                loss_total += loss.detach()
            optimizer.step()

            # Selection-split accuracy (same rounding as compute_acc_and_f1)
            head.eval()
            with torch.no_grad():
                correct = sum((head(sel_P[:, s:s + eval_chunk])[0].argmax(dim=1) == sel_y[s:s + eval_chunk]).sum()
                              for s in range(0, sel_y.shape[0], eval_chunk))
                sel_acc = round(correct.item() / sel_y.shape[0] * 100.0, 2)
            if sel_acc > best_sel_acc:
                best_sel_acc, best_epoch = sel_acc, epoch
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

            if epoch % max(1, epochs // 5) == 0 and does_report_training_process:
                print(f"Epoch {epoch}/{epochs}, loss: {float(loss_total):.4f}, {select_split}_acc: {sel_acc:.4f}")

        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            ensemble_logits, w = _chunked_forward(head, P, eval_chunk)
        weights = w.cpu().numpy().tolist()
        print(f"\nBest model from epoch {best_epoch} with {select_split} accuracy: {best_sel_acc:.4f}")
        del head, optimizer, best_state

    ensemble_predictions = ensemble_logits.argmax(dim=1)
    results = {}
    for split_name, mask in masks.items():
        pred = ensemble_predictions[mask].cpu().numpy()
        gt = y[mask].cpu().numpy()
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(pred, gt)
        error_rate, error_number = compute_error_rate_and_number(pred, gt)
        results[split_name] = {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
                               'error_rate': error_rate, 'error_number': error_number}
        print(f"[{split_name.upper()}] Acc: {acc:.2f}, Macro-F1: {macro_f1:.2f}, Weighted-F1: {weighted_f1:.2f}")
    return ensemble_predictions, results, weights


def _free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _report(reporter, title, approach, best_hyperparams, best_results, grid):
    cost = best_results['cost']
    r = best_results
    lines = [
        "Ensemble Hyperparameter Tuning Results (memory-efficient, from saved logits):", "",
        f"Ensemble Weight Approach: {approach}"
        f"{' (no learnable weights; hyperparameters below are unused)' if approach == 'uniform' else ''}", "",
        "Best Hyperparameters:",
        f"  • Number of Layers: {best_hyperparams['num_layers']}",
        f"  • Hidden Dimension: {best_hyperparams['hidden_dim']}",
        f"  • Dropout: {best_hyperparams['dropout']}",
        f"  • Learning Rate: {best_hyperparams['learning_rate']}", "",
        "Best Performance Metrics:",
    ]
    for split, name in (('test', 'Test'), ('val', 'Validation'), ('train', 'Train')):
        lines += [f"  • {name} Accuracy: {r[split]['accuracy']:.4f}",
                  f"  • {name} Macro-F1: {r[split]['macro_f1']:.4f}",
                  f"  • {name} Weighted-F1: {r[split]['weighted_f1']:.4f}"]
    lines += [
        "", "Hyperparameter Search Space:",
        f"  • Number of Layers: {grid[0]}", f"  • Hidden Dimensions: {grid[1]}",
        f"  • Dropout Options: {grid[2]}", f"  • Learning Rates: {grid[3]}",
        f"  • Total Combinations Tested: {len(grid[0]) * len(grid[1]) * len(grid[2]) * len(grid[3])}", "",
        "Cost:",
        f"  • Tuning time (all configs): {cost['tuning_time_sec']:.2f}s ({cost['n_configs']} configs, "
        f"avg {cost['avg_config_time_sec']}s/config)",
        f"  • Tuning peak VRAM: {cost['tuning_peak_vram_mb']} MB (+{cost['tuning_peak_vram_delta_mb']} MB during tuning)",
        f"  • Selected ensemble fit time: {cost['best_fit_time_sec']}s",
        f"  • Selected ensemble fit peak VRAM: {cost['best_fit_peak_vram_mb']} MB "
        f"(+{cost['best_fit_peak_vram_delta_mb']} MB)",
    ]
    reporter.report(f"Ensemble Hyperparameter Tuning - {title}" if title else "Ensemble Hyperparameter Tuning",
                    "\n".join(lines))


def tune_ensemble_hyperparameter_efficient(dataset_name, model_list=None, custom_embeddings=None,
                                           modified_datasets_list=None, does_report_training_process=False,
                                           title="", reporter=None, llm_name='llama_3.2_1B', peft_type='lora',
                                           title_list=None, visualize_best=False, ensemble_approaches='learnable',
                                           save_dir=None, show=False, predictions_list=None, supervised=True,
                                           seed=None, select_split='val', max_chunk_mb=DEFAULT_MAX_CHUNK_MB):
    """
    Signature-compatible with tune_ensemble_hyperparameter (so it can be passed as tune_fn to
    run_ensembles / run_m_ablation), for the precomputed-logits path only.
    Returns (best_results, best_hyperparams, None).
    """
    if predictions_list is None:
        raise ValueError("tune_ensemble_hyperparameter_efficient needs predictions_list (precomputed logits)")
    if modified_datasets_list is not None:
        raise ValueError("modified_datasets_list is not supported by the logits-only tuner")
    if visualize_best:
        print("[efficient ensemble] visualize_best ignored (no GNNs/embeddings in the logits pipeline)")

    approach = ensemble_approaches
    device = predictions_list[0].device
    targets = load_targets(dataset_name, supervised, seed, llm_name, device)
    P = torch.stack([p.float() for p in predictions_list])  # [M, N, C]
    if P.shape[1] != targets["y"].shape[0]:
        raise ValueError(f"Logits have {P.shape[1]} nodes but the dataset has {targets['y'].shape[0]}")

    grid = (NUM_LAYERS_OPTIONS, HIDDEN_DIM_OPTIONS, DROPOUT_OPTIONS, LEARNING_RATE_OPTIONS)
    if approach == 'uniform':
        grid = ([2], [64], [0.3], [0.01])
    configs = [(l, h, d, lr) for l in grid[0] for h in grid[1] for d in grid[2] for lr in grid[3]]
    print(f"Starting memory-efficient ensemble tuning | approach={approach} | {len(configs)} configs | "
          f"models={P.shape[0]} nodes={P.shape[1]} classes={P.shape[2]} | max_chunk_mb={max_chunk_mb}")
    print("=" * 80)

    best_acc, best_results, best_hyperparams, best_fit_cost = -1.0, {}, {}, None
    config_times, error_messages = [], []
    split_cache = {}
    tuning_monitor = ResourceMonitor().__enter__()
    for idx, (num_layers, hidden_dim, dropout, lr) in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Testing: layers={num_layers}, hidden={hidden_dim}, dropout={dropout}, lr={lr}")
        try:
            with ResourceMonitor() as fit_monitor:
                _, results, _ = ensemble_fit(P, targets, approach, num_layers, hidden_dim, dropout, lr,
                                             select_split=select_split, max_chunk_mb=max_chunk_mb,
                                             does_report_training_process=does_report_training_process,
                                             split_cache=split_cache)
            config_times.append(fit_monitor.time_sec)
            sel_acc = results[select_split]['accuracy']
            print(f"  ✓ {select_split} Acc: {sel_acc:.4f} | Test Acc: {results['test']['accuracy']:.4f} | "
                  f"peak VRAM {fit_monitor.metrics()['peak_vram_mb']} MB")
            if sel_acc > best_acc:
                best_acc, best_results = sel_acc, results
                best_hyperparams = {'num_layers': num_layers, 'hidden_dim': hidden_dim,
                                    'dropout': dropout, 'learning_rate': lr}
                best_fit_cost = fit_monitor.metrics()
                print(f"  ★ New best {select_split} accuracy: {best_acc:.4f}")
        except Exception as e:
            error_messages.append(f"Combination {idx}: layers={num_layers}, hidden={hidden_dim}, "
                                  f"dropout={dropout}, lr={lr} - Error: {e}")
            print(f"  ✗ Error: {str(e)[:100]}")
        finally:
            _free_cuda()

    if not best_results:
        print("\nWARNING: every ensemble config failed:")
        for msg in error_messages[-5:]:
            print(f"  - {msg}")
        tuning_monitor.__exit__(None, None, None)
        del P, split_cache
        _free_cuda()
        return {}, {}, None

    tuning_monitor.__exit__(None, None, None)
    tuning_cost = tuning_monitor.metrics()
    best_results = dict(best_results)
    best_results['cost'] = {
        'tuning_time_sec': tuning_cost['time_sec'],
        'tuning_peak_vram_mb': tuning_cost['peak_vram_mb'],
        'tuning_peak_vram_delta_mb': tuning_cost['peak_vram_delta_mb'],
        'n_configs': len(config_times),
        'avg_config_time_sec': round(sum(config_times) / len(config_times), 3) if config_times else None,
        'best_fit_time_sec': best_fit_cost.get('time_sec'),
        'best_fit_peak_vram_mb': best_fit_cost.get('peak_vram_mb'),
        'best_fit_peak_vram_delta_mb': best_fit_cost.get('peak_vram_delta_mb'),
    }
    cost = best_results['cost']
    print(f"\nBest Hyperparameters: {best_hyperparams}")
    print(f"Best Test Accuracy: {best_results['test']['accuracy']:.4f} | "
          f"Macro-F1: {best_results['test']['macro_f1']:.4f}")
    print(f"Tuning cost: {cost['tuning_time_sec']:.2f}s for {cost['n_configs']} configs, "
          f"peak VRAM {cost['tuning_peak_vram_mb']} MB")
    print("=" * 80)
    if reporter is not None:
        _report(reporter, title, approach, best_hyperparams, best_results, grid)

    del P, split_cache
    _free_cuda()
    return best_results, best_hyperparams, None
