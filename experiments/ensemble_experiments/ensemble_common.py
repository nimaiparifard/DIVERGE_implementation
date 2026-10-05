# Shared helpers for the adaptive-ensemble experiments (reviewer R2-C5, also
# R1-C6 / R4-C3):
#
#   "The proposed adaptive ensemble module introduces additional learnable
#    parameters, but the paper does not provide sufficient analysis on how the
#    learned weights behave or why they improve over simple averaging."
#
#   exp1_disagreement_stratified.py  - where does the gain come from? (+ oracle gap)
#   exp2_weak_view_injection.py      - corrupt / duplicate one view on purpose
#   exp3_fusion_baselines.py         - is learning needed? (non-learned fusion rules + #params)
#   exp4_stability_significance.py   - weight stability across seeds + significance tests
#   exp5_node_level_routing.py       - does the node-conditioned MLP route nodes to the right view?
#   exp6_per_class_weights.py        - do views specialise per class?
#   exp7_logit_scale_calibration.py  - is part of the gain just recalibration?
#   exp8_label_efficiency.py         - how many labels does the fusion head need?
#   exp9_view_contribution.py        - leave-one-view-out + Shapley vs learned weights
"""
Leak-free fusion protocol used by every experiment
--------------------------------------------------
ensemble/ensemble_gnns_learning.py trains the WeightLearner on *train-mask*
logits (nodes the GNNs were fit on, so their logits are nearly perfect) and keeps
the epoch with the best *validation* accuracy. The test nodes are never used for
fitting or selection under any protocol here; they are scored exactly once:

  protocol='val_split' (default)
      the validation nodes are split (per --fuse_seeds seed) into
        fit    : the fusion head is trained on these nodes' GNN logits
        select : early stopping / model selection for the fusion head
      and the test nodes are scored exactly once.
  protocol='train_fit'
      fit on the train-mask logits, select on the full validation set (still
      no test access) - shows how much fitting on over-confident train logits
      hurts compared with 'val_split'.
  protocol='paper'
      reproduces ensemble/ensemble_gnns_learning.py: fit on the train-mask
      logits and pick the epoch (and, with --tune 1, the hyperparameters) with
      the best VALIDATION accuracy. (Older versions selected on the test nodes,
      which leaked test labels; that is no longer possible. It is now the same
      node assignment as 'train_fit' and is kept as an alias for old scripts.)

--tune 1 grid-searches the learned heads' hyperparameters over the same space
as ensemble_gnns_learning.tune_ensemble_hyperparameter (layers x hidden x
dropout x lr, see TUNE_GRID) and keeps the config with the best accuracy on the
selection nodes (validation nodes, never test nodes).

The GNNs themselves are trained exactly as in the paper (train mask, early
stopping on val) - one GNN per LoRA-init view per --gnn_seeds seed - and their
full-graph logits are cached under <out_dir>/gnn_views/<tag>/.

Every fusion rule consumes the stacked view logits L [M, n, C] and returns
fused class probabilities for all nodes, so all rules are compared on exactly
the same GNNs, nodes and labels.
"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import set_mgnn_cfg, GNNTrainer
from ensemble.weight_learner_ensemble import WeightLearner, WeightLearnerPerClass
from experiments.cka import APPROACHES, TITLES
from experiments.diversity_common import TextReport, append_rows, parse_list

DEFAULT_OUT_DIR = 'results/ensemble_experiments'

# Fixed colours: a view / method keeps its colour in every figure.
VIEW_COLORS = {'pissa': '#2a78d6', 'orthogonal': '#eb6834', 'gaussian': '#1baf7a', 'loftq': '#eda100',
               'eva': '#e87ba4'}
METHOD_COLORS = {'uniform_logits': '#898781', 'scalar': '#2a78d6', 'per_class': '#eb6834', 'mlp': '#1baf7a',
                 'stacking_lr': '#4a3aa7', 'temp_uniform_probs': '#eda100', 'oracle': '#0b0b0b'}
METHOD_TITLES = {
    'uniform_logits': 'Uniform (logits)', 'uniform_probs': 'Uniform (probs)', 'majority_vote': 'Majority vote',
    'max_confidence': 'Max confidence', 'entropy_weighted': 'Entropy-weighted', 'val_acc_weighted': 'Val-acc-weighted',
    'temp_uniform_probs': 'Temp.-scaled uniform', 'scalar': 'Learned scalar (DIVERGE)',
    'per_class': 'Learned per-class (DIVERGE)', 'mlp': 'Learned node-MLP (DIVERGE)', 'stacking_lr': 'Stacking (LogReg)',
}
PARAM_FREE = ['uniform_logits', 'uniform_probs', 'majority_vote', 'max_confidence', 'entropy_weighted']
LABEL_USING = ['val_acc_weighted', 'temp_uniform_probs']
LEARNED = ['scalar', 'per_class', 'mlp', 'stacking_lr']
ALL_METHODS = PARAM_FREE + LABEL_USING + LEARNED
PROTOCOLS = ['paper', 'val_split', 'train_fit']
EVAL_SETS = ['test', 'val']

# Same search space as ensemble_gnns_learning.tune_ensemble_hyperparameter
TUNE_GRID = {'mlp_layers': [2, 3], 'mlp_hidden': [64, 128, 256], 'mlp_dropout': [0.3, 0.5],
             'lr': [0.001, 0.01, 0.0001]}


# ----------------------------------------------------------------------------- CLI
def add_common_args(parser):
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                        help=f'Comma-separated subset of {APPROACHES} (the ensemble views)')
    parser.add_argument('--split', type=int, default=1,
                        help='1 = supervised embeddings/split (paper setting), 0 = semi-supervised')
    parser.add_argument('--lora_seed', type=int, default=None,
                        help='Seed of the LoRA adapters / embedding caches (default: cfg.dataset.seed)')
    parser.add_argument('--gnn_seeds', type=str, default='42,43,44',
                        help='Comma-separated GNN seeds; one GNN per view per seed (cached)')
    parser.add_argument('--fuse_seeds', type=str, default='0,1,2',
                        help='Comma-separated seeds for the val fit/select split and fusion-head init')
    parser.add_argument('--protocol', type=str, default='val_split', choices=PROTOCOLS,
                        help="val_split = fit/select on two halves of val; train_fit = fit on train, select on val; "
                             "paper = alias of train_fit (as ensemble_gnns_learning.py). Test nodes are never "
                             "used for fitting or selection.")
    parser.add_argument('--fit_frac', type=float, default=0.5,
                        help="val_split only: share of validation nodes used to fit the fusion head (rest = selection)")
    parser.add_argument('--tune', type=int, default=0,
                        help='1 = grid-search the learned heads like tune_ensemble_hyperparameter (TUNE_GRID)')
    parser.add_argument('--gnn_model_name', type=str, default=None,
                        help='GNN encoder (default: the one in the tuned hyperparameter JSON)')
    parser.add_argument('--retrain_gnn', action='store_true', help='Ignore cached view GNNs and retrain')
    parser.add_argument('--epochs', type=int, default=200, help='Fusion-head training epochs (paper: 200)')
    parser.add_argument('--lr', type=float, default=0.01, help='Fusion-head learning rate')
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_hidden', type=int, default=64)
    parser.add_argument('--mlp_dropout', type=float, default=0.3)
    parser.add_argument('--mlp_weight_decay', type=float, default=5e-4)
    parser.add_argument('--eval_set', type=str, default='test', choices=EVAL_SETS,
                        help="Nodes the fused predictions are scored on. 'val' keeps the fusion/selection unchanged, "
                             "only scores the validation nodes; output files get a '_validation' suffix "
                             "(labels inside figures/reports still say 'test')")
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    return parser


def fusion_hp(args):
    return {'epochs': args.epochs, 'lr': args.lr, 'mlp_layers': args.mlp_layers, 'mlp_hidden': args.mlp_hidden,
            'mlp_dropout': args.mlp_dropout, 'mlp_weight_decay': args.mlp_weight_decay, 'tune': bool(args.tune)}


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------------- view GNNs
def train_gnn_logits(gnn_cfg, emb, verbose=False):
    """Train one GNN exactly like the paper and return (full-graph logits [n, C] on cpu, results, meta)."""
    trainer = GNNTrainer(gnn_cfg, emb, verbose=verbose)
    results, _, _ = trainer.train()
    with torch.no_grad():
        trainer.model.eval()
        logits = trainer.model(trainer.features, trainer.graph).float().cpu()
    d = trainer.data
    meta = {'y': d.y.detach().cpu().long(), 'train_mask': d.train_mask.detach().cpu().bool(),
            'val_mask': d.val_mask.detach().cpu().bool(), 'test_mask': d.test_mask.detach().cpu().bool()}
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, results, meta


class ViewSet:
    """The DIVERGE ensemble views (one LoRA init each) for one (dataset, llm, lora seed)."""

    def __init__(self, args):
        self.args = args
        self.dataset_name, self.llm_name, self.peft_type = args.dataset_name, args.llm_name, args.peft_type
        self.supervised = args.split == 1
        self.cfg = setup_finetuning_cfg(self.dataset_name, self.llm_name, self.peft_type)
        self.lora_seed = args.lora_seed if args.lora_seed is not None else getattr(self.cfg.dataset, 'seed', None)
        requested = parse_list(args.approaches)
        self.approaches = [a for a in APPROACHES if a in requested]
        self.titles = [TITLES[a] for a in self.approaches]
        self.out_dir = args.out_dir
        self._emb = None

    def gnn_cfg(self, gnn_seed):
        # gnn_seed only seeds the GNN weights; the node split is the one the LoRA adapters were fine-tuned on
        # (lora_seed), otherwise the GNN test nodes would include LLM training nodes.
        return set_mgnn_cfg(self.dataset_name, supervised=self.supervised, seed=gnn_seed, llm_name=self.llm_name,
                            gnn_model_name=self.args.gnn_model_name, split_seed=self.lora_seed)

    def tag(self, gnn_seed=None):
        """With gnn_seed: cache key of the view GNNs (protocol-independent). Without: output-file prefix,
        which includes the protocol so 'paper' and 'val_split' runs never overwrite each other."""
        split = 'sup' if self.supervised else 'semi'
        gnn = self.gnn_cfg(gnn_seed if gnn_seed is not None else 42).gnn_model_name
        base = f'{self.dataset_name}_{self.cfg.llm.model_name}_{self.peft_type}_{split}_{gnn}_lora{self.lora_seed}'
        if gnn_seed is not None:
            return f'{base}_gnnseed{gnn_seed}'
        tuned = '_tuned' if getattr(self.args, 'tune', 0) else ''
        return f'{base}_{self.args.protocol}{tuned}{eval_suffix(self.args)}'

    def embeddings(self):
        """{approach: [n, d] float cpu tensor} for the selected approaches."""
        if self._emb is None:
            data = get_init_dataset_for_gnn(self.cfg, supervised=self.supervised, seed=self.lora_seed)
            by_name = dict(zip(APPROACHES, data))  # same order as get_init_dataset_for_gnn's return tuple
            self._emb = {a: get_embedding_from_data(by_name[a]).detach().float().cpu() for a in self.approaches}
            del data
        return self._emb

    def load(self, gnn_seed, retrain=False, verbose=True):
        """
        Returns dict: logits [M, n, C] float, y [n], train/val/test masks [n] bool,
        val_acc / test_acc [M] (fractions), approaches, titles, gnn_seed.
        """
        pool_dir = os.path.join(self.out_dir, 'gnn_views', self.tag(gnn_seed))
        os.makedirs(pool_dir, exist_ok=True)
        meta_path = os.path.join(pool_dir, 'meta.pt')
        meta = torch.load(meta_path, weights_only=False) if os.path.exists(meta_path) and not retrain else None
        gnn_cfg = None
        records = []
        for i, a in enumerate(self.approaches):
            path = os.path.join(pool_dir, f'{a}.pt')
            if meta is not None and os.path.exists(path) and not retrain:
                records.append(torch.load(path, weights_only=False))
                continue
            gnn_cfg = gnn_cfg or self.gnn_cfg(gnn_seed)
            if verbose:
                print(f'\n--- [views] training {gnn_cfg.gnn_model_name} on {TITLES[a]} '
                      f'({i + 1}/{len(self.approaches)}), gnn_seed={gnn_seed} ---')
            logits, results, m = train_gnn_logits(gnn_cfg, self.embeddings()[a], verbose=verbose)
            rec = {'logits': logits, 'val_acc': float(results['val_acc']), 'test_acc': float(results['test_acc'])}
            torch.save(rec, path)
            records.append(rec)
            if meta is None:
                meta = m
                torch.save(meta, meta_path)
        on_val = getattr(self.args, 'eval_set', 'test') == 'val'
        return {
            'logits': torch.stack([r['logits'] for r in records]),
            # GNNTrainer reports percent; everything in these experiments uses fractions
            'val_acc': np.array([r['val_acc'] for r in records]) / 100.0,
            # eval_set='val': the per-view "test" accuracy is the val accuracy (scored nodes = val nodes)
            'test_acc': np.array([r['val_acc' if on_val else 'test_acc'] for r in records]) / 100.0,
            'approaches': list(self.approaches), 'titles': list(self.titles), 'gnn_seed': gnn_seed,
            'eval_set': 'val' if on_val else 'test',
            **meta,
        }

    def run_info(self):
        return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'dataset_name': self.dataset_name,
                'llm_name': self.llm_name, 'peft_type': self.peft_type, 'supervised': self.supervised,
                'lora_seed': self.lora_seed, 'protocol': self.args.protocol,
                'selection_on_test': False, 'tuned': bool(getattr(self.args, 'tune', 0)),
                'fit_frac': self.args.fit_frac, 'eval_set': getattr(self.args, 'eval_set', 'test'),
                'views': ','.join(self.titles)}


# ----------------------------------------------------------------------------- protocol
def protocol_indices(views, protocol='val_split', fit_frac=0.5, split_seed=0):
    """{'fit', 'sel', 'test'} LongTensor node indices. No protocol fits or selects on the test nodes.
    With views['eval_set'] == 'val', idx['test'] (the scored nodes) is the validation set instead; fit/sel
    are unchanged, so the fusion heads are identical to the test-set run."""
    idx = _protocol_indices(views, protocol, fit_frac, split_seed)
    test = torch.where(views['test_mask'])[0]
    for k in ('fit', 'sel'):
        if torch.isin(idx[k], test).any():
            raise RuntimeError(f"Leakage: '{k}' nodes overlap the test nodes under protocol={protocol!r}")
    if views.get('eval_set', 'test') == 'val':
        idx['test'] = torch.where(views['val_mask'])[0]
    return idx


def _protocol_indices(views, protocol, fit_frac, split_seed):
    val = torch.where(views['val_mask'])[0]
    test = torch.where(views['test_mask'])[0]
    if protocol in ('paper', 'train_fit'):
        # as ensemble/ensemble_gnns_learning.py: fit on train logits, pick the epoch on VALIDATION
        return {'fit': torch.where(views['train_mask'])[0], 'sel': val, 'test': test}
    if protocol != 'val_split':
        raise ValueError(f'Unknown protocol: {protocol}. Options: {PROTOCOLS}')
    g = torch.Generator().manual_seed(int(split_seed))
    val = val[torch.randperm(len(val), generator=g)]
    k = max(1, min(len(val) - 1, int(round(fit_frac * len(val)))))
    return {'fit': val[:k].sort().values, 'sel': val[k:].sort().values, 'test': test}


def iter_runs(viewset, args, verbose=True):
    """Yield (views, idx, gnn_seed, fuse_seed) for every GNN seed x fusion seed."""
    for gnn_seed in parse_list(args.gnn_seeds, int):
        views = viewset.load(gnn_seed, retrain=args.retrain_gnn, verbose=verbose)
        for fuse_seed in parse_list(args.fuse_seeds, int):
            idx = protocol_indices(views, args.protocol, args.fit_frac, split_seed=fuse_seed)
            yield views, idx, gnn_seed, fuse_seed


# ----------------------------------------------------------------------------- metrics
def acc_f1(pred, y, idx):
    p, t = pred[idx].cpu().numpy(), y[idx].cpu().numpy()
    return float((p == t).mean()), float(f1_score(t, p, average='macro', zero_division=0))


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return total


def nll(probs, y):
    return float(-torch.log(probs[torch.arange(len(y)), y].clamp_min(1e-12)).mean())


def oracle_correct(logits, y):
    """[n] bool: at least one view predicts the correct label."""
    return (logits.argmax(dim=2) == y.unsqueeze(0)).any(dim=0)


def view_correct(logits, y):
    """[M, n] bool: view m is correct on node i."""
    return logits.argmax(dim=2) == y.unsqueeze(0)


def agreement_count(logits):
    """[n] size of the largest group of views that agree on a label (M = unanimous, 1 = all differ)."""
    preds = logits.argmax(dim=2)  # [M, n]
    C = int(logits.shape[2])
    counts = F.one_hot(preds, C).sum(dim=0)  # [n, C]
    return counts.max(dim=1).values


# ----------------------------------------------------------------------------- fusion rules
def _probs(L):
    return torch.softmax(L, dim=-1)


def fit_temperatures(L, y, fit_idx, steps=100):
    """One temperature per view minimising the NLL on the fit nodes."""
    temps = []
    for m in range(L.shape[0]):
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)
        x, t = L[m][fit_idx], y[fit_idx]

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(x / log_t.exp(), t)
            loss.backward()
            return loss

        opt.step(closure)
        temps.append(float(log_t.exp().item()))
    return np.array(temps)


def _count_params(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _train_learner(module, L, y, idx, hp, seed, weight_decay=0.0):
    """Adam on fit-node logits, keep the epoch with the best accuracy on the select nodes
    (idx['sel']: held-out validation nodes; never the test nodes)."""
    torch.manual_seed(seed)
    dev = device()
    module = module.to(dev)
    views = [L[m].to(dev) for m in range(L.shape[0])]
    yd = y.to(dev)
    fit, sel = idx['fit'].to(dev), idx['sel'].to(dev)
    opt = torch.optim.Adam(module.parameters(), lr=hp['lr'], weight_decay=weight_decay)
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(hp['epochs']):
        module.train()
        opt.zero_grad()
        out, _ = module(*[v[fit] for v in views])
        F.cross_entropy(out, yd[fit]).backward()
        opt.step()
        module.eval()
        with torch.no_grad():
            out, _ = module(*[v[sel] for v in views])
            acc = (out.argmax(1) == yd[sel]).float().mean().item()
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    module.load_state_dict(best_state)
    module.eval()
    with torch.no_grad():
        fused, _ = module(*views)
        fit_out, _ = module(*[v[fit] for v in views])
        fit_acc = (fit_out.argmax(1) == yd[fit]).float().mean().item()
    return module, fused.cpu(), {'sel_acc': best_acc, 'fit_acc': fit_acc, 'best_epoch': best_epoch}


def _tune_grid(method):
    """Configs searched for one learned head; layers/hidden/dropout only matter for the MLP."""
    import itertools
    keys = ['mlp_layers', 'mlp_hidden', 'mlp_dropout', 'lr'] if method == 'mlp' else ['lr']
    return [dict(zip(keys, vals)) for vals in itertools.product(*(TUNE_GRID[k] for k in keys))]


def _fit_head(method, make_module, L, y, idx, hp, seed, weight_decay_key=None):
    """Train one learned head; with hp['tune'] run the grid and keep the best selection-node accuracy."""
    configs = _tune_grid(method) if hp.get('tune') else [{}]
    best = None
    for cfg in configs:
        h = {**hp, **cfg}
        wd = h[weight_decay_key] if weight_decay_key else 0.0
        module, fused, info = _train_learner(make_module(h), L, y, idx, h, seed, weight_decay=wd)
        if best is None or info['sel_acc'] > best[2]['sel_acc']:  # strict '>' keeps the first best, as in the paper
            best = (module, fused, info, h)
    module, fused, info, h = best
    info = {**info, 'n_configs': len(configs), 'tuned_hp': {k: h[k] for k in configs[0]}}
    return module, fused, info


def mlp_node_weights(module, L):
    """[n, M] per-node softmax weights of a multi-layer WeightLearner."""
    dev = device()
    with torch.no_grad():
        cat = torch.cat([L[m].to(dev) for m in range(L.shape[0])], dim=-1)
        return torch.softmax(module.mlp(cat), dim=-1).cpu()


def fuse(method, L, y, idx, seed=0, hp=None):
    """
    Apply one fusion rule. L: [M, n, C] view logits. Returns dict with
      probs [n, C], pred [n], n_params, view_weights [M] or None, class_weights [M, C] or None,
      node_weights [n, M] or None, temperatures [M] or None, fit_acc / sel_acc (learned rules).
    """
    hp = hp or {'epochs': 200, 'lr': 0.01, 'mlp_layers': 2, 'mlp_hidden': 64, 'mlp_dropout': 0.3,
                'mlp_weight_decay': 5e-4, 'tune': False}
    torch.manual_seed(seed)
    np.random.seed(seed)
    M, n, C = L.shape
    P = _probs(L)
    out = {'n_params': 0, 'view_weights': None, 'class_weights': None, 'node_weights': None, 'temperatures': None}

    if method == 'uniform_logits':
        probs = _probs(L.mean(0))
        out['view_weights'] = np.full(M, 1.0 / M)
    elif method == 'uniform_probs':
        probs = P.mean(0)
        out['view_weights'] = np.full(M, 1.0 / M)
    elif method == 'majority_vote':
        votes = F.one_hot(L.argmax(2), C).float().sum(0)
        probs = votes / M + 1e-3 * P.mean(0)  # mean probability only breaks ties
        probs = probs / probs.sum(1, keepdim=True)
    elif method == 'max_confidence':
        pick = P.max(dim=2).values.argmax(dim=0)  # [n]
        probs = P[pick, torch.arange(n)]
    elif method == 'entropy_weighted':
        H = -(P * torch.log(P.clamp_min(1e-12))).sum(2)  # [M, n]
        w = 1.0 / (H + 1e-6)
        w = w / w.sum(0, keepdim=True)
        probs = (w.unsqueeze(2) * P).sum(0)
        out['node_weights'] = w.T
    elif method == 'val_acc_weighted':
        acc = (L[:, idx['fit']].argmax(2) == y[idx['fit']].unsqueeze(0)).float().mean(1)
        w = acc / acc.sum()
        probs = _probs((w.view(M, 1, 1) * L).sum(0))
        out['view_weights'] = w.numpy()
    elif method == 'temp_uniform_probs':
        T = fit_temperatures(L, y, idx['fit'])
        probs = _probs(L / torch.tensor(T, dtype=L.dtype).view(M, 1, 1)).mean(0)
        out['temperatures'] = T
        out['n_params'] = M
    elif method == 'scalar':
        module, fused, info = _fit_head('scalar', lambda h: WeightLearner(M, num_layers=1), L, y, idx, hp, seed)
        probs = _probs(fused)
        out['view_weights'] = torch.softmax(module.weights.detach(), 0).cpu().numpy()
        out['n_params'] = _count_params(module)
        out.update(info)
    elif method == 'per_class':
        module, fused, info = _fit_head('per_class', lambda h: WeightLearnerPerClass(M, C, num_layers=1),
                                        L, y, idx, hp, seed)
        probs = _probs(fused)
        cw = torch.softmax(module.weights.detach(), 0).cpu().numpy()  # [M, C]
        out['class_weights'] = cw
        out['view_weights'] = cw.mean(1)
        out['n_params'] = _count_params(module)
        out.update(info)
    elif method == 'mlp':
        def make_mlp(h):
            return WeightLearner(M, num_classes=C, hidden_dim=h['mlp_hidden'], num_layers=max(2, h['mlp_layers']),
                                 dropout=h['mlp_dropout'])
        module, fused, info = _fit_head('mlp', make_mlp, L, y, idx, hp, seed, weight_decay_key='mlp_weight_decay')
        probs = _probs(fused)
        nw = mlp_node_weights(module, L)
        out['node_weights'] = nw
        out['view_weights'] = nw[idx['test']].mean(0).numpy()
        out['n_params'] = _count_params(module)
        out.update(info)
    elif method == 'stacking_lr':
        from sklearn.linear_model import LogisticRegression
        X = torch.log_softmax(L, dim=2).permute(1, 0, 2).reshape(n, M * C).numpy()
        yn = y.numpy()
        fit, sel = idx['fit'].numpy(), idx['sel'].numpy()
        best = None
        for c in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(X[fit], yn[fit])
            acc = float((clf.predict(X[sel]) == yn[sel]).mean())
            if best is None or acc > best[0]:
                best = (acc, clf)
        clf = best[1]
        proba = np.zeros((n, C), dtype=np.float32)
        proba[:, clf.classes_] = clf.predict_proba(X)
        probs = torch.from_numpy(proba)
        out['n_params'] = int(clf.coef_.size + clf.intercept_.size)
        out['sel_acc'] = best[0]
        out['fit_acc'] = float((clf.predict(X[fit]) == yn[fit]).mean())
    else:
        raise ValueError(f'Unknown fusion method: {method}. Options: {ALL_METHODS}')

    out['probs'] = probs.float()
    out['pred'] = probs.argmax(1)
    return out


def evaluate(result, y, idx):
    acc, f1 = acc_f1(result['pred'], y, idx['test'])
    return {'test_acc': acc, 'test_macro_f1': f1}


# ----------------------------------------------------------------------------- output helpers
def exp_dir(args, name):
    path = os.path.join(args.out_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def eval_suffix(args):
    return '_validation' if getattr(args, 'eval_set', 'test') == 'val' else ''


def out_file(args, out_dir, name):
    """<out_dir>/<name>, with '_validation' before the extension when --eval_set val."""
    stem, ext = os.path.splitext(name)
    return os.path.join(out_dir, f'{stem}{eval_suffix(args)}{ext}')


def mean_std(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return (float(v.mean()), float(v.std())) if v.size else (float('nan'), float('nan'))


def plt_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e1e0d9', 'grid.linewidth': 0.6, 'axes.edgecolor': '#c3c2b7',
                         'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.frameon': False})
    return plt


def save_fig(plt, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] figure saved to: {path}')


__all__ = ['add_common_args', 'fusion_hp', 'ViewSet', 'protocol_indices', 'iter_runs', 'fuse', 'evaluate', 'acc_f1',
           'ece', 'nll', 'oracle_correct', 'view_correct', 'agreement_count', 'fit_temperatures', 'train_gnn_logits',
           'mlp_node_weights', 'exp_dir', 'out_file', 'eval_suffix', 'mean_std', 'plt_module', 'save_fig', 'TextReport', 'append_rows',
           'parse_list', 'APPROACHES', 'TITLES', 'VIEW_COLORS', 'METHOD_COLORS', 'METHOD_TITLES', 'PARAM_FREE',
           'LABEL_USING', 'LEARNED', 'ALL_METHODS', 'PROTOCOLS', 'EVAL_SETS', 'TUNE_GRID', 'device']
