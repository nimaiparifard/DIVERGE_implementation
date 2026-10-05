# Reviewer R4-C5 (Problem D): "DIVERGE should be tested against baselines rather than only
# fine-tuned vs. base embeddings."
"""
Significance of DIVERGE against its comparators. No LLM is fine-tuned or run here: every
embedding comes from artifacts/cache; only GNNs (seconds per view) and ensemble heads are
trained, and both are cached so the statistics can be recomputed for free.

Leak-free runs
--------------
A run uses the LoRA embeddings fine-tuned on split s together with a GNN / ensemble on the
SAME split s (verify_lora_split() checks this from the LLM logits stored in every cache:
train accuracy > test accuracy only on the split the adapter was trained on).
    cora, citeseer : s = 42, 43, 44, 45
    pubmed         : s = 43          (only seed with all five views cached)
    wikics         : s = 42
    arxiv          : s = 50          (un-seeded cache; verify_lora_split shows split 50)
The primary run of a dataset is its first seed. Ensemble heads are fitted on the train-node
logits and both the epoch and the configuration are chosen on validation accuracy
(paper Section "Implementation Detail for Ensemble Learning"); test nodes are scored once.

Part A - paired per-node tests (same test nodes, exact two-sided McNemar)
    DIVERGE (Method 1/2/3) vs
      base       frozen LLaMA embeddings + GNN       (no fine-tuning; not ArXiv: no cache)
      best_view  single LoRA view with best val acc  (no ensemble)
      uniform    uniform average of the view logits  (no learned head)
      seed_only  same head on 5 Gaussian LoRA seeds  (no init diversity; Cora split 42 only)
      tape       TAPE on the base LLM with GCN, same split (TAPE/run_tape_base.py, scripts/run_tape_base.bat;
                 per-node predictions in results/tape/preds; not PubMed: no GPT-4o texts)
    Primary family = Method 3 on each dataset's primary run; Holm over that family.
    Cora/CiteSeer also: per-seed tests and a t-test over the 4 per-seed gains.

Part B - unpaired tests against the numbers reported in Table 5 (no per-node predictions)
    Two-proportion z-test (DIVERGE primary-run accuracy vs reported accuracy, both on n test
    nodes). It accounts for test-node sampling only; the baseline's training variance is unknown
    (all baselines are reported with +/-0.00). Holm over all (dataset, baseline) pairs.
    TAPE is the exception: its value is our TAPE (GCN) run on the primary split when it exists.
    Cora/CiteSeer also: one-sample t-test of the 4 per-seed DIVERGE accuracies vs the value.

Outputs (results/significance_baselines/):
    preds/<dataset>_seed<s>.npz   per-node predictions of every model (reused unless --refit)
    lora_split_check.csv, paired_tests.csv, seed_level_tests.csv, baseline_tests.csv,
    run_accuracy.csv, report.txt
"""
import argparse
import itertools
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import f1_score

from dataset.data_utils import get_based_model_dataset, get_cached_dir, get_embedding_from_data
from ensemble.weight_learner_ensemble import WeightLearner, WeightLearnerPerClass, PerClassMLPEnsemble
from experiments.diversity_common import TextReport
from experiments.ensemble_experiments import ensemble_common
from experiments.ensemble_experiments.ensemble_common import ViewSet, _train_learner
from experiments.sparse_sage import load_sparse_graph, train_sparse_gnn_logits

OUT_DIR = 'results/significance_baselines'
GNN_VIEW_DIR = 'results/ensemble_experiments'  # shares the view-GNN cache with the ensemble experiments
LLM = 'llama_3.2_1B'
RUNS = {'cora': [42, 43, 44, 45], 'citeseer': [42, 43, 44, 45], 'pubmed': [43], 'wikics': [42], 'arxiv': [50]}
DATASETS = list(RUNS)
SPARSE_DATASETS = {'arxiv', 'computer'}  # >1.1M edges: full-batch SAGE does not fit in 12 GB
SEED_ONLY = {('cora', 42): [142, 242, 342, 442, 542]}  # Gaussian LoRA seeds fine-tuned on split 42

# Paper ensemble-head search space (Table tab:search_space); epoch and config chosen on val
HEAD_GRID = {'layers': [2, 3], 'hidden': [64, 128, 256], 'dropout': [0.3, 0.5], 'lr': [1e-4, 1e-3, 1e-2]}
HEADS = {
    'M1': lambda M, C, h: WeightLearner(M, num_classes=C, hidden_dim=h['hidden'], num_layers=h['layers'],
                                        dropout=h['dropout']),
    'M2': lambda M, C, h: WeightLearnerPerClass(M, C, hidden_dim=h['hidden'], num_layers=h['layers'],
                                                dropout=h['dropout']),
    'M3': lambda M, C, h: PerClassMLPEnsemble(M, C, hidden_dim=h['hidden'], num_layers=h['layers'],
                                              dropout=h['dropout']),
}
PRIMARY_METHOD = 'M3'
COMPARATORS = ['base', 'best_view', 'uniform', 'seed_only', 'tape']
TAPE_PREDS_DIR = 'results/tape/preds'  # TAPE/run_tape_base.py --preds_dir
TAPE_GNN = 'GCN'

# Table tab:diverge_node_classification_accuracy_sota (accuracy, %)
REPORTED = {
    'GCN': [87.32, 74.52, 89.24, 82.59, 70.92], 'SAGE': [88.01, 74.15, 90.77, 84.55, 71.31],
    'GAT': [86.28, 72.33, 87.95, 83.24, 71.42], 'SenBERT-66M': [80.52, 73.16, 93.23, 86.91, 72.24],
    'RoBERTa-355M': [83.86, 74.89, 93.63, 87.13, 70.22], 'ENGINE': [85.10, 73.22, 90.42, 85.13, 73.29],
    'TAPE': [89.32, 77.65, 93.98, 87.24, 75.92], 'GraphGPT': [82.29, 74.67, 93.54, 82.54, 75.15],
    'LLaGA': [87.55, 76.73, 90.28, 84.03, 74.49], 'E-LLaGNN': [86.13, 74.02, 90.03, 85.79, 71.33],
    'UltraTAG': [87.22, 76.69, 90.59, 85.93, 72.84], 'LLM-BP': [80.29, 70.79, 88.58, 82.11, 69.18],
    'STAGE': [88.90, 76.44, 93.49, 86.73, 74.27], 'SKETCH': [82.30, 71.51, 90.01, 84.25, 72.44],
}
REPORTED_ORDER = ['cora', 'citeseer', 'pubmed', 'wikics', 'arxiv']
PAPER_DIVERGE = {'M1': [89.80, 79.47, 95.26, 88.47, 77.89], 'M2': [89.90, 79.39, 95.26, 88.72, 77.76],
                 'M3': [89.94, 79.50, 95.23, 88.59, 77.81]}


# ----------------------------------------------------------------------------- leakage check
def verify_lora_split(dataset, lora_seed, split_seed):
    """LLM accuracy of the cached fine-tuned model on the train / test nodes of `split_seed`."""
    cache = get_cached_dir()
    rows = []
    for init in ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']:
        path = os.path.join(cache, f'{LLM}_{dataset}_seqcls_lora_init-{init}_pool-mean_seed{lora_seed}_semi_supervised.pt')
        if not os.path.exists(path):
            path = os.path.join(cache, f'{LLM}_{dataset}_seqcls_lora_init-{init}_pool-mean.pt')
        d = torch.load(path, map_location='cpu', weights_only=False)
        ok = (d['logits'].argmax(1) == d['labels'].view(-1)).numpy()
        z = np.load(f'seed_snapshot/splits/{dataset}_split1_seed{split_seed}.npz')
        rows.append({'dataset': dataset, 'split_seed': split_seed, 'view': init, 'cache': os.path.basename(path),
                     'llm_train_acc': ok[z['train_idx']].mean() * 100, 'llm_test_acc': ok[z['test_idx']].mean() * 100})
    return rows


# ----------------------------------------------------------------------------- collection
def viewset(dataset, seed):
    args = SimpleNamespace(dataset_name=dataset, llm_name=LLM, peft_type='lora', split=1, lora_seed=seed,
                           approaches='pissa,orthogonal,gaussian,loftq,eva', out_dir=GNN_VIEW_DIR, gnn_model_name=None,
                           protocol='train_fit', tune=1, eval_set='test')
    return ViewSet(args)


@contextmanager
def gnn_backend(dataset, gnn_cfg):
    """ArXiv: full-batch edge gathering does not fit in 12 GB; train the same model via spmm instead."""
    if dataset not in SPARSE_DATASETS:
        yield
        return
    graph = load_sparse_graph(gnn_cfg)
    original = ensemble_common.train_gnn_logits
    ensemble_common.train_gnn_logits = lambda cfg, emb, verbose=False: train_sparse_gnn_logits(cfg, emb, graph)
    try:
        yield
    finally:
        ensemble_common.train_gnn_logits = original


def cached_gnn(path, gnn_cfg, emb):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    logits, results, _ = ensemble_common.train_gnn_logits(gnn_cfg, emb, verbose=False)
    rec = {'logits': logits, 'val_acc': float(results['val_acc']), 'test_acc': float(results['test_acc'])}
    torch.save(rec, path)
    return rec


def fit_head(method, L, y, idx, seed=0):
    """Paper grid; keep the config with the best val accuracy (first best on ties). Returns pred [n], cfg."""
    M, _, C = L.shape
    best = None
    for vals in itertools.product(*HEAD_GRID.values()):
        h = dict(zip(HEAD_GRID, vals))
        _, fused, info = _train_learner(HEADS[method](M, C, h), L, y, idx, {'lr': h['lr'], 'epochs': 200}, seed)
        if best is None or info['sel_acc'] > best[0]:
            best = (info['sel_acc'], fused.argmax(1), h)
    return best[1], best[2]


def load_views_and_base(dataset, seed):
    """
    One GNN per LoRA view and one on the base (not fine-tuned) LLM embeddings, all trained on split `seed`
    (cached in GNN_VIEW_DIR). Returns (viewset, views as ViewSet.load, base logits [n, C] or None).
    """
    vs = viewset(dataset, seed)
    gnn_cfg = vs.gnn_cfg(seed)
    with gnn_backend(dataset, gnn_cfg):
        views = vs.load(seed, verbose=False)
        path = os.path.join(GNN_VIEW_DIR, 'gnn_views', vs.tag(seed), 'base_model.pt')
        try:
            base_emb = None if os.path.exists(path) else \
                get_embedding_from_data(get_based_model_dataset(vs.cfg)).detach().float().cpu()
            base = cached_gnn(path, gnn_cfg, base_emb)['logits']
        except FileNotFoundError as e:
            print(f'  [skip base] {e.args[0].splitlines()[0]}')
            base = None
    return vs, views, base


def collect_run(dataset, seed, refit=False):
    path = os.path.join(OUT_DIR, 'preds', f'{dataset}_seed{seed}.npz')
    if os.path.exists(path) and not refit:
        return dict(np.load(path, allow_pickle=True))
    print(f'\n=== {dataset} seed {seed} ===')
    vs, views, base = load_views_and_base(dataset, seed)
    L, y = views['logits'], views['y']
    idx = {'fit': torch.where(views['train_mask'])[0], 'sel': torch.where(views['val_mask'])[0]}
    out = {'y': y.numpy(), 'test_idx': torch.where(views['test_mask'])[0].numpy(),
           'views': np.array(vs.approaches), 'view_val_acc': views['val_acc'], 'view_test_acc': views['test_acc']}

    for m in HEADS:
        pred, cfg = fit_head(m, L, y, idx)
        out[f'pred_{m}'] = pred.numpy()
        out[f'cfg_{m}'] = np.array(str(cfg))
        print(f'  {m}: test acc {(pred[views["test_mask"]] == y[views["test_mask"]]).float().mean() * 100:.2f}  {cfg}')
    out['pred_uniform'] = L.mean(0).argmax(1).numpy()
    best = int(np.argmax(views['val_acc']))  # first best on ties
    out['pred_best_view'] = L[best].argmax(1).numpy()
    out['best_view'] = np.array(vs.approaches[best])

    if base is not None:
        out['pred_base'] = base.argmax(1).numpy()

    if (dataset, seed) in SEED_ONLY:
        gnn_cfg = vs.gnn_cfg(seed)
        cache = get_cached_dir()
        sdir = os.path.join(OUT_DIR, 'gnn_views', f'{dataset}_{LLM}_seedonly_split{seed}_gnnseed{seed}')
        os.makedirs(sdir, exist_ok=True)
        recs = []
        for ls in SEED_ONLY[(dataset, seed)]:
            f = os.path.join(cache, f'{LLM}_{dataset}_seqcls_lora_init-gaussian_pool-mean_seed{ls}_split{seed}_semi_supervised.pt')
            emb = get_embedding_from_data(torch.load(f, map_location='cpu', weights_only=False)).float()
            recs.append(cached_gnn(os.path.join(sdir, f'gaussian_seed{ls}.pt'), gnn_cfg, emb))
        Ls = torch.stack([r['logits'] for r in recs])
        pred, cfg = fit_head(PRIMARY_METHOD, Ls, y, idx)
        out['pred_seed_only'] = pred.numpy()
        out['pred_seed_only_uniform'] = Ls.mean(0).argmax(1).numpy()
        out['cfg_seed_only'] = np.array(str(cfg))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def add_tape(run, dataset, seed):
    """Attach TAPE's per-node predictions on the same split (kept out of the cached npz, so a TAPE re-run
    is picked up without refitting the heads)."""
    path = os.path.join(TAPE_PREDS_DIR, f'{dataset}_{LLM}_{TAPE_GNN}_seed{seed}.npz')
    if not os.path.exists(path):
        print(f'  [skip tape] {path} not found (scripts/run_tape_base.bat)')
        return run
    t = np.load(path)
    if not (np.array_equal(t['test_idx'], run['test_idx']) and np.array_equal(t['y'], run['y'])):
        raise ValueError(f'{path}: test nodes / labels differ from the DIVERGE run on split {seed}')
    run['pred_tape'] = t['pred_tape']
    return run


# ----------------------------------------------------------------------------- statistics
def mcnemar(ok_a, ok_b):
    """a = DIVERGE, b = comparator. n10 = only DIVERGE correct, n01 = only comparator correct."""
    n10 = int((ok_a & ~ok_b).sum())
    n01 = int((~ok_a & ok_b).sum())
    n = n10 + n01
    p_exact = stats.binomtest(n10, n, 0.5).pvalue if n else 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / n if n else 0.0
    return n10, n01, float(p_exact), float(chi2), float(stats.chi2.sf(chi2, 1)) if n else 1.0


def paired_bootstrap_ci(ok_a, ok_b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = ok_a.astype(float) - ok_b.astype(float)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def correct(run, name):
    t = run['test_idx']
    return run[f'pred_{name}'][t] == run['y'][t]


def macro_f1(run, name):
    t = run['test_idx']
    return f1_score(run['y'][t], run[f'pred_{name}'][t], average='macro') * 100


def analyze(runs, report):
    present = [d for d in DATASETS if (d, RUNS[d][0]) in runs]
    # --- per-run accuracies
    acc_rows = []
    for (ds, s), run in runs.items():
        for name in [k[5:] for k in run if k.startswith('pred_')]:
            ok = correct(run, name)
            acc_rows.append({'dataset': ds, 'seed': s, 'model': name, 'n_test': len(ok), 'acc': ok.mean() * 100,
                             'macro_f1': macro_f1(run, name)})
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(os.path.join(OUT_DIR, 'run_accuracy.csv'), index=False)

    report.section('Test accuracy (%) per dataset: primary run | mean +/- std over seeds')
    for ds in present:
        a = acc[acc.dataset == ds]
        line = []
        for m in list(HEADS) + ['uniform', 'best_view', 'base', 'seed_only', 'tape']:
            r = a[a.model == m]
            if len(r):
                prim = r[r.seed == RUNS[ds][0]].acc.iloc[0]
                line.append(f'{m} {prim:.2f}' + (f' ({r.acc.mean():.2f}+/-{r.acc.std(ddof=0):.2f})' if len(r) > 1 else ''))
        report.line(f'  {ds:<9}' + ' | '.join(line))

    # --- Part A: paired tests
    rows = []
    for (ds, s), run in runs.items():
        for m in HEADS:
            ok_d = correct(run, m)
            for c in COMPARATORS:
                if f'pred_{c}' not in run:
                    continue
                ok_c = correct(run, c)
                n10, n01, p, chi2, p_chi2 = mcnemar(ok_d, ok_c)
                lo, hi = paired_bootstrap_ci(ok_d, ok_c)
                rows.append({'dataset': ds, 'seed': s, 'primary_run': s == RUNS[ds][0], 'method': m, 'comparator': c,
                             'n_test': len(ok_d), 'diverge_acc': ok_d.mean() * 100, 'comparator_acc': ok_c.mean() * 100,
                             'diff_pp': (ok_d.mean() - ok_c.mean()) * 100, 'ci_low_pp': lo, 'ci_high_pp': hi,
                             'only_diverge_correct': n10, 'only_comparator_correct': n01,
                             'p_exact': p, 'chi2_cc': chi2, 'p_chi2_cc': p_chi2})
    pt = pd.DataFrame(rows)
    pt['p_holm'] = np.nan
    pt['p_bonf'] = np.nan
    for m in HEADS:  # one family per method: its primary-run tests over datasets x comparators
        fam = pt.primary_run & (pt.method == m)
        pt.loc[fam, 'p_holm'] = holm(pt.loc[fam, 'p_exact'])
        pt.loc[fam, 'p_bonf'] = np.minimum(1.0, pt.loc[fam, 'p_exact'] * fam.sum())
    pt.to_csv(os.path.join(OUT_DIR, 'paired_tests.csv'), index=False)

    fam = pt[pt.primary_run & (pt.method == PRIMARY_METHOD)]
    report.section(f'Part A - DIVERGE {PRIMARY_METHOD} vs in-house comparators, primary run, exact McNemar '
                   f'(Holm over {len(fam)} tests)')
    report.line(f'  {"dataset":<9}{"vs":<11}{"n":>6}{"DIVERGE":>9}{"comp.":>8}{"diff pp [95% CI]":>23}'
                f'{"n10/n01":>10}{"p":>11}{"p_Holm":>10}')
    for _, r in fam.iterrows():
        report.line(f'  {r.dataset:<9}{r.comparator:<11}{r.n_test:>6}{r.diverge_acc:>9.2f}{r.comparator_acc:>8.2f}'
                    f'{r.diff_pp:>+8.2f} [{r.ci_low_pp:+.2f},{r.ci_high_pp:+.2f}]'
                    f'{f"{r.only_diverge_correct}/{r.only_comparator_correct}":>10}{r.p_exact:>11.3g}{r.p_holm:>10.3g}'
                    f'{"  *" if r.p_holm < 0.05 else ""}')
    report.line('  n10/n01: test nodes only DIVERGE / only the comparator classifies correctly. * = Holm p < 0.05')

    report.section('Part A - same tests for all three heads (primary run): diff pp (Holm p within each head)')
    for ds in present:
        for c in COMPARATORS:
            sub = pt[pt.primary_run & (pt.dataset == ds) & (pt.comparator == c)]
            if len(sub):
                report.line(f'  {ds:<9}{c:<11}' + '  '.join(
                    f'{r.method} {r.diff_pp:+.2f} (p={r.p_holm:.2g})' for _, r in sub.iterrows()))

    # --- seed level (Cora / CiteSeer)
    srows = []
    for ds in [d for d in present if len(RUNS[d]) > 1]:
        for m in HEADS:
            for c in COMPARATORS:
                sub = pt[(pt.dataset == ds) & (pt.method == m) & (pt.comparator == c)]
                if len(sub) < 2:
                    continue
                g = sub.diff_pp.values
                srows.append({'dataset': ds, 'method': m, 'comparator': c, 'n_seeds': len(g), 'gain_mean_pp': g.mean(),
                              'gain_std_pp': g.std(ddof=1), 'seeds_positive': int((g > 0).sum()),
                              'seeds_mcnemar_sig': int((sub.p_exact < 0.05).sum()),
                              'ttest_p': float(stats.ttest_1samp(g, 0).pvalue) if g.std() > 0 else np.nan})
    st = pd.DataFrame(srows)
    st.to_csv(os.path.join(OUT_DIR, 'seed_level_tests.csv'), index=False)
    report.section(f'Part A - across the 4 seeds (Cora, CiteSeer), DIVERGE {PRIMARY_METHOD}: gain per seed')
    for _, r in st[st.method == PRIMARY_METHOD].iterrows():
        report.line(f'  {r.dataset:<9}{r.comparator:<11}gain {r.gain_mean_pp:+.2f}+/-{r.gain_std_pp:.2f} pp  '
                    f'positive {r.seeds_positive}/{r.n_seeds}  McNemar p<0.05 in {r.seeds_mcnemar_sig}/{r.n_seeds}  '
                    f't-test p={r.ttest_p:.3g}')

    # --- Part B: reported baselines
    brows = []
    for ds in present:
        j = REPORTED_ORDER.index(ds)
        prim = runs[(ds, RUNS[ds][0])]
        ok = correct(prim, PRIMARY_METHOD)
        n, p1 = len(ok), ok.mean()
        seed_accs = acc[(acc.dataset == ds) & (acc.model == PRIMARY_METHOD)].acc.values
        for b, vals in REPORTED.items():
            p2, source = vals[j] / 100, 'reported'
            if b == 'TAPE' and 'pred_tape' in prim:  # our TAPE (GCN) run on the same split
                p2, source = correct(prim, 'tape').mean(), f'measured ({TAPE_GNN}, split {RUNS[ds][0]})'
            se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
            z = (p1 - p2) / se
            row = {'dataset': ds, 'baseline': b, 'baseline_acc': p2 * 100, 'baseline_source': source,
                   'diverge_acc': p1 * 100,
                   'paper_diverge_acc': PAPER_DIVERGE[PRIMARY_METHOD][j], 'n_test': n, 'diff_pp': (p1 - p2) * 100,
                   'z': z, 'p_z': float(2 * stats.norm.sf(abs(z)))}
            if len(seed_accs) > 1:
                row['diverge_seed_mean'] = seed_accs.mean()
                row['ttest_seeds_p'] = float(stats.ttest_1samp(seed_accs, p2 * 100).pvalue)
            brows.append(row)
    bt = pd.DataFrame(brows)
    bt['p_holm'] = holm(bt.p_z)
    bt['p_bonf'] = np.minimum(1.0, bt.p_z * len(bt))
    bt.to_csv(os.path.join(OUT_DIR, 'baseline_tests.csv'), index=False)

    report.section(f'Part B - DIVERGE {PRIMARY_METHOD} (primary run) vs reported baselines, two-proportion z-test '
                   f'(Holm over {len(bt)} tests)')
    for ds in present:
        sub = bt[bt.dataset == ds].sort_values('baseline_acc', ascending=False)
        r0 = sub.iloc[0]
        report.line(f'  {ds} (n={r0.n_test}, DIVERGE {r0.diverge_acc:.2f}, paper {r0.paper_diverge_acc:.2f})')
        for _, r in sub.head(4).iterrows():
            extra = f'  seeds t-test p={r.ttest_seeds_p:.3g}' if 'ttest_seeds_p' in r and not np.isnan(r.ttest_seeds_p) else ''
            src = '' if r.baseline_source == 'reported' else f'  [{r.baseline_source}]'
            report.line(f'    vs {r.baseline:<13}{r.baseline_acc:>6.2f}{src}  diff {r.diff_pp:+.2f} pp  p={r.p_z:.3g}  '
                        f'Holm {r.p_holm:.3g}{"  *" if r.p_holm < 0.05 else ""}{extra}')
        n_sig = int((sub.p_holm < 0.05).sum())
        report.line(f'    significant after Holm vs {n_sig}/{len(sub)} baselines; '
                    f'not significant vs: {", ".join(sub[sub.p_holm >= 0.05].baseline) or "-"}')
    return pt, st, bt


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = [d for d in DATASETS if d in args.datasets.split(',')]
    report = TextReport(os.path.join(OUT_DIR, 'report.txt'),
                        'R4-C5 / Problem D - DIVERGE vs comparators and reported baselines (LLaMA-3.2-1B, SAGE)')

    check = []
    for ds in datasets:
        lora = RUNS[ds][0]
        for s in sorted(set(RUNS[ds]) | {42, 43, 44, 45} | ({50} if ds == 'arxiv' else set())):
            check += verify_lora_split(ds, lora, s)
    ck = pd.DataFrame(check)
    ck.to_csv(os.path.join(OUT_DIR, 'lora_split_check.csv'), index=False)
    report.section('Leakage check: LLM accuracy (mean of 5 views) on train / test nodes of each split '
                   '(gap only on the adapter\'s own split)')
    for ds, g in ck.groupby('dataset', sort=False):
        m = g.groupby('split_seed')[['llm_train_acc', 'llm_test_acc']].mean()
        report.line(f'  {ds:<9}LoRA seed {RUNS[ds][0]}: ' + ' | '.join(
            f's{s} {r.llm_train_acc:.1f}/{r.llm_test_acc:.1f}' for s, r in m.iterrows()))

    runs = {}
    for ds in datasets:
        for s in RUNS[ds]:
            if args.stage != 'analyze' or os.path.exists(os.path.join(OUT_DIR, 'preds', f'{ds}_seed{s}.npz')):
                runs[(ds, s)] = add_tape(collect_run(ds, s, refit=args.refit), ds, s)
    if args.stage != 'collect':
        analyze(runs, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    p.add_argument('--stage', type=str, default='all', choices=['all', 'collect', 'analyze'])
    p.add_argument('--refit', action='store_true', help='Refit the ensemble heads (view GNNs stay cached)')
    main(p.parse_args())
