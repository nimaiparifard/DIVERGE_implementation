# Q2 experiment (R1-C2, R3-C2): is the diversity between LoRA initializations LARGER than the
# diversity between training runs of the SAME initialization?
"""
Why a new pool is needed
------------------------
In the pools of experiments/seed_vs_init_diversity.py one "LoRA seed" changed three things at
once: the fine-tuning split, the random draw of the LoRA init (A of Gaussian, Orthogonal and
default-LoRA "EVA") and the classification-head init, while the HF Trainer seed (data order,
dropout) was 42 for every run. So a "seed-only" pair of Gaussian members was in fact two
different random initializations on two different label sets, not two training runs of one
initialization. That inflates the seed-only distance (Q2 ratio <= 1) and leaks test labels.

Controlled pool (one per dataset x LLM)
---------------------------------------
  5 init strategies x S training seeds (default 4), and for every member
    split seed       = SPLIT_SEED (same 60/20/20 split for everyone; the GNN uses it too)
    LoRA init seed   = INIT_SEED  (same starting adapter for every member of one strategy)
    training seed    = the member's seed: classification head, data order, dropout
  Trained with   train_llm.train_other_lora_init_apporaches --split 1 --seed s
                 --split_seed SPLIT_SEED --init_seed INIT_SEED
  so a seed-only pair is exactly what R3-C2 asks for: the same LoRA initialization trained with
  a different random seed on the same data.

Stages and pre-registered primary distance (one per stage)
  1 adapter weights   1 - cosine of the adapter updates B A (mean over LoRA layers)
  2 LLM embeddings    1 - linear CKA
  3 GNN embeddings    1 - linear CKA of the GNN penultimate layer (fixed GNN seed = split seed)
  4 GNN predictions   label disagreement on the test nodes
  secondary (--knn):  1 - mean 10-NN Jaccard on the LLM and on the GNN embeddings

Tests per stage (pairs: seed-only = same init / different seed, init-only = different init /
same seed, both = different init and seed)
  Q1  init adds diversity on top of training noise: mean(both) - mean(seed-only), permutation of
      the init labels within each seed (diversity_common.within_between_test)
  Q2  ratio = mean(init-only) / mean(seed-only)
      - crossed (pigeonhole) bootstrap over inits AND seeds: both factors are resampled with
        replacement and every pair is weighted by the product of its members' multiplicities,
        which respects that pairs sharing a member are dependent. 95% CI and one-sided
        p = P(ratio <= 1). This is the per-pool Q2 test.
      - one-sided Mann-Whitney on the pair values (approximate; as in the earlier reports)
  PERMANOVA R^2 of init and of seed

Claim across pools: experiments/diversity_experiment/aggregate_q2.py (sign test + Wilcoxon over
pools on log ratio).

Outputs (under --out_dir, default results/diversity_q2/)
  <tag>_report.txt, q2_summary.csv (one row per pool x stage), matrices/, gnn_pool/
"""
import itertools
import os

import numpy as np
import torch

from config import setup_finetuning_cfg
from dataset.data_utils import get_cached_dir
from gnns.gnn_mtrainer import set_mgnn_cfg
from experiments.cka import APPROACHES, TITLES
from experiments.diversity_common import (
    Pool, parse_list, cka_distance_matrix, disagreement_matrix, cached_matrix, within_between_test, permanova,
    pair_groups, save_pool_heatmap, save_matrix_csv, run_info, append_rows, TextReport,
)
from experiments.lora_subspace_analysis import load_adapter
from experiments.knn_neighborhood_analysis import knn_indices, knn_jaccard
from adapter_dir import lora_run_tag

DEFAULT_OUT_DIR = 'results/diversity_q2'
STAGES = ['adapter_update', 'llm_embedding_cka', 'gnn_embedding_cka', 'gnn_prediction_disagreement']
SECONDARY = ['llm_knn', 'gnn_knn']


# ----------------------------------------------------------------------------- pool
class ControlledPool(Pool):
    """Pool of members trained on one split with one LoRA init draw; only the training seed varies."""

    def __init__(self, dataset_name, llm_name, peft_type, approaches, seeds, split_seed, init_seed,
                 gnn_model_name=None, out_dir=DEFAULT_OUT_DIR):
        self.dataset_name, self.llm_name, self.peft_type = dataset_name, llm_name, peft_type
        self.split_seed, self.init_seed = split_seed, init_seed
        # The GNN uses the split the LoRAs were fine-tuned on: no member has seen a test label.
        self.supervised, self.gnn_seed, self.own_split, self.out_dir = True, split_seed, False, out_dir
        self.gnn_model_name_arg = gnn_model_name
        self.cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
        self.model_name = self.cfg.llm.model_name
        self.gnn_cfg = set_mgnn_cfg(dataset_name, supervised=True, seed=split_seed, llm_name=llm_name,
                                    gnn_model_name=gnn_model_name)
        self.gnn_name = self.gnn_cfg.gnn_model_name

        cache_dir = get_cached_dir()
        self.members, self.missing = [], []
        for init in [a for a in APPROACHES if a in approaches]:
            for seed in seeds:
                tag = lora_run_tag(seed, split_seed, init_seed)
                m = {'init': init, 'seed': seed, 'label': f'{TITLES[init]}-{seed}',
                     'path': os.path.join(cache_dir, f'{self.model_name}_{dataset_name}_seqcls_{peft_type}'
                                                     f'_init-{init}_pool-mean{tag}_semi_supervised.pt'),
                     'adapter': os.path.join('artifacts', f'{self.model_name}_{dataset_name}_seqcls_{peft_type}'
                                                          f'_semi_supervised_init_type_{init}{tag}')}
                (self.members if os.path.exists(m['path']) else self.missing).append(m)
        if len(self.members) < 2:
            raise FileNotFoundError(f'Fewer than 2 controlled-pool caches for {dataset_name}/{llm_name}. Missing:\n'
                                    + '\n'.join(f"  - {m['path']}" for m in self.missing))
        self._index()

    def _index(self):
        self.inits = np.array([m['init'] for m in self.members])
        self.seeds = np.array([m['seed'] for m in self.members])
        self.labels = [m['label'] for m in self.members]

    def drop(self, labels):
        self.members = [m for m in self.members if m['label'] not in set(labels)]
        self._index()

    @property
    def tag(self):
        return (f'{self.dataset_name}_{self.model_name}_{self.peft_type}_q2ctl_split{self.split_seed}'
                f'_init{self.init_seed}_{self.gnn_name}')

    def describe(self):
        lines = [f'Controlled pool: {len(self)} members ({len(set(self.inits))} inits x {len(set(self.seeds))} '
                 f'training seeds), split seed {self.split_seed} for LoRA and GNN, LoRA init seed '
                 f'{self.init_seed}, GNN={self.gnn_name} (GNN seed {self.gnn_seed})']
        for init in dict.fromkeys(self.inits):
            lines.append(f'  {TITLES[init]}: seeds {sorted(int(s) for s in self.seeds[self.inits == init])}')
        if self.missing:
            lines.append(f"  Skipped (no cache): {', '.join(m['label'] for m in self.missing)}")
        return lines


# ----------------------------------------------------------------------------- distances
@torch.no_grad()
def adapter_update_distance(paths, device):
    """[M, M] 1 - cosine between the adapter updates B A, averaged over the shared LoRA layers."""
    adapters = [load_adapter(p, device) for p in paths]
    names = sorted(set.intersection(*(set(a) for a in adapters)))
    gram = [{n: (a[n][1].T @ a[n][1], a[n][0] @ a[n][0].T) for n in names} for a in adapters]
    norms = [{n: torch.trace(g[n][0] @ g[n][1]).sqrt() for n in names} for g in gram]
    M = len(adapters)
    D = np.zeros((M, M))
    for i, j in itertools.combinations(range(M), 2):
        cos = []
        for n in names:
            (Ai, Bi), (Aj, Bj) = adapters[i][n], adapters[j][n]
            inner = torch.trace((Bi.T @ Bj) @ (Aj @ Ai.T))
            den = norms[i][n] * norms[j][n]
            cos.append((inner / den).item() if den > 0 else 0.0)
        D[i, j] = D[j, i] = 1.0 - float(np.mean(cos))
    return D, len(names)


def knn_distance_matrix(mats, k, device):
    nbrs = [knn_indices(x, k, device) for x in mats]
    M = len(nbrs)
    D = np.zeros((M, M))
    for i, j in itertools.combinations(range(M), 2):
        D[i, j] = D[j, i] = 1.0 - knn_jaccard(nbrs[i], nbrs[j])
    return D


# ----------------------------------------------------------------------------- Q2 test
def crossed_bootstrap_ratio(D, inits, seeds, n_boot=5000, rng=None):
    """
    Ratio mean(init-only) / mean(seed-only) with a crossed (pigeonhole) bootstrap: inits and
    seeds are resampled independently with replacement; member (a, s) gets weight
    w_init[a] * w_seed[s] and a pair the product of its members' weights. Draws in which one of
    the two pair groups gets zero weight are skipped.
    Returns ratio, 95% CI, one-sided p = P*(ratio <= 1) and the number of valid draws.
    """
    rng = rng or np.random.default_rng(0)
    ui, ii = np.unique(inits, return_inverse=True)
    us, si = np.unique(seeds, return_inverse=True)
    g = pair_groups(inits, seeds)
    so, io = np.array(g['seed_only']), np.array(g['init_only'])
    d_so, d_io = D[so[:, 0], so[:, 1]], D[io[:, 0], io[:, 1]]
    ratio = d_io.mean() / d_so.mean()
    boots = []
    for _ in range(n_boot):
        w = rng.multinomial(len(ui), np.full(len(ui), 1 / len(ui)))[ii] * \
            rng.multinomial(len(us), np.full(len(us), 1 / len(us)))[si]
        w_so, w_io = w[so[:, 0]] * w[so[:, 1]], w[io[:, 0]] * w[io[:, 1]]
        if w_so.sum() == 0 or w_io.sum() == 0:
            continue
        den = w_so @ d_so / w_so.sum()
        if den > 0:
            boots.append((w_io @ d_io / w_io.sum()) / den)
    boots = np.array(boots)
    return {'ratio': float(ratio), 'ci_low': float(np.percentile(boots, 2.5)),
            'ci_high': float(np.percentile(boots, 97.5)),
            'boot_p': float((np.sum(boots <= 1.0) + 1) / (len(boots) + 1)), 'n_boot_valid': int(len(boots))}


# ----------------------------------------------------------------------------- main
def flag_collapsed(pool, gnn, drop_pp):
    """Members whose GNN validation accuracy is more than drop_pp below the pool median.
    (Validation, not test: flagged members can be excluded from the analysis, i.e. this is a selection step.)"""
    med = float(np.median(gnn['val_acc']))
    return [pool.labels[i] for i, a in enumerate(gnn['val_acc']) if a < med - drop_pp]


def main(args):
    pool = ControlledPool(args.dataset_name, args.llm_name, args.peft_type, parse_list(args.approaches),
                          parse_list(args.seeds, int), args.split_seed, args.init_seed,
                          gnn_model_name=args.gnn_model_name, out_dir=args.out_dir)
    report = TextReport(os.path.join(args.out_dir, f'{pool.tag}_report.txt'),
                        f'Q2: initialization vs training noise | {pool.dataset_name} | {pool.llm_name}')
    for line in pool.describe():
        report.line(line)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    report.section('Member GNN test accuracy')
    for lbl, acc in zip(pool.labels, gnn['test_acc']):
        report.line(f'  {lbl:<16}{acc:6.2f}')
    collapsed = flag_collapsed(pool, gnn, args.collapse_pp)
    if collapsed:
        report.line(f'  COLLAPSED (> {args.collapse_pp} pp below the median): {", ".join(collapsed)}')
        if args.exclude_collapsed:
            keep = [i for i, l in enumerate(pool.labels) if l not in collapsed]
            gnn = {**gnn, 'probs': gnn['probs'][keep], 'preds': gnn['preds'][keep],
                   'emb': [gnn['emb'][i] for i in keep], 'test_acc': gnn['test_acc'][keep],
                   'val_acc': gnn['val_acc'][keep], 'test_masks': gnn['test_masks'][keep]}
            pool.drop(collapsed)
            report.line('  -> excluded from every analysis below')

    # ---- distance matrix of every stage (same member order everywhere)
    mats = {}
    have_adapters = all(os.path.exists(os.path.join(m['adapter'], 'adapter_model.safetensors'))
                        for m in pool.members)
    if have_adapters:
        mats['adapter_update'], n_layers = adapter_update_distance([m['adapter'] for m in pool.members], device)
        report.line(f'\nAdapter stage: {n_layers} LoRA layers')
    else:
        report.line('\nAdapter stage skipped: not every member has adapter_model.safetensors')
    mats['llm_embedding_cka'] = cached_matrix(pool, 'llm_cka_dist',
                                              lambda: cka_distance_matrix(pool.load_llm_embeddings()))
    mats['gnn_embedding_cka'] = cached_matrix(pool, 'gnn_cka_dist', lambda: cka_distance_matrix(gnn['emb']),
                                              recompute=args.retrain_gnn)
    mats['gnn_prediction_disagreement'] = disagreement_matrix(gnn['preds'], gnn['test_mask'])
    if args.knn:
        mats['llm_knn'] = cached_matrix(pool, f'llm_knn{args.k}_dist',
                                        lambda: knn_distance_matrix(pool.load_llm_embeddings(), args.k, device))
        mats['gnn_knn'] = cached_matrix(pool, f'gnn_knn{args.k}_dist',
                                        lambda: knn_distance_matrix(gnn['emb'], args.k, device),
                                        recompute=args.retrain_gnn)

    # ---- Q1 / Q2 per stage
    info = dict(run_info(pool), split_seed=pool.split_seed, init_seed=pool.init_seed,
                excluded=','.join(collapsed) if args.exclude_collapsed else '')
    rng = np.random.default_rng(0)
    rows = []
    report.section('Q1 / Q2 per stage (distance: larger = more diverse)')
    for stage in [s for s in STAGES + SECONDARY if s in mats]:
        D = mats[stage]
        wb = within_between_test(D, pool.inits, pool.seeds, n_perm=args.n_perm, rng=rng)
        bs = crossed_bootstrap_ratio(D, pool.inits, pool.seeds, n_boot=args.n_boot, rng=rng)
        pi = permanova(D, pool.inits, n_perm=args.n_perm, rng=rng)
        ps = permanova(D, pool.seeds, n_perm=args.n_perm, rng=rng)
        q2_pass = bs['ratio'] > 1 and bs['boot_p'] < 0.05
        report.within_between(f"{stage}{' (secondary)' if stage in SECONDARY else ''}", wb)
        report.line(f"  Q2 crossed bootstrap: ratio {bs['ratio']:.3f} [95% CI {bs['ci_low']:.3f}, "
                    f"{bs['ci_high']:.3f}] p(ratio<=1)={bs['boot_p']:.4g} ({bs['n_boot_valid']} draws) "
                    f"-> {'PASS' if q2_pass else 'fail'}")
        report.line(f"  PERMANOVA R2 init {pi['R2']:.3f} (p={pi['p_value']:.3g}) | "
                    f"seed {ps['R2']:.3f} (p={ps['p_value']:.3g})")
        rows.append(dict(info, stage=stage, primary=stage in STAGES, **TextReport.within_between_row(wb),
                         q2_ratio=bs['ratio'], q2_ci_low=bs['ci_low'], q2_ci_high=bs['ci_high'],
                         q2_boot_p=bs['boot_p'], q2_pass=q2_pass, q1_pass=wb['init_effect_p'] < 0.05,
                         permanova_init_R2=pi['R2'], permanova_init_p=pi['p_value'],
                         permanova_seed_R2=ps['R2'], permanova_seed_p=ps['p_value']))
        save_matrix_csv(D, pool.labels, os.path.join(args.out_dir, 'matrices_csv', f'{pool.tag}_{stage}.csv'))
        if not args.no_plots:
            save_pool_heatmap(D, pool.labels, pool.inits,
                              os.path.join(args.out_dir, 'heatmaps', f'{pool.tag}_{stage}.png'),
                              title=f'{stage}\n{pool.dataset_name} / {pool.llm_name} (controlled pool)',
                              cbar_label='distance')

    append_rows(rows, os.path.join(args.out_dir, 'q2_summary.csv'))
    report.line(f'\nReport: {report.path}')


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Q2: init-only vs seed-only diversity on a controlled pool '
                                            '(fixed split and LoRA init draw, only the training seed varies)')
    p.add_argument('--dataset_name', type=str, default='cora')
    p.add_argument('--llm_name', type=str, default='gemma2_2B')
    p.add_argument('--peft_type', type=str, default='lora')
    p.add_argument('--approaches', type=str, default=','.join(APPROACHES))
    p.add_argument('--seeds', type=str, default='42,43,44,45', help='Training seeds of the pool')
    p.add_argument('--split_seed', type=int, default=42, help='Fine-tuning AND GNN split of every member')
    p.add_argument('--init_seed', type=int, default=0, help='LoRA init seed shared by every member')
    p.add_argument('--gnn_model_name', type=str, default=None)
    p.add_argument('--retrain_gnn', action='store_true')
    p.add_argument('--knn', action='store_true', help='Also run the secondary k-NN distances')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--collapse_pp', type=float, default=15.0,
                   help='Flag members whose GNN test accuracy is this many pp below the pool median')
    p.add_argument('--exclude_collapsed', action='store_true', help='Drop flagged members from the analysis')
    p.add_argument('--n_perm', type=int, default=2000)
    p.add_argument('--n_boot', type=int, default=5000)
    p.add_argument('--no_plots', action='store_true')
    p.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    main(p.parse_args())
