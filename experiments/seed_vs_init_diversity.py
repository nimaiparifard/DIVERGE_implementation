# Experiment #1 (R1-C2 / R2-C1): is the diversity between LoRA inits larger than
# the diversity produced by random training variance (different seeds)?
"""
For one (dataset, llm), build the pool of init x seed members (e.g. 5 x 4 = 20)
and compute three pool-level diversity matrices (as distances):

  1) LLM embedding    : 1 - linear CKA between the cached LoRA embeddings
  2) GNN embedding    : 1 - linear CKA between the GNN penultimate layers
                        (one GNN per member, fixed GNN seed/split)
  3) GNN predictions  : label disagreement on the test nodes

For each matrix:
  - seed-only pairs (same init, different seed) -> pure seed / training noise
    (the LoRA seed sets the classifier-head init, data order, dropout and the
     random A matrix of gaussian/orthogonal)
  - init-only pairs (different init, same seed) -> pure initialization effect
  - init effect exists: among cross-seed pairs, are different-init pairs
    farther apart than same-init pairs? (permutation test, init labels
    shuffled within each seed)
  - init-only / seed-only ratio: is the init effect larger than seed noise?
  - PERMANOVA R^2 for the init factor and for the seed factor
    ("init explains X% of the representation variance, seed Y%")
  - heatmap sorted by init: block structure = init dominates seed

Outputs (under --out_dir, default results/diversity/seed_vs_init/):
  seed_vs_init_summary.csv   one row per (run, level)
  <tag>_<level>_matrix.csv   full pool matrix
  <tag>_<level>_heatmap.png
  <tag>_report.txt
"""
import os

import numpy as np

from experiments.diversity_common import (
    add_common_args, pool_from_args, cka_distance_matrix, disagreement_matrix, cached_matrix, pair_node_counts,
    within_between_test, permanova, save_pool_heatmap, save_matrix_csv, run_info, append_rows, TextReport,
)


def main(args):
    pool = pool_from_args(args)
    out_dir = os.path.join(args.out_dir, 'seed_vs_init')
    report = TextReport(os.path.join(out_dir, f'{pool.tag}_report.txt'),
                        f'Seed noise vs. initialization diversity | {pool.dataset_name} | {pool.llm_name}')
    for line in pool.describe():
        report.line(line)

    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    test_masks = gnn['test_masks']

    levels = {
        'llm_embedding_cka': (
            lambda: cached_matrix(pool, 'llm_cka_dist', lambda: cka_distance_matrix(pool.load_llm_embeddings()),
                                  recompute=args.recompute),
            '1 - CKA (LLM embeddings)'),
        'gnn_embedding_cka': (
            lambda: cached_matrix(pool, 'gnn_cka_dist', lambda: cka_distance_matrix(gnn['emb']),
                                  recompute=args.recompute or args.retrain_gnn),
            '1 - CKA (GNN embeddings, before logits)'),
        'gnn_prediction_disagreement': (
            lambda: disagreement_matrix(gnn['preds'], test_masks),
            'Prediction disagreement (test nodes)'),
    }

    report.section('Individual GNN test accuracy (' + ('own split, own test nodes' if pool.own_split
                                                      else 'fixed GNN seed') + ')')
    for lbl, acc in zip(pool.labels, gnn['test_acc']):
        report.line(f'  {lbl:<16} {acc:.2f}')
    if pool.own_split:
        cnt = pair_node_counts(test_masks)
        same = pool.seeds[:, None] == pool.seeds[None, :]
        off = ~np.eye(len(pool), dtype=bool)
        report.line(f'Shared test nodes per pair: same seed {cnt[same & off].mean():.0f}, '
                    f'different seed {cnt[~same].mean():.0f} (min {cnt[~same].min()})')

    rows = []
    rng = np.random.default_rng(0)
    for level, (fn, cbar) in levels.items():
        D = fn()
        wb = within_between_test(D, pool.inits, pool.seeds, n_perm=args.n_perm, rng=rng)
        pm_init = permanova(D, pool.inits, n_perm=args.n_perm, rng=rng)
        pm_seed = permanova(D, pool.seeds, n_perm=args.n_perm, rng=rng)

        report.section(level)
        report.within_between('Diversity', wb)
        report.line(f"PERMANOVA init: R2={pm_init['R2']:.3f} F={pm_init['F']:.2f} p={pm_init['p_value']:.4g} | "
                    f"seed: R2={pm_seed['R2']:.3f} F={pm_seed['F']:.2f} p={pm_seed['p_value']:.4g}")

        # per-init seed noise (how stable each init is across seeds)
        for init in dict.fromkeys(pool.inits):
            idx = np.where(pool.inits == init)[0]
            if len(idx) > 1:
                sub = D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
                report.line(f'  within {init:<11}: mean {sub.mean():.4f} (max {sub.max():.4f})')

        save_matrix_csv(D, pool.labels, os.path.join(out_dir, f'{pool.tag}_{level}_matrix.csv'))
        if not args.no_plots:
            save_pool_heatmap(D, pool.labels, pool.inits, os.path.join(out_dir, f'{pool.tag}_{level}_heatmap.png'),
                              title=f'{cbar}\n{pool.dataset_name} / {pool.llm_name}', cbar_label=cbar)

        rows.append(dict(
            run_info(pool), level=level, **TextReport.within_between_row(wb),
            permanova_init_R2=pm_init['R2'], permanova_init_F=pm_init['F'], permanova_init_p=pm_init['p_value'],
            permanova_seed_R2=pm_seed['R2'], permanova_seed_F=pm_seed['F'], permanova_seed_p=pm_seed['p_value'],
        ))

    append_rows(rows, os.path.join(out_dir, 'seed_vs_init_summary.csv'))
    report.line(f'\nReport: {report.path}')
    return rows


if __name__ == '__main__':
    import argparse

    parser = add_common_args(argparse.ArgumentParser(
        description='Within-init (seed) vs between-init diversity of LLM embeddings, GNN embeddings and predictions'))
    parser.add_argument('--recompute', action='store_true', help='Recompute cached CKA matrices')
    parser.add_argument('--no_plots', action='store_true')
    main(parser.parse_args())
