# Experiment #2 (R1-C2): are the differences between LoRA inits *systematic*
# (reproducible across seeds) or just random training variance?
"""
If an init gives a genuinely different "semantic view", its mistakes should be
reproducible: the same init should fail on the same nodes and the same classes
across LoRA seeds, while a different init fails on different nodes/classes.
Pure noise would give no such structure.

Uses the shared GNN pool (one GNN per init x seed, fixed GNN seed/split):

  a) Error-set distance: 1 - Jaccard(errors_i, errors_j) on test nodes,
     seed-only vs init-only pairs + permutation test (same test as #1).
  b) Error persistence per init: of the test nodes an init gets wrong in at
     least one seed, the share it gets wrong in *every* seed.
  c) Systematic complementarity: nodes that init A gets wrong in every seed
     while init B gets right in every seed (A -> B "rescued" nodes).
  d) Per-class accuracy, two-way ANOVA  init x class  with the seeds as
     replicates (balanced init x seed grid). The residual term is exactly
     the seed noise, so a significant init x class interaction means the
     inits have class-specific strengths that exceed seed variance.

Own-split mode (--gnn_seed -1): a) compares each pair on the test nodes shared by
both members, d) uses every member's own test nodes, and b)/c) (which follow the
same nodes across all seeds) run on the nodes that are test nodes for every
member of the grid, and are skipped when there are fewer than --min_eval_nodes.

Outputs (under --out_dir, default results/diversity/error_consistency/):
  error_consistency_summary.csv      one row per run
  error_consistency_per_class.csv    per (run, init, class) mean/std accuracy
  error_consistency_rescue.csv       per (run, init A, init B) rescued nodes
  <tag>_error_jaccard_heatmap.png, <tag>_per_class_heatmap.png, <tag>_report.txt
"""
import os

import numpy as np
from scipy.stats import f as f_dist

from experiments.cka import TITLES
from experiments.diversity_common import (
    add_common_args, pool_from_args, within_between_test, save_pool_heatmap, save_matrix_csv, run_info,
    append_rows, TextReport,
)


def error_jaccard_distance(correct, masks):
    """
    [M, M] 1 - Jaccard overlap of the error sets. `correct` [M, n] bool per node,
    `masks` [M, n] test nodes per member; a pair is compared on masks[i] & masks[j].
    """
    err = ~correct
    M = err.shape[0]
    D = np.zeros((M, M))
    for i in range(M):
        m = masks[i] & masks                                       # [M, n]
        inter = (err[i] & err & m).sum(axis=1)
        union = ((err[i] | err) & m).sum(axis=1)
        D[i] = 1.0 - np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return D


def two_way_anova(Y):
    """Balanced two-way ANOVA with replication. Y[a, b, n]: factor A (init), factor B (class), n replicates (seeds)."""
    a, b, n = Y.shape
    gm = Y.mean()
    mA, mB, mAB = Y.mean(axis=(1, 2)), Y.mean(axis=(0, 2)), Y.mean(axis=2)
    ss = {
        'init': b * n * ((mA - gm) ** 2).sum(),
        'class': a * n * ((mB - gm) ** 2).sum(),
        'init_x_class': n * ((mAB - mA[:, None] - mB[None, :] + gm) ** 2).sum(),
        'residual_seed': ((Y - mAB[:, :, None]) ** 2).sum(),
    }
    df = {'init': a - 1, 'class': b - 1, 'init_x_class': (a - 1) * (b - 1), 'residual_seed': a * b * (n - 1)}
    ms_res = ss['residual_seed'] / df['residual_seed'] if df['residual_seed'] > 0 else float('nan')
    out = {}
    for k in ('init', 'class', 'init_x_class'):
        F = (ss[k] / df[k]) / ms_res if ms_res and ms_res > 0 else float('nan')
        out[k] = {'SS': ss[k], 'df': df[k], 'F': F,
                  'p': float(f_dist.sf(F, df[k], df['residual_seed'])) if np.isfinite(F) else float('nan'),
                  'partial_eta2': ss[k] / (ss[k] + ss['residual_seed']) if ss[k] + ss['residual_seed'] > 0 else 0.0}
    out['residual_seed'] = {'SS': ss['residual_seed'], 'df': df['residual_seed']}
    return out


def main(args):
    pool = pool_from_args(args)
    out_dir = os.path.join(args.out_dir, 'error_consistency')
    report = TextReport(os.path.join(out_dir, f'{pool.tag}_report.txt'),
                        f'Error consistency across seeds | {pool.dataset_name} | {pool.llm_name}')
    for line in pool.describe():
        report.line(line)

    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    y_all = gnn['y'].numpy()
    test_masks = gnn['test_masks'].numpy()                          # [M, n] own test nodes
    correct_all = gnn['preds'].numpy() == y_all[None, :]            # [M, n]
    info = run_info(pool)

    # a) error-set distance
    D = error_jaccard_distance(correct_all, test_masks)
    wb = within_between_test(D, pool.inits, pool.seeds, n_perm=args.n_perm)
    report.section('a) Error-set distance (1 - Jaccard of misclassified test nodes)')
    report.within_between('Error-set distance', wb)
    save_matrix_csv(D, pool.labels, os.path.join(out_dir, f'{pool.tag}_error_jaccard_matrix.csv'))
    if not args.no_plots:
        save_pool_heatmap(D, pool.labels, pool.inits, os.path.join(out_dir, f'{pool.tag}_error_jaccard_heatmap.png'),
                          title=f'Error-set distance (1 - Jaccard)\n{pool.dataset_name} / {pool.llm_name}',
                          cbar_label='1 - Jaccard')

    # b) + c) need every init with the same seeds
    idx, inits, seeds = pool.balanced_subset()
    report.section(f'Balanced grid for b)-d): {len(inits)} inits x {len(seeds)} seeds {seeds}')
    if len(seeds) < 2:
        report.line('Need >= 2 common seeds for the persistence / ANOVA analyses; stopping here.')
        append_rows([dict(info, **TextReport.within_between_row(wb, 'error_jaccard_'))],
                    os.path.join(out_dir, 'error_consistency_summary.csv'))
        return
    # b) + c) follow the same nodes across seeds: nodes that are test nodes for the whole grid
    test = test_masks[idx].all(axis=0)
    n_test = int(test.sum())
    correct = correct_all[:, test]
    grid = {init: np.stack([correct[pool.labels.index(f'{TITLES[init]}-{s}')] for s in seeds]) for init in inits}
    run_bc = not (pool.own_split and n_test < args.min_eval_nodes)
    persistence, rescue_rows = {}, []
    persistent_any = persistent_shared = np.zeros(0, dtype=bool)
    if pool.own_split:
        report.line(f'Own split: b) and c) use the {n_test} nodes that are test nodes for every grid member'
                    + ('' if run_bc else f' (< {args.min_eval_nodes}: skipped)'))

    if run_bc:
        persistence, rescue_rows, persistent_any, persistent_shared = _persistence_and_rescues(
            report, grid, inits, info, n_test)

    # d) per-class two-way ANOVA (each member on its own test nodes)
    classes = np.unique(y_all[test_masks.any(axis=0)])
    Y = np.zeros((len(inits), len(classes), len(seeds)))
    for ai, init in enumerate(inits):
        for si, s in enumerate(seeds):
            k = pool.labels.index(f'{TITLES[init]}-{s}')
            for ci, c in enumerate(classes):
                m = test_masks[k] & (y_all == c)
                Y[ai, ci, si] = correct_all[k, m].mean() * 100.0 if m.any() else np.nan
    anova = two_way_anova(Y)
    report.section('d) Per-class test accuracy, two-way ANOVA (init x class, seeds = replicates)')
    for k in ('init', 'class', 'init_x_class'):
        r = anova[k]
        report.line(f"  {k:<13} F({r['df']},{anova['residual_seed']['df']})={r['F']:.2f} p={r['p']:.4g} "
                    f"partial eta2={r['partial_eta2']:.3f}")
    report.line('')
    report.line(f"{'class':<8}" + ''.join(f'{TITLES[i]:>16}' for i in inits) + '   best (wins/seeds)')
    per_class_rows = []
    for ci, c in enumerate(classes):
        means, stds = Y[:, ci, :].mean(axis=1), Y[:, ci, :].std(axis=1)
        best = int(np.argmax(means))
        wins = int((Y[:, ci, :].argmax(axis=0) == best).sum())
        report.line(f'{int(c):<8}' + ''.join(f'{m:>10.1f}+/-{s:<4.1f}' for m, s in zip(means, stds)) +
                    f'   {TITLES[inits[best]]} ({wins}/{len(seeds)})')
        n_c = int(np.mean([(test_masks[k] & (y_all == c)).sum() for k in idx]))
        for ai, init in enumerate(inits):
            per_class_rows.append(dict(info, init=TITLES[init], class_id=int(c), n_test_nodes=n_c,
                                       acc_mean=means[ai], acc_std=stds[ai], is_best_init=ai == best,
                                       best_wins_over_seeds=wins if ai == best else ''))
    if not args.no_plots:
        mean_mat = Y.mean(axis=2) - Y.mean(axis=(0, 2))[None, :]  # deviation from the class mean
        _per_class_plot(mean_mat, inits, classes, os.path.join(out_dir, f'{pool.tag}_per_class_heatmap.png'), pool)

    summary = dict(info, n_test_nodes=n_test, balanced_seeds=','.join(map(str, seeds)),
                   **TextReport.within_between_row(wb, 'error_jaccard_'),
                   **{f'persistence_{TITLES[i]}': persistence[i] for i in persistence},
                   persistent_errors_any_init=int(persistent_any.sum()) if run_bc else '',
                   persistent_errors_all_inits=int(persistent_shared.sum()) if run_bc else '',
                   anova_init_F=anova['init']['F'], anova_init_p=anova['init']['p'],
                   anova_class_F=anova['class']['F'], anova_class_p=anova['class']['p'],
                   anova_interaction_F=anova['init_x_class']['F'], anova_interaction_p=anova['init_x_class']['p'],
                   anova_interaction_partial_eta2=anova['init_x_class']['partial_eta2'])
    append_rows([summary], os.path.join(out_dir, 'error_consistency_summary.csv'))
    append_rows(per_class_rows, os.path.join(out_dir, 'error_consistency_per_class.csv'))
    if rescue_rows:
        append_rows(rescue_rows, os.path.join(out_dir, 'error_consistency_rescue.csv'))
    report.line(f'\nReport: {report.path}')


def _persistence_and_rescues(report, grid, inits, info, n_test):
    report.section('b) Error persistence (wrong in every seed / wrong in at least one seed)')
    persistence = {}
    for init in inits:
        always_wrong = (~grid[init]).all(axis=0)
        ever_wrong = (~grid[init]).any(axis=0)
        persistence[init] = always_wrong.sum() / max(ever_wrong.sum(), 1)
        report.line(f'  {TITLES[init]:<11} persistent {always_wrong.sum():>5} / ever-wrong {ever_wrong.sum():>5} '
                    f'= {persistence[init]:.3f}')

    report.section('c) Systematic rescues: nodes init A always gets wrong but init B always gets right')
    rescue_rows = []
    report.line('A (always wrong) / B (always right)'.ljust(36) + ''.join(f'{TITLES[b]:>12}' for b in inits))
    for a in inits:
        a_wrong = (~grid[a]).all(axis=0)
        cells = []
        for b in inits:
            n = int((a_wrong & grid[b].all(axis=0)).sum()) if a != b else 0
            cells.append(n)
            if a != b:
                rescue_rows.append(dict(info, init_a=TITLES[a], init_b=TITLES[b], a_always_wrong=int(a_wrong.sum()),
                                        rescued_by_b=n, rescued_share_of_a_errors=n / max(int(a_wrong.sum()), 1),
                                        rescued_share_of_test=n / n_test))
        report.line(f'{TITLES[a]:<36}' + ''.join(f'{c:>12}' for c in cells))
    all_wrong = np.stack([(~grid[i]).all(axis=0) for i in inits])
    persistent_any = all_wrong.any(axis=0)
    persistent_shared = all_wrong.all(axis=0)
    report.line(f'Nodes persistently wrong for at least one init: {persistent_any.sum()} | for every init: '
                f'{persistent_shared.sum()} -> {persistent_any.sum() - persistent_shared.sum()} persistent errors are '
                f'init-specific (fixable by another view)')

    return persistence, rescue_rows, persistent_any, persistent_shared


def _per_class_plot(mean_mat, inits, classes, save_path, pool):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lim = max(np.abs(mean_mat).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(0.6 * len(classes) + 3, 0.5 * len(inits) + 2))
    im = ax.imshow(mean_mat, cmap='RdBu', vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(int(c)) for c in classes])
    ax.set_yticks(range(len(inits)))
    ax.set_yticklabels([TITLES[i] for i in inits])
    for i in range(mean_mat.shape[0]):
        for j in range(mean_mat.shape[1]):
            ax.text(j, i, f'{mean_mat[i, j]:+.1f}', ha='center', va='center', fontsize=7)
    ax.set_xlabel('class')
    ax.set_title(f'Per-class accuracy vs. class mean (avg over seeds, pp)\n{pool.dataset_name} / {pool.llm_name}',
                 fontsize=10)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] heatmap saved to: {save_path}')


if __name__ == '__main__':
    import argparse

    parser = add_common_args(argparse.ArgumentParser(
        description='Are LoRA-init differences systematic across seeds? (error overlap, rescues, per-class ANOVA)'))
    parser.add_argument('--no_plots', action='store_true')
    main(parser.parse_args())
