# Experiment #9 (R2-C5): what does each view CONTRIBUTE, and do the learned weights reflect it?
"""
For every non-empty subset S of the M views we refit the ensemble on S only and
measure its test accuracy v(S) (v(empty) = accuracy of predicting the most
frequent fit-node class). From v we compute, for each fusion rule:

  - leave-one-view-out drop:   v(all) - v(all minus view i)
  - exact Shapley value:       phi_i = sum_S |S|!(M-|S|-1)!/M! [v(S + i) - v(S)]
    (sum_i phi_i = v(all) - v(empty); 2^M - 1 = 31 subsets for M = 5)

and compare with the learned scalar weights and with the views' standalone
accuracy (Spearman). If the learned weight tracks the Shapley contribution
rather than the standalone accuracy, the head rewards *complementary* views,
not just strong ones - which simple averaging cannot express.

Outputs (under <out_dir>/exp9_view_contribution/):
  exp9_subset_rows.csv     one row per (run, method, subset): v(S)
  exp9_view_rows.csv       one row per (run, method, view): LOO drop, Shapley, weight, standalone acc
  exp9_summary.csv         one row per (dataset, method): correlations
  <tag>_shapley.png        Shapley per view per rule | learned scalar weight per view
  <tag>_report.txt
"""
import argparse
import itertools
import math
import os

import numpy as np
from scipy import stats

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, evaluate, exp_dir, mean_std, plt_module, save_fig,
    TextReport, append_rows, parse_list, VIEW_COLORS, METHOD_COLORS, METHOD_TITLES,
)


def subset_value(method, L, y, idx, S, seed, hp):
    if len(S) == 0:
        fit_y = y[idx['fit']]
        majority = int(fit_y.mode().values)
        return float((y[idx['test']] == majority).float().mean())
    if len(S) == 1:  # every rule reduces to the single view
        return float((L[S[0]][idx['test']].argmax(1) == y[idx['test']]).float().mean())
    return evaluate(fuse(method, L[list(S)], y, idx, seed=seed, hp=hp), y, idx)['test_acc']


def shapley(values, M):
    phi = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for r in range(M):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                phi[i] += w * (values[tuple(sorted(S + (i,)))] - values[S])
    return phi


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp9_view_contribution')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp9 - leave-one-out and Shapley contribution of each view | {vs.dataset_name} | {vs.llm_name}')
    methods = ['uniform_logits'] + parse_list(args.methods)
    hp = fusion_hp(args)
    titles = vs.titles
    M = len(titles)

    subset_rows, view_rows = [], []
    PHI = {m: [] for m in methods}
    LOO = {m: [] for m in methods}
    WEIGHTS, ALONE = [], []
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y = views['logits'], views['y']
        info = {**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed}
        alone = np.array([float((L[j][idx['test']].argmax(1) == y[idx['test']]).float().mean()) for j in range(M)])
        scalar_w = np.asarray(fuse('scalar', L, y, idx, seed=fuse_seed, hp=hp)['view_weights'], dtype=float)
        WEIGHTS.append(scalar_w)
        ALONE.append(alone)
        for m in methods:
            values = {}
            for r in range(M + 1):
                for S in itertools.combinations(range(M), r):
                    values[S] = subset_value(m, L, y, idx, S, fuse_seed, hp)
                    subset_rows.append({**info, 'method': m, 'subset': '+'.join(titles[j] for j in S) or 'EMPTY',
                                        'size': r, 'test_acc': values[S]})
            full = tuple(range(M))
            phi = shapley(values, M)
            loo = np.array([values[full] - values[tuple(j for j in full if j != i)] for i in range(M)])
            PHI[m].append(phi)
            LOO[m].append(loo)
            for j, t in enumerate(titles):
                view_rows.append({**info, 'method': m, 'view': t, 'shapley': float(phi[j]), 'loo_drop': float(loo[j]),
                                  'standalone_test_acc': float(alone[j]), 'learned_scalar_weight': float(scalar_w[j]),
                                  'full_ensemble_acc': values[full]})

    append_rows(subset_rows, out_file(args, out_dir, 'exp9_subset_rows.csv'))
    append_rows(view_rows, out_file(args, out_dir, 'exp9_view_rows.csv'))

    W_mean, A_mean = np.mean(WEIGHTS, 0), np.mean(ALONE, 0)
    summary = []
    for m in methods:
        P, D = np.array(PHI[m]), np.array(LOO[m])
        report.section(f'{METHOD_TITLES[m]} (mean over {len(P)} runs)')
        report.line(f"{'view':<11} {'Shapley (pp)':>13} {'LOO drop (pp)':>14} {'alone acc':>10} {'scalar w':>9}")
        for j, t in enumerate(titles):
            report.line(f'{t:<11} {P[:, j].mean() * 100:>+13.2f} {D[:, j].mean() * 100:>+14.2f} '
                        f'{A_mean[j] * 100:>10.2f} {W_mean[j]:>9.3f}')
        rho_w_phi = [stats.spearmanr(w, p)[0] for w, p in zip(WEIGHTS, P)]
        rho_w_alone = [stats.spearmanr(w, a)[0] for w, a in zip(WEIGHTS, ALONE)]
        rho_phi_alone = [stats.spearmanr(p, a)[0] for p, a in zip(P, ALONE)]
        s = {'rho_scalar_weight_vs_shapley': float(np.nanmean(rho_w_phi)),
             'rho_scalar_weight_vs_standalone_acc': float(np.nanmean(rho_w_alone)),
             'rho_shapley_vs_standalone_acc': float(np.nanmean(rho_phi_alone)),
             'max_loo_drop': float(D.mean(0).max()), 'min_loo_drop': float(D.mean(0).min())}
        report.line(f"  Spearman: weight~Shapley={s['rho_scalar_weight_vs_shapley']:+.3f}  "
                    f"weight~standalone={s['rho_scalar_weight_vs_standalone_acc']:+.3f}  "
                    f"Shapley~standalone={s['rho_shapley_vs_standalone_acc']:+.3f}")
        summary.append({**vs.run_info(), 'method': m, 'n_runs': len(P), **s,
                        **{f'shapley_{t}': float(P[:, j].mean()) for j, t in enumerate(titles)},
                        **{f'loo_drop_{t}': float(D[:, j].mean()) for j, t in enumerate(titles)}})
    report.line('')
    report.line('A negative LOO drop means the ensemble is better WITHOUT that view; compare how often this '
                'happens under uniform averaging vs the learned heads.')
    append_rows(summary, out_file(args, out_dir, 'exp9_summary.csv'))

    # ---- figure
    plt = plt_module()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [2, 1]})
    x = np.arange(M)
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        P = np.array(PHI[m]) * 100
        ax1.bar(x + (k - (len(methods) - 1) / 2) * width, P.mean(0), width * 0.92, yerr=P.std(0), capsize=2,
                color=METHOD_COLORS.get(m), edgecolor='white', label=METHOD_TITLES[m])
    ax1.axhline(0, color='#898781', lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(titles, rotation=20)
    ax1.set_ylabel('Shapley contribution (pp of test accuracy)')
    ax1.set_title('Contribution of each view')
    ax1.legend(fontsize=8)
    ax2.bar(x, W_mean, 0.6, color=[VIEW_COLORS[a] for a in vs.approaches], edgecolor='white')
    ax2.axhline(1 / M, color='#898781', ls='--', lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(titles, rotation=20)
    ax2.set_ylabel('Learned scalar weight (dashed = uniform)')
    ax2.set_title('Learned scalar weight')
    save_fig(plt, os.path.join(out_dir, f'{tag}_shapley.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='scalar,per_class',
                   help='Learned rules to compute Shapley values for (each is refit on all 2^M - 1 subsets)')
    main(p.parse_args())
