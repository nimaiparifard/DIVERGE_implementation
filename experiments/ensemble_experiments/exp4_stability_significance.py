# Experiment #4 (R2-C5, also R4-C5): are the learned weights STABLE and is the gain SIGNIFICANT?
"""
Repeated over gnn_seeds x fuse_seeds (GNN retraining noise x fusion-head
split/initialisation noise). For every learned head (scalar / per-class /
node-MLP) and every run we record the per-view weights and the test gain over
uniform-logit averaging.

Stability of the weights
  - mean +/- std and coefficient of variation of each view's weight
  - Kendall's W (concordance of the view ranking across runs; 1 = identical
    ranking in every run, 0 = no agreement)
  - variance decomposition: share of the weight variance explained by the GNN
    seed (between-gnn-seed variance / total)

Significance of the gain over uniform averaging
  - per run: exact McNemar test on the test nodes (discordant pairs) and a
    node-bootstrap 95% CI of the accuracy gain
  - across runs: paired t-test and Wilcoxon signed-rank test on the per-run
    gains, both on all runs and on GNN-seed means (the conservative version:
    runs sharing a GNN seed are not independent)
  - Holm-Bonferroni correction over the learned heads

Outputs (under <out_dir>/exp4_stability/):
  exp4_weight_rows.csv    one row per (run, method, view)
  exp4_run_rows.csv       one row per (run, method): gain, McNemar p, bootstrap CI
  exp4_summary.csv        one row per (dataset, method)
  <tag>_weights.png       per-view weight distribution across runs, one panel per head
  <tag>_gains.png         per-run gain over uniform, one strip per head
  <tag>_report.txt
"""
import argparse
import os

import numpy as np
from scipy import stats

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, evaluate, exp_dir, mean_std, plt_module, save_fig,
    TextReport, append_rows, parse_list, VIEW_COLORS, METHOD_COLORS, METHOD_TITLES,
)


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired boolean correctness vectors."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(p)


def bootstrap_gain_ci(correct_a, correct_b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    diff = correct_a.astype(float) - correct_b.astype(float)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def kendalls_w(weight_matrix):
    """weight_matrix [runs, M] -> Kendall's coefficient of concordance of the view rankings."""
    R = np.array([stats.rankdata(-w) for w in weight_matrix])
    m, n = R.shape
    if m < 2 or n < 2:
        return float('nan')
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp4_stability')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp4 - weight stability and significance | {vs.dataset_name} | {vs.llm_name}')
    methods = parse_list(args.methods)
    hp = fusion_hp(args)
    titles = vs.titles

    weight_rows, run_rows = [], []
    W = {m: [] for m in methods}          # per method: list of [M] weight vectors
    W_gnn = {m: [] for m in methods}      # matching gnn seed per run
    gains = {m: [] for m in methods}
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y, test = views['logits'], views['y'], idx['test']
        uni = fuse('uniform_logits', L, y, idx)
        uni_ok = (uni['pred'][test] == y[test]).numpy()
        uni_acc = uni_ok.mean()
        for m in methods:
            res = fuse(m, L, y, idx, seed=fuse_seed, hp=hp)
            ok = (res['pred'][test] == y[test]).numpy()
            b, c, p = mcnemar_exact(ok, uni_ok)
            lo, hi = bootstrap_gain_ci(ok, uni_ok, n_boot=args.n_boot, seed=fuse_seed)
            gain = float(ok.mean() - uni_acc)
            gains[m].append((gnn_seed, gain))
            W[m].append(np.asarray(res['view_weights'], dtype=float))
            W_gnn[m].append(gnn_seed)
            run_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed, 'method': m,
                             'uniform_test_acc': float(uni_acc), **evaluate(res, y, idx), 'gain': gain,
                             'mcnemar_adaptive_only_correct': b, 'mcnemar_uniform_only_correct': c,
                             'mcnemar_p': p, 'gain_ci_low': lo, 'gain_ci_high': hi})
            for t, w in zip(titles, res['view_weights']):
                weight_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed, 'method': m,
                                    'view': t, 'weight': float(w)})

    append_rows(weight_rows, out_file(args, out_dir, 'exp4_weight_rows.csv'))
    append_rows(run_rows, out_file(args, out_dir, 'exp4_run_rows.csv'))

    # ---- stability
    summary = {}
    for m in methods:
        Wm = np.array(W[m])
        g = np.array(W_gnn[m])
        report.section(f'{METHOD_TITLES[m]} - weights over {len(Wm)} runs')
        total_var = Wm.var(0)
        between = np.array([np.var([Wm[g == s, j].mean() for s in np.unique(g)]) for j in range(Wm.shape[1])])
        for j, t in enumerate(titles):
            mu, sd = Wm[:, j].mean(), Wm[:, j].std()
            share = between[j] / total_var[j] if total_var[j] > 0 else float('nan')
            report.line(f'  {t:<11} {mu:.4f} +/- {sd:.4f}  CV={sd / mu if mu else float("nan"):.3f}  '
                        f'var share from GNN seed={share:.2f}')
        kw = kendalls_w(Wm)
        report.line(f"  Kendall's W (ranking concordance across runs) = {kw:.3f}")
        summary[m] = {'kendalls_w': kw, 'mean_weight_std': float(Wm.std(0).mean()),
                      'max_weight_minus_min': float((Wm.mean(0).max() - Wm.mean(0).min()))}

    # ---- significance
    report.section('Gain over uniform-logit averaging')
    p_t, p_w, p_t_g, p_w_g = [], [], [], []
    for m in methods:
        gv = np.array([x for _, x in gains[m]])
        gs = np.array([s for s, _ in gains[m]])
        gm = np.array([gv[gs == s].mean() for s in np.unique(gs)])
        pt = stats.ttest_1samp(gv, 0.0).pvalue if len(gv) > 1 and gv.std() > 0 else float('nan')
        pw = stats.wilcoxon(gv).pvalue if np.any(gv != 0) else 1.0
        ptg = stats.ttest_1samp(gm, 0.0).pvalue if len(gm) > 1 and gm.std() > 0 else float('nan')
        pwg = stats.wilcoxon(gm).pvalue if len(gm) > 1 and np.any(gm != 0) else float('nan')
        p_t.append(pt), p_w.append(pw), p_t_g.append(ptg), p_w_g.append(pwg)
        mr = [r for r in run_rows if r['method'] == m]
        n_sig = sum(r['mcnemar_p'] < 0.05 and r['gain'] > 0 for r in mr)
        summary[m].update({'n_runs': len(gv), 'gain_mean': float(gv.mean()), 'gain_std': float(gv.std()),
                           'runs_positive': int((gv > 0).sum()), 'runs_mcnemar_sig_positive': n_sig,
                           'ttest_p': pt, 'wilcoxon_p': pw, 'ttest_p_gnnseed_means': ptg,
                           'wilcoxon_p_gnnseed_means': pwg})

    def safe_holm(ps):
        ps = np.array(ps, dtype=float)
        out = np.full(len(ps), np.nan)
        ok = ~np.isnan(ps)
        if ok.any():
            out[ok] = holm(ps[ok])
        return out

    for key, ps in (('ttest_p_holm', p_t), ('wilcoxon_p_holm', p_w), ('ttest_p_gnnseed_means_holm', p_t_g),
                    ('wilcoxon_p_gnnseed_means_holm', p_w_g)):
        for m, p in zip(methods, safe_holm(ps)):
            summary[m][key] = float(p)

    rows = []
    for m in methods:
        s = summary[m]
        report.line(f"  {METHOD_TITLES[m]:<30} gain={s['gain_mean'] * 100:+.2f}+/-{s['gain_std'] * 100:.2f} pp  "
                    f"positive in {s['runs_positive']}/{s['n_runs']} runs, McNemar-significant in "
                    f"{s['runs_mcnemar_sig_positive']}  | t-test p={s['ttest_p']:.3g} (Holm {s['ttest_p_holm']:.3g})  "
                    f"Wilcoxon p={s['wilcoxon_p']:.3g} (Holm {s['wilcoxon_p_holm']:.3g})  "
                    f"| on GNN-seed means: t p={s['ttest_p_gnnseed_means']:.3g}")
        rows.append({**vs.run_info(), 'method': m, **s})
    report.line('Runs that share a GNN seed are not independent; the GNN-seed-mean tests are the conservative ones.')
    append_rows(rows, out_file(args, out_dir, 'exp4_summary.csv'))

    # ---- figures
    plt = plt_module()
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[0], methods):
        Wm = np.array(W[m])
        for j, a in enumerate(vs.approaches):
            jitter = rng.uniform(-0.15, 0.15, len(Wm))
            ax.scatter(np.full(len(Wm), j) + jitter, Wm[:, j], s=40, color=VIEW_COLORS[a], edgecolor='white',
                       linewidth=0.8, alpha=0.9)
            ax.hlines(Wm[:, j].mean(), j - 0.3, j + 0.3, color='#0b0b0b', lw=2)
        ax.axhline(1 / len(titles), color='#898781', ls='--', lw=1)
        ax.set_xticks(range(len(titles)))
        ax.set_xticklabels(titles, rotation=30, ha='right', fontsize=8)
        ax.set_title(f"{METHOD_TITLES[m]}\nKendall's W={summary[m]['kendalls_w']:.2f}", fontsize=9)
    axes[0][0].set_ylabel('Learned weight (dashed = uniform)')
    save_fig(plt, os.path.join(out_dir, f'{tag}_weights.png'))

    plt.figure(figsize=(1.8 * len(methods) + 2, 4))
    for i, m in enumerate(methods):
        gv = np.array([x for _, x in gains[m]]) * 100
        plt.scatter(np.full(len(gv), i) + rng.uniform(-0.15, 0.15, len(gv)), gv, s=40,
                    color=METHOD_COLORS.get(m), edgecolor='white', linewidth=0.8)
        plt.hlines(gv.mean(), i - 0.3, i + 0.3, color='#0b0b0b', lw=2)
    plt.axhline(0, color='#898781', lw=1)
    plt.xticks(range(len(methods)), [METHOD_TITLES[m] for m in methods], rotation=20, ha='right', fontsize=8)
    plt.ylabel('Test accuracy gain over uniform (pp)')
    plt.title(f'{vs.dataset_name.upper()}: per-run gain ({len(run_rows) // len(methods)} runs)')
    save_fig(plt, os.path.join(out_dir, f'{tag}_gains.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='scalar,per_class,mlp')
    p.add_argument('--n_boot', type=int, default=1000, help='Node-bootstrap resamples for the per-run gain CI')
    main(p.parse_args())
