# Experiment #5 (R2-C5): does the node-conditioned MLP ROUTE each node to the right view?
"""
Averaging the MLP's per-node weights over all nodes (what a single weight table
shows) hides what makes it node-conditioned. On the test nodes we instead ask:

  1) Correct vs wrong: for every view, the mean weight it receives on nodes
     where that view is correct vs nodes where it is wrong (Mann-Whitney U,
     one-sided: correct > wrong). A positive gap = the MLP trusts a view where
     that view is actually right, which averaging cannot do.
  2) Confidence: Spearman correlation between a view's per-node weight and its
     own confidence (max softmax) and margin (top-1 minus top-2 probability).
  3) Routing accuracy: on "contested" nodes (views disagree and at least one is
     correct) - is the highest-weighted view correct? Compared with picking a
     view at random (mean share of correct views) and with max-confidence
     picking.
  4) Weight concentration: normalised entropy of the per-node weight vector
     (1 = uniform, 0 = all weight on one view) on unanimous vs contested nodes.

Outputs (under <out_dir>/exp5_routing/):
  exp5_view_rows.csv      one row per (run, view): weights correct/wrong, p-values, correlations
  exp5_run_rows.csv       one row per run: routing accuracy, entropy
  <tag>_correct_vs_wrong.png, <tag>_entropy.png
  <tag>_report.txt
"""
import argparse
import os

import numpy as np
import torch
from scipy import stats

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, evaluate, view_correct, agreement_count, exp_dir,
    mean_std, plt_module, save_fig, TextReport, append_rows, VIEW_COLORS,
)


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp5_routing')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp5 - node-level routing of the MLP ensemble | {vs.dataset_name} | {vs.llm_name}')
    hp = fusion_hp(args)
    titles = vs.titles
    M = len(titles)

    view_rows, run_rows = [], []
    pooled_w = {t: {'correct': [], 'wrong': []} for t in titles}
    pooled_ent = {'unanimous': [], 'contested': []}
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y, test = views['logits'], views['y'], idx['test']
        res = fuse('mlp', L, y, idx, seed=fuse_seed, hp=hp)
        Wn = res['node_weights'][test].numpy()                  # [n_test, M]
        ok = view_correct(L, y)[:, test].numpy()                # [M, n_test]
        P = torch.softmax(L[:, test], dim=2)
        top2 = P.topk(2, dim=2).values
        conf = top2[..., 0].numpy()
        margin = (top2[..., 0] - top2[..., 1]).numpy()
        info = {**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed}

        for j, t in enumerate(titles):
            wc, ww = Wn[ok[j], j], Wn[~ok[j], j]
            p = stats.mannwhitneyu(wc, ww, alternative='greater').pvalue if len(wc) and len(ww) else float('nan')
            pooled_w[t]['correct'].append(wc)
            pooled_w[t]['wrong'].append(ww)
            view_rows.append({**info, 'view': t, 'view_test_acc': float(ok[j].mean()),
                              'weight_when_correct': float(wc.mean()) if len(wc) else float('nan'),
                              'weight_when_wrong': float(ww.mean()) if len(ww) else float('nan'),
                              'mwu_p_correct_gt_wrong': float(p),
                              'spearman_weight_confidence': float(stats.spearmanr(Wn[:, j], conf[j])[0]),
                              'spearman_weight_margin': float(stats.spearmanr(Wn[:, j], margin[j])[0])})

        agree = agreement_count(L)[test].numpy()
        any_ok, all_ok = ok.any(0), ok.all(0)
        contested = any_ok & ~all_ok
        top_view = Wn.argmax(1)
        routed_ok = ok[top_view, np.arange(len(top_view))]
        maxconf_ok = ok[conf.argmax(0), np.arange(len(top_view))]
        ent = -(Wn * np.log(np.clip(Wn, 1e-12, None))).sum(1) / np.log(M)
        pooled_ent['unanimous'].append(ent[agree == M])
        pooled_ent['contested'].append(ent[contested])
        run_rows.append({**info, **evaluate(res, y, idx), 'n_contested': int(contested.sum()),
                         'routing_acc_contested': float(routed_ok[contested].mean()) if contested.any() else float('nan'),
                         'random_view_acc_contested': float(ok[:, contested].mean()) if contested.any() else float('nan'),
                         'max_confidence_acc_contested': float(maxconf_ok[contested].mean()) if contested.any()
                         else float('nan'),
                         'entropy_unanimous': float(ent[agree == M].mean()) if (agree == M).any() else float('nan'),
                         'entropy_contested': float(ent[contested].mean()) if contested.any() else float('nan'),
                         'entropy_all': float(ent.mean())})

    append_rows(view_rows, out_file(args, out_dir, 'exp5_view_rows.csv'))
    append_rows(run_rows, out_file(args, out_dir, 'exp5_run_rows.csv'))

    report.section('Per view: mean MLP weight on nodes where the view is correct vs wrong (mean over runs)')
    for t in titles:
        vr = [r for r in view_rows if r['view'] == t]
        wc = mean_std([r['weight_when_correct'] for r in vr])
        ww = mean_std([r['weight_when_wrong'] for r in vr])
        n_sig = sum(r['mwu_p_correct_gt_wrong'] < 0.05 for r in vr)
        rc = mean_std([r['spearman_weight_confidence'] for r in vr])
        rm = mean_std([r['spearman_weight_margin'] for r in vr])
        report.line(f'  {t:<11} correct={wc[0]:.4f}  wrong={ww[0]:.4f}  diff={wc[0] - ww[0]:+.4f}  '
                    f'(MWU p<0.05 in {n_sig}/{len(vr)} runs) | rho(weight, confidence)={rc[0]:+.3f} '
                    f'rho(weight, margin)={rm[0]:+.3f}')

    report.section('Routing on contested nodes (views disagree, at least one correct)')
    for key, label in (('routing_acc_contested', 'highest-weighted view correct'),
                       ('max_confidence_acc_contested', 'most confident view correct'),
                       ('random_view_acc_contested', 'random view correct (baseline)')):
        mu, sd = mean_std([r[key] for r in run_rows])
        report.line(f'  {label:<34} {mu * 100:.2f} +/- {sd * 100:.2f}')
    report.line(f"  contested nodes per run: {np.mean([r['n_contested'] for r in run_rows]):.1f}")
    for key in ('entropy_unanimous', 'entropy_contested', 'entropy_all'):
        mu, sd = mean_std([r[key] for r in run_rows])
        report.line(f'  normalised weight entropy ({key[8:]}): {mu:.3f} +/- {sd:.3f}  (1 = uniform)')

    # ---- figures
    plt = plt_module()
    plt.figure(figsize=(8, 4.2))
    x = np.arange(M)
    for j, (t, a) in enumerate(zip(titles, vs.approaches)):
        wc = np.concatenate(pooled_w[t]['correct']) if pooled_w[t]['correct'] else np.array([])
        ww = np.concatenate(pooled_w[t]['wrong']) if pooled_w[t]['wrong'] else np.array([])
        parts = plt.violinplot([wc if len(wc) else [np.nan], ww if len(ww) else [np.nan]],
                               positions=[j - 0.18, j + 0.18], widths=0.32, showmeans=True, showextrema=False)
        for body, alpha in zip(parts['bodies'], (0.85, 0.3)):
            body.set_facecolor(VIEW_COLORS[a])
            body.set_edgecolor(VIEW_COLORS[a])
            body.set_alpha(alpha)
        parts['cmeans'].set_color('#0b0b0b')
    plt.axhline(1 / M, color='#898781', ls='--', lw=1)
    plt.xticks(x, titles, rotation=20)
    plt.ylabel('Per-node MLP weight on the view')
    plt.title(f'{vs.dataset_name.upper()}: weight when the view is correct (solid) vs wrong (faded)')
    save_fig(plt, os.path.join(out_dir, f'{tag}_correct_vs_wrong.png'))

    plt.figure(figsize=(6.5, 4))
    bins = np.linspace(0, 1, 31)
    for key, color in (('unanimous', '#86b6ef'), ('contested', '#eb6834')):
        vals = np.concatenate(pooled_ent[key]) if pooled_ent[key] else np.array([])
        if len(vals):
            plt.hist(vals, bins=bins, density=True, alpha=0.6, color=color, edgecolor='white', label=f'{key} nodes')
    plt.xlabel('Normalised entropy of per-node weights (1 = uniform)')
    plt.ylabel('Density')
    plt.title(f'{vs.dataset_name.upper()}: how concentrated are the MLP weights?')
    plt.legend(fontsize=8)
    save_fig(plt, os.path.join(out_dir, f'{tag}_entropy.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    main(p.parse_args())
