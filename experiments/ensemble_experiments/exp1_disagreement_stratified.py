# Experiment #1 (R2-C5): WHERE does the adaptive ensemble beat simple averaging?
"""
Test nodes are bucketed by how many of the M views agree on the plurality label
(agreement = M means unanimous, 1 means every view predicts a different label).
For every bucket we report the accuracy of uniform averaging, of each learned
fusion head (scalar / per-class / node-MLP) and of the oracle (a node counts as
correct if ANY view is correct).

Expected story: on unanimous nodes the rules are identical (a convex combination
of logits - uniform, scalar or node-MLP weights - cannot overturn a unanimous
vote; only per-class weights can, and rarely do); the adaptive gain
concentrates on disagreement nodes.

The headline number is the share of the uniform -> oracle gap that the learned
head recovers:
    gap_recovered = (acc_adaptive - acc_uniform) / (acc_oracle - acc_uniform)

Outputs (under <out_dir>/exp1_disagreement/):
  exp1_bucket_rows.csv      one row per (run, bucket, method)
  exp1_summary.csv          one row per (dataset, method): overall + gap recovered, mean/std over runs
  <tag>_buckets.png         grouped bars per bucket (mean over runs) + node share per bucket
  <tag>_report.txt
"""
import argparse
import os

import numpy as np
import torch

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, oracle_correct, agreement_count, exp_dir, mean_std,
    plt_module, save_fig, TextReport, append_rows, parse_list, METHOD_COLORS, METHOD_TITLES,
)


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp1_disagreement')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp1 - disagreement-stratified gain | {vs.dataset_name} | {vs.llm_name}')
    methods = ['uniform_logits'] + parse_list(args.methods)
    hp = fusion_hp(args)

    bucket_rows, summary = [], {m: {'acc': [], 'gap': []} for m in methods}
    summary['oracle'] = {'acc': [], 'gap': []}
    per_bucket = {}  # (bucket, method) -> list of acc
    bucket_share = {}

    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y, test = views['logits'], views['y'], idx['test']
        M = L.shape[0]
        agree = agreement_count(L)[test]
        oracle = oracle_correct(L, y)[test]
        correct = {'oracle': oracle}
        for m in methods:
            res = fuse(m, L, y, idx, seed=fuse_seed, hp=hp)
            correct[m] = (res['pred'][test] == y[test])

        acc_u = correct['uniform_logits'].float().mean().item()
        acc_o = oracle.float().mean().item()
        for m, c in correct.items():
            acc = c.float().mean().item()
            gap = (acc - acc_u) / (acc_o - acc_u) if acc_o > acc_u else float('nan')
            summary[m]['acc'].append(acc)
            summary[m]['gap'].append(gap)

        for b in range(1, M + 1):
            mask = agree == b
            n_b = int(mask.sum())
            bucket_share.setdefault(b, []).append(n_b / len(test))
            for m, c in correct.items():
                acc_b = c[mask].float().mean().item() if n_b else float('nan')
                per_bucket.setdefault((b, m), []).append(acc_b)
                bucket_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed,
                                    'agreement': f'{b}/{M}', 'n_nodes': n_b, 'share': n_b / len(test),
                                    'method': m, 'accuracy': acc_b})

    M = len(vs.approaches)
    report.section('Accuracy per agreement bucket (mean over runs)')
    header = f"{'agreement':>10} {'share':>7} " + ' '.join(f'{m[:14]:>14}' for m in ['oracle'] + methods)
    report.line(header)
    for b in range(M, 0, -1):
        cells = ' '.join(f'{np.nanmean(per_bucket[(b, m)]) * 100:>14.2f}' for m in ['oracle'] + methods)
        report.line(f'{b:>8}/{M} {np.mean(bucket_share[b]) * 100:>6.1f}% {cells}')

    report.section('Overall test accuracy and share of the uniform->oracle gap recovered')
    rows = []
    for m in ['oracle'] + methods:
        a_mu, a_sd = mean_std(summary[m]['acc'])
        g_mu, g_sd = mean_std(summary[m]['gap'])
        report.line(f'  {METHOD_TITLES.get(m, m):<30} acc={a_mu * 100:.2f}+/-{a_sd * 100:.2f}  '
                    f'gap recovered={g_mu * 100:+.1f}%+/-{g_sd * 100:.1f}')
        dis = [x for b in range(1, M) for x in per_bucket[(b, m)] if not np.isnan(x)]
        rows.append({**vs.run_info(), 'method': m, 'n_runs': len(summary[m]['acc']), 'test_acc_mean': a_mu,
                     'test_acc_std': a_sd, 'gap_recovered_mean': g_mu, 'gap_recovered_std': g_sd,
                     'unanimous_acc_mean': float(np.nanmean(per_bucket[(M, m)])),
                     'disagreement_acc_mean': float(np.mean(dis)) if dis else float('nan'),
                     'unanimous_share': float(np.mean(bucket_share[M]))})
    report.line('Scalar and node-MLP weights are convex combinations, so they can only change predictions on '
                'nodes where the views disagree; the gain over uniform should therefore sit in buckets < M/M.')

    append_rows(bucket_rows, out_file(args, out_dir, 'exp1_bucket_rows.csv'))
    append_rows(rows, out_file(args, out_dir, 'exp1_summary.csv'))

    # ---- figure: accuracy per bucket (bars) + node share per bucket (separate panel, no dual axis)
    plt = plt_module()
    buckets = list(range(M, 0, -1))
    shown = ['uniform_logits'] + [m for m in methods if m != 'uniform_logits'] + ['oracle']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    x = np.arange(len(buckets))
    width = 0.8 / len(shown)
    for k, m in enumerate(shown):
        vals = [np.nanmean(per_bucket[(b, m)]) * 100 for b in buckets]
        ax1.bar(x + (k - (len(shown) - 1) / 2) * width, vals, width * 0.92, label=METHOD_TITLES.get(m, 'Oracle'),
                color=METHOD_COLORS.get(m, '#52514e'), edgecolor='white', linewidth=0.8)
    ax1.set_ylabel('Test accuracy (%)')
    ax1.set_title(f'{vs.dataset_name.upper()}: accuracy by view agreement')
    ax1.legend(fontsize=8, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.07))
    ax1.set_ylim(0, 105)
    ax2.bar(x, [np.mean(bucket_share[b]) * 100 for b in buckets], 0.5, color='#86b6ef', edgecolor='white')
    ax2.set_ylabel('% of test nodes')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{b}/{M}' for b in buckets])
    ax2.set_xlabel('Views agreeing on the plurality label')
    save_fig(plt, os.path.join(out_dir, f'{tag}_buckets.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='scalar,per_class,mlp',
                   help='Learned fusion heads to compare against uniform averaging')
    main(p.parse_args())
