# Q2 claim across pools: is init-only diversity > seed-only diversity in general, per stage?
"""
Reads q2_summary.csv written by init_vs_training_noise.py (one row per pool x stage; the last
row of a pool/stage wins if a pool was rerun) and tests, per stage, over the pools:

  sign test   H1: P(ratio > 1) > 0.5            (binomial, one-sided)
  Wilcoxon    H1: median log(ratio) > 0         (signed-rank, one-sided)

Decision rule (fixed before the runs): a stage is claimed only if BOTH tests have p < 0.05.
With 10 pools the sign test needs >= 9 pools with ratio > 1 (p = 0.011; 8 of 10 gives 0.055).
Every pool is listed in the output, including the ones that fail.
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

from experiments.diversity_experiment.init_vs_training_noise import DEFAULT_OUT_DIR, STAGES, SECONDARY


def load_rows(path):
    latest = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            latest[(r['dataset_name'], r['llm_name'], r['split_seed'], r['init_seed'], r['stage'])] = r
    return list(latest.values())


def main(args):
    rows = load_rows(os.path.join(args.out_dir, 'q2_summary.csv'))
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r['stage']].append(r)

    lines = [f'Q2 across pools ({args.out_dir}/q2_summary.csv)', '']
    out_rows = []
    for stage in [s for s in STAGES + SECONDARY if s in by_stage]:
        rs = sorted(by_stage[stage], key=lambda r: (r['dataset_name'], r['llm_name']))
        ratios = np.array([float(r['q2_ratio']) for r in rs])
        n, k = len(ratios), int(np.sum(ratios > 1))
        sign_p = binomtest(k, n, 0.5, alternative='greater').pvalue
        wil_p = wilcoxon(np.log(ratios), alternative='greater').pvalue if n >= 2 else float('nan')
        claim = sign_p < 0.05 and wil_p < 0.05
        gmean = float(np.exp(np.log(ratios).mean()))
        lines.append(f"{stage}{' (secondary)' if stage in SECONDARY else ''}: {k}/{n} pools ratio > 1, "
                     f"geometric mean {gmean:.3f}, sign p={sign_p:.4g}, Wilcoxon p={wil_p:.4g} "
                     f"-> {'CLAIM' if claim else 'no claim'}")
        for r in rs:
            lines.append(f"    {r['dataset_name']:<10}{r['llm_name']:<16}ratio {float(r['q2_ratio']):.3f} "
                         f"[{float(r['q2_ci_low']):.3f}, {float(r['q2_ci_high']):.3f}]  "
                         f"pool p={float(r['q2_boot_p']):.3g}  Q1 p={float(r['init_effect_p']):.3g}")
        out_rows.append({'stage': stage, 'n_pools': n, 'n_ratio_gt_1': k, 'geo_mean_ratio': gmean,
                         'sign_p': sign_p, 'wilcoxon_p': wil_p, 'claim': claim})

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.out_dir, 'q2_across_pools.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(os.path.join(args.out_dir, 'q2_across_pools.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Sign test + Wilcoxon of the Q2 ratio over all controlled pools')
    p.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    main(p.parse_args())
