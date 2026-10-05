# Experiment #8 (R2-C5): how many LABELS does the fusion head need, and does it overfit?
"""
The extra parameters of the adaptive ensemble are fit on labelled nodes. We
subsample the fit nodes (the train nodes under 'paper' / 'train_fit', the fit
half of the validation set under 'val_split') to --fractions of their size, repeat --repeats times
per fraction, and refit every head. The selection nodes and test nodes are
kept fixed.

Reported per fraction and head:
  - test accuracy (and gain over uniform averaging, which uses no labels)
  - fit accuracy minus test accuracy (generalisation gap -> overfitting)
  - number of labelled nodes and number of parameters (labels per parameter)

Outputs (under <out_dir>/exp8_label_efficiency/):
  exp8_rows.csv             one row per (run, fraction, repeat, method)
  exp8_summary.csv          one row per (dataset, fraction, method)
  <tag>_curve.png           test accuracy vs #fit labels (log x), with uniform as reference
  <tag>_gap.png             fit - test accuracy vs #fit labels
  <tag>_report.txt
"""
import argparse
import os

import numpy as np
import torch

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, evaluate, exp_dir, mean_std, plt_module, save_fig,
    TextReport, append_rows, parse_list, METHOD_COLORS, METHOD_TITLES,
)


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp8_label_efficiency')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp8 - label efficiency of the fusion head | {vs.dataset_name} | {vs.llm_name}')
    methods = parse_list(args.methods)
    fractions = parse_list(args.fractions, float)
    hp = fusion_hp(args)

    rows = []
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y = views['logits'], views['y']
        uni_acc = evaluate(fuse('uniform_logits', L, y, idx), y, idx)['test_acc']
        n_fit_full = len(idx['fit'])
        for frac in fractions:
            reps = 1 if frac >= 1.0 else args.repeats
            for rep in range(reps):
                g = torch.Generator().manual_seed(1000 * fuse_seed + rep)
                k = max(len(torch.unique(y[idx['fit']])), int(round(frac * n_fit_full)))
                sub = idx['fit'][torch.randperm(n_fit_full, generator=g)[:k]].sort().values
                sidx = {**idx, 'fit': sub}
                for m in methods:
                    res = fuse(m, L, y, sidx, seed=fuse_seed * 100 + rep, hp=hp)
                    ev = evaluate(res, y, sidx)
                    rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed, 'fraction': frac,
                                 'repeat': rep, 'n_fit_labels': int(k), 'method': m, 'n_params': res['n_params'],
                                 'labels_per_param': k / max(res['n_params'], 1), **ev,
                                 'fit_acc': res.get('fit_acc', float('nan')),
                                 'generalisation_gap': res.get('fit_acc', float('nan')) - ev['test_acc'],
                                 'uniform_test_acc': uni_acc, 'gain_over_uniform': ev['test_acc'] - uni_acc})

    append_rows(rows, out_file(args, out_dir, 'exp8_rows.csv'))

    summary = []
    report.section('Test accuracy / gain over uniform / fit-test gap vs number of fit labels (mean over runs)')
    for frac in fractions:
        fr = [r for r in rows if r['fraction'] == frac]
        n_lab = np.mean([r['n_fit_labels'] for r in fr])
        report.line(f'  fraction={frac:g} (~{n_lab:.0f} labelled nodes)')
        for m in methods:
            mr = [r for r in fr if r['method'] == m]
            acc = mean_std([r['test_acc'] for r in mr])
            gain = mean_std([r['gain_over_uniform'] for r in mr])
            gap = mean_std([r['generalisation_gap'] for r in mr])
            report.line(f'    {METHOD_TITLES[m]:<30} acc={acc[0] * 100:.2f}+/-{acc[1] * 100:.2f}  '
                        f'gain={gain[0] * 100:+.2f}  fit-test gap={gap[0] * 100:+.2f}  '
                        f'#params={mr[0]["n_params"]}  labels/param={np.mean([r["labels_per_param"] for r in mr]):.1f}')
            summary.append({**vs.run_info(), 'fraction': frac, 'n_fit_labels_mean': n_lab, 'method': m,
                            'n_params': mr[0]['n_params'], 'test_acc_mean': acc[0], 'test_acc_std': acc[1],
                            'gain_mean': gain[0], 'gain_std': gain[1], 'gap_mean': gap[0], 'gap_std': gap[1]})
    uni = mean_std([r['uniform_test_acc'] for r in rows if r['method'] == methods[0] and r['fraction'] == fractions[0]])
    report.line(f'  Uniform averaging (no labels): {uni[0] * 100:.2f}+/-{uni[1] * 100:.2f}')
    append_rows(summary, out_file(args, out_dir, 'exp8_summary.csv'))

    # ---- figures
    plt = plt_module()
    for key, ylabel, fname, ref in (('test_acc_mean', 'Test accuracy (%)', 'curve', uni[0] * 100),
                                    ('gap_mean', 'Fit accuracy - test accuracy (pp)', 'gap', 0.0)):
        plt.figure(figsize=(6.5, 4.2))
        for m in methods:
            s = [r for r in summary if r['method'] == m]
            xs = [r['n_fit_labels_mean'] for r in s]
            ys = [r[key] * 100 for r in s]
            err = [r[key.replace('_mean', '_std')] * 100 for r in s]
            plt.errorbar(xs, ys, yerr=err, marker='o', ms=6, lw=2, capsize=3, label=METHOD_TITLES[m],
                         color=METHOD_COLORS.get(m))
        plt.axhline(ref, color='#898781', ls='--', lw=1,
                    label='Uniform averaging (no labels)' if fname == 'curve' else None)
        plt.xscale('log')
        plt.xlabel('Labelled nodes used to fit the fusion head (log)')
        plt.ylabel(ylabel)
        plt.title(f'{vs.dataset_name.upper()}: fusion head vs label budget')
        plt.legend(fontsize=8)
        save_fig(plt, os.path.join(out_dir, f'{tag}_{fname}.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='scalar,per_class,mlp,stacking_lr')
    p.add_argument('--fractions', type=str, default='0.05,0.1,0.25,0.5,1.0',
                   help='Fractions of the fit nodes used to train the fusion head')
    p.add_argument('--repeats', type=int, default=3, help='Random subsamples per fraction (< 1.0)')
    main(p.parse_args())
