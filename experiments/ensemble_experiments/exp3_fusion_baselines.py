# Experiment #3 (R2-C5): is LEARNING needed, and what do the extra parameters buy?
"""
The same M view GNNs are fused with every rule in ensemble_common.ALL_METHODS:

  parameter-free   uniform logits, uniform probabilities, majority vote,
                   max confidence, entropy-weighted averaging
  label-using, no gradient training
                   validation-accuracy-weighted averaging, per-view temperature
                   scaling + uniform probabilities
  learned          DIVERGE scalar weights, DIVERGE per-class weights,
                   DIVERGE node-conditioned MLP, stacking (logistic regression)

For each rule: number of extra parameters, test accuracy / macro-F1
(mean +/- std over gnn_seeds x fuse_seeds), gain over uniform-logit averaging,
and paired win/tie/loss counts vs uniform across runs.

--compare_protocols runs the same table under other protocols as well
(default: val_split,train_fit next to the main --protocol). No protocol uses
the test nodes for fitting or selection; the report ends with a "fit-node
effect" section: accuracy under 'train_fit' (head fit on train-mask logits)
minus accuracy under 'val_split' (head fit on half of the val nodes).

Outputs (under <out_dir>/exp3_fusion_baselines/):
  exp3_run_rows.csv        one row per (run, method, protocol)
  exp3_summary.csv         one row per (dataset, method, protocol)
  <tag>_table.tex          LaTeX table body (acc / F1 / #params)
  <tag>_bars.png           accuracy per method (mean +/- std), grouped by family
  <tag>_report.txt
"""
import argparse
import copy
import os

import numpy as np

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, evaluate, exp_dir, mean_std, plt_module, save_fig,
    TextReport, append_rows, parse_list, ALL_METHODS, PARAM_FREE, LABEL_USING, LEARNED, METHOD_TITLES,
)

FAMILY_COLORS = {'param-free': '#898781', 'label-using': '#eda100', 'learned': '#2a78d6'}


def family(m):
    return 'param-free' if m in PARAM_FREE else 'label-using' if m in LABEL_USING else 'learned'


def run_protocol(vs, args, methods, hp, protocol):
    a = copy.copy(args)
    a.protocol = protocol
    rows = []
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, a):
        y = views['y']
        best_single = float(views['test_acc'][int(np.argmax(views['val_acc']))])
        for m in methods:
            res = fuse(m, views['logits'], y, idx, seed=fuse_seed, hp=hp)
            rows.append({**vs.run_info(), 'protocol': protocol, 'selection_on_test': False,
                         'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed,
                         'method': m, 'family': family(m), 'n_params': res['n_params'],
                         'fit_acc': res.get('fit_acc', float('nan')), 'sel_acc': res.get('sel_acc', float('nan')),
                         'best_epoch': res.get('best_epoch', ''), 'tuned_hp': str(res.get('tuned_hp', '')),
                         'best_single_view_test_acc': best_single, **evaluate(res, y, idx)})
    return rows


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp3_fusion_baselines')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp3 - fusion rules vs learned ensemble | {vs.dataset_name} | {vs.llm_name}')
    methods = [m for m in ALL_METHODS if m in parse_list(args.methods)] if args.methods != 'all' else ALL_METHODS
    if 'uniform_logits' not in methods:
        methods = ['uniform_logits'] + methods
    hp = fusion_hp(args)

    protocols = [args.protocol] + [p for p in parse_list(args.compare_protocols) if p != args.protocol]
    all_rows, summary_rows = [], []
    for protocol in protocols:
        rows = run_protocol(vs, args, methods, hp, protocol)
        all_rows += rows
        runs = sorted({(r['gnn_seed'], r['fuse_seed']) for r in rows})
        uni = {(r['gnn_seed'], r['fuse_seed']): r['test_acc'] for r in rows if r['method'] == 'uniform_logits'}

        report.section(f'Protocol = {protocol}  ({len(runs)} runs)')
        report.line(f"{'method':<30} {'family':<12} {'#params':>8} {'test acc':>14} {'macro-F1':>14} "
                    f"{'gain':>8} {'W/T/L':>9}")
        for m in methods:
            mr = [r for r in rows if r['method'] == m]
            acc_mu, acc_sd = mean_std([r['test_acc'] for r in mr])
            f1_mu, f1_sd = mean_std([r['test_macro_f1'] for r in mr])
            gains = [r['test_acc'] - uni[(r['gnn_seed'], r['fuse_seed'])] for r in mr]
            w = sum(g > 1e-9 for g in gains)
            l = sum(g < -1e-9 for g in gains)
            t = len(gains) - w - l
            n_params = int(np.median([r['n_params'] for r in mr]))
            report.line(f'{METHOD_TITLES[m]:<30} {family(m):<12} {n_params:>8} {acc_mu * 100:>7.2f}+/-{acc_sd * 100:<5.2f} '
                        f'{f1_mu * 100:>7.2f}+/-{f1_sd * 100:<5.2f} {np.mean(gains) * 100:>+8.2f} {w:>3}/{t}/{l}')
            summary_rows.append({**vs.run_info(), 'protocol': protocol, 'selection_on_test': False,
                                 'method': m, 'family': family(m),
                                 'n_params': n_params, 'n_runs': len(mr), 'test_acc_mean': acc_mu,
                                 'test_acc_std': acc_sd, 'test_macro_f1_mean': f1_mu, 'test_macro_f1_std': f1_sd,
                                 'gain_over_uniform_mean': float(np.mean(gains)), 'wins': w, 'ties': t, 'losses': l,
                                 'fit_acc_mean': mean_std([r['fit_acc'] for r in mr])[0]})
        bs = mean_std([r['best_single_view_test_acc'] for r in rows if r['method'] == 'uniform_logits'])
        report.line(f"{'Best single view (by val)':<30} {'':<12} {'':>8} {bs[0] * 100:>7.2f}+/-{bs[1] * 100:<5.2f}")

    # ---- effect of the fit nodes (train-mask logits vs held-out val logits); selection is on val in both
    protos = {r['protocol'] for r in summary_rows}
    fit_train = 'train_fit' if 'train_fit' in protos else 'paper' if 'paper' in protos else None
    if fit_train and 'val_split' in protos:
        report.section(f"Fit-node effect: acc('{fit_train}') - acc('val_split'), pp (no test-node selection in either)")
        acc = {(r['protocol'], r['method']): r['test_acc_mean'] for r in summary_rows}
        for m in methods:
            report.line(f"  {METHOD_TITLES[m]:<30} {(acc[(fit_train, m)] - acc[('val_split', m)]) * 100:+.2f}")

    append_rows(all_rows, out_file(args, out_dir, 'exp3_run_rows.csv'))
    append_rows(summary_rows, out_file(args, out_dir, 'exp3_summary.csv'))

    # ---- LaTeX table body for the main protocol
    main_rows = [r for r in summary_rows if r['protocol'] == args.protocol]
    tex = os.path.join(out_dir, f'{tag}_table.tex')
    with open(tex, 'w', encoding='utf-8') as f:
        f.write(f'% {vs.dataset_name} | protocol={args.protocol} | mean $\\pm$ std over {main_rows[0]["n_runs"]} runs\n')
        f.write('% Method & Family & #Params & Acc & Macro-F1 \\\\\n')
        for r in main_rows:
            f.write(f"{METHOD_TITLES[r['method']]} & {r['family']} & {r['n_params']} & "
                    f"{r['test_acc_mean'] * 100:.2f} $\\pm$ {r['test_acc_std'] * 100:.2f} & "
                    f"{r['test_macro_f1_mean'] * 100:.2f} $\\pm$ {r['test_macro_f1_std'] * 100:.2f} \\\\\n")
    print(f'[OK] LaTeX rows written to: {tex}')

    # ---- figure (main protocol)
    plt = plt_module()
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(main_rows))
    for i, r in enumerate(main_rows):
        plt.bar(i, r['test_acc_mean'] * 100, 0.7, yerr=r['test_acc_std'] * 100, capsize=3,
                color=FAMILY_COLORS[r['family']], edgecolor='white', linewidth=0.8)
    for fam, c in FAMILY_COLORS.items():
        plt.bar([], [], color=c, label=fam)
    lo = min(r['test_acc_mean'] for r in main_rows) * 100
    hi = max(r['test_acc_mean'] + r['test_acc_std'] for r in main_rows) * 100
    plt.ylim(max(0, lo - 3), min(100, hi + 1))
    plt.xticks(x, [METHOD_TITLES[r['method']] for r in main_rows], rotation=35, ha='right', fontsize=8)
    plt.ylabel('Test accuracy (%)')
    plt.title(f'{vs.dataset_name.upper()}: fusion rules on the same {len(vs.approaches)} view GNNs')
    plt.legend(fontsize=8)
    save_fig(plt, os.path.join(out_dir, f'{tag}_bars.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='all', help=f"'all' or comma-separated subset of {ALL_METHODS}")
    p.add_argument('--compare_protocols', type=str, default='val_split,train_fit',
                   help="Extra protocols to run next to --protocol (comma-separated; '' = none)")
    main(p.parse_args())
