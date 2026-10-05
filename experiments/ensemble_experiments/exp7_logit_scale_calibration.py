# Experiment #7 (R2-C5): is part of the "adaptive" gain simply RECALIBRATION?
"""
Uniform averaging of raw logits is not an equal vote: a view whose GNN outputs
larger (more confident) logits dominates the sum. A learned scalar weight can
partly act as an inverse temperature. To separate "learned which view to
trust" from "learned to undo logit-scale differences" we report:

Per view (test nodes)
  - logit scale: mean over nodes of the std of the class logits
  - effective share of the uniform-logit sum: ||centred logits_m|| / sum_k ||.||
  - mean confidence (max softmax), ECE, NLL, accuracy
  - fitted temperature T_m (NLL on the fit nodes)

Per ensemble rule: accuracy, ECE, NLL for
  uniform logits, uniform probabilities, temperature-scaled uniform
  probabilities, temperature-scaled uniform logits (mean of logits / T_m),
  and the learned scalar / per-class / node-MLP heads.

Headline: share of the learned heads' gain over uniform-logit averaging that a
per-view temperature alone already recovers:
    recovered = (acc_TS-uniform - acc_uniform) / (acc_learned - acc_uniform)
plus the Spearman correlation between the learned scalar weights and 1/T_m.

Outputs (under <out_dir>/exp7_calibration/):
  exp7_view_rows.csv, exp7_method_rows.csv, exp7_summary.csv
  <tag>_view_scale.png      per-view logit scale and effective share (two panels)
  <tag>_reliability.png     reliability diagrams of uniform / TS-uniform / learned scalar
  <tag>_report.txt
"""
import argparse
import os

import numpy as np
import torch
from scipy import stats

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, evaluate, fit_temperatures, ece, nll, exp_dir, mean_std,
    plt_module, save_fig, TextReport, append_rows, parse_list, VIEW_COLORS, METHOD_COLORS, METHOD_TITLES,
)

EXTRA_TITLES = {'temp_uniform_logits': 'Temp.-scaled uniform (logits)'}


def reliability(probs, y, n_bins=10):
    conf, pred = probs.max(1)
    ok = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    xs, accs, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean().item())
            accs.append(ok[m].mean().item())
            cnt.append(int(m.sum()))
    return np.array(xs), np.array(accs), np.array(cnt)


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp7_calibration')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp7 - logit scale, temperature and calibration | {vs.dataset_name} | {vs.llm_name}')
    hp = fusion_hp(args)
    learned = parse_list(args.methods)
    titles = vs.titles
    rules = ['uniform_logits', 'uniform_probs', 'temp_uniform_probs', 'temp_uniform_logits'] + learned

    view_rows, method_rows, rho_wT = [], [], []
    rel = {}
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y, test = views['logits'], views['y'], idx['test']
        M = L.shape[0]
        info = {**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed}
        T = fit_temperatures(L, y, idx['fit'])
        Lc = L[:, test] - L[:, test].mean(2, keepdim=True)
        norms = Lc.norm(dim=2)                                      # [M, n_test]
        share = (norms / norms.sum(0, keepdim=True)).mean(1).numpy()
        for j, t in enumerate(titles):
            P = torch.softmax(L[j, test], 1)
            view_rows.append({**info, 'view': t, 'logit_scale': float(L[j, test].std(1).mean()),
                              'uniform_effective_share': float(share[j]), 'temperature': float(T[j]),
                              'mean_confidence': float(P.max(1).values.mean()), 'ece': ece(P, y[test]),
                              'nll': nll(P, y[test]), 'test_acc': float((P.argmax(1) == y[test]).float().mean())})

        results = {}
        for r in rules:
            if r == 'temp_uniform_logits':
                probs = torch.softmax((L / torch.tensor(T, dtype=L.dtype).view(M, 1, 1)).mean(0), 1)
                res = {'probs': probs, 'pred': probs.argmax(1), 'n_params': M, 'view_weights': None}
            else:
                res = fuse(r, L, y, idx, seed=fuse_seed, hp=hp)
            results[r] = res
            method_rows.append({**info, 'method': r, 'n_params': res['n_params'], **evaluate(res, y, idx),
                                'ece': ece(res['probs'][test], y[test]), 'nll': nll(res['probs'][test], y[test])})
            if r in ('uniform_logits', 'temp_uniform_probs', 'scalar'):
                xs, accs, cnt = reliability(res['probs'][test], y[test])
                rel.setdefault(r, []).append((xs, accs, cnt))
        if 'scalar' in results and results['scalar']['view_weights'] is not None:
            rho_wT.append(stats.spearmanr(results['scalar']['view_weights'], 1.0 / T)[0])

    append_rows(view_rows, out_file(args, out_dir, 'exp7_view_rows.csv'))
    append_rows(method_rows, out_file(args, out_dir, 'exp7_method_rows.csv'))

    report.section('Per view (mean over runs)')
    report.line(f"{'view':<11} {'logit scale':>11} {'uni share':>9} {'T':>6} {'conf':>6} {'ECE':>6} {'NLL':>6} "
                f"{'acc':>6}")
    vagg = {}
    for t in titles:
        vr = [r for r in view_rows if r['view'] == t]
        a = {k: np.mean([r[k] for r in vr]) for k in ('logit_scale', 'uniform_effective_share', 'temperature',
                                                      'mean_confidence', 'ece', 'nll', 'test_acc')}
        vagg[t] = a
        report.line(f"{t:<11} {a['logit_scale']:>11.3f} {a['uniform_effective_share']:>9.3f} {a['temperature']:>6.2f} "
                    f"{a['mean_confidence']:>6.3f} {a['ece']:>6.3f} {a['nll']:>6.3f} {a['test_acc'] * 100:>6.2f}")
    report.line(f'An equal vote would give every view a share of {1 / len(titles):.3f}.')

    report.section('Ensemble rules (mean +/- std over runs)')
    magg = {}
    for r in rules:
        mr = [x for x in method_rows if x['method'] == r]
        magg[r] = {k: mean_std([x[k] for x in mr]) for k in ('test_acc', 'ece', 'nll')}
        name = METHOD_TITLES.get(r, EXTRA_TITLES.get(r, r))
        report.line(f"  {name:<32} acc={magg[r]['test_acc'][0] * 100:.2f}+/-{magg[r]['test_acc'][1] * 100:.2f}  "
                    f"ECE={magg[r]['ece'][0]:.4f}  NLL={magg[r]['nll'][0]:.4f}")

    summary = {**vs.run_info(), 'n_runs': len([x for x in method_rows if x['method'] == 'uniform_logits'])}
    report.section('How much of the learned gain does temperature scaling alone recover?')
    base = magg['uniform_logits']['test_acc'][0]
    for ts in ('temp_uniform_probs', 'temp_uniform_logits'):
        for m in learned:
            denom = magg[m]['test_acc'][0] - base
            rec = (magg[ts]['test_acc'][0] - base) / denom if abs(denom) > 1e-9 else float('nan')
            summary[f'{ts}_recovers_{m}'] = rec
            report.line(f'  {EXTRA_TITLES.get(ts, METHOD_TITLES.get(ts)):<32} recovers {rec * 100:+.1f}% of the '
                        f'{METHOD_TITLES[m]} gain ({denom * 100:+.2f} pp)')
    if rho_wT:
        summary['spearman_scalar_weight_vs_inverse_T'] = float(np.nanmean(rho_wT))
        report.line(f'  Spearman rho(learned scalar weight, 1/T) = {np.nanmean(rho_wT):+.3f} (mean over runs); '
                    f'high rho = the scalar head mostly re-scales over-/under-confident views')
    append_rows([summary], out_file(args, out_dir, 'exp7_summary.csv'))

    # ---- figures
    plt = plt_module()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(titles))
    colors = [VIEW_COLORS[a] for a in vs.approaches]
    ax1.bar(x, [vagg[t]['logit_scale'] for t in titles], 0.6, color=colors, edgecolor='white')
    ax1.set_ylabel('Logit scale (std across classes)')
    ax1.set_title('Raw logit scale per view')
    ax2.bar(x, [vagg[t]['uniform_effective_share'] for t in titles], 0.6, color=colors, edgecolor='white')
    ax2.axhline(1 / len(titles), color='#898781', ls='--', lw=1, label='Equal vote')
    ax2.set_ylabel('Share of the uniform logit sum')
    ax2.set_title('Effective vote under uniform logit averaging')
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=20)
    save_fig(plt, os.path.join(out_dir, f'{tag}_view_scale.png'))

    plt.figure(figsize=(5.2, 5))
    plt.plot([0, 1], [0, 1], color='#c3c2b7', lw=1, ls='--', label='Perfect calibration')
    for r, runs in rel.items():
        xs = np.concatenate([a for a, _, _ in runs])
        ys = np.concatenate([b for _, b, _ in runs])
        order = np.argsort(xs)
        bins = np.linspace(0, 1, 11)
        which = np.digitize(xs[order], bins)
        bx = [xs[order][which == b].mean() for b in np.unique(which)]
        by = [ys[order][which == b].mean() for b in np.unique(which)]
        plt.plot(bx, by, marker='o', ms=6, lw=2, color=METHOD_COLORS.get(r), label=METHOD_TITLES[r])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'{vs.dataset_name.upper()}: reliability (test)')
    plt.legend(fontsize=8)
    save_fig(plt, os.path.join(out_dir, f'{tag}_reliability.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='scalar,per_class,mlp', help='Learned heads to compare')
    main(p.parse_args())
