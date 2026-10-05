# Experiment #6 (R2-C5, supports R1-C2 / R2-C1): do the views SPECIALISE per class?
"""
The per-class head learns a weight matrix W [M views x C classes] (softmax over
views for each class). We put it next to the per-class accuracy of each view
(recall of class c on the test nodes) and ask:

  - within each class, does the head give more weight to the views that are
    better on that class? (Spearman rho between W[:, c] and per-class accuracy,
    averaged over classes; per-class accuracy measured on the fit nodes to
    avoid circularity, and on the test nodes for reporting)
  - specialisation: in how many classes is the top-weighted view NOT the
    globally best view? how many different views are top-weighted?
  - per-class gain of the per-class head over uniform averaging, and its
    correlation with how much the views disagree on that class

Outputs (under <out_dir>/exp6_per_class/):
  exp6_class_rows.csv       one row per (run, class): weights, per-view accuracy, gains
  exp6_summary.csv          one row per dataset
  <tag>_heatmaps.png        mean W next to per-class view accuracy (test), same view x class grid
  <tag>_class_gain.png      per-class gain over uniform vs per-class view disagreement
  <tag>_report.txt
"""
import argparse
import os

import numpy as np
from scipy import stats

from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, iter_runs, fuse, view_correct, exp_dir, mean_std, plt_module, save_fig,
    TextReport, append_rows,
)


def per_class_view_acc(ok, y, nodes, C):
    """[M, C] accuracy of each view on the nodes of each class (nan if the class is absent)."""
    acc = np.full((ok.shape[0], C), np.nan)
    yn = y[nodes].numpy()
    okn = ok[:, nodes].numpy()
    for c in range(C):
        m = yn == c
        if m.any():
            acc[:, c] = okn[:, m].mean(1)
    return acc


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp6_per_class')
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp6 - per-class weights and view specialisation | {vs.dataset_name} | {vs.llm_name}')
    hp = fusion_hp(args)
    titles = vs.titles

    class_rows, Ws, ACC_test, rho_fit_all, spec_rows = [], [], [], [], []
    gain_c, disagree_c = [], []
    for views, idx, gnn_seed, fuse_seed in iter_runs(vs, args):
        L, y, test = views['logits'], views['y'], idx['test']
        C = L.shape[2]
        ok = view_correct(L, y)
        res = fuse('per_class', L, y, idx, seed=fuse_seed, hp=hp)
        uni = fuse('uniform_logits', L, y, idx)
        W = res['class_weights']                                     # [M, C]
        acc_fit = per_class_view_acc(ok, y, idx['fit'], C)
        acc_test = per_class_view_acc(ok, y, test, C)
        Ws.append(W)
        ACC_test.append(acc_test)

        global_best = int(np.argmax(np.nanmean(acc_fit, 1)))
        preds = L.argmax(2)
        rhos = []
        for c in range(C):
            nodes_c = test[y[test] == c]
            if len(nodes_c) == 0:
                continue
            ok_pc = (res['pred'][nodes_c] == y[nodes_c]).float().mean().item()
            ok_uc = (uni['pred'][nodes_c] == y[nodes_c]).float().mean().item()
            p = preds[:, nodes_c]
            M = p.shape[0]
            # mean pairwise disagreement over the M*(M-1) ordered view pairs (diagonal is always 0)
            dis = float((p.unsqueeze(0) != p.unsqueeze(1)).float().sum() / (M * (M - 1) * p.shape[1]))
            rho = stats.spearmanr(W[:, c], acc_fit[:, c])[0] if not np.isnan(acc_fit[:, c]).any() else np.nan
            rhos.append(rho)
            gain_c.append(ok_pc - ok_uc)
            disagree_c.append(dis)
            class_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed, 'class': c,
                               'n_test_nodes': int(len(nodes_c)), 'top_weighted_view': titles[int(W[:, c].argmax())],
                               'best_view_on_class_fit': titles[int(np.nanargmax(acc_fit[:, c]))]
                               if not np.isnan(acc_fit[:, c]).all() else '',
                               'spearman_weight_vs_fit_acc': float(rho), 'per_class_head_acc': ok_pc,
                               'uniform_acc': ok_uc, 'gain': ok_pc - ok_uc, 'view_disagreement': dis,
                               **{f'w_{t}': float(W[j, c]) for j, t in enumerate(titles)},
                               **{f'test_acc_{t}': float(acc_test[j, c]) for j, t in enumerate(titles)}})
        top = W.argmax(0)
        rho_fit_all.append(np.nanmean(rhos))
        spec_rows.append({'n_classes_top_not_global_best': int((top != global_best).sum()),
                          'n_distinct_top_views': int(len(set(top.tolist()))), 'C': C})

    append_rows(class_rows, out_file(args, out_dir, 'exp6_class_rows.csv'))

    rho_mu, rho_sd = mean_std(rho_fit_all)
    nb_mu = np.mean([s['n_classes_top_not_global_best'] for s in spec_rows])
    nd_mu = np.mean([s['n_distinct_top_views'] for s in spec_rows])
    C = spec_rows[0]['C']
    g_rho, g_p = stats.spearmanr(disagree_c, gain_c)
    report.section('Specialisation')
    report.line(f"  mean Spearman rho(class weight, view's accuracy on that class, fit nodes) = "
                f'{rho_mu:+.3f} +/- {rho_sd:.3f}')
    report.line(f'  classes whose top-weighted view is NOT the globally best view: {nb_mu:.1f} / {C}')
    report.line(f'  distinct views that are top-weighted for at least one class: {nd_mu:.1f} / {len(titles)}')
    report.line(f'  Spearman rho(per-class view disagreement, per-class gain over uniform) = {g_rho:+.3f} '
                f'(p={g_p:.3g})')
    report.section('Mean class-weight matrix (rows = views, columns = classes)')
    W_mean = np.mean(Ws, 0)
    for j, t in enumerate(titles):
        report.line(f'  {t:<11} ' + ' '.join(f'{v:.2f}' for v in W_mean[j]))
    append_rows([{**vs.run_info(), 'n_runs': len(Ws), 'n_classes': C, 'spearman_weight_vs_acc_mean': rho_mu,
                  'spearman_weight_vs_acc_std': rho_sd, 'classes_top_not_global_best': nb_mu,
                  'distinct_top_views': nd_mu, 'spearman_disagreement_vs_gain': float(g_rho),
                  'spearman_disagreement_vs_gain_p': float(g_p)}],
                out_file(args, out_dir, 'exp6_summary.csv'))

    # ---- figures
    plt = plt_module()
    A_mean = np.nanmean(ACC_test, 0)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.45 * C * 2 + 4), 3.6))
    for ax, mat, title, cmap, label in ((axes[0], W_mean, 'Learned per-class weight', 'Blues', 'weight'),
                                        (axes[1], A_mean * 100, 'View accuracy on the class (test)', 'Blues',
                                         'accuracy (%)')):
        im = ax.imshow(mat, aspect='auto', cmap=cmap)
        ax.set_yticks(range(len(titles)))
        ax.set_yticklabels(titles, fontsize=8)
        ax.set_xticks(range(C))
        ax.set_xticklabels(range(C), fontsize=7 if C > 15 else 8)
        ax.set_xlabel('Class')
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        top = np.nanargmax(np.where(np.isnan(mat), -np.inf, mat), axis=0)
        ax.scatter(range(C), top, marker='o', s=18, facecolor='none', edgecolor='#eb6834', linewidth=1.2)
        fig.colorbar(im, ax=ax, label=label)
    fig.suptitle(f'{vs.dataset_name.upper()}: orange ring = top view per class', fontsize=10)
    save_fig(plt, os.path.join(out_dir, f'{tag}_heatmaps.png'))

    plt.figure(figsize=(6, 4.2))
    plt.scatter(disagree_c, np.array(gain_c) * 100, s=36, color='#2a78d6', edgecolor='white', linewidth=0.8, alpha=0.8)
    plt.axhline(0, color='#898781', lw=1)
    plt.xlabel('Mean pairwise view disagreement on the class (test)')
    plt.ylabel('Per-class head gain over uniform (pp)')
    plt.title(f'{vs.dataset_name.upper()}: gain vs disagreement (Spearman rho={g_rho:+.2f})')
    save_fig(plt, os.path.join(out_dir, f'{tag}_class_gain.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    main(p.parse_args())
