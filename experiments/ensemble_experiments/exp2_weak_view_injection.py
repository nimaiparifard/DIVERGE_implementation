# Experiment #2 (R2-C5): do the learned weights ADAPT when one view becomes weak?
"""
A controlled intervention instead of a correlation: one view (the target) is
deliberately degraded, its GNN is retrained on the degraded embeddings with the
same GNN seed/split, and all fusion rules are refit.

Part A - corruption of the target view's LLM embeddings
  noise:<s>  X + s * std(X) * N(0, 1)       for s in --noise_levels
  shuffle    rows of X permuted (text <-> node link destroyed, graph kept)
  random     pure Gaussian features with the same std
  base       embeddings of the base (not fine-tuned) LLM, if cached
For each setting we record the target view's standalone accuracy, the test
accuracy of uniform averaging vs the learned heads, and the weight the learned
heads give the target view.
Expected: uniform accuracy falls as the view degrades; the learned heads push
the target's weight toward 0 and stay close to the 4-view ensemble.

Part B - redundancy: the target view's logits are duplicated k times
(M + k views). Uniform averaging gives the duplicated view (1 + k) / (M + k)
of the vote; a learned head can undo that. We report accuracy and the total
weight on the duplicated group.

Outputs (under <out_dir>/exp2_weak_view/):
  exp2_corruption_rows.csv, exp2_duplication_rows.csv   one row per (run, setting, method)
  <tag>_noise_curve.png, <tag>_corruption_bars.png, <tag>_duplication.png
  <tag>_report.txt
Corrupted-view GNN logits are cached in <out_dir>/exp2_weak_view/gnn_cache/.
"""
import argparse
import os

import numpy as np
import torch

from dataset.data_utils import get_based_model_dataset, get_embedding_from_data
from experiments.ensemble_experiments.ensemble_common import (
    out_file, add_common_args, fusion_hp, ViewSet, protocol_indices, fuse, evaluate, train_gnn_logits, exp_dir, mean_std,
    plt_module, save_fig, TextReport, append_rows, parse_list, TITLES, METHOD_COLORS, METHOD_TITLES,
)


def corrupt(emb, kind, gen, base_emb=None):
    std = emb.std()
    if kind.startswith('noise:'):
        s = float(kind.split(':')[1])
        return emb + s * std * torch.randn(emb.shape, generator=gen)
    if kind == 'shuffle':
        return emb[torch.randperm(emb.shape[0], generator=gen)]
    if kind == 'random':
        return std * torch.randn(emb.shape, generator=gen)
    if kind == 'base':
        return base_emb
    raise ValueError(kind)


def pick_targets(spec, views):
    names = views['approaches']
    if spec == 'all':
        return list(range(len(names)))
    out = []
    for t in parse_list(spec):
        if t == 'best':
            out.append(int(np.argmax(views['val_acc'])))
        elif t == 'worst':
            out.append(int(np.argmin(views['val_acc'])))
        else:
            out.append(names.index(t))
    return list(dict.fromkeys(out))


def corrupted_logits(vs, args, gnn_seed, target_name, kind, base_emb, cache_dir):
    safe = kind.replace(':', '')
    path = os.path.join(cache_dir, f'{vs.tag(gnn_seed)}_{target_name}_{safe}.pt')
    if os.path.exists(path) and not args.retrain_gnn:
        return torch.load(path, weights_only=False)
    gen = torch.Generator().manual_seed(gnn_seed)
    emb = corrupt(vs.embeddings()[target_name], kind, gen, base_emb)
    print(f'\n--- [exp2] retraining GNN on {TITLES[target_name]} with corruption={kind} (gnn_seed={gnn_seed}) ---')
    logits, results, _ = train_gnn_logits(vs.gnn_cfg(gnn_seed), emb, verbose=False)
    rec = {'logits': logits, 'test_acc': float(results['test_acc']), 'val_acc': float(results['val_acc'])}
    torch.save(rec, path)
    return rec


def main(args):
    vs = ViewSet(args)
    out_dir = exp_dir(args, 'exp2_weak_view')
    cache_dir = os.path.join(out_dir, 'gnn_cache')
    os.makedirs(cache_dir, exist_ok=True)
    tag = vs.tag()
    report = TextReport(os.path.join(out_dir, f'{tag}_report.txt'),
                        f'Exp2 - weak-view injection | {vs.dataset_name} | {vs.llm_name}')
    methods = ['uniform_logits'] + parse_list(args.methods)
    hp = fusion_hp(args)

    kinds = [f'noise:{s:g}' for s in parse_list(args.noise_levels, float)] + parse_list(args.corruptions)
    base_emb = None
    if 'base' in kinds:
        try:
            base_emb = get_embedding_from_data(get_based_model_dataset(vs.cfg)).detach().float().cpu()
        except (FileNotFoundError, RuntimeError) as e:
            report.line(f'[skip] base-model embeddings unavailable: {e}')
            kinds.remove('base')

    corr_rows, dup_rows = [], []
    for gnn_seed in parse_list(args.gnn_seeds, int):
        views = vs.load(gnn_seed, retrain=args.retrain_gnn)
        L0, y = views['logits'], views['y']
        targets = pick_targets(args.target, views)

        # ---------------- Part A: corruption
        for t in targets:
            tname = views['approaches'][t]
            for kind in kinds:
                if kind == 'noise:0':
                    rec = {'logits': L0[t]}
                else:
                    rec = corrupted_logits(vs, args, gnn_seed, tname, kind, base_emb, cache_dir)
                L = L0.clone()
                L[t] = rec['logits']
                for fuse_seed in parse_list(args.fuse_seeds, int):
                    idx = protocol_indices(views, args.protocol, args.fit_frac, split_seed=fuse_seed)
                    view_acc = float((L[t][idx['test']].argmax(1) == y[idx['test']]).float().mean())
                    for m in methods:
                        res = fuse(m, L, y, idx, seed=fuse_seed, hp=hp)
                        w = res['view_weights']
                        corr_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed,
                                          'target_view': TITLES[tname], 'corruption': kind,
                                          'target_view_test_acc': view_acc, 'method': m,
                                          'target_weight': float(w[t]) if w is not None else float('nan'),
                                          **evaluate(res, y, idx)})
            # reference: ensemble without the target view at all
            keep = [i for i in range(L0.shape[0]) if i != t]
            for fuse_seed in parse_list(args.fuse_seeds, int):
                idx = protocol_indices(views, args.protocol, args.fit_frac, split_seed=fuse_seed)
                for m in methods:
                    res = fuse(m, L0[keep], y, idx, seed=fuse_seed, hp=hp)
                    corr_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed,
                                      'target_view': TITLES[tname], 'corruption': 'removed',
                                      'target_view_test_acc': float('nan'), 'method': m,
                                      'target_weight': 0.0, **evaluate(res, y, idx)})

        # ---------------- Part B: duplication of the weakest view
        t = int(np.argmin(views['val_acc']))
        M = L0.shape[0]
        for k in parse_list(args.dup_counts, int):
            L = torch.cat([L0, L0[t:t + 1].repeat(k, 1, 1)], dim=0) if k else L0
            group = [t] + list(range(M, M + k))
            for fuse_seed in parse_list(args.fuse_seeds, int):
                idx = protocol_indices(views, args.protocol, args.fit_frac, split_seed=fuse_seed)
                for m in methods:
                    res = fuse(m, L, y, idx, seed=fuse_seed, hp=hp)
                    w = res['view_weights']
                    dup_rows.append({**vs.run_info(), 'gnn_seed': gnn_seed, 'fuse_seed': fuse_seed,
                                     'duplicated_view': TITLES[views['approaches'][t]], 'extra_copies': k,
                                     'method': m, 'group_weight': float(np.sum(w[group])) if w is not None
                                     else float('nan'), 'uniform_group_share': (1 + k) / (M + k),
                                     **evaluate(res, y, idx)})

    append_rows(corr_rows, out_file(args, out_dir, 'exp2_corruption_rows.csv'))
    append_rows(dup_rows, out_file(args, out_dir, 'exp2_duplication_rows.csv'))

    # ---------------- aggregate + report
    def agg(rows, key_fields, value):
        out = {}
        for r in rows:
            out.setdefault(tuple(r[f] for f in key_fields), []).append(r[value])
        return {k: mean_std(v) for k, v in out.items()}

    acc = agg(corr_rows, ['target_view', 'corruption', 'method'], 'test_acc')
    wt = agg(corr_rows, ['target_view', 'corruption', 'method'], 'target_weight')
    vacc = agg(corr_rows, ['target_view', 'corruption', 'method'], 'target_view_test_acc')
    target_titles = list(dict.fromkeys(r['target_view'] for r in corr_rows))
    for tt in target_titles:
        report.section(f'Part A - corrupting {tt}')
        report.line(f"{'corruption':<12} {'view acc':>9} " + ' '.join(f'{m[:12]:>12}' for m in methods)
                    + '   target weight (' + ', '.join(m for m in methods if m != 'uniform_logits') + ')')
        for kind in kinds + ['removed']:
            accs = ' '.join(f'{acc[(tt, kind, m)][0] * 100:>12.2f}' for m in methods)
            wts = ', '.join(f'{wt[(tt, kind, m)][0]:.3f}' for m in methods if m != 'uniform_logits')
            va = vacc[(tt, kind, methods[0])][0]
            report.line(f'{kind:<12} {va * 100 if va == va else float("nan"):>9.2f} {accs}   {wts}')

    dacc = agg(dup_rows, ['extra_copies', 'method'], 'test_acc')
    dwt = agg(dup_rows, ['extra_copies', 'method'], 'group_weight')
    report.section('Part B - duplicating the weakest view (mean over runs)')
    for k in parse_list(args.dup_counts, int):
        report.line(f'  +{k} copies: ' + ' | '.join(
            f'{m}: acc={dacc[(k, m)][0] * 100:.2f} group_w={dwt[(k, m)][0]:.3f}' for m in methods))

    # ---------------- figures
    plt = plt_module()
    noise = [k for k in kinds if k.startswith('noise:')]
    levels = [float(k.split(':')[1]) for k in noise]
    for tt in target_titles:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for m in methods:
            mu = [acc[(tt, k, m)][0] * 100 for k in noise]
            sd = [acc[(tt, k, m)][1] * 100 for k in noise]
            ax1.errorbar(levels, mu, yerr=sd, marker='o', ms=6, lw=2, capsize=3, label=METHOD_TITLES[m],
                         color=METHOD_COLORS.get(m))
        ax1.plot(levels, [vacc[(tt, k, methods[0])][0] * 100 for k in noise], ls=':', marker='s', ms=6, lw=2,
                 color='#52514e', label=f'{tt} alone')
        ax1.set_xlabel(f'Noise scale s (x std) on {tt} embeddings')
        ax1.set_ylabel('Test accuracy (%)')
        ax1.set_title('Accuracy under a degrading view')
        ax1.legend(fontsize=8)
        for m in methods:
            if m == 'uniform_logits':
                continue
            ax2.plot(levels, [wt[(tt, k, m)][0] for k in noise], marker='o', ms=6, lw=2, label=METHOD_TITLES[m],
                     color=METHOD_COLORS.get(m))
        ax2.axhline(1 / len(vs.approaches), color='#898781', ls='--', lw=1, label='Uniform weight')
        ax2.set_xlabel(f'Noise scale s (x std) on {tt} embeddings')
        ax2.set_ylabel(f'Weight on {tt}')
        ax2.set_title('Learned weight on the degraded view')
        ax2.legend(fontsize=8)
        save_fig(plt, os.path.join(out_dir, f'{tag}_{tt}_noise_curve.png'))

        cats = [k for k in kinds if not k.startswith('noise:')] + ['removed']
        if cats:
            plt.figure(figsize=(8, 4))
            x = np.arange(len(cats))
            width = 0.8 / len(methods)
            for i, m in enumerate(methods):
                plt.bar(x + (i - (len(methods) - 1) / 2) * width, [acc[(tt, c, m)][0] * 100 for c in cats],
                        width * 0.92, label=METHOD_TITLES[m], color=METHOD_COLORS.get(m), edgecolor='white')
            plt.xticks(x, cats)
            plt.ylabel('Test accuracy (%)')
            plt.title(f'{vs.dataset_name.upper()}: replacing {tt} embeddings')
            plt.legend(fontsize=8)
            save_fig(plt, os.path.join(out_dir, f'{tag}_{tt}_corruption_bars.png'))

    ks = parse_list(args.dup_counts, int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in methods:
        ax1.plot(ks, [dacc[(k, m)][0] * 100 for k in ks], marker='o', ms=6, lw=2, label=METHOD_TITLES[m],
                 color=METHOD_COLORS.get(m))
        ax2.plot(ks, [dwt[(k, m)][0] for k in ks], marker='o', ms=6, lw=2, label=METHOD_TITLES[m],
                 color=METHOD_COLORS.get(m))
    ax1.set_xlabel('Extra copies of the weakest view')
    ax1.set_ylabel('Test accuracy (%)')
    ax1.set_title('Accuracy with a duplicated view')
    ax1.legend(fontsize=8)
    ax2.set_xlabel('Extra copies of the weakest view')
    ax2.set_ylabel('Total weight on the duplicated view')
    ax2.set_title('Weight given to the duplicated group')
    ax2.legend(fontsize=8)
    save_fig(plt, os.path.join(out_dir, f'{tag}_duplication.png'))


if __name__ == '__main__':
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    p.add_argument('--methods', type=str, default='scalar,per_class,mlp')
    p.add_argument('--target', type=str, default='best',
                   help="View(s) to degrade: 'best', 'worst', 'all' or comma-separated approach names")
    p.add_argument('--noise_levels', type=str, default='0,0.5,1,2,4,8')
    p.add_argument('--corruptions', type=str, default='shuffle,random,base',
                   help='Extra corruptions besides Gaussian noise (shuffle, random, base)')
    p.add_argument('--dup_counts', type=str, default='0,1,2,3,4',
                   help='Extra copies of the weakest view for the redundancy test')
    main(p.parse_args())
