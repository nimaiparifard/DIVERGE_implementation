# Experiment #3 (R1-C2 / R2-C1 / R2-C2 / R4-C1): is the init diversity *useful*?
# Complementarity of the members and ensemble gain, multi-init vs seed-only ensembles.
"""
Disagreement alone is not enough (a weak model disagrees a lot). Here we check
that the diversity is complementary and turns into ensemble gain.

Uses the shared GNN pool (one GNN per init x seed, fixed GNN seed/split).
All ensembles are UNIFORM averages of the members' class probabilities, so no
ensemble weights are learned and nothing is fitted on test nodes.

  a) Pairwise complementarity on test nodes: double-fault rate (both wrong),
     oracle accuracy (at least one right), oracle gain over the better member,
     Yule's Q statistic (Kuncheva & Whitaker, 2003). Seed-only vs init-only
     pairs, same test as experiment #1.
  b) Seed-only ensembles (one init, k seeds) vs multi-init ensembles (k inits,
     one seed) at the SAME size k = min(#inits, #seeds), plus the full-size
     versions. For each ensemble, the ambiguity decomposition
     (Krogh & Vedelsby, 1995) with squared loss on the probability vectors:
          ensemble error = average member error - ambiguity (diversity)
     which is an exact identity for a uniform average, and links diversity
     to the gain directly.
  c) Diversity vs gain: every k-subset of the pool (sampled if too many),
     mean pairwise disagreement vs gain over the mean member, Spearman
     correlation, and the average gain grouped by number of distinct inits.

Own-split mode (--gnn_seed -1): every pair / ensemble / subset is evaluated only on
the nodes that are test nodes for all of its members (so no member was fine-tuned
on them). Ensembles and subsets with fewer than --min_eval_nodes such nodes are
skipped; in practice this drops groups that span many seeds (e.g. seed-only
ensembles of 4 seeds share only ~0.2% of the nodes).

Outputs (under --out_dir, default results/diversity/complementarity/):
  complementarity_summary.csv     one row per run
  complementarity_ensembles.csv   one row per seed-only / multi-init ensemble
  complementarity_subsets.csv     diversity-vs-gain summary per (k, #distinct inits)
  <tag>_*_heatmap.png, <tag>_diversity_vs_gain_k<k>.png, <tag>_report.txt
"""
import itertools
import math
import os

import numpy as np
import torch
from scipy.stats import spearmanr, mannwhitneyu

from experiments.diversity_common import (
    add_common_args, pool_from_args, parse_list, within_between_test, save_pool_heatmap, save_matrix_csv,
    run_info, append_rows, TextReport, common_mask,
)


def pairwise_complementarity(correct, preds, masks):
    """
    Pairwise matrices; `correct`/`preds` are [M, n] over all nodes and `masks` [M, n]
    the test nodes of each member. A pair (i, j) is evaluated on masks[i] & masks[j].
    """
    M = correct.shape[0]
    out = {k: np.zeros((M, M)) for k in ('double_fault', 'oracle_acc', 'oracle_gain', 'q_statistic',
                                         'disagreement')}
    for i in range(M):
        for j in range(M):
            m = masks[i] & masks[j]
            ci, cj = correct[i, m], correct[j, m]
            n = max(int(m.sum()), 1)
            n11, n00 = (ci & cj).sum(), (~ci & ~cj).sum()
            n10, n01 = (ci & ~cj).sum(), (~ci & cj).sum()
            denom = n11 * n00 + n01 * n10
            oracle = 1 - n00 / n
            out['double_fault'][i, j] = n00 / n
            out['oracle_acc'][i, j] = oracle
            out['oracle_gain'][i, j] = oracle - max(ci.mean(), cj.mean())
            out['q_statistic'][i, j] = (n11 * n00 - n01 * n10) / denom if denom > 0 else 1.0
            out['disagreement'][i, j] = (preds[i, m] != preds[j, m]).mean()
    return out


def ensemble_stats(probs, y_onehot, y, members):
    """Uniform-average ensemble of `members` (indices into probs [M, n, C]) on the given nodes."""
    P = probs[members]                                   # [k, n, C]
    p_bar = P.mean(dim=0)
    member_acc = (P.argmax(dim=2) == y[None]).float().mean(dim=1)
    member_err = ((P - y_onehot[None]) ** 2).sum(dim=2).mean(dim=1)
    ambiguity = ((P - p_bar[None]) ** 2).sum(dim=2).mean(dim=1)
    ens_acc = (p_bar.argmax(dim=1) == y).float().mean().item()
    return {
        'ens_acc': ens_acc * 100,
        'mean_member_acc': member_acc.mean().item() * 100,
        'best_member_acc': member_acc.max().item() * 100,
        'gain_vs_mean': (ens_acc - member_acc.mean().item()) * 100,
        'gain_vs_best': (ens_acc - member_acc.max().item()) * 100,
        'ens_sq_err': ((p_bar - y_onehot) ** 2).sum(dim=1).mean().item(),
        'avg_member_sq_err': member_err.mean().item(),
        'ambiguity': ambiguity.mean().item(),
    }


def subset_sweep(probs, y, masks, seeds, inits, k, max_subsets, min_nodes, rng, chunk=512):
    """
    Diversity vs gain over k-subsets of the pool. `probs` [M, n, C] and `y` [n] over
    all nodes; each subset is evaluated on the nodes that are test nodes for all of its
    members (shared by every subset with the same set of seeds). Subsets with fewer
    than `min_nodes` such nodes are dropped.
    """
    M = probs.shape[0]
    total = math.comb(M, k)
    if total <= max_subsets:
        subsets = np.array(list(itertools.combinations(range(M), k)))
    else:
        seen = set()
        while len(seen) < max_subsets:
            seen.add(tuple(sorted(rng.choice(M, size=k, replace=False))))
        subsets = np.array(sorted(seen))
    groups = {}
    for r, s in enumerate(subsets):
        groups.setdefault(frozenset(int(x) for x in seeds[s]), []).append(r)
    keep, ens_acc, div, mean_acc, best_acc = [], [], [], [], []
    iu = np.triu_indices(k, 1)
    for rows in groups.values():
        m = common_mask(masks, subsets[rows[0]])
        if int(m.sum()) < min_nodes:
            continue
        pm, ym = probs[:, m], y[m]
        pred = pm.argmax(dim=2)                          # [M, n_m]
        acc = (pred == ym[None]).float().mean(dim=1).numpy() * 100
        sub = subsets[rows]
        for s in range(0, len(sub), chunk):
            idx = torch.as_tensor(sub[s:s + chunk])
            p_bar = pm[idx].mean(dim=1)                  # [B, n_m, C]
            ens_acc.append((p_bar.argmax(dim=2) == ym[None]).float().mean(dim=1).numpy() * 100)
        for s in sub:
            ps = pred[torch.as_tensor(s)]
            d = (ps[:, None] != ps[None, :]).float().mean(dim=2).numpy()
            div.append(d[iu].mean())
        mean_acc.append(acc[sub].mean(axis=1))
        best_acc.append(acc[sub].max(axis=1))
        keep.extend(rows)
    if not keep:
        return None
    n_dropped = len(subsets) - len(keep)
    subsets = subsets[keep]
    ens_acc, mean_acc, best_acc = np.concatenate(ens_acc), np.concatenate(mean_acc), np.concatenate(best_acc)
    n_inits = np.array([len(set(inits[s])) for s in subsets])
    return {'subsets': subsets, 'diversity': np.array(div), 'ens_acc': ens_acc, 'gain_vs_mean': ens_acc - mean_acc,
            'gain_vs_best': ens_acc - best_acc, 'n_inits': n_inits, 'sampled': total > max_subsets, 'total': total,
            'n_dropped': n_dropped}


def plot_div_vs_gain(res, k, save_path, pool):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in sorted(set(res['n_inits'])):
        m = res['n_inits'] == n
        ax.scatter(res['diversity'][m] * 100, res['gain_vs_mean'][m], s=8, alpha=0.5, label=f'{n} distinct init(s)')
    ax.set_xlabel('mean pairwise disagreement (%)')
    ax.set_ylabel('ensemble gain over mean member (pp)')
    ax.set_title(f'Diversity vs ensemble gain, k={k}\n{pool.dataset_name} / {pool.llm_name}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] plot saved to: {save_path}')


def main(args):
    pool = pool_from_args(args)
    out_dir = os.path.join(args.out_dir, 'complementarity')
    report = TextReport(os.path.join(out_dir, f'{pool.tag}_report.txt'),
                        f'Complementarity & ensemble gain | {pool.dataset_name} | {pool.llm_name}')
    for line in pool.describe():
        report.line(line)
    info = run_info(pool)
    rng = np.random.default_rng(0)

    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    masks = gnn['test_masks']                            # [M, n] own test nodes
    y_all = gnn['y']
    probs_all = gnn['probs']                             # [M, n, C]
    preds_all = gnn['preds'].numpy()
    correct_all = preds_all == y_all.numpy()[None]
    min_nodes = args.min_eval_nodes if pool.own_split else 1

    # a) pairwise complementarity
    pw = pairwise_complementarity(correct_all, preds_all, masks.numpy())
    report.section('a) Pairwise complementarity (' + ('shared test nodes of each pair' if pool.own_split
                                                        else 'test nodes') + ')')
    summary = dict(info, n_test_nodes=int(masks[0].sum()))
    # orientation for the test: larger = more diverse / complementary
    oriented = {'double_fault_complement': 1 - pw['double_fault'], 'oracle_gain': pw['oracle_gain'],
                'q_distance': (1 - pw['q_statistic']) / 2}
    for name, D in oriented.items():
        wb = within_between_test(D, pool.inits, pool.seeds, n_perm=args.n_perm, rng=rng)
        report.within_between(name, wb)
        summary.update(TextReport.within_between_row(wb, f'{name}_'))
    for name in ('double_fault', 'oracle_acc', 'q_statistic'):
        save_matrix_csv(pw[name], pool.labels, os.path.join(out_dir, f'{pool.tag}_{name}_matrix.csv'))
        if not args.no_plots:
            save_pool_heatmap(pw[name], pool.labels, pool.inits,
                              os.path.join(out_dir, f'{pool.tag}_{name}_heatmap.png'),
                              title=f'{name}\n{pool.dataset_name} / {pool.llm_name}', cbar_label=name)

    # b) seed-only vs multi-init ensembles at matched size
    idx, inits, seeds = pool.balanced_subset()
    ens_rows = []
    if len(inits) >= 2 and len(seeds) >= 2:
        lab = {(m['init'], m['seed']): i for i, m in enumerate(pool.members)}
        k = min(len(inits), len(seeds))
        families = {'seed_only_matched': [], 'multi_init_matched': [], 'seed_only_full': [], 'multi_init_full': []}
        for init in inits:
            for combo in itertools.combinations(seeds, k):
                families['seed_only_matched'].append([lab[(init, s)] for s in combo])
            families['seed_only_full'].append([lab[(init, s)] for s in seeds])
        for s in seeds:
            for combo in itertools.combinations(inits, k):
                families['multi_init_matched'].append([lab[(i, s)] for i in combo])
            families['multi_init_full'].append([lab[(i, s)] for i in inits])
        # full-size families are identical to the matched ones when #inits == #seeds
        if len(seeds) == k:
            families.pop('seed_only_full')
        if len(inits) == k:
            families.pop('multi_init_full')

        report.section(f'b) Uniform-average ensembles: seed-only vs multi-init (matched size k={k})')
        report.line(f"{'family':<20}{'n':>4}{'size':>6}{'ens acc':>10}{'gain/mean':>11}{'gain/best':>11}"
                    f"{'avg err':>10}{'ambiguity':>11}{'ens err':>10}")
        fam_acc = {}
        n_cls = probs_all.shape[2]

        def eval_ensemble(members):
            m = common_mask(masks, members)
            if int(m.sum()) < min_nodes:
                return None
            y = y_all[m]
            st = ensemble_stats(probs_all[:, m], torch.nn.functional.one_hot(y, n_cls).float(), y,
                                torch.as_tensor(members))
            st['n_eval_nodes'] = int(m.sum())
            return st

        for fam, member_lists in list(families.items()):
            evaluated = [(m, eval_ensemble(m)) for m in member_lists]
            if pool.own_split:
                n_skip = sum(st is None for _, st in evaluated)
                if n_skip:
                    report.line(f'  {fam}: {n_skip}/{len(evaluated)} ensembles skipped (< {min_nodes} shared test nodes)')
            evaluated = [(m, st) for m, st in evaluated if st is not None]
            if not evaluated:
                continue
            member_lists = [m for m, _ in evaluated]
            stats = [st for _, st in evaluated]
            for m, st in zip(member_lists, stats):
                ens_rows.append(dict(info, family=fam, size=len(m), members='|'.join(pool.labels[i] for i in m),
                                     n_distinct_inits=len(set(pool.inits[m])), **st))
            fam_acc[fam] = np.array([st['ens_acc'] for st in stats])
            avg = {key: np.mean([st[key] for st in stats]) for key in stats[0]}
            report.line(f"{fam:<20}{len(stats):>4}{len(member_lists[0]):>6}{avg['ens_acc']:>10.2f}"
                        f"{avg['gain_vs_mean']:>+11.2f}{avg['gain_vs_best']:>+11.2f}{avg['avg_member_sq_err']:>10.4f}"
                        f"{avg['ambiguity']:>11.4f}{avg['ens_sq_err']:>10.4f}")
            summary.update({f'{fam}_{key}': avg[key] for key in ('ens_acc', 'gain_vs_mean', 'gain_vs_best',
                                                                    'ambiguity', 'avg_member_sq_err', 'ens_sq_err')})
            summary[f'{fam}_n'] = len(stats)
        if 'multi_init_matched' in fam_acc and 'seed_only_matched' in fam_acc:
            p = mannwhitneyu(fam_acc['multi_init_matched'], fam_acc['seed_only_matched'],
                             alternative='greater').pvalue
            summary['matched_multi_init_vs_seed_only_mwu_p'] = p
            report.line(f'Multi-init > seed-only (matched size, Mann-Whitney one-sided): p={p:.4g}')
        else:
            report.line('Multi-init vs seed-only test skipped: one family has no evaluable ensemble')
        report.line('(ensemble error = avg member error - ambiguity; higher ambiguity at equal member error = '
                    'more useful diversity)')
    else:
        report.line('b) skipped: need >= 2 inits and >= 2 common seeds')

    # c) diversity vs gain over k-subsets
    report.section('c) Diversity vs gain over k-subsets of the pool')
    subset_rows = []
    for k in parse_list(args.subset_sizes, int):
        if k < 2 or k > len(pool):
            continue
        res = subset_sweep(probs_all, y_all, masks, pool.seeds, pool.inits, k, args.max_subsets, min_nodes, rng)
        if res is None or len(res['subsets']) < 3:
            report.line(f'k={k}: skipped (no subset with >= {min_nodes} shared test nodes)')
            continue
        rho, p = spearmanr(res['diversity'], res['gain_vs_mean'])
        dropped = f", {res['n_dropped']} dropped (< {min_nodes} shared test nodes)" if res['n_dropped'] else ''
        report.line(f"k={k}: {len(res['subsets'])} subsets{' (sampled of ' + str(res['total']) + ')' if res['sampled'] else ''}"
                    f"{dropped} | Spearman(diversity, gain/mean) = {rho:.3f} (p={p:.3g})")
        for n in sorted(set(res['n_inits'])):
            m = res['n_inits'] == n
            report.line(f"    {n} distinct init(s): n={m.sum():>6}  diversity {res['diversity'][m].mean() * 100:6.2f}%  "
                        f"ens acc {res['ens_acc'][m].mean():6.2f}  gain/mean {res['gain_vs_mean'][m].mean():+6.2f}  "
                        f"gain/best {res['gain_vs_best'][m].mean():+6.2f}")
            subset_rows.append(dict(info, k=k, n_distinct_inits=int(n), n_subsets=int(m.sum()),
                                    sampled=res['sampled'], spearman_div_gain=rho, spearman_p=p,
                                    mean_diversity=res['diversity'][m].mean(), mean_ens_acc=res['ens_acc'][m].mean(),
                                    mean_gain_vs_mean=res['gain_vs_mean'][m].mean(),
                                    mean_gain_vs_best=res['gain_vs_best'][m].mean(),
                                    max_ens_acc=res['ens_acc'][m].max()))
        summary[f'k{k}_spearman_div_gain'] = rho
        if not args.no_plots:
            plot_div_vs_gain(res, k, os.path.join(out_dir, f'{pool.tag}_diversity_vs_gain_k{k}.png'), pool)

    append_rows([summary], os.path.join(out_dir, 'complementarity_summary.csv'))
    if ens_rows:
        append_rows(ens_rows, os.path.join(out_dir, 'complementarity_ensembles.csv'))
    if subset_rows:
        append_rows(subset_rows, os.path.join(out_dir, 'complementarity_subsets.csv'))
    report.line(f'\nReport: {report.path}')


if __name__ == '__main__':
    import argparse

    parser = add_common_args(argparse.ArgumentParser(
        description='Complementarity of LoRA-init views and ensemble gain (multi-init vs seed-only)'))
    parser.add_argument('--subset_sizes', type=str, default='2,3,4,5',
                        help='Subset sizes k for the diversity-vs-gain sweep')
    parser.add_argument('--max_subsets', type=int, default=20000,
                        help='Sample at most this many k-subsets per k')
    parser.add_argument('--no_plots', action='store_true')
    main(parser.parse_args())
