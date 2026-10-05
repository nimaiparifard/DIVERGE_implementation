# Experiment #5 (R1-C2 / R2-C1): do the LoRA inits arrange the *local* semantic
# neighbourhoods differently? (complements the global CKA of experiment #1)
"""
CKA compares the global geometry of two embedding sets. Here we compare local
structure: for every node, its k nearest neighbours (cosine) in view A vs in
view B.

For the LLM embeddings and/or the GNN penultimate embeddings of every pool
member (init x seed):
  a) k-NN overlap distance: 1 - mean_node Jaccard(kNN_A(v), kNN_B(v)),
     seed-only vs init-only pairs, same test as experiment #1.
  b) k-NN label homophily per member: share of a node's k neighbours that
     have its label (all nodes and test nodes), i.e. how well each view places
     same-class nodes together. Also reported per init (mean +/- std over seeds).
  c) Neighbourhood complementarity: share of test nodes whose k-NN majority
     label is correct in at least one of g views, for g different inits
     (same seed) vs g different seeds (same init) - does combining inits give
     more nodes a "good neighbourhood" than combining seeds?

Outputs (under --out_dir, default results/diversity/knn_neighborhood/):
  knn_summary.csv       one row per (run, level)
  knn_homophily.csv     one row per (run, level, member)
  <tag>_<level>_knn_heatmap.png, <tag>_report.txt
"""
import os

import numpy as np
import torch

from experiments.cka import TITLES
from experiments.diversity_common import (
    add_common_args, pool_from_args, parse_list, within_between_test, save_pool_heatmap, save_matrix_csv,
    run_info, append_rows, TextReport, common_mask,
)


@torch.no_grad()
def knn_indices(X, k, device, chunk=2048):
    """[n, k] indices of the k nearest neighbours (cosine, self excluded)."""
    Xn = torch.nn.functional.normalize(X.float().to(device), dim=1)
    n = Xn.shape[0]
    out = torch.empty((n, k), dtype=torch.long)
    for s in range(0, n, chunk):
        sim = Xn[s:s + chunk] @ Xn.T
        rows = torch.arange(s, min(s + chunk, n), device=device)
        sim[rows - s, rows] = -float('inf')
        out[s:s + chunk] = sim.topk(k, dim=1).indices.cpu()
    return out


def knn_jaccard(a, b):
    """Mean over nodes of the Jaccard overlap between two [n, k] neighbour sets."""
    k = a.shape[1]
    both = torch.sort(torch.cat([a, b], dim=1), dim=1).values
    inter = (both[:, 1:] == both[:, :-1]).sum(dim=1).float()
    return (inter / (2 * k - inter)).mean().item()


def main(args):
    pool = pool_from_args(args)
    out_dir = os.path.join(args.out_dir, 'knn_neighborhood')
    report = TextReport(os.path.join(out_dir, f'{pool.tag}_report.txt'),
                        f'k-NN neighbourhood agreement | {pool.dataset_name} | {pool.llm_name} | k={args.k}')
    for line in pool.describe():
        report.line(line)
    info = dict(run_info(pool), k=args.k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(0)

    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    y = gnn['y']
    masks = gnn['test_masks']                                         # [M, n] own test nodes
    min_nodes = args.min_eval_nodes if pool.own_split else 1
    summary_rows, homo_rows = [], []

    for level in parse_list(args.levels):
        if level == 'llm':
            nbrs = [knn_indices(pool.load_llm_embedding(i), args.k, device) for i in range(len(pool))]
        elif level == 'gnn':
            nbrs = [knn_indices(e, args.k, device) for e in gnn['emb']]
        else:
            raise ValueError(f"Unknown level '{level}' (use llm and/or gnn)")
        M = len(nbrs)

        # a) k-NN overlap distance
        D = np.zeros((M, M))
        for i in range(M):
            for j in range(i + 1, M):
                D[i, j] = D[j, i] = 1.0 - knn_jaccard(nbrs[i], nbrs[j])
        wb = within_between_test(D, pool.inits, pool.seeds, n_perm=args.n_perm, rng=rng)
        report.section(f'[{level}] a) k-NN overlap distance (1 - Jaccard of the {args.k} nearest neighbours)')
        report.within_between('k-NN distance', wb)
        save_matrix_csv(D, pool.labels, os.path.join(out_dir, f'{pool.tag}_{level}_knn_matrix.csv'))
        if not args.no_plots:
            save_pool_heatmap(D, pool.labels, pool.inits, os.path.join(out_dir, f'{pool.tag}_{level}_knn_heatmap.png'),
                              title=f'{level} k-NN distance (1 - Jaccard, k={args.k})\n'
                                    f'{pool.dataset_name} / {pool.llm_name}', cbar_label='1 - Jaccard')

        # b) homophily
        same = [(y[nb] == y[:, None]).float() for nb in nbrs]            # [n, k] per member
        homo_all = np.array([s.mean().item() for s in same])
        homo_test = np.array([s[masks[i]].mean().item() for i, s in enumerate(same)])
        report.section(f'[{level}] b) k-NN label homophily (share of neighbours with the same label)')
        for init in dict.fromkeys(pool.inits):
            m = pool.inits == init
            report.line(f'  {TITLES[init]:<11} all {homo_all[m].mean():.4f} +/- {homo_all[m].std():.4f} | '
                        f'test {homo_test[m].mean():.4f} +/- {homo_test[m].std():.4f}')
        for i, lbl in enumerate(pool.labels):
            homo_rows.append(dict(info, level=level, member=lbl, init=TITLES[pool.inits[i]], seed=int(pool.seeds[i]),
                                  homophily_all=homo_all[i], homophily_test=homo_test[i]))

        # c) neighbourhood complementarity (k-NN majority label correct), seed-matched
        #    Compared at equal group size with the seed-only counterpart (same init, several seeds).
        good = torch.stack([(s.mean(dim=1) > 0.5) for s in same])                  # [M, n]
        idx_b, inits_b, seeds_b = pool.balanced_subset()
        g = min(len(inits_b), len(seeds_b))
        lab = {(m['init'], m['seed']): i for i, m in enumerate(pool.members)}

        def union_share(members):
            # share of the group's shared test nodes with a correct k-NN majority in >= 1 view
            m = common_mask(masks, members)
            return good[members][:, m].any(dim=0).float().mean().item() if int(m.sum()) >= min_nodes else None

        union_init = [union_share([lab[(i, s)] for i in inits_b[:g]]) for s in seeds_b]
        union_seed = [union_share([lab[(i, s)] for s in seeds_b[:g]]) for i in inits_b]
        union_init = [u for u in union_init if u is not None]
        union_seed = [u for u in union_seed if u is not None]
        single = np.mean([good[i, masks[i]].float().mean().item() for i in range(M)])
        if g >= 2:
            report.section(f'[{level}] c) Test nodes with a same-label k-NN majority (groups of {g} views)')
            report.line(f'  single view (avg): {single:.4f}')
            for name, u in ((f'any of {g} inits  (same seed)', union_init), (f'any of {g} seeds  (same init)', union_seed)):
                report.line(f'  {name}: {np.mean(u):.4f}  (+{np.mean(u) - single:.4f})' if u else
                            f'  {name}: skipped (< {min_nodes} shared test nodes)')

        summary_rows.append(dict(info, level=level, **TextReport.within_between_row(wb),
                                 homophily_all_mean=homo_all.mean(), homophily_test_mean=homo_test.mean(),
                                 good_neighbourhood_single=single, group_size=g,
                                 good_neighbourhood_any_init=np.mean(union_init) if g >= 2 and union_init else '',
                                 good_neighbourhood_any_seed=np.mean(union_seed) if g >= 2 and union_seed else ''))

    append_rows(summary_rows, os.path.join(out_dir, 'knn_summary.csv'))
    append_rows(homo_rows, os.path.join(out_dir, 'knn_homophily.csv'))
    report.line(f'\nReport: {report.path}')


if __name__ == '__main__':
    import argparse

    parser = add_common_args(argparse.ArgumentParser(
        description='Local k-NN neighbourhood agreement between LoRA-init views (init vs seed)'))
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbours')
    parser.add_argument('--levels', type=str, default='llm,gnn', help='Comma-separated: llm and/or gnn')
    parser.add_argument('--no_plots', action='store_true')
    main(parser.parse_args())
