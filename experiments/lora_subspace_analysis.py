# Experiment #4 (R2-C1): mechanism - how does initialization diversity become
# representation diversity? Adapter weight subspaces + stage-to-stage chain.
"""
Works directly on the trained LoRA adapters of the pool (one per init x seed,
artifacts/<llm>_<dataset>_seqcls_<peft>_semi_supervised_init_type_<init>_seed<seed>/).

  a) Adapter subspace overlap between members. For every target layer the
     adapter writes into col(B) (d_out x r) and reads from row(A) (r x d_in).
     Overlap of two r-dim subspaces = ||Q_i^T Q_j||_F^2 / r  in [0, 1]
     (1 = same subspace, r/d = random subspaces), averaged over layers.
     Plus the cosine between the adapter products B_i A_i and B_j A_j.
     Seed-only vs init-only pairs, same test as experiment #1: if the init
     decides *where* in weight space the adapter lives, init-only pairs are
     farther apart than seed-only pairs.

  b) Spectral position w.r.t. the pretrained weight W of each layer: overlap
     of col(B) with the top-r / bottom-r left singular vectors of W and of
     row(A) with the top-r / bottom-r right singular vectors (random = r/d).
     PiSSA starts exactly on the top-r singular subspace, so its overlap after
     training measures how much of the init persists; random inits
     (gaussian/orthogonal) should sit near r/d; EVA / LoftQ in between.

  c) Stage chain: Mantel correlations (Spearman, permutation p-value) between
     the pool distance matrices of consecutive stages
        adapter weights -> LLM embeddings (1-CKA) -> GNN embeddings (1-CKA)
        -> GNN predictions (disagreement)
     A positive chain shows that the members that differ in weight space are
     the ones that differ in representation and in prediction.

Caveat: for PiSSA / LoftQ the base weight is replaced by a residual, so the
saved B A is not the update w.r.t. the original W; (a) therefore compares the
adapter factors themselves (the subspace the adapter reads/writes), not
W_eff - W.

Outputs (under --out_dir, default results/diversity/lora_subspace/):
  lora_subspace_summary.csv    one row per (run, measure) with the seed/init test
  lora_subspace_spectral.csv   one row per (run, member) with spectral overlaps
  lora_subspace_mantel.csv     one row per (run, stage pair)
  <tag>_*_heatmap.png, <tag>_report.txt
"""
import itertools
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from experiments.cka import TITLES
from experiments.diversity_common import (
    add_common_args, pool_from_args, cka_distance_matrix, disagreement_matrix, cached_matrix, within_between_test,
    mantel_test, save_pool_heatmap, save_matrix_csv, run_info, append_rows, TextReport,
)
from train_llm.peft_model import get_model_path


def adapter_dir(pool, member):
    # Pool members carry their adapter dir (it depends on --split_seed / --init_seed).
    if 'adapter' in member:
        return member['adapter']
    return os.path.join('artifacts', f"{pool.model_name}_{pool.dataset_name}_seqcls_{pool.peft_type}"
                                     f"_semi_supervised_init_type_{member['init']}_seed{member['seed']}")


def load_adapter(path, device):
    """{layer_name: (A [r, d_in], B [d_out, r])} for every LoRA target layer."""
    f = safe_open(os.path.join(path, 'adapter_model.safetensors'), 'pt')
    layers = {}
    for key in f.keys():
        if key.endswith('.lora_A.weight'):
            name = key[:-len('.lora_A.weight')]
            A = f.get_tensor(key).float().to(device)
            B = f.get_tensor(f'{name}.lora_B.weight').float().to(device)
            layers[name] = (A, B)
    return layers


def orth(M):
    return torch.linalg.qr(M)[0]


def base_weight_name(layer_name):
    # base_model.model.model.layers.0.self_attn.q_proj -> model.layers.0.self_attn.q_proj.weight
    name = layer_name[len('base_model.model.'):] if layer_name.startswith('base_model.model.') else layer_name
    return f'{name}.weight'


def base_spectral_subspaces(model_name, layer_names, r, cache_path, device):
    """Top-r / bottom-r singular subspaces of the pretrained weight of every target layer (cached)."""
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if set(layer_names) <= set(cached):
            return cached
    model_path = get_model_path(model_name)
    index = os.path.join(model_path, 'model.safetensors.index.json')
    if os.path.exists(index):
        weight_map = json.load(open(index))['weight_map']
    else:
        single = os.path.join(model_path, 'model.safetensors')
        with safe_open(single, 'pt') as f:
            weight_map = {k: 'model.safetensors' for k in f.keys()}
    handles, out = {}, {}
    for i, name in enumerate(layer_names):
        wname = base_weight_name(name)
        fname = weight_map[wname]
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(model_path, fname), 'pt')
        W = handles[fname].get_tensor(wname).float().to(device)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        out[name] = {'U_top': U[:, :r].cpu(), 'U_bot': U[:, -r:].cpu(),
                     'V_top': Vh[:r].T.cpu(), 'V_bot': Vh[-r:].T.cpu(), 'd_out': W.shape[0], 'd_in': W.shape[1]}
        if (i + 1) % 16 == 0:
            print(f'  base SVD {i + 1}/{len(layer_names)}')
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(out, cache_path)
    return out


def overlap(Q1, Q2):
    return (torch.linalg.matrix_norm(Q1.T @ Q2) ** 2).item() / Q1.shape[1]


def main(args):
    pool = pool_from_args(args)
    out_dir = os.path.join(args.out_dir, 'lora_subspace')
    report = TextReport(os.path.join(out_dir, f'{pool.tag}_report.txt'),
                        f'LoRA adapter subspaces & stage chain | {pool.dataset_name} | {pool.llm_name}')
    for line in pool.describe():
        report.line(line)
    info = run_info(pool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # members that have a trained adapter on disk
    keep = [i for i, m in enumerate(pool.members)
            if os.path.exists(os.path.join(adapter_dir(pool, m), 'adapter_model.safetensors'))]
    missing = [pool.labels[i] for i in range(len(pool)) if i not in keep]
    if missing:
        report.line(f'No adapter found for: {", ".join(missing)} (excluded from this analysis)')
    if len(keep) < 2:
        raise FileNotFoundError('Fewer than 2 pool members have a saved adapter')
    labels = [pool.labels[i] for i in keep]
    inits, seeds = pool.inits[keep], pool.seeds[keep]

    # ---- load adapter bases
    bases = []
    for i in keep:
        layers = load_adapter(adapter_dir(pool, pool.members[i]), device)
        bases.append({n: {'QA': orth(A.T), 'QB': orth(B), 'A': A, 'B': B} for n, (A, B) in layers.items()})
    layer_names = sorted(set.intersection(*(set(b) for b in bases)))
    r = bases[0][layer_names[0]]['A'].shape[0]
    report.line(f'{len(layer_names)} LoRA layers, rank r={r}')

    # ---- a) pairwise adapter distances
    M = len(keep)
    D = {'adapter_A_subspace': np.zeros((M, M)), 'adapter_B_subspace': np.zeros((M, M)),
         'adapter_product': np.zeros((M, M))}
    for i, j in itertools.combinations(range(M), 2):
        ova, ovb, cos = [], [], []
        for n in layer_names:
            bi, bj = bases[i][n], bases[j][n]
            ova.append(overlap(bi['QA'], bj['QA']))
            ovb.append(overlap(bi['QB'], bj['QB']))
            inner = torch.trace((bi['B'].T @ bj['B']) @ (bj['A'] @ bi['A'].T))
            ni = torch.trace((bi['B'].T @ bi['B']) @ (bi['A'] @ bi['A'].T)).sqrt()
            nj = torch.trace((bj['B'].T @ bj['B']) @ (bj['A'] @ bj['A'].T)).sqrt()
            cos.append((inner / (ni * nj)).item() if ni * nj > 0 else 0.0)
        D['adapter_A_subspace'][i, j] = D['adapter_A_subspace'][j, i] = 1 - np.mean(ova)
        D['adapter_B_subspace'][i, j] = D['adapter_B_subspace'][j, i] = 1 - np.mean(ovb)
        D['adapter_product'][i, j] = D['adapter_product'][j, i] = 1 - np.mean(cos)

    report.section('a) Adapter weight-space distance (seed-only vs init-only)')
    rng = np.random.default_rng(0)
    summary_rows = []
    for name, mat in D.items():
        wb = within_between_test(mat, inits, seeds, n_perm=args.n_perm, rng=rng)
        report.within_between(name, wb)
        per_init = {}
        for init in dict.fromkeys(inits):
            idx = np.where(inits == init)[0]
            if len(idx) > 1:
                per_init[init] = float(mat[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean())
        report.line('  seed-only distance per init: ' +
                    ', '.join(f'{TITLES[i]} {v:.3f}' for i, v in per_init.items()))
        summary_rows.append(dict(info, measure=name, **TextReport.within_between_row(wb),
                                 **{f'seed_only_{TITLES[i]}': v for i, v in per_init.items()}))
        save_matrix_csv(mat, labels, os.path.join(out_dir, f'{pool.tag}_{name}_matrix.csv'))
        if not args.no_plots:
            save_pool_heatmap(mat, labels, inits, os.path.join(out_dir, f'{pool.tag}_{name}_heatmap.png'),
                              title=f'{name} distance\n{pool.dataset_name} / {pool.llm_name}', cbar_label='distance')

    # ---- b) spectral position relative to the pretrained weights
    report.section('b) Spectral position of the adapters in the pretrained weights (overlap; random = r/d)')
    spec = base_spectral_subspaces(pool.model_name, layer_names, r,
                                   os.path.join(args.out_dir, 'base_svd', f'{pool.model_name}_r{r}.pt'), device)
    spectral_rows = []
    per_member = []
    for k, i in enumerate(keep):
        vals = {'B_top': [], 'B_bottom': [], 'A_top': [], 'A_bottom': [], 'random_B': [], 'random_A': []}
        for n in layer_names:
            s, b = spec[n], bases[k][n]
            vals['B_top'].append(overlap(s['U_top'].to(device), b['QB']))
            vals['B_bottom'].append(overlap(s['U_bot'].to(device), b['QB']))
            vals['A_top'].append(overlap(s['V_top'].to(device), b['QA']))
            vals['A_bottom'].append(overlap(s['V_bot'].to(device), b['QA']))
            vals['random_B'].append(r / s['d_out'])
            vals['random_A'].append(r / s['d_in'])
        row = {k2: float(np.mean(v)) for k2, v in vals.items()}
        per_member.append(row)
        spectral_rows.append(dict(info, member=labels[k], init=TITLES[inits[k]], seed=int(seeds[k]),
                                  **{f'overlap_{k2}': v for k2, v in row.items()}))
    cols = ('B_top', 'B_bottom', 'A_top', 'A_bottom')
    report.line(f"{'init':<12}" + ''.join(f'{c:>16}' for c in cols) + f"{'random B':>10}{'random A':>10}"
                "   (mean +/- std over seeds)")
    for init in dict.fromkeys(inits):
        rows = [per_member[k] for k in range(M) if inits[k] == init]
        cells = ''.join(f"{np.mean([x[c] for x in rows]):.3f} +/- {np.std([x[c] for x in rows]):.3f}".rjust(16)
                        for c in cols)
        report.line(f'{TITLES[init]:<12}{cells}{rows[0]["random_B"]:>10.4f}{rows[0]["random_A"]:>10.4f}')

    # ---- c) Mantel chain across stages
    report.section('c) Stage chain: Mantel correlation between pool distance matrices')
    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    llm = cached_matrix(pool, 'llm_cka_dist', lambda: cka_distance_matrix(pool.load_llm_embeddings()))
    gnn_cka = cached_matrix(pool, 'gnn_cka_dist', lambda: cka_distance_matrix(gnn['emb']),
                            recompute=args.retrain_gnn)
    pred = disagreement_matrix(gnn['preds'], gnn['test_masks'])
    sub = np.ix_(keep, keep)
    stages = {'adapter_A_subspace': D['adapter_A_subspace'], 'adapter_B_subspace': D['adapter_B_subspace'],
              'adapter_product': D['adapter_product'], 'llm_embedding_cka': llm[sub],
              'gnn_embedding_cka': gnn_cka[sub], 'gnn_prediction_disagreement': pred[sub]}
    chain = [('adapter_B_subspace', 'llm_embedding_cka'), ('adapter_product', 'llm_embedding_cka'),
             ('llm_embedding_cka', 'gnn_embedding_cka'), ('gnn_embedding_cka', 'gnn_prediction_disagreement'),
             ('llm_embedding_cka', 'gnn_prediction_disagreement'),
             ('adapter_B_subspace', 'gnn_prediction_disagreement')]
    mantel_rows = []
    for a, b in chain:
        res = mantel_test(stages[a], stages[b], n_perm=args.n_perm, rng=rng)
        report.line(f'  {a:<28} -> {b:<28} rho={res["r"]:+.3f}  p={res["p_value"]:.4g}')
        mantel_rows.append(dict(info, stage_a=a, stage_b=b, mantel_spearman=res['r'], p_value=res['p_value']))

    append_rows(summary_rows, os.path.join(out_dir, 'lora_subspace_summary.csv'))
    append_rows(spectral_rows, os.path.join(out_dir, 'lora_subspace_spectral.csv'))
    append_rows(mantel_rows, os.path.join(out_dir, 'lora_subspace_mantel.csv'))
    report.line(f'\nReport: {report.path}')


if __name__ == '__main__':
    import argparse

    parser = add_common_args(argparse.ArgumentParser(
        description='LoRA adapter subspace analysis (init vs seed), spectral position and stage-chain Mantel tests'))
    parser.add_argument('--no_plots', action='store_true')
    main(parser.parse_args())
