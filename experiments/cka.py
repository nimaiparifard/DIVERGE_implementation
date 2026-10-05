# calculate cka metric for one dartaset dffirent embedding initilizatrion
r"""
Yes. In ML, **CKA** usually means **Centered Kernel Alignment** (Kornblith et al., 2019). It measures how similar two representation matrices are, even when they have different dimensions or are related by a linear transform.

That is how it shows up in your DIVERGE paper notes: pairwise CKA between embedding sets, as a diversity / representation-similarity metric.

**What you feed it.** For a shared set of \(n\) examples (nodes, sentences, etc.), you have two matrices \(X \in \mathbb{R}^{n \times d_x}\) and \(Y \in \mathbb{R}^{n \times d_y}\) (rows = items, columns = features). CKA does **not** compare example \(i\) in \(X\) to example \(i\) in \(Y\) one-to-one in feature space. It compares the **pairwise similarity structure** among the \(n\) items.

**Linear CKA (the usual practical version).**

1. Center the features (subtract the column mean).
2. Form Gram matrices \(K = XX^\top\) and \(L = YY^\top\) (or equivalently HSIC on centered features).
3. Score:

\[
\text{CKA}(X,Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \,\|Y^\top Y\|_F}
\]

(after centering). That is HSIC\((X,Y) / \sqrt{\text{HSIC}(X,X)\,\text{HSIC}(Y,Y)}\).

**What the number means.**

- **1**: same similarity geometry (up to isotropic scaling; linear CKA is invariant to orthogonal transforms and isotropic scaling).
- **0**: uncorrelated similarity structures.
- For embeddings, **lower CKA** between two GNN/init runs usually means they occupy more different subspaces — i.e. more diversity. **Higher CKA** means they are more interchangeable.

**Why people use it instead of cosine.**

- Cosine is typically **per-vector** (or mean cosine of aligned pairs). It cares about coordinates after you pick a pairing.
- CKA is **set-level**: it asks whether the *relative* arrangement of points is the same. Different width, permutation of hidden units, and rotation of the feature basis do not wreck linear CKA the way raw cosine on concatenated features would.

**Kernel CKA** replaces \(XX^\top\) with a kernel (e.g. RBF). Linear CKA is what most papers report for hidden activations because it is cheap and often enough.

**Caveats.**

- You must use the **same rows** (same nodes/examples) in both matrices.
- Sample size matters; CKA on few nodes is noisy.
- Linear CKA can still be high if two embeddings share a large shared subspace even when some directions differ.
- It is not a diversity loss by itself — only a diagnostic, unless you train against it.

In your setting, the experiment is: take the five embedding sets (different LoRA/init approaches), compute a **5×5 CKA matrix**, and show that structured inits land in different subspaces than random ones — complementary to pairwise cosine and GNN prediction disagreement.
"""
# save it in propoer way with report and also in csv file
# run with diffirent llm and dataset name
import os
from datetime import datetime

import numpy as np
import torch

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import append_gnn_result_csv
from report.reporter import ReportResults

# Order must match dataset.data_utils.get_init_dataset_for_gnn's return tuple
APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
TITLES = {'pissa': 'PISSA', 'orthogonal': 'ORTHOGONAL', 'gaussian': 'GAUSSIAN', 'loftq': 'LOFTQ', 'eva': 'EVA'}


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear Centered Kernel Alignment (Kornblith et al., 2019) between two
    representation matrices X in R^{n x dx}, Y in R^{n x dy} sharing the same
    n rows/examples:

        CKA(X, Y) = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)

    where X_c, Y_c are column-centered (equivalently HSIC(X,Y) / sqrt(HSIC(X,X)*HSIC(Y,Y))).
    1.0 = identical similarity geometry (up to orthogonal transform / isotropic
    scale), 0.0 = uncorrelated similarity structures. Computed in the feature
    space (d x d) rather than the n x n Gram-matrix space, so it stays cheap
    even for large node counts.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA requires the same number of rows/examples, got {X.shape[0]} vs {Y.shape[0]}")
    X = X.detach().to(dtype=torch.float64, device='cpu')
    Y = Y.detach().to(dtype=torch.float64, device='cpu')
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = torch.norm(Y.T @ X, p='fro') ** 2
    hsic_xx = torch.norm(X.T @ X, p='fro')
    hsic_yy = torch.norm(Y.T @ Y, p='fro')
    denom = (hsic_xx * hsic_yy).item()
    if denom == 0:
        return float('nan')
    return (hsic_xy / denom).item()


def compute_cka_matrix(emb: dict, approaches: list) -> dict:
    """Pairwise linear CKA for every (approach_i, approach_j) pair, including the
    diagonal (=1.0). Returns {(approach_i, approach_j): cka_value}, symmetric."""
    cka_matrix = {}
    for a in approaches:
        for b in approaches:
            if a == b:
                cka_matrix[(a, b)] = 1.0
            elif (b, a) in cka_matrix:
                cka_matrix[(a, b)] = cka_matrix[(b, a)]
            else:
                cka_matrix[(a, b)] = linear_cka(emb[a], emb[b])
    return cka_matrix


def save_cka_heatmap(cka_matrix: dict, approaches: list, save_path: str,
                     title: str = "Linear CKA between LoRA-init embeddings") -> str:
    import matplotlib.pyplot as plt

    n = len(approaches)
    labels = [TITLES[a] for a in approaches]
    mat = np.array([[cka_matrix[(a, b)] for b in approaches] for a in approaches])

    fig, ax = plt.subplots(figsize=(1.4 * n + 2, 1.2 * n + 2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                     color='white' if mat[i, j] < 0.6 else 'black', fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='CKA')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] CKA heatmap saved to: {save_path}")
    return save_path


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         approaches=None, reporter_index=0, csv_path='results/cka/cka_pairwise_results.csv',
         summary_csv_path='results/cka/cka_summary_results.csv', save_heatmap=True,
         heatmap_dir='results/cka'):
    """
    Compute the pairwise linear-CKA matrix between the cached LLM embeddings of
    every LoRA-init approach, for one (dataset, llm, seed) combination.

    Args:
        seed: Seed the LoRA adapters / embedding caches were trained with (must match
              --seed used in train_llm/train_other_lora_init_apporaches.py and
              cache/cache_embedding_with_diffrent_init_weights.py). If None, uses
              configs/dataset/<name>.json's seed.
        supervised: True for the supervised embedding cache, False for the
              semi-supervised (seed-tagged) one.
        approaches: subset of ['pissa','orthogonal','gaussian','loftq','eva'] to
              include (default: all 5, giving the paper's 5x5 matrix).
        csv_path: long-format CSV (one row per ordered approach pair) - easy to
              pivot/aggregate across datasets/llms/seeds later.
        summary_csv_path: one row per run with the mean off-diagonal CKA (a single
              "ensemble diversity" number) plus the most/least similar pair.

    Returns:
        dict with the CKA matrix (keyed by (approach_i, approach_j)) and the
        ranked approach list used.
    """
    approaches = approaches or APPROACHES
    if len(approaches) < 2:
        raise ValueError("Need at least 2 approaches to compute pairwise CKA")

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    print("=" * 80)
    print(f"CKA | dataset={dataset_name} llm={llm_name} peft={peft_type} "
          f"seed={resolved_seed} supervised={supervised} approaches={approaches}")
    print("=" * 80)

    reporter = ReportResults(
        cfg,
        save_dir=f"results/train_info/{dataset_name}_cka_seed{resolved_seed}",
        index_run=reporter_index,
    )
    reporter.report_title("CKA (Centered Kernel Alignment) - Embedding Diversity Analysis")
    reporter.report_txt(f"Dataset: {dataset_name}")
    reporter.report_txt(f"LLM: {llm_name}")
    reporter.report_txt(f"PEFT Type: {peft_type}")
    reporter.report_txt(f"Seed: {resolved_seed}")
    reporter.report_txt(f"Supervised: {supervised}")
    reporter.report_txt(f"Approaches: {[TITLES[a] for a in approaches]}")
    reporter.report_txt("")

    # Load the seed-tagged embeddings for every requested init approach
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    emb = {name: get_embedding_from_data(all_data[name]) for name in approaches}
    print("Loaded embeddings: " + ", ".join(f"{name}={tuple(emb[name].shape)}" for name in approaches))

    n_rows = {name: emb[name].shape[0] for name in approaches}
    if len(set(n_rows.values())) > 1:
        raise ValueError(f"Embedding sets have mismatched row counts, CKA requires the same examples: {n_rows}")

    # Pairwise 5x5 (or len(approaches)^2) CKA matrix
    cka_matrix = compute_cka_matrix(emb, approaches)

    # ---- report as a formatted matrix ----
    reporter.report_title("CKA Matrix")
    header = " " * 12 + "".join(f"{TITLES[a]:>12}" for a in approaches)
    print(header)
    reporter.report_txt(header)
    for a in approaches:
        row = f"{TITLES[a]:<12}" + "".join(f"{cka_matrix[(a, b)]:>12.4f}" for b in approaches)
        print(row)
        reporter.report_txt(row)
    reporter.report_txt("")

    # ---- diversity summary: mean off-diagonal CKA (lower = more diverse ensemble) ----
    off_diag_pairs = [(a, b) for a in approaches for b in approaches if a != b]
    off_diag_values = [cka_matrix[p] for p in off_diag_pairs]
    mean_off_diag = sum(off_diag_values) / len(off_diag_values)
    most_similar_pair = max(off_diag_pairs, key=lambda p: cka_matrix[p])
    most_diverse_pair = min(off_diag_pairs, key=lambda p: cka_matrix[p])

    summary_lines = [
        f"Mean off-diagonal CKA (lower = more diverse ensemble members): {mean_off_diag:.4f}",
        f"Most similar pair (highest CKA):  {TITLES[most_similar_pair[0]]} vs {TITLES[most_similar_pair[1]]} "
        f"= {cka_matrix[most_similar_pair]:.4f}",
        f"Most diverse pair (lowest CKA):   {TITLES[most_diverse_pair[0]]} vs {TITLES[most_diverse_pair[1]]} "
        f"= {cka_matrix[most_diverse_pair]:.4f}",
    ]
    reporter.report_title("Diversity Summary")
    for line in summary_lines:
        print(line)
        reporter.report_txt(line)

    # ---- heatmap visualization (paper figure) ----
    heatmap_path = None
    if save_heatmap:
        heatmap_path = os.path.join(
            heatmap_dir, f"{dataset_name}_{llm_name}_{peft_type}_seed{resolved_seed}_cka_heatmap.png")
        save_cka_heatmap(cka_matrix, approaches, heatmap_path)
        reporter.report_txt(f"\nCKA heatmap saved to: {heatmap_path}")

    # ---- CSV: one row per ordered pair (long format, easy to pivot/aggregate) ----
    if csv_path:
        for a in approaches:
            for b in approaches:
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_name": dataset_name,
                    "llm_name": llm_name,
                    "peft_type": peft_type,
                    "seed": resolved_seed,
                    "supervised": supervised,
                    "approach_i": TITLES[a],
                    "approach_j": TITLES[b],
                    "cka": cka_matrix[(a, b)],
                }
                append_gnn_result_csv(row, csv_path)
        print(f"[OK] CKA pairwise results appended to: {csv_path}")

    # ---- CSV: one summary row per run ----
    if summary_csv_path:
        summary_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "llm_name": llm_name,
            "peft_type": peft_type,
            "seed": resolved_seed,
            "supervised": supervised,
            "approaches": ",".join(TITLES[a] for a in approaches),
            "mean_off_diagonal_cka": mean_off_diag,
            "most_similar_pair": f"{TITLES[most_similar_pair[0]]}-{TITLES[most_similar_pair[1]]}",
            "most_similar_cka": cka_matrix[most_similar_pair],
            "most_diverse_pair": f"{TITLES[most_diverse_pair[0]]}-{TITLES[most_diverse_pair[1]]}",
            "most_diverse_cka": cka_matrix[most_diverse_pair],
            "heatmap_path": heatmap_path or "",
        }
        append_gnn_result_csv(summary_row, summary_csv_path)
        print(f"[OK] CKA summary appended to: {summary_csv_path}")

    return {
        "cka_matrix": cka_matrix,
        "approaches": approaches,
        "mean_off_diagonal_cka": mean_off_diag,
        "most_similar_pair": most_similar_pair,
        "most_diverse_pair": most_diverse_pair,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the pairwise linear-CKA matrix between LoRA-init embedding sets for one dataset/llm/seed")
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches (must match --seed used in '
                              'train_llm/train_other_lora_init_apporaches.py and the caching step)')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embedding cache, 0 = semi-supervised (seed-tagged) cache')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                         help=f"Comma-separated subset of {APPROACHES} to include")
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/cka/cka_pairwise_results.csv')
    parser.add_argument('--summary_csv_path', type=str, default='results/cka/cka_summary_results.csv')
    parser.add_argument('--no_heatmap', action='store_true', help='Skip saving the CKA heatmap PNG')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        approaches=[a.strip() for a in args.approaches.split(',') if a.strip()],
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
        summary_csv_path=args.summary_csv_path,
        save_heatmap=not args.no_heatmap,
    )
