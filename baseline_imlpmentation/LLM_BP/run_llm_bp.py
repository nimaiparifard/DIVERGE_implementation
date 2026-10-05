"""
LLM-BP (Wang et al., "Generalization Principles for Inference over Text-Attributed
Graphs with Large Language Models", ICML 2025) adapted to supervised node classification.

Pipeline (paper Algorithm 1, few-shot branch, with every training node as a "shot"):
  1. Task-adaptive node embeddings h_i from a frozen LLM with a class-conditional
     prompt (LLM_BP.embedding) -- computed once per (dataset, llm, encoding) and cached.
  2. Class embeddings q_k = mean of the training-node embeddings of class k.
  3. Node potentials p_i^(0) = softmax(cos(h_i, q_k) / tau) (Eq. 5); training nodes
     are clamped to their labels.
  4. Edge potential from the homophily level r measured on training edges (Eq. 6),
     or a full class-compatibility matrix (supervised extension).
  5. Loopy BP for L layers (Eq. 4) or its one-layer linear approximation (Eq. 7).
tau and L are picked on the validation set (the paper fixes them: tau=0.025 / 0.01, L=5).
Nothing is trained, so every seed runs in seconds once the embeddings are cached.

Usage (from the repo root):
    python -m LLM_BP.run_llm_bp --dataset cora --llm_name llama_3.2_1B --seeds 0 1 2 3
    python -m LLM_BP.run_llm_bp --dataset citeseer --encoding vanilla --no_tune
"""
import argparse
import itertools
import json
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import load_graph_dataset, set_seed
from common.metrics import compute_acc_and_f1

from LLM_BP.bp import (
    bp_approx, clamp_labeled, class_embeddings, homophily_from_counts, log_node_prior,
    loopy_bp, matrix_edge_potential, prepare_edges, scalar_edge_potential, train_edge_statistics,
)
from LLM_BP.embedding import ENCODINGS, get_or_build_embeddings

LLM_BP_DIR = os.path.dirname(os.path.abspath(__file__))

METHODS = ["Raw", "BP", "BP (appr.)", "BP (matrix)"]
TAU_GRID = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
LAYER_GRID = [1, 2, 3, 4, 5]
# Paper Appendix B.2
PAPER_TAU = {"BP": 0.025, "BP (appr.)": 0.01, "BP (matrix)": 0.025, "Raw": 0.025}
PAPER_LAYERS = 5


def resolve_path_prefix():
    cwd = os.path.abspath(os.getcwd())
    if cwd == os.path.abspath(REPO_ROOT):
        return "."
    return os.path.normpath(os.path.relpath(REPO_ROOT, start=cwd))


def load_graph(dataset_name, device, seed, split):
    """split='repo' -> 60/20/20 per seed (honours seed_snapshot/splits); 'default' -> the dataset's own split."""
    data = load_graph_dataset(
        dataset_name, device, re_split=(split == "repo"), path_prefix=resolve_path_prefix(), seed=seed,
    )
    data.y = data.y.squeeze().long()
    return data


def accuracy(log_probs, y, mask):
    return (log_probs[mask].argmax(dim=-1) == y[mask]).float().mean().item()


def run_method(method, log_prior_fn, graph, tau, num_layers):
    src, dst, rev, psi_scalar, psi_matrix, r, clamp_mask = graph
    log_prior = log_prior_fn(tau)
    if method == "Raw":
        return log_prior
    if method == "BP":
        return loopy_bp(log_prior, src, dst, rev, psi_scalar, num_layers, clamp_mask)
    if method == "BP (matrix)":
        return loopy_bp(log_prior, src, dst, rev, psi_matrix, num_layers, clamp_mask)
    if method == "BP (appr.)":
        return bp_approx(log_prior, src, dst, r)
    raise ValueError(method)


def run_one_seed(args, embeddings, seed, device):
    set_seed(seed)
    data = load_graph(args.dataset, device, seed, args.split)
    y, train_mask, val_mask, test_mask = data.y, data.train_mask, data.val_mask, data.test_mask
    num_classes = int(y.max()) + 1
    emb = embeddings.to(device)

    q = class_embeddings(emb, y, train_mask, num_classes)
    clamp_mask = None if args.no_clamp_train else train_mask

    def log_prior_fn(tau):
        log_prior = log_node_prior(emb, q, tau)
        if clamp_mask is not None:
            log_prior = clamp_labeled(log_prior, y, clamp_mask, args.label_smoothing)
        return log_prior

    src, dst, rev = prepare_edges(data.edge_index, data.num_nodes)
    counts = train_edge_statistics(src, dst, y, train_mask, num_classes)
    r = args.homophily if args.homophily is not None else homophily_from_counts(counts)
    graph = (
        src, dst, rev,
        scalar_edge_potential(r, num_classes),
        matrix_edge_potential(counts, args.matrix_alpha),
        r, clamp_mask,
    )
    true_r = float((y[src] == y[dst]).float().mean())
    print(f"  seed={seed} train/val/test={int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())} "
          f"r(train edges)={r:.3f} r(all edges, reference only)={true_r:.3f}")

    results = {}
    for method in args.methods:
        if args.no_tune:
            candidates = [(PAPER_TAU[method], PAPER_LAYERS)]
        elif method in ("Raw", "BP (appr.)"):
            candidates = [(tau, 1) for tau in TAU_GRID]
        else:
            candidates = list(itertools.product(TAU_GRID, LAYER_GRID))

        best = None
        for tau, num_layers in candidates:
            log_probs = run_method(method, log_prior_fn, graph, tau, num_layers)
            val_acc = accuracy(log_probs, y, val_mask)
            if best is None or val_acc > best[0]:
                best = (val_acc, tau, num_layers, log_probs)

        val_acc, tau, num_layers, log_probs = best
        pred = log_probs.argmax(dim=-1)
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(
            pred[test_mask].cpu().numpy(), y[test_mask].cpu().numpy()
        )
        results[method] = {
            "test_acc": acc, "test_macro_f1": macro_f1, "test_weighted_f1": weighted_f1,
            "val_acc": round(val_acc * 100, 2), "tau": tau,
            "num_layers": None if method in ("Raw", "BP (appr.)") else num_layers,
        }
        print(f"  seed={seed} method={method:12s} test_acc={acc:.2f} val_acc={val_acc * 100:.2f} "
              f"tau={tau} L={results[method]['num_layers']}")

    results["homophily_r"] = r
    return results


def aggregate(all_seed_results, methods):
    summary = {}
    for name in methods:
        accs = [r[name]["test_acc"] for r in all_seed_results]
        macro_f1s = [r[name]["test_macro_f1"] for r in all_seed_results]
        weighted_f1s = [r[name]["test_weighted_f1"] for r in all_seed_results]
        summary[name] = {
            "test_acc_mean": round(statistics.mean(accs), 2),
            "test_acc_std": round(statistics.pstdev(accs), 2) if len(accs) > 1 else 0.0,
            "test_macro_f1_mean": round(statistics.mean(macro_f1s), 2),
            "test_weighted_f1_mean": round(statistics.mean(weighted_f1s), 2),
        }
    return summary


def print_summary_table(args, summary):
    print(f"\n{'=' * 72}")
    print(f"LLM-BP results -- dataset={args.dataset} llm={args.llm_name} encoding={args.encoding} split={args.split}")
    print(f"{'=' * 72}")
    print(f"{'Method':12s} {'Test Acc (%)':>18s} {'Macro F1':>12s} {'Weighted F1':>14s}")
    for name, m in summary.items():
        acc_str = f"{m['test_acc_mean']:.2f} +/- {m['test_acc_std']:.2f}"
        print(f"{name:12s} {acc_str:>18s} {m['test_macro_f1_mean']:>12.2f} {m['test_weighted_f1_mean']:>14.2f}")
    print(f"{'=' * 72}\n")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-BP (supervised adaptation) on a text-attributed graph.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--llm_name", type=str, default="llama_3.2_1B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--split", type=str, default="repo", choices=["repo", "default"],
                        help="repo: 60/20/20 per seed (seed_snapshot); default: the dataset's own split")
    parser.add_argument("--encoding", type=str, default="task_adaptive", choices=ENCODINGS)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--max_text_tokens", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--no_tune", action="store_true", help="use the paper's fixed tau / L instead of validation tuning")
    parser.add_argument("--no_clamp_train", action="store_true",
                        help="do not clamp training nodes to their labels (paper's few-shot behaviour)")
    parser.add_argument("--label_smoothing", type=float, default=1e-3)
    parser.add_argument("--homophily", type=float, default=None, help="override r instead of measuring it on training edges")
    parser.add_argument("--matrix_alpha", type=float, default=1.0, help="Laplace smoothing of the class-compatibility matrix")
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--results_dir", type=str, default=os.path.join(LLM_BP_DIR, "results"))
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    # Text and class names are static across seeds, so embeddings are built once
    data = load_graph(args.dataset, device, args.seeds[0], args.split)
    class_names = [str(c) for c in data.label_name]
    print(f"Encoding {len(data.raw_texts)} node texts with frozen LLM '{args.llm_name}' "
          f"(dataset={args.dataset}, encoding={args.encoding}, pooling={args.pooling})...")
    embeddings = get_or_build_embeddings(
        args.dataset, data.raw_texts, class_names,
        llm_name=args.llm_name, encoding=args.encoding, pooling=args.pooling, device=device,
        max_text_tokens=args.max_text_tokens, batch_size=args.batch_size,
        force_rebuild=args.force_rebuild_embeddings,
    )
    print(f"Embeddings ready: {tuple(embeddings.shape)}")
    del data

    all_seed_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        all_seed_results.append(run_one_seed(args, embeddings, seed, device))

    summary = aggregate(all_seed_results, args.methods)
    print_summary_table(args, summary)

    tune_tag = "fixed" if args.no_tune else "tuned"
    output_path = os.path.join(
        args.results_dir,
        f"{args.dataset}_{args.llm_name}_{args.encoding}_{args.pooling}_{args.split}_{tune_tag}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"args": vars(args), "per_seed_results": all_seed_results, "summary": summary},
            f, indent=2,
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
