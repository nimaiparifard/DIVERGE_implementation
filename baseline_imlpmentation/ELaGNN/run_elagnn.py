"""
E-LLaGNN end-to-end runner for supervised node classification.

Jaiswal et al., "All Against Some: Efficient Integration of Large Language Models for
Message Passing in Graph Neural Networks", arXiv:2407.14996, 2024.

Pipeline:
  1. Encode the original node texts with Sentence-BERT (x_i).
  2. Active node selection picks `budget` nodes (pagerank / clustering / text_length / degree / random).
  3. The LLM enhances only those nodes with every prompt in the catalog (cached, on demand).
  4. Train a plain mean-SAGE with per-epoch random mixing of original / enhanced
     features and top-K cosine-re-ranked neighbourhoods; evaluate LLM-free.

Usage (from the repo root):
    python -m ELaGNN.run_elagnn --dataset cora --budget 0.25 --heuristic pagerank --seeds 42 43 44 45 46
    python -m ELaGNN.run_elagnn --dataset cora --budget 0                 # no-LLM reference (0% column of Table 5)
    python -m ELaGNN.run_elagnn --dataset pubmed --n_layers 8 --budget 0.25
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import load_graph_dataset, set_seed

from ELaGNN.encoding import DEFAULT_ENCODER, SentenceEncoder, get_augmented_embeddings, get_original_embeddings
from ELaGNN.generation import DEFAULT_REPO_ID, ensure_augmentations
from ELaGNN.prompts import PROMPT_CATALOG
from ELaGNN.selection import HEURISTICS, select_nodes
from ELaGNN.trainer import train_elagnn

ELAGNN_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path_prefix():
    cwd = os.path.abspath(os.getcwd())
    if cwd == os.path.abspath(REPO_ROOT):
        return "."
    return os.path.normpath(os.path.relpath(REPO_ROOT, start=cwd))


def load_graph(dataset_name, split, seed):
    """
    split="repo":      the 60/20/20 seeded supervised split used across DIVERGE (default).
    split="planetoid": the dataset's own fixed masks (public split for cora/pubmed/citeseer).
    """
    data = load_graph_dataset(dataset_name, "cpu", re_split=(split == "repo"),
                              path_prefix=resolve_path_prefix(), seed=seed)
    data.y = data.y.squeeze()
    return data


def summarize(per_seed):
    keys = ["val_acc", "test_acc", "test_macro_f1", "test_weighted_f1", "test_acc_llm_incorporated"]
    out = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = round(statistics.mean(vals), 2)
        out[f"{k}_std"] = round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description="E-LLaGNN: on-demand LLM enhancement for GNN message passing.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--split", type=str, default="repo", choices=["repo", "planetoid"])
    # LLM budget / selection
    parser.add_argument("--budget", type=float, default=0.25,
                        help="nodes selected for LLM enhancement: fraction of all nodes if <= 1, else an absolute count")
    parser.add_argument("--heuristic", type=str, default="pagerank", choices=HEURISTICS)
    parser.add_argument("--prompts", type=str, nargs="+", default=list(PROMPT_CATALOG.keys()), choices=list(PROMPT_CATALOG.keys()))
    parser.add_argument("--p", type=float, default=0.5, help="probability an eligible node is shown enhanced in an epoch")
    parser.add_argument("--llm_repo", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--gen_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    # neighbourhood
    parser.add_argument("--top_k", type=int, default=5, help="K: neighbours kept after cosine re-ranking")
    parser.add_argument("--fanout", type=int, default=25, help="uniform GraphSAGE sample size before re-ranking")
    # GNN
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--generate_only", action="store_true", help="only run selection + LLM enhancement")
    parser.add_argument("--results_dir", type=str, default=os.path.join(ELAGNN_DIR, "results"))
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    data = load_graph(args.dataset, args.split, args.seeds[0])
    num_classes = len(data.label_name)
    budget = int(round(args.budget * data.num_nodes)) if args.budget <= 1 else int(args.budget)
    print(f"Dataset {args.dataset}: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges, "
          f"{num_classes} classes; LLM budget = {budget} nodes ({100 * budget / data.num_nodes:.1f}%)")

    encoder = SentenceEncoder(args.encoder, device=device)
    x_orig = get_original_embeddings(args.dataset, data.raw_texts, encoder)

    # Active node selection is label-free and split-independent, so it is done once
    sel = select_nodes(args.heuristic, budget, data, x_orig, num_classes, seed=0)
    t_gen = 0.0
    if len(sel) > 0:
        del encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        t0 = time.time()
        augmentations = ensure_augmentations(
            args.dataset, data.raw_texts, list(data.label_name), sel.tolist(), args.prompts,
            repo_id=args.llm_repo, max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size,
            temperature=args.temperature)
        t_gen = time.time() - t0
        encoder = SentenceEncoder(args.encoder, device=device)
        aug_emb, avail = get_augmented_embeddings(args.dataset, augmentations, sel.tolist(), args.prompts,
                                                  encoder, args.llm_repo.replace("/", "__"))
    else:
        aug_emb = torch.zeros(0, len(args.prompts), x_orig.shape[1])
        avail = torch.zeros(0, len(args.prompts), dtype=torch.bool)
    del encoder
    if args.generate_only:
        print(f"Generated/cached enhancements for {len(sel)} nodes.")
        return

    sel_ids = torch.from_numpy(sel)
    hp = {k: getattr(args, k) for k in ["n_layers", "hidden_dim", "dropout", "lr", "weight_decay", "epochs",
                                        "patience", "p", "top_k", "fanout"]}
    per_seed = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        data = load_graph(args.dataset, args.split, seed)
        t0 = time.time()
        res = train_elagnn(data, x_orig, sel_ids, aug_emb, avail, num_classes, hp, seed, device)
        res["seed"], res["train_time_s"] = seed, round(time.time() - t0, 1)
        print(f"  seed={seed} val={res['val_acc']:.2f} test(LLM-free)={res['test_acc']:.2f} "
              f"test(LLM-incorporated)={res['test_acc_llm_incorporated']:.2f} best_epoch={res['best_epoch']}")
        per_seed.append(res)

    summary = summarize(per_seed)
    print(f"\n{'=' * 72}\nE-LLaGNN -- dataset={args.dataset} layers={args.n_layers} budget={budget} "
          f"heuristic={args.heuristic} split={args.split}")
    print(f"Val acc                  {summary['val_acc_mean']:.2f} +/- {summary['val_acc_std']:.2f}")
    print(f"Test acc (LLM-free)      {summary['test_acc_mean']:.2f} +/- {summary['test_acc_std']:.2f}")
    print(f"Test acc (LLM-incorp.)   {summary['test_acc_llm_incorporated_mean']:.2f} +/- "
          f"{summary['test_acc_llm_incorporated_std']:.2f}\n{'=' * 72}")

    budget_tag = f"{args.budget:g}"
    out_path = os.path.join(args.results_dir, f"{args.dataset}_{args.split}_L{args.n_layers}_{args.heuristic}"
                                              f"_b{budget_tag}_K{args.top_k}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "num_enhanced_nodes": int(len(sel)), "generation_time_s": round(t_gen, 1),
                   "per_seed_results": per_seed, "summary": summary}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
