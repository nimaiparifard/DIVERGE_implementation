"""
SKETCH end-to-end runner for supervised node classification.

Pipeline (Zhou et al., ACL 2025, "Taming Language Models for Text-attributed Graph
Learning with Decoupled Aggregation"):
  1. Semantic retrieval: sentence-transformer embeddings + FAISS cosine top-k (SKETCH.semantic).
  2. Structural retrieval: MinHash-estimated multi-hop Jaccard ranking of K-hop
     neighbours (SKETCH.structural, paper Algorithm 1).
  3. Weighted rank fusion G = R_sem + w * R_struct and concatenation of the retrieved
     texts after the anchor text up to L tokens (SKETCH.aggregate).
  4. A long-context LM (LoRA) is fine-tuned on the enriched texts with the node labels
     of the train split, early-stopped on validation accuracy, and evaluated on test
     (SKETCH.classifier). No GNN is involved.

Steps 1-2 only use node texts and edges (never labels), are identical for every seed
and are cached under SKETCH/cache/.

Usage (from the repo root):
    python -m SKETCH.run_sketch --dataset cora --llm_name llama_3.2_1B --seeds 42
    python -m SKETCH.run_sketch --dataset cora --strategy original          # Table 2 "Original"
    python -m SKETCH.run_sketch --dataset pubmed --split paper --seeds 42 43  # 80/10/10 as in the paper
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
from common.dataloader import re_split_data

from SKETCH.aggregate import STRATEGIES, build_contexts, order_candidates, truncate_texts
from SKETCH.classifier import build_lora_classifier, load_tokenizer, model_max_positions, train_and_evaluate
from SKETCH.semantic import DEFAULT_SENTENCE_ENCODER, get_cache_dir, get_or_build_sentence_embeddings, semantic_topk
from SKETCH.structural import khop_neighbours, structural_topk

SKETCH_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path_prefix():
    cwd = os.path.abspath(os.getcwd())
    if cwd == os.path.abspath(REPO_ROOT):
        return "."
    return os.path.normpath(os.path.relpath(REPO_ROOT, start=cwd))


def load_graph(dataset_name, split, seed, device):
    """
    split="repo":  the 60/20/20 supervised split used by the rest of DIVERGE
                   (load_graph_dataset(re_split=True), honours seed_snapshot/splits).
    split="paper": the 80/10/10 split of the SKETCH paper (Appendix A.1), drawn from `seed`.
    """
    data = load_graph_dataset(dataset_name, device, re_split=True, path_prefix=resolve_path_prefix(), seed=seed)
    data.y = data.y.squeeze()
    if split == "paper":
        data.train_mask, data.val_mask, data.test_mask = re_split_data(
            data.num_nodes, train_percent=0.8, val_percent=0.1, test_percent=0.1, device=device, seed=seed)
    return data


def get_rankings(dataset_name, data, args, device):
    tag = (f"{dataset_name}_{args.sentence_encoder.replace('/', '__')}_k{args.num_hops}"
           f"_l{args.num_hashes_l}_m{args.num_hashes_m}_top{args.top_k}")
    cache_path = os.path.join(get_cache_dir(), f"{tag}_rankings.pt")
    if os.path.exists(cache_path) and not args.force_rebuild:
        print(f"Loading cached rankings from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    t0 = time.time()
    emb = get_or_build_sentence_embeddings(dataset_name, data.raw_texts, args.sentence_encoder, device=device)
    sem_idx, _ = semantic_topk(emb, args.top_k)
    t_sem = time.time() - t0
    print(f"Semantic retrieval done in {t_sem:.1f}s")

    t0 = time.time()
    struct_idx, _, struct_hop = structural_topk(
        data.edge_index, data.num_nodes, args.top_k, num_hops=args.num_hops,
        num_hashes_l=args.num_hashes_l, num_hashes_m=args.num_hashes_m, seed=0)
    t_struct = time.time() - t0
    print(f"Structural (MinHash) retrieval done in {t_struct:.1f}s")

    rankings = {"sem_idx": sem_idx, "struct_idx": struct_idx, "struct_hop": struct_hop,
                "time_semantic_s": t_sem, "time_structural_s": t_struct}
    torch.save(rankings, cache_path)
    return rankings


def aggregate_summary(per_seed):
    summary = {}
    for key in ["val_acc", "test_acc", "test_macro_f1", "test_weighted_f1"]:
        values = [r[key] for r in per_seed]
        summary[f"{key}_mean"] = round(statistics.mean(values), 2)
        summary[f"{key}_std"] = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser(description="SKETCH (semantic + structural retrieval -> LM) for supervised node classification.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--llm_name", type=str, default="llama_3.2_1B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--split", type=str, default="repo", choices=["repo", "paper"])
    parser.add_argument("--strategy", type=str, default="sketch", choices=STRATEGIES)
    # retrieval
    parser.add_argument("--sentence_encoder", type=str, default=DEFAULT_SENTENCE_ENCODER)
    parser.add_argument("--top_k", type=int, default=32, help="candidates kept per retriever")
    parser.add_argument("--num_hops", type=int, default=2, help="k in the paper")
    parser.add_argument("--num_hashes_l", type=int, default=64, help="MinHash length for the 1-hop/1-hop intersection")
    parser.add_argument("--num_hashes_m", type=int, default=16, help="MinHash length for the other hop combinations (m < l)")
    parser.add_argument("--structural_weight", type=float, default=1.5, help="w in G = R_sem + w * R_struct")
    # context
    parser.add_argument("--max_length", type=int, default=2048, help="L: token budget of the concatenated context")
    parser.add_argument("--anchor_tokens", type=int, default=512, help="max tokens kept from the anchor text")
    parser.add_argument("--neighbor_tokens", type=int, default=192, help="max tokens kept from each retrieved text")
    # LM training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--evals_per_epoch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true", help="recompute cached embeddings/rankings")
    parser.add_argument("--retrieval_only", action="store_true", help="build and save contexts, skip LM training")
    parser.add_argument("--results_dir", type=str, default=os.path.join(SKETCH_DIR, "results"))
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    max_pos = model_max_positions(args.llm_name)
    if max_pos is not None and args.max_length > max_pos:
        print(f"[warn] {args.llm_name} supports {max_pos} positions; clipping --max_length {args.max_length} -> {max_pos}")
        args.max_length = max_pos

    data = load_graph(args.dataset, args.split, args.seeds[0], device)
    num_classes = len(data.label_name)
    print(f"Dataset {args.dataset}: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, {num_classes} classes")

    rankings = get_rankings(args.dataset, data, args, device)
    need_khop = args.strategy in ("onehop", "multihop", "combined")
    onehop_lists = khop_neighbours(data.edge_index, data.num_nodes, 1) if args.strategy == "onehop" else None
    khop_lists = khop_neighbours(data.edge_index, data.num_nodes, args.num_hops) if need_khop else None

    tokenizer = load_tokenizer(args.llm_name)
    anchor_texts, anchor_lengths = truncate_texts(data.raw_texts, tokenizer, args.anchor_tokens)
    neighbor_texts, neighbor_lengths = truncate_texts(data.raw_texts, tokenizer, args.neighbor_tokens)
    labels = data.y.cpu().tolist()

    run_name = f"{args.dataset}_{args.llm_name}_{args.strategy}_{args.split}_L{args.max_length}_w{args.structural_weight}"
    per_seed = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        data = load_graph(args.dataset, args.split, seed, device)
        orders = order_candidates(args.strategy, data.num_nodes, rankings["sem_idx"], rankings["struct_idx"],
                                  khop_lists, onehop_lists, args.structural_weight, np.random.default_rng(seed))
        contexts, used = build_contexts(anchor_texts, anchor_lengths, neighbor_texts, neighbor_lengths,
                                        orders, args.max_length)
        print(f"Retrieved texts per node: mean={np.mean(used):.1f} min={np.min(used)} max={np.max(used)}")

        if args.retrieval_only:
            ctx_path = os.path.join(args.results_dir, f"{run_name}_seed{seed}_contexts.json")
            with open(ctx_path, "w", encoding="utf-8") as f:
                json.dump({"contexts": contexts, "retrieved_ids": [o[:u] for o, u in zip(orders, used)]}, f)
            print(f"Saved contexts to {ctx_path}")
            continue

        train_idx = data.train_mask.nonzero().squeeze(-1).cpu().tolist()
        val_idx = data.val_mask.nonzero().squeeze(-1).cpu().tolist()
        test_idx = data.test_mask.nonzero().squeeze(-1).cpu().tolist()
        print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        set_seed(seed)
        model = build_lora_classifier(args.llm_name, num_classes, tokenizer, lora_rank=args.lora_rank,
                                      lora_alpha=args.lora_alpha,
                                      gradient_checkpointing=not args.no_gradient_checkpointing, device=device)
        t0 = time.time()
        results = train_and_evaluate(
            model, tokenizer, contexts, labels, train_idx, val_idx, test_idx, args.max_length,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
            eval_batch_size=args.eval_batch_size, evals_per_epoch=args.evals_per_epoch,
            patience=args.patience, seed=seed, device=device)
        results["train_time_s"] = round(time.time() - t0, 1)
        results["seed"] = seed
        results["mean_retrieved_texts"] = float(np.mean(used))
        print(f"  seed={seed} val_acc={results['val_acc']:.2f} test_acc={results['test_acc']:.2f} "
              f"macro_f1={results['test_macro_f1']:.2f}")
        per_seed.append(results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.retrieval_only:
        return

    summary = aggregate_summary(per_seed)
    print(f"\n{'=' * 70}\nSKETCH -- dataset={args.dataset} llm={args.llm_name} strategy={args.strategy} split={args.split}")
    print(f"Val acc  {summary['val_acc_mean']:.2f} +/- {summary['val_acc_std']:.2f}")
    print(f"Test acc {summary['test_acc_mean']:.2f} +/- {summary['test_acc_std']:.2f}  "
          f"macro-F1 {summary['test_macro_f1_mean']:.2f}  weighted-F1 {summary['test_weighted_f1_mean']:.2f}\n{'=' * 70}")

    out_path = os.path.join(args.results_dir, f"{run_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "retrieval_time_s": {"semantic": rankings["time_semantic_s"], "structural": rankings["time_structural_s"]},
                   "per_seed_results": per_seed, "summary": summary}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
