"""
UltraTAG-S end-to-end experiment runner.

Usage (from the repo root):
    python -m ULTRATAG.run_ultratag --dataset cora   --ratios 0.0 0.8 --seeds 42 43 44
    python -m ULTRATAG.run_ultratag --dataset pubmed --ratios 0.0 0.8 --seeds 42 43 44

Per (dataset, ratio), the expensive LLM-based augmentation (text propagation,
LLM summary/keyword/soft-label generation, virtual-edge construction, PageRank
node selection, LLM edge reconfiguration) is run ONCE using `--augmentation_seed`
and cached; each of `--seeds` then only re-runs the stochastic, comparatively
cheap LM-finetuning + dual-GNN training on top of that fixed augmented graph.
This mirrors STAGE's embedding-caching strategy and is necessary here because
every augmentation run makes O(N) to O(N + |E_important|) local LLM generation
calls -- repeating that per training seed would multiply an already significant
cost for no scientific benefit (the augmentation and split are held fixed; only
model initialization/training stochasticity is what "seeds" should vary).
"""
import argparse
import json
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ULTRATAG.generation import InstructLLM, DEFAULT_REPO_ID
from ULTRATAG.pipeline import build_augmented_graph, get_augmentation_cache_path, run_downstream
from STAGE.embedding import StageTextEmbedder

DEFAULT_HP = dict(
    lm_lr=5e-5, lm_epochs=3, lm_batch_size=8, lm_dropout=0.3,          # paper Section 5.4
    gnn_hidden_dim=64, gnn_lr=1e-2, gnn_weight_decay=5e-4,             # paper Section 5.4
    gnn_dropout=0.5, gnn_epochs=100, gnn_patience=20,
    gnn_topk=10,  # our top-k sparsification of the learned similarity matrix (see dual_gnn.py docstring)
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def aggregate(seed_results):
    keys = seed_results[0].keys()
    summary = {}
    for key in keys:
        values = [r[key] for r in seed_results]
        summary[f"{key}_mean"] = round(statistics.mean(values), 2)
        summary[f"{key}_std"] = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the UltraTAG-S pipeline on a text-attributed graph dataset.")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed"])
    parser.add_argument("--llm_name", type=str, default="llama_3.2_1B", help="LM finetuned in module 4.2 (LoRA).")
    parser.add_argument("--generator_repo", type=str, default=DEFAULT_REPO_ID, help="Instruct LLM for text/edge generation.")
    parser.add_argument("--embedder_llm_name", type=str, default="llama_3.2_1B", help="Frozen LM for virtual-edge embeddings.")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--augmentation_seed", type=int, default=None, help="Defaults to the first --seeds value.")
    parser.add_argument("--force_rebuild_augmentation", action="store_true")
    args = parser.parse_args()

    augmentation_seed = args.augmentation_seed if args.augmentation_seed is not None else args.seeds[0]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for ratio in args.ratios:
        print(f"\n{'=' * 70}\nDataset={args.dataset}  Sparsity ratio={ratio:.0%}\n{'=' * 70}")

        cache_path = get_augmentation_cache_path(args.dataset, ratio, augmentation_seed)
        if os.path.exists(cache_path) and not args.force_rebuild_augmentation:
            # Cache hit: skip loading the generator/embedder models entirely.
            bundle = build_augmented_graph(args.dataset, ratio, augmentation_seed, None, None, device)
        else:
            # IMPORTANT: the generator (InstructLLM) and embedder (StageTextEmbedder)
            # must be freed before downstream LM-finetuning/dual-GNN training below.
            # Keeping all three models resident pushed a 12GB card to ~96% VRAM use,
            # which (empirically measured -- see ULTRATAG/README.md "Runtime") caused
            # catastrophic per-step slowdown (~330s/step vs. ~0.15s/step in isolation),
            # almost certainly from CUDA falling back to memory-pressure paging.
            print(f"Loading generation LLM '{args.generator_repo}' and embedding LM '{args.embedder_llm_name}'...")
            generator = InstructLLM(repo_id=args.generator_repo)
            embedder = StageTextEmbedder(llm_name=args.embedder_llm_name, device=device)

            bundle = build_augmented_graph(
                args.dataset, ratio, augmentation_seed, generator, embedder, device,
                force_rebuild=args.force_rebuild_augmentation,
            )

            del generator, embedder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Augmented graph ready: {len(bundle['augmented_texts'])} nodes, "
              f"{bundle['final_edge_index'].shape[1]} edges, "
              f"{len(bundle['important_nodes'])} important nodes.")

        seed_results = []
        for seed in args.seeds:
            results = run_downstream(bundle, args.llm_name, seed, device, DEFAULT_HP)
            print(f"  seed={seed} test_acc={results['test_acc']:.2f} val_acc={results['val_acc']:.2f}")
            seed_results.append(results)

        summary = aggregate(seed_results)
        all_results[f"ratio_{ratio}"] = {"per_seed": seed_results, "summary": summary}
        print(f"Ratio {ratio:.0%} -> test_acc = {summary['test_acc_mean']:.2f} +/- {summary['test_acc_std']:.2f}")

    output_path = os.path.join(RESULTS_DIR, f"{args.dataset}_{args.llm_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "llm_name": args.llm_name,
                "generator_repo": args.generator_repo,
                "embedder_llm_name": args.embedder_llm_name,
                "seeds": args.seeds,
                "augmentation_seed": augmentation_seed,
                "hyperparameters": DEFAULT_HP,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
