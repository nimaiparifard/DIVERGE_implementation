"""
STAGE end-to-end experiment runner.

Reproduces the paper's core Table-1-style pipeline:
  1. Encode each node's raw text (title/abstract) with a frozen, zero-shot LLM
     (STAGE.embedding) -- computed once per (dataset, llm) and cached.
  2. Train an ensemble of GCN / SAGE / RevGAT / MLP on those embeddings + the
     graph's adjacency structure (STAGE.trainer / STAGE.models).
  3. Mean-pool the softmax predictions across the ensemble (STAGE.ensemble).
  4. Repeat over several seeds (each with an independent 60/20/20 re-split, as in
     the paper) and report mean +/- std accuracy per model and for the ensemble.

Usage (from the repo root):
    python -m STAGE.run_stage --dataset cora --llm_name llama_3.2_1B --seeds 0 1 2 3
    python -m STAGE.run_stage --dataset pubmed --llm_name qwen2.5_1.5B --seeds 0 1 2 3
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

from common import load_graph_dataset_for_tape, set_seed
from common.dataloader import re_split_data

from STAGE.embedding import get_or_build_stage_embeddings
from STAGE.ensemble import ensemble_mean_probs
from STAGE.models import MODEL_BUILDERS
from STAGE.trainer import StageTrainer
from common.metrics import compute_acc_and_f1


# Shared hyperparameters per dataset. Not exhaustively tuned per-architecture (the
# paper does not report per-architecture GNN hyperparameters either); values follow
# this repo's existing tuned defaults for these datasets (gnn_hyperparameters/*.json).
DATASET_HYPERPARAMS = {
    "cora": dict(hidden_dim=256, n_layers=3, dropout=0.3, lr=0.005, weight_decay=5e-4, epochs=200, patience=30),
    "pubmed": dict(hidden_dim=256, n_layers=3, dropout=0.3, lr=0.005, weight_decay=5e-4, epochs=200, patience=30),
}

TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT = 0.6, 0.2, 0.2


def resolve_path_prefix():
    cwd = os.path.abspath(os.getcwd())
    if cwd == os.path.abspath(REPO_ROOT):
        return "."
    return os.path.normpath(os.path.relpath(REPO_ROOT, start=cwd))


def load_stage_graph(dataset_name, device, seed):
    path_prefix = resolve_path_prefix()
    data, num_classes, raw_texts = load_graph_dataset_for_tape(
        dataset_name, device, re_split=False, path_prefix=path_prefix, seed=seed,
    )
    data.y = data.y.squeeze()
    data.train_mask, data.val_mask, data.test_mask = re_split_data(
        data.num_nodes,
        train_percent=TRAIN_PERCENT,
        val_percent=VAL_PERCENT,
        test_percent=TEST_PERCENT,
        device=device,
        seed=seed,
    )
    return data, num_classes, raw_texts


def run_one_seed(dataset_name, embeddings, seed, hp, device):
    set_seed(seed)
    data, num_classes, _ = load_stage_graph(dataset_name, device, seed)
    features = embeddings.to(device)

    results_per_model = {}
    probs_per_model = {}

    for name, builder in MODEL_BUILDERS.items():
        set_seed(seed)
        model = builder(
            input_dim=features.shape[1],
            output_dim=num_classes,
            hidden_dim=hp["hidden_dim"],
            n_layers=hp["n_layers"],
            dropout=hp["dropout"],
        )
        trainer = StageTrainer(
            model, features, data,
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            epochs=hp["epochs"], patience=hp["patience"], device=device,
        )
        results, _, probs = trainer.train()
        print(f"  seed={seed} model={name:6s} test_acc={results['test_acc']:.2f} val_acc={results['val_acc']:.2f}")
        results_per_model[name] = results
        probs_per_model[name] = probs

    ensemble_probs = ensemble_mean_probs(list(probs_per_model.values()))
    pred = ensemble_probs.argmax(dim=-1)
    acc, macro_f1, weighted_f1 = compute_acc_and_f1(
        pred[data.test_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy()
    )
    results_per_model["Ensemble"] = {"test_acc": acc, "test_macro_f1": macro_f1, "test_weighted_f1": weighted_f1}
    print(f"  seed={seed} model=Ensemble test_acc={acc:.2f}")

    return results_per_model


def aggregate(all_seed_results):
    model_names = list(all_seed_results[0].keys())
    summary = {}
    for name in model_names:
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


def print_summary_table(dataset_name, llm_name, summary):
    print(f"\n{'=' * 70}")
    print(f"STAGE results -- dataset={dataset_name} llm={llm_name}")
    print(f"{'=' * 70}")
    print(f"{'Model':10s} {'Test Acc (%)':>18s} {'Macro F1':>12s} {'Weighted F1':>14s}")
    for name, m in summary.items():
        acc_str = f"{m['test_acc_mean']:.2f} +/- {m['test_acc_std']:.2f}"
        print(f"{name:10s} {acc_str:>18s} {m['test_macro_f1_mean']:>12.2f} {m['test_weighted_f1_mean']:>14.2f}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the STAGE pipeline on a text-attributed graph dataset.")
    parser.add_argument("--dataset", type=str, default="cora", choices=list(DATASET_HYPERPARAMS.keys()))
    parser.add_argument("--llm_name", type=str, default="llama_3.2_1B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--instruction", type=str, default=None, help="Optional task instruction to bias embeddings.")
    parser.add_argument("--instruction_tag", type=str, default="no_instruction")
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    # Text is static across seeds/splits, so raw_texts + embeddings only need to be
    # generated once per (dataset, llm, instruction).
    _, _, raw_texts = load_stage_graph(args.dataset, device, seed=args.seeds[0])
    print(f"Encoding {len(raw_texts)} node texts with frozen LLM '{args.llm_name}' (dataset={args.dataset})...")
    embeddings = get_or_build_stage_embeddings(
        args.dataset, raw_texts,
        llm_name=args.llm_name, device=device,
        instruction=args.instruction, instruction_tag=args.instruction_tag,
        force_rebuild=args.force_rebuild_embeddings,
    )
    print(f"Embeddings ready: {tuple(embeddings.shape)}")

    hp = DATASET_HYPERPARAMS[args.dataset]
    all_seed_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        all_seed_results.append(run_one_seed(args.dataset, embeddings, seed, hp, device))

    summary = aggregate(all_seed_results)
    print_summary_table(args.dataset, args.llm_name, summary)

    output_path = os.path.join(args.results_dir, f"{args.dataset}_{args.llm_name}_{args.instruction_tag}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "llm_name": args.llm_name,
                "instruction": args.instruction,
                "seeds": args.seeds,
                "hyperparameters": hp,
                "per_seed_results": all_seed_results,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
