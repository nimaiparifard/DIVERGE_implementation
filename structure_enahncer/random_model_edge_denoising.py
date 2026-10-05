# Random Edge Denoising Baseline
# Randomly remove a percentage of edges from the graph (no edge predictor).
# Train GNN on original and modified datasets, then compare node classification results.
# Uses the same pipeline as uncertainty_edge_detection.py but without edge predictors.

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import (
    set_mgnn_cfg,
    gnn_train_and_report_with_modified_dataset,
    get_datasets_path,
)
from common import load_graph_dataset_for_tape
from LLMReasoner.TAPE.peft_tape_finetuning.reporter import ReportResults


def create_random_modified_dataset(dataset, remove_percentage, seed=42):
    """
    Create a modified dataset by randomly removing a percentage of edges.

    Args:
        dataset: Original PyG dataset (has .edge_index).
        remove_percentage: Percentage of edges to remove (0–100).
        seed: Random seed for reproducibility.

    Returns:
        modified_dataset: Dataset with randomly removed edges.
        removal_stats: Dict with original_edges, removed_edges, kept_edges,
                       removal_percentage; removed_uncertainty_mean and
                       kept_uncertainty_mean are None (random removal).
    """
    generator = torch.Generator().manual_seed(seed)
    original_edge_index = dataset.edge_index
    num_original_edges = original_edge_index.shape[1]

    num_to_remove = int(num_original_edges * remove_percentage / 100)
    num_to_remove = min(num_to_remove, num_original_edges)
    num_to_keep = num_original_edges - num_to_remove

    # Random permutation and select indices to remove
    perm = torch.randperm(num_original_edges, generator=generator)
    edges_to_remove_indices = perm[:num_to_remove]
    edges_to_keep_mask = torch.ones(num_original_edges, dtype=torch.bool)
    edges_to_keep_mask[edges_to_remove_indices] = False

    kept_edge_index = original_edge_index[:, edges_to_keep_mask]

    modified_dataset = dataset.clone()
    modified_dataset.edge_index = kept_edge_index

    removal_stats = {
        "original_edges": num_original_edges,
        "removed_edges": num_to_remove,
        "kept_edges": num_to_keep,
        "removal_percentage": (num_to_remove / num_original_edges) * 100,
        "removed_uncertainty_mean": None,
        "kept_uncertainty_mean": None,
    }

    print(f"\n{'='*80}")
    print("Random Edge Removal")
    print(f"{'='*80}")
    print(f"Remove percentage: {remove_percentage}%")
    print(f"Original edges: {num_original_edges}")
    print(f"Removed edges: {num_to_remove}")
    print(f"Kept edges: {num_to_keep}")
    print(f"{'='*80}\n")

    return modified_dataset, removal_stats


def compare_results(
    original_results,
    modified_results,
    removal_stats,
    cfg,
    save_dir="results/random_edge_denoising",
):
    """
    Compare and visualize results before and after random edge removal.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("COMPARISON: Original vs Random Edge Denoised Dataset")
    print(f"{'='*80}")
    print("\nORIGINAL DATASET:")
    print(f"  Test Accuracy:       {original_results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {original_results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {original_results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {original_results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {original_results['val_f1']:.4f}")

    print("\nMODIFIED DATASET (After Random Edge Removal):")
    print(f"  Test Accuracy:       {modified_results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {modified_results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {modified_results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {modified_results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {modified_results['val_f1']:.4f}")

    acc_delta = modified_results["test_acc"] - original_results["test_acc"]
    f1_delta = modified_results["test_f1"] - original_results["test_f1"]

    print("\nCHANGE:")
    print(f"  Test Accuracy: {acc_delta:+.4f} ({acc_delta*100:+.2f}%)")
    print(f"  Test F1:      {f1_delta:+.4f}")
    print(f"{'='*80}\n")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ["Test Acc", "Val Acc"]
    original_accs = [original_results["test_acc"], original_results["val_acc"]]
    modified_accs = [modified_results["test_acc"], modified_results["val_acc"]]
    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width / 2, original_accs, width, label="Original", color="skyblue", edgecolor="black")
    axes[0].bar(x + width / 2, modified_accs, width, label="Random denoised", color="lightcoral", edgecolor="black")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    f1_metrics = ["Test F1", "Val F1"]
    original_f1s = [original_results["test_f1"], original_results["val_f1"]]
    modified_f1s = [modified_results["test_f1"], modified_results["val_f1"]]
    axes[1].bar(x - width / 2, original_f1s, width, label="Original", color="skyblue", edgecolor="black")
    axes[1].bar(x + width / 2, modified_f1s, width, label="Random denoised", color="lightcoral", edgecolor="black")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(f1_metrics)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    uncertainty_text = ""
    if removal_stats.get("removed_uncertainty_mean") is not None:
        uncertainty_text = f"""
Uncertainty (removed): {removal_stats['removed_uncertainty_mean']:.4f}
Uncertainty (kept):   {removal_stats['kept_uncertainty_mean']:.4f}"""
    else:
        uncertainty_text = "\nUncertainty: N/A (random removal)"

    summary_text = f"""
Comparison Summary (Random Edge Denoising):
{'='*50}

Edge statistics:
  Original edges:    {removal_stats['original_edges']}
  Removed edges:     {removal_stats['removed_edges']}
  Removal percentage: {removal_stats['removal_percentage']:.2f}%
  Kept edges:        {removal_stats['kept_edges']}

Performance change:
  Test Acc Δ: {acc_delta:+.4f}
  Test F1 Δ:  {f1_delta:+.4f}
  Val Acc Δ:  {modified_results['val_acc'] - original_results['val_acc']:+.4f}
  Val F1 Δ:   {modified_results['val_f1'] - original_results['val_f1']:+.4f}
{uncertainty_text}
"""
    axes[2].axis("off")
    axes[2].text(
        0.05, 0.95, summary_text, transform=axes[2].transAxes,
        fontsize=10, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    dataset_name = cfg.dataset if isinstance(cfg.dataset, str) else getattr(cfg.dataset, "name", "unknown")
    save_path = os.path.join(save_dir, f"comparison_random_{dataset_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved comparison plot to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random edge denoising baseline (no edge predictor)")
    parser.add_argument("--dataset_name", type=str, default="cora", help="Dataset name")
    parser.add_argument("--llm_name", type=str, default="llama_3.2_1B", help="LLM name")
    parser.add_argument("--peft_type", type=str, default="lora", help="PEFT type")
    parser.add_argument(
        "--init_weight_approach",
        type=str,
        default="loftq",
        choices=["pissa", "orthogonal", "guassian", "loftq", "eva"],
        help="Initialization weight approach",
    )
    parser.add_argument(
        "--remove_percentage",
        type=float,
        default=30.0,
        help="Percentage of edges to remove at random (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for edge removal")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("RANDOM EDGE DENOISING (baseline, no edge predictor)")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"LLM: {args.llm_name}")
    print(f"Init: {args.init_weight_approach}")
    print(f"Remove: {args.remove_percentage}% of edges (seed={args.seed})")
    print("=" * 80 + "\n")

    cfg = setup_finetuning_cfg(args.dataset_name, args.llm_name, args.peft_type)
    gnn_cfg = set_mgnn_cfg(args.dataset_name)

    # Path for loading graph dataset (same logic as uncertainty_edge_detection)
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, "datasets")):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            path_prefix = "."
        else:
            path_prefix = os.path.relpath(repo_root, start=os.getcwd())
            path_prefix = os.path.normpath(path_prefix)
    else:
        path_prefix = "../.."

    device = torch.device("cuda:0" if gnn_cfg.device > 0 else "cpu")
    dataset, _, _ = load_graph_dataset_for_tape(
        gnn_cfg.dataset,
        device,
        re_split=gnn_cfg.re_split,
        path_prefix=path_prefix,
        seed=gnn_cfg.split_seed,
    )

    # Embeddings from finetuning data
    _ = load_dataset(cfg)
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    if args.init_weight_approach == "pissa":
        embedding = get_embedding_from_data(data_pissa)
    elif args.init_weight_approach == "orthogonal":
        embedding = get_embedding_from_data(data_orthogonal)
    elif args.init_weight_approach == "guassian":
        embedding = get_embedding_from_data(data_guassian)
    elif args.init_weight_approach == "loftq":
        embedding = get_embedding_from_data(data_loftq)
    elif args.init_weight_approach == "eva":
        embedding = get_embedding_from_data(data_eva)
    else:
        embedding = get_embedding_from_data(data_loftq)

    # Step 1: Baseline GNN on original graph
    print("\n" + "=" * 80)
    print("STEP 1: Training baseline GNN (original graph)")
    print("=" * 80)
    original_results, _ = gnn_train_and_report_with_modified_dataset(
        modified_dataset=dataset,
        embedding=embedding,
        dataset_name=args.dataset_name,
    )

    # Step 2: Random edge removal
    print("\n" + "=" * 80)
    print("STEP 2: Random edge removal")
    print("=" * 80)
    modified_dataset, removal_stats = create_random_modified_dataset(
        dataset=dataset,
        remove_percentage=args.remove_percentage,
        seed=args.seed,
    )

    # Step 3: GNN on modified graph
    print("\n" + "=" * 80)
    print("STEP 3: Training GNN on modified (random denoised) graph")
    print("=" * 80)
    modified_results, _ = gnn_train_and_report_with_modified_dataset(
        modified_dataset=modified_dataset,
        embedding=embedding,
        dataset_name=args.dataset_name,
    )

    # Step 4: Compare and save
    print("\n" + "=" * 80)
    print("STEP 4: Compare and report")
    print("=" * 80)
    compare_results(original_results, modified_results, removal_stats, cfg)

    results_dir = "results/random_edge_denoising"
    os.makedirs(results_dir, exist_ok=True)

    # Report random edge denoising results using ReportResults (filename includes init_weight_approach)
    reporter = ReportResults(
        cfg,
        save_dir=results_dir,
        index_run=0,
        init_approach=f"{args.init_weight_approach}_random_{args.remove_percentage}pct",
    )
    reporter.report_title("RANDOM EDGE DENOISING RESULTS")
    reporter.report_txt(f"Dataset: {args.dataset_name}")
    reporter.report_txt(f"LLM: {args.llm_name}")
    reporter.report_txt(f"Init weight approach: {args.init_weight_approach}")
    reporter.report_txt(f"Remove percentage: {args.remove_percentage}%")
    reporter.report_txt(f"Seed: {args.seed}")
    reporter.report_txt("")
    reporter.report_title("ORIGINAL DATASET")
    reporter.report_txt(f"  Test Accuracy:       {original_results['test_acc']:.4f}")
    reporter.report_txt(f"  Test F1 (Macro):     {original_results['test_f1']:.4f}")
    reporter.report_txt(f"  Test F1 (Weighted):  {original_results['test_weight_f1']:.4f}")
    reporter.report_txt(f"  Val Accuracy:        {original_results['val_acc']:.4f}")
    reporter.report_txt(f"  Val F1 (Macro):      {original_results['val_f1']:.4f}")
    reporter.report_txt("")
    reporter.report_title("MODIFIED DATASET (RANDOM DENOISED)")
    reporter.report_txt(f"  Test Accuracy:       {modified_results['test_acc']:.4f}")
    reporter.report_txt(f"  Test F1 (Macro):     {modified_results['test_f1']:.4f}")
    reporter.report_txt(f"  Test F1 (Weighted):  {modified_results['test_weight_f1']:.4f}")
    reporter.report_txt(f"  Val Accuracy:        {modified_results['val_acc']:.4f}")
    reporter.report_txt(f"  Val F1 (Macro):      {modified_results['val_f1']:.4f}")
    reporter.report_txt("")
    reporter.report_title("EDGE REMOVAL STATISTICS")
    reporter.report_txt(f"  Original edges:      {removal_stats['original_edges']}")
    reporter.report_txt(f"  Removed edges:       {removal_stats['removed_edges']}")
    reporter.report_txt(f"  Kept edges:          {removal_stats['kept_edges']}")
    reporter.report_txt(f"  Removal percentage:  {removal_stats['removal_percentage']:.2f}%")
    reporter.report_txt("")
    acc_delta = modified_results["test_acc"] - original_results["test_acc"]
    f1_delta = modified_results["test_f1"] - original_results["test_f1"]
    reporter.report_title("IMPROVEMENT (MODIFIED - ORIGINAL)")
    reporter.report_txt(f"  Test Accuracy Δ:     {acc_delta:+.4f}")
    reporter.report_txt(f"  Test F1 Δ:           {f1_delta:+.4f}")
    reporter.report_txt(f"  Val Accuracy Δ:      {modified_results['val_acc'] - original_results['val_acc']:+.4f}")
    reporter.report_txt(f"  Val F1 Δ:            {modified_results['val_f1'] - original_results['val_f1']:+.4f}")
    print(f"[OK] Report saved to: {reporter.file_path}")

    results_summary = {
        "dataset": args.dataset_name,
        "llm_name": args.llm_name,
        "init_weight_approach": args.init_weight_approach,
        "remove_percentage": args.remove_percentage,
        "seed": args.seed,
        "original_results": original_results,
        "modified_results": modified_results,
        "removal_stats": removal_stats,
        "improvement": {
            "test_acc": modified_results["test_acc"] - original_results["test_acc"],
            "test_f1": modified_results["test_f1"] - original_results["test_f1"],
            "val_acc": modified_results["val_acc"] - original_results["val_acc"],
            "val_f1": modified_results["val_f1"] - original_results["val_f1"],
        },
    }
    results_path = os.path.join(
        results_dir,
        f"{args.dataset_name}_{args.init_weight_approach}_random_remove_{args.remove_percentage}pct_seed{args.seed}.json",
    )
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[OK] Results saved to: {results_path}")
    print("\n" + "=" * 80)
    print("RANDOM EDGE DENOISING DONE")
    print("=" * 80)
