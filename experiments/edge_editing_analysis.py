import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from experiments.edge_editing import (
    keep_train_mistakes,
    calculate_cosine_similarities,
    sample_edges,
    get_precomputed_node_embeddings,
    create_modified_dataset_for_mistakes,
)
from gnns.gnn_mtrainer import (
    get_datasets_path,
    gnn_train_and_report,
    gnn_train_and_report_with_modified_dataset,
)
from report.reporter import ReportResults
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter, ensemble_gnn_learning
from visualize.visualize_gnn_mistakes import get_gnn_embedding_mistakes
from dataset.dataset_loader import load_dataset
from common import load_graph_dataset_for_tape


def plot_cosine_similarity_box_plot(edge_info_lists, similarities_list, titles, cfg, show: bool = False):
    """
    Plot box plots of cosine similarities for multiple embedding approaches.

    For each approach we create **two** boxplots in the same axis:
        - one for edges connecting nodes with the **same label**
        - one for edges connecting nodes with **different labels**

    We also compute mean and std for both groups and save them as a CSV file
    whose name contains the dataset name.

    Args:
        edge_info_lists: list of edge_info lists (as returned by calculate_cosine_similarities)
        similarities_list: list of similarity lists (kept for completeness, not strictly needed)
        titles: list of titles (e.g. ['Pissa', 'Orthogonal', 'Eva'])
        cfg: configuration object with dataset.name
        show: whether to call plt.show()
    """

    num_methods = len(edge_info_lists)
    if num_methods == 0:
        return

    # Prepare directory for saving figures and CSV
    save_dir = os.path.join("results", "edge_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # Create subplots: one column per method, shared y-axis for easier comparison
    fig, axes = plt.subplots(
        1, num_methods,
        figsize=(5 * num_methods, 6),
        sharey=True
    )
    if num_methods == 1:
        axes = [axes]

    # CSV for statistics
    csv_path = os.path.join(
        save_dir,
        f"edge_similarity_boxplot_stats_{cfg.dataset.name}.csv"
    )
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "method", "group", "count", "mean", "std"]
        )

        for idx, (edge_info, sims, title) in enumerate(
            zip(edge_info_lists, similarities_list, titles)
        ):
            # Split similarities by same / different label using edge_info
            same_label_sims = [
                info["similarity"] for info in edge_info if info["same_label"]
            ]
            diff_label_sims = [
                info["similarity"] for info in edge_info if not info["same_label"]
            ]

            ax = axes[idx]
            box_data = [same_label_sims, diff_label_sims]
            bp = ax.boxplot(
                box_data,
                labels=["Same Label", "Different Label"],
                patch_artist=True,
            )
            # Color the boxes
            if len(bp["boxes"]) >= 2:
                bp["boxes"][0].set_facecolor("lightgreen")
                bp["boxes"][1].set_facecolor("lightcoral")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Cosine Similarity", fontsize=10)
            ax.grid(alpha=0.3, axis="y")

            # Compute stats and write to CSV
            for group_name, values in [
                ("same_label", same_label_sims),
                ("different_label", diff_label_sims),
            ]:
                if len(values) == 0:
                    mean_val, std_val = float("nan"), float("nan")
                else:
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values))

                writer.writerow(
                    [
                        cfg.dataset.name,
                        title,
                        group_name,
                        len(values),
                        mean_val,
                        std_val,
                    ]
                )

    fig.suptitle(
        f"Edge Cosine Similarity Boxplots - {cfg.dataset.name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join(
        save_dir,
        f"edge_similarity_boxplots_{cfg.dataset.name}.png"
    )
    try:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Boxplot figure saved to: {fig_path}")
        print(f"Statistics CSV saved to: {csv_path}")
    except Exception as e:
        print(f"Error saving boxplot figure: {e}")

    if show:
        plt.show()


def threshold_analysis(
    cfg,
    reporter,
    base_dataset,
    gnn_dataset,
    emb,
    model,
    approach_name="PISSA",
    th_start=0.1,
    th_end=0.95,
    th_step=0.02,
):
    """
    Sweep edge-similarity thresholds for a given GNN + embedding approach
    and plot accuracy / edge-removal trade-offs.

    Args:
        cfg: finetuning configuration
        reporter: ReportResults instance (or None)
        base_dataset: graph dataset whose edges are edited (same as in edge_editing.main)
        gnn_dataset: dataset used by the GNN trainer (load_dataset(cfg))
        emb: node embeddings (e.g. emb_pissa)
        model: trained GNN model corresponding to `emb`
        approach_name: label for plots / reports (e.g. "PISSA")
        th_start, th_end, th_step: range for similarity threshold sweep
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Precomputed node embeddings (one vector per node)
    node_embeddings = get_precomputed_node_embeddings(base_dataset, emb)

    # Get misclassified node IDs for the given GNN
    _, _, _, mistakes = get_gnn_embedding_mistakes(
        gnn_dataset,
        model,
        emb,
        output_mask="all",
    )
    print(f"Total misclassified nodes for {approach_name}: {len(mistakes)}")
    # only train-node mistakes may drive the edit (val/test mistakes are found with their labels)
    mistakes = keep_train_mistakes(mistakes, cfg.dataset.name, supervised=True)

    # Range of similarity thresholds to evaluate (same as in get_best_similarity_threshold by default)
    th_range = np.arange(th_start, th_end, th_step)

    test_accs = []
    val_accs = []
    removed_edges = []
    modified_edges = []

    print(f"\nRunning threshold analysis for {approach_name} over {len(th_range)} thresholds...")

    for th in th_range:
        print(f"\n=== {approach_name} | Threshold {th:.2f} ===")

        # Remove low-similarity edges incident to misclassified nodes
        modified_dataset, stats = create_modified_dataset_for_mistakes(
            mistakes=mistakes,
            dataset=gnn_dataset,
            node_embeddings=node_embeddings,
            tokenizer=None,           # not used because all node embeddings are precomputed
            model=None,               # not used because all node embeddings are precomputed
            device=device,
            similarity_threshold=th,
            sentence_transformer_used=False,
        )

        # Train/evaluate GNN on the modified graph
        results_mod, _ = gnn_train_and_report_with_modified_dataset(
            modified_dataset=modified_dataset,
            embedding=emb,
            dataset_name=cfg.dataset.name,
        )

        test_accs.append(results_mod["test_acc"])
        val_accs.append(results_mod["val_acc"])
        removed_edges.append(stats["removed"])
        modified_edges.append(stats["modified_edges"])

    test_accs = np.array(test_accs)
    val_accs = np.array(val_accs)
    removed_edges = np.array(removed_edges)
    modified_edges = np.array(modified_edges)

    # Derive some summary statistics
    best_idx = int(np.argmax(val_accs))
    best_th = float(th_range[best_idx])
    best_test_acc = float(test_accs[best_idx])
    best_val_acc = float(val_accs[best_idx])

    print(f"\nThreshold analysis summary ({approach_name}):")
    print(f"  Best threshold (by val acc): {best_th:.4f}")
    print(f"  Best validation accuracy:    {best_val_acc:.4f}")
    print(f"  Corresponding test accuracy: {best_test_acc:.4f}")

    # Report results via reporter
    if reporter is not None:
        report_lines = [
            f"Threshold Analysis - {approach_name} Embeddings",
            "",
            f"Best Threshold (by validation accuracy): {best_th:.4f}",
            f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc:.2%})",
            f"Corresponding Test Accuracy: {best_test_acc:.4f} ({best_test_acc:.2%})",
        ]
        reporter.report(f"Threshold Analysis - {approach_name}", "\n".join(report_lines))

    # Save detailed CSV of the sweep
    save_dir = os.path.join("results", "edge_analysis")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"threshold_analysis_{approach_name.lower()}_{cfg.dataset.name}.csv")

    original_edges = int(gnn_dataset.edge_index.shape[1])
    edge_reduction_pct = (original_edges - modified_edges) / original_edges * 100.0

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "val_acc",
                "test_acc",
                "removed_edges",
                "modified_edges",
                "original_edges",
                "edge_reduction_pct",
            ]
        )
        for th, va, ta, rem, mod, red_pct in zip(
            th_range, val_accs, test_accs, removed_edges, modified_edges, edge_reduction_pct
        ):
            writer.writerow(
                [
                    float(th),
                    float(va),
                    float(ta),
                    int(rem),
                    int(mod),
                    original_edges,
                    float(red_pct),
                ]
            )

    print(f"Threshold analysis CSV saved to: {csv_path}")

    # Plot accuracy vs threshold and edge reduction vs threshold
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(th_range, test_accs, label="Test Accuracy", color="tab:blue", marker="o")
    ax1.plot(th_range, val_accs, label="Val Accuracy", color="tab:green", marker="s", linestyle="--")
    ax1.set_xlabel("Similarity Threshold")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{approach_name} Threshold Sweep - {cfg.dataset.name}")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(th_range, edge_reduction_pct, label="Edge Reduction (%)", color="tab:red", marker="^")
    ax2.set_ylabel("Edge Reduction (%)")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()

    fig_path = os.path.join(save_dir, f"threshold_analysis_{approach_name.lower()}_{cfg.dataset.name}.png")
    try:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Threshold analysis figure saved to: {fig_path}")
    except Exception as e:
        print(f"Error saving threshold analysis figure: {e}")

    # Do not force plt.show(); keep script non-blocking in batch runs

if __name__ == "__main__":
    dataset_name = 'cora'

    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    reporter_index = dataset_name + "_" + "error_rate"
    reporter = ReportResults(cfg, index_run=reporter_index)
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        # If running from project root, use current directory
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            path_prefix = '.'
        else:
            # Otherwise, use relative path from current directory to repo root
            path_prefix = os.path.relpath(repo_root, start=os.getcwd())
            # Normalize path to avoid double slashes
            path_prefix = os.path.normpath(path_prefix)
    else:
        # Fallback to default
        path_prefix = '../..'
    dataset, _, _ = load_graph_dataset_for_tape(dataset_name, 'cuda:0', re_split=1, path_prefix=path_prefix,
                                                seed=cfg.dataset.seed)
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian = get_embedding_from_data(
        data_pissa), get_embedding_from_data(data_orthogonal), get_embedding_from_data(
        data_loftq), get_embedding_from_data(data_eva), get_embedding_from_data(data_guassian)
    sampled_edges = sample_edges(40000, dataset)
    edge_pissa_info, simiarities_pissa = calculate_cosine_similarities(sampled_edges, emb_pissa, dataset,
                                                                       sentence_transformer_used=False)
    edge_orthogonal_info, simiarities_orthogonal = calculate_cosine_similarities(sampled_edges,
                                                                                 emb_orthogonal, dataset,
                                                                                 sentence_transformer_used=False)
    edge_loftq_info, simiarities_loftq = calculate_cosine_similarities(sampled_edges, emb_loftq, dataset,
                                                                       sentence_transformer_used=False)
    edge_eva_info, simiarities_eva = calculate_cosine_similarities(sampled_edges, emb_eva, dataset,
                                                                   sentence_transformer_used=False)
    edge_guassian_info, simiarities_guassian = calculate_cosine_similarities(sampled_edges, emb_guassian,
                                                                             dataset, sentence_transformer_used=False)
    print("edge_pissa_info", edge_pissa_info)

    # you should plot two box plot for each emb and info the box plot should be the same label and different label
    # the ayre in the same image
    # also report mean and std of the same label and different label ans save it in csv wih name of the dataset
    edge_info_lists = [edge_pissa_info, edge_orthogonal_info, edge_eva_info, edge_guassian_info, edge_loftq_info]
    similarites_list = [simiarities_pissa, simiarities_orthogonal, simiarities_eva, simiarities_guassian, simiarities_loftq]
    titles  = ['Pissa', 'Orthogonal', 'Eva', 'Guassian', 'Loftq']
    # plot_cosine_similarity_box_plot(edge_info_lists, similarites_list, titles, cfg)

    # bring on the tran med and embedding and see thresult after moving threhold
    # plot accuracy with diffrent threhold, number of removed edges
    results_pissa, model_pissa = gnn_train_and_report(dataset_name, emb_pissa, title="PISSA",
                                                      does_print_training_process=False, reporter=reporter)
    results_orthogonal, model_orthogonal = gnn_train_and_report(dataset_name, emb_orthogonal, title="ORTHOGONAL",
                                                                does_print_training_process=False, reporter=reporter)
    results_loftq, model_loftq = gnn_train_and_report(dataset_name, emb_loftq, title="LOFTQ",
                                                      does_print_training_process=False, reporter=reporter)
    results_eva, model_eva = gnn_train_and_report(dataset_name, emb_eva, title="EVA", does_print_training_process=False,
                                                  reporter=reporter)
    results_guassian, model_guassian = gnn_train_and_report(dataset_name, emb_guassian, title="GUASSIAN",
                                                            does_print_training_process=False, reporter=reporter)

    # Dataset used by the GNN trainer (same convention as in edge_editing.main)
    gnn_dataset = load_dataset(cfg)

    emb_list = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
    model_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_guassian]
    titles = ['Pissa', 'Orthogonal', 'Loftq', 'Eva', 'Guassian']
    # Run threshold analysis for the PISSA approach
    for emb, model, approach_name in zip(emb_list, model_list, titles):
        threshold_analysis(
            cfg=cfg,
            reporter=reporter,
            base_dataset=dataset,
            gnn_dataset=gnn_dataset,
            emb=emb,
            model=model,
            approach_name=approach_name,
        )
