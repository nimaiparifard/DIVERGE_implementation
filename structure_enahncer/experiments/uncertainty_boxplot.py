# Uncertainty edge detection — box plot experiment
# Box plots for uncertainty of edges that exist in the graph:
#   - Std: standard deviation of edge probabilities across K models (high std → high uncertainty)
#   - Entropy: binary entropy of mean probability H = -p*log(p) - (1-p)*log(1-p)
# No edge removal; only compute both metrics and visualize distributions. Save in results folder.

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repo root is on path (experiments -> structure_hybrid_enahncer -> TAPE -> LLMReasoner -> repo root)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import set_mgnn_cfg
from LLMReasoner.TAPE.structure_hybrid_enahncer.uncertainty_edge_detection import NoisyEdgeDetector

# Default results directory (structure_hybrid_enahncer/results or experiments/results)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCTURE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(STRUCTURE_DIR, 'results', 'uncertainty_boxplot')
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_uncertainty_boxplots(std_scores, entropy_scores, dataset_name, save_dir=None):
    """
    Create box plots for Std and Entropy uncertainty over all edges (no removal).

    Args:
        std_scores: tensor or array [num_edges] — std across K models per edge
        entropy_scores: tensor or array [num_edges] — binary entropy of mean p per edge
        dataset_name: str, for title and filename
        save_dir: directory to save figures (default: RESULTS_DIR)
    """
    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    std_np = np.asarray(std_scores).flatten()
    entropy_np = np.asarray(entropy_scores).flatten()

    # --- Figure 1: Side-by-side box plots (raw values, each on its own scale) ---
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Std
    axes[0].boxplot(
        std_np,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    axes[0].set_ylabel('Standard deviation', fontsize=12)
    axes[0].set_title('Std across K models\n(high → high uncertainty)', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels([f'Edges (n={len(std_np)})'])
    axes[0].grid(axis='y', alpha=0.3)

    # Entropy
    axes[1].boxplot(
        entropy_np,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    axes[1].set_ylabel('Binary entropy', fontsize=12)
    axes[1].set_title('Entropy of mean p\nH = -p log p - (1-p) log(1-p)', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels([f'Edges (n={len(entropy_np)})'])
    axes[1].grid(axis='y', alpha=0.3)

    fig1.suptitle(f'Uncertainty distribution — {dataset_name} (existing edges)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path1 = os.path.join(save_dir, f'uncertainty_boxplot_raw_{dataset_name}.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved raw box plots: {path1}")

    # --- Figure 2: Single plot with both metrics normalized to [0,1] for comparison ---
    std_min, std_max = std_np.min(), std_np.max()
    entropy_min, entropy_max = entropy_np.min(), entropy_np.max()
    std_norm = (std_np - std_min) / (std_max - std_min + 1e-10)
    entropy_norm = (entropy_np - entropy_min) / (entropy_max - entropy_min + 1e-10)

    fig2, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [std_norm, entropy_norm],
        labels=['Std (norm.)', 'Entropy (norm.)'],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    ax.set_ylabel('Normalized uncertainty [0, 1]', fontsize=12)
    ax.set_title(f'Uncertainty comparison (normalized) — {dataset_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.8)
    plt.tight_layout()
    path2 = os.path.join(save_dir, f'uncertainty_boxplot_normalized_{dataset_name}.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved normalized comparison: {path2}")

    # --- Figure 3: Seaborn-style combined distribution (violin or box) ---
    # Build long-format dataframe for seaborn
    n = len(std_np)
    metric = np.array(['Std'] * n + ['Entropy'] * n)
    values = np.concatenate([std_norm, entropy_norm])

    fig3, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=metric, y=values, ax=ax, palette=['#4e79a7', '#f28e2b'], width=0.5)
    ax.set_ylabel('Normalized uncertainty [0, 1]', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title(f'Uncertainty metrics — {dataset_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(save_dir, f'uncertainty_boxplot_seaborn_{dataset_name}.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved seaborn box plot: {path3}")


def main():
    parser = argparse.ArgumentParser(description='Uncertainty box plot experiment (std vs entropy, no edge removal)')
    parser.add_argument('--dataset_name', type=str, default='cora', help='Dataset name')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', help='LLM name')
    parser.add_argument('--peft_type', type=str, default='lora', help='PEFT type')
    parser.add_argument('--init_weight_approach', type=str, default='loftq',
                        choices=['pissa', 'orthogonal', 'guassian', 'loftq', 'eva'],
                        help='Initialization weight approach')
    parser.add_argument('--num_models', type=int, default=10, help='Number of edge predictor models (K)')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots (default: structure_hybrid_enahncer/results/uncertainty_boxplot)')
    args = parser.parse_args()

    save_dir = args.save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 60)
    print('Uncertainty box plot experiment (std & entropy, no removal)')
    print('=' * 60)
    print(f"Dataset: {args.dataset_name}, LLM: {args.llm_name}, Init: {args.init_weight_approach}")
    print(f"K models: {args.num_models}, Decoder: {args.decoder_type}")
    print('=' * 60)

    # Config and data
    cfg = setup_finetuning_cfg(args.dataset_name, args.llm_name, args.peft_type)
    gnn_cfg = set_mgnn_cfg(args.dataset_name)
    dataset = load_dataset(cfg)
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)

    init_to_data = {
        'pissa': data_pissa,
        'orthogonal': data_orthogonal,
        'guassian': data_guassian,
        'loftq': data_loftq,
        'eva': data_eva,
    }
    embedding = get_embedding_from_data(init_to_data[args.init_weight_approach])
    dataset.x = embedding

    # Detector and train K edge predictors
    detector = NoisyEdgeDetector(
        cfg=gnn_cfg,
        features=embedding,
        num_models=args.num_models,
        decoder_type=args.decoder_type,
        training_strategy='with_splitting',
        negative_sampling_ratio=1.0,
    )
    detector.train_edge_predictors()

    # Uncertainty scores for all existing edges (no removal)
    std_scores = detector.calculate_standard_deviation()
    entropy_scores = detector.calculate_entropy()

    # Box plots
    plot_uncertainty_boxplots(
        std_scores,
        entropy_scores,
        dataset_name=args.dataset_name,
        save_dir=save_dir,
    )

    print('=' * 60)
    print('Done. Plots saved under:', save_dir)
    print('=' * 60)


if __name__ == '__main__':
    main()
