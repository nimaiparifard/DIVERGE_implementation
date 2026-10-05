# Experiment: Line plot for entropy threshold different ranges and node classification metrics
# 
# This experiment:
# 1. Trains K edge predictors to compute entropy uncertainty scores for all edges
# 2. Sweeps through different threshold values (e.g., 0.1, 0.2, 0.3, ..., 0.7)
# 3. For each threshold:
#    - Removes edges with entropy uncertainty > threshold
#    - Trains GNN on modified graph and records node classification metrics
# 4. Trains baseline GNN once on original graph (no edge removal)
# 5. Creates line plot with three lines:
#    - Line 1: Baseline node classification (original graph, constant)
#    - Line 2: Node classification after edge removal (varies with threshold)
#    - Line 3: Improvement delta (after - baseline)
#
# X-axis: Threshold value (entropy uncertainty)
# Y-axis: Node classification metric (test accuracy, test F1, etc.)

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure repo root is on path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import set_mgnn_cfg, gnn_train_and_report_with_modified_dataset
from LLMReasoner.TAPE.structure_hybrid_enahncer.uncertainty_edge_detection import NoisyEdgeDetector, create_modified_dataset

# Default results directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCTURE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(STRUCTURE_DIR, 'results', 'line_plot_entropy_threshold')
os.makedirs(RESULTS_DIR, exist_ok=True)


def sweep_threshold_and_train(
    detector,
    embedding,
    dataset_name,
    thresholds,
    metric='test_acc'
):
    """
    Sweep through different threshold values, remove edges, train GNN, and collect metrics.
    
    Args:
        detector: NoisyEdgeDetector instance (already trained edge predictors)
        embedding: Node embeddings tensor
        dataset_name: Dataset name string
        thresholds: List of threshold values to try (e.g., [0.1, 0.2, 0.3])
        metric: Metric to track ('test_acc', 'test_f1', 'val_acc', etc.)
    
    Returns:
        baseline_result: dict with baseline metrics (original graph)
        threshold_results: list of dicts, one per threshold, with metrics and removal stats
    """
    print(f"\n{'='*80}")
    print(f"SWEEPING ENTROPY THRESHOLD VALUES")
    print(f"{'='*80}")
    print(f"Threshold values to test: {thresholds}")
    print(f"Tracking metric: {metric}")
    print(f"{'='*80}\n")
    
    # Step 1: Compute entropy uncertainty scores (already computed by detector)
    entropy_scores = detector.calculate_entropy()
    
    # Step 2: Train baseline on original graph (only once)
    print(f"\n{'='*80}")
    print(f"BASELINE: Training on Original Graph (no removal)")
    print(f"{'='*80}")
    baseline_result, baseline_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=detector.dataset,  # Original dataset
        embedding=embedding,
        dataset_name=dataset_name
    )
    baseline_metric = baseline_result[metric]
    print(f"Baseline {metric}: {baseline_metric:.4f}\n")
    
    # Step 3: Sweep through threshold values
    threshold_results = []
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"THRESHOLD = {threshold:.4f}")
        print(f"{'='*80}")
        
        # Remove edges with entropy > threshold
        modified_dataset, removal_stats = create_modified_dataset(
            dataset=detector.dataset,
            uncertainty_scores=entropy_scores,
            threshold=threshold,
            removal_strategy='threshold'
        )
        
        # Train GNN on modified graph
        modified_result, modified_model = gnn_train_and_report_with_modified_dataset(
            modified_dataset=modified_dataset,
            embedding=embedding,
            dataset_name=dataset_name
        )
        
        modified_metric = modified_result[metric]
        improvement = modified_metric - baseline_metric
        
        result_entry = {
            'threshold': threshold,
            'removal_percentage': removal_stats['removal_percentage'],
            'removed_edges': removal_stats['removed_edges'],
            'kept_edges': removal_stats['kept_edges'],
            'metrics': modified_result,
            'metric_value': modified_metric,
            'improvement': improvement,
            'removal_stats': removal_stats
        }
        threshold_results.append(result_entry)
        
        print(f"Threshold {threshold:.4f}: {metric} = {modified_metric:.4f}, "
              f"Removed {removal_stats['removal_percentage']:.2f}%, "
              f"Improvement = {improvement:+.4f}")
    
    return baseline_result, threshold_results


def plot_line_chart(
    baseline_result,
    threshold_results,
    metric='test_acc',
    dataset_name='cora',
    save_dir=None
):
    """
    Create line plot with three lines:
    1. Baseline (constant)
    2. After edge removal (varies with threshold)
    3. Improvement delta (after - baseline)
    
    Args:
        baseline_result: dict with baseline metrics
        threshold_results: list of dicts from sweep_threshold_and_train
        metric: metric name to plot
        dataset_name: dataset name for title
        save_dir: directory to save plot
    """
    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    thresholds = [r['threshold'] for r in threshold_results]
    removal_percentages = [r['removal_percentage'] for r in threshold_results]
    metric_values = [r['metric_value'] for r in threshold_results]
    improvements = [r['improvement'] for r in threshold_results]
    baseline_value = baseline_result[metric]
    
    # Create figure with two subplots: main plot and improvement plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- Plot 1: Main metrics comparison ---
    ax1.plot(
        thresholds,
        [baseline_value] * len(thresholds),
        '--',
        label='Baseline (Original Graph)',
        linewidth=2,
        color='gray',
        alpha=0.7
    )
    ax1.plot(
        thresholds,
        metric_values,
        '-o',
        label=f'After Edge Removal ({metric})',
        linewidth=2,
        color='steelblue',
        markersize=6
    )
    ax1.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax1.set_title(f'Node Classification: Baseline vs After Edge Removal (Entropy Threshold) — {dataset_name}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    if len(thresholds) > 1:
        ax1.set_xlim([min(thresholds) - (max(thresholds) - min(thresholds)) * 0.05, 
                      max(thresholds) + (max(thresholds) - min(thresholds)) * 0.05])
    
    # --- Plot 2: Improvement delta ---
    ax2.plot(
        thresholds,
        improvements,
        '-s',
        label=f'Improvement ({metric})',
        linewidth=2,
        color='coral',
        markersize=6
    )
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Entropy Threshold (Uncertainty)', fontsize=12)
    ax2.set_ylabel(f'Improvement ({metric.replace("_", " ").title()})', fontsize=12)
    ax2.set_title('Improvement Over Baseline', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    if len(thresholds) > 1:
        ax2.set_xlim([min(thresholds) - (max(thresholds) - min(thresholds)) * 0.05, 
                      max(thresholds) + (max(thresholds) - min(thresholds)) * 0.05])
    
    plt.tight_layout()
    
    # Save plot
    metric_clean = metric.replace('_', '')
    save_path = os.path.join(save_dir, f'line_plot_{metric_clean}_entropy_threshold_{dataset_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved line plot: {save_path}")
    plt.close()
    
    # --- Additional: Combined single plot with all three lines ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        thresholds,
        [baseline_value] * len(thresholds),
        '--',
        label='Baseline (Original)',
        linewidth=2,
        color='gray',
        alpha=0.7
    )
    ax.plot(
        thresholds,
        metric_values,
        '-o',
        label=f'After Removal ({metric})',
        linewidth=2,
        color='steelblue',
        markersize=6
    )
    
    # Improvement on secondary y-axis
    ax2_twin = ax.twinx()
    ax2_twin.plot(
        thresholds,
        improvements,
        '-s',
        label='Improvement',
        linewidth=2,
        color='coral',
        markersize=6,
        alpha=0.8
    )
    ax2_twin.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2_twin.set_ylabel(f'Improvement ({metric.replace("_", " ").title()})', fontsize=11, color='coral')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    
    ax.set_xlabel('Entropy Threshold (Uncertainty)', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Node Classification Metrics vs Entropy Threshold — {dataset_name}', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if len(thresholds) > 1:
        ax.set_xlim([min(thresholds) - (max(thresholds) - min(thresholds)) * 0.05, 
                     max(thresholds) + (max(thresholds) - min(thresholds)) * 0.05])
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f'line_plot_combined_{metric_clean}_entropy_threshold_{dataset_name}.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved combined plot: {save_path2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Line plot experiment: entropy threshold vs node classification metrics'
    )
    parser.add_argument('--dataset_name', type=str, default='cora', help='Dataset name')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', help='LLM name')
    parser.add_argument('--peft_type', type=str, default='lora', help='PEFT type')
    parser.add_argument('--init_weight_approach', type=str, default='orthogonal',
                        choices=['pissa', 'orthogonal', 'guassian', 'loftq', 'eva'],
                        help='Initialization weight approach')
    parser.add_argument('--num_models', type=int, default=10,
                        help='Number of edge predictor models (K)')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type')
    parser.add_argument('--threshold_start', type=float, default=0.1,
                        help='Starting threshold value (default: 0.1)')
    parser.add_argument('--threshold_end', type=float, default=0.8,
                        help='Ending threshold value (default: 0.7)')
    parser.add_argument('--threshold_step', type=float, default=0.01,
                        help='Step size for threshold (default: 0.05)')
    parser.add_argument('--metric', type=str, default='test_acc',
                        choices=['test_acc', 'test_f1', 'val_acc', 'val_f1', 'test_weight_f1'],
                        help='Metric to plot (default: test_acc)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    save_dir = args.save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate threshold values
    thresholds = np.arange(args.threshold_start, args.threshold_end + args.threshold_step, args.threshold_step).tolist()
    thresholds = [round(x, 4) for x in thresholds]  # Round to avoid float precision issues
    
    print('=' * 80)
    print('LINE PLOT EXPERIMENT: ENTROPY THRESHOLD vs NODE CLASSIFICATION')
    print('=' * 80)
    print(f"Dataset: {args.dataset_name}, LLM: {args.llm_name}, Init: {args.init_weight_approach}")
    print(f"K models: {args.num_models}, Decoder: {args.decoder_type}")
    print(f"Threshold range: {args.threshold_start} to {args.threshold_end} (step: {args.threshold_step})")
    print(f"Threshold values: {thresholds[:5]}...{thresholds[-5:] if len(thresholds) > 10 else thresholds}")
    print(f"Metric: {args.metric}")
    print('=' * 80)
    
    # Setup config and load data
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
    
    # Create detector and train K edge predictors
    print(f"\n{'='*80}")
    print("STEP 1: Training K Edge Predictors for Entropy Uncertainty")
    print(f"{'='*80}")
    detector = NoisyEdgeDetector(
        cfg=gnn_cfg,
        features=embedding,
        num_models=args.num_models,
        decoder_type=args.decoder_type,
        training_strategy='with_splitting',
        negative_sampling_ratio=1.0,
    )
    detector.train_edge_predictors()
    
    # Sweep threshold values and train GNNs
    print(f"\n{'='*80}")
    print("STEP 2: Sweeping Threshold Values and Training GNNs")
    print(f"{'='*80}")
    baseline_result, threshold_results = sweep_threshold_and_train(
        detector=detector,
        embedding=embedding,
        dataset_name=args.dataset_name,
        thresholds=thresholds,
        metric=args.metric
    )
    
    # Create line plots
    print(f"\n{'='*80}")
    print("STEP 3: Creating Line Plots")
    print(f"{'='*80}")
    plot_line_chart(
        baseline_result=baseline_result,
        threshold_results=threshold_results,
        metric=args.metric,
        dataset_name=args.dataset_name,
        save_dir=save_dir
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline ({args.metric}): {baseline_result[args.metric]:.4f}")
    print(f"\nThreshold Results:")
    for r in threshold_results:
        print(f"  Threshold {r['threshold']:.4f}: {r['metric_value']:.4f} "
              f"(removed {r['removal_percentage']:.2f}%, improvement: {r['improvement']:+.4f})")
    
    best_result = max(threshold_results, key=lambda x: x['metric_value'])
    print(f"\nBest result: Threshold {best_result['threshold']:.4f} "
          f"→ {best_result['metric_value']:.4f} "
          f"(removed {best_result['removal_percentage']:.2f}%, "
          f"improvement: {best_result['improvement']:+.4f})")
    
    print(f"\n{'='*80}")
    print(f"Plots saved to: {save_dir}")
    print('=' * 80)


if __name__ == '__main__':
    main()
