# Experiment: Analyze influence of number of models (K) on node classification metrics using std uncertainty
# 
# This experiment:
# 1. Sweeps through different numbers of edge predictor models (K): 2, 5, 7, 10, 13, 15, 20, 30, 40, 50
# 2. For each K:
#    - Trains K edge predictors to compute std uncertainty scores for all edges
#    - Removes edges based on a fixed removal strategy (top-k % or threshold)
#    - Trains GNN on modified graph and records node classification metrics
# 3. Trains baseline GNN once on original graph (no edge removal)
# 4. Creates line plot showing:
#    - How node classification metrics vary with number of models K
#    - Improvement over baseline for each K
#
# X-axis: Number of models (K)
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
RESULTS_DIR = os.path.join(STRUCTURE_DIR, 'results', 'line_plot_std_num_models')
os.makedirs(RESULTS_DIR, exist_ok=True)


def sweep_num_models_and_train(
    gnn_cfg,
    embedding,
    dataset_name,
    num_models_list,
    removal_strategy='top_k',
    removal_threshold=10.0,  # 10% for top_k, or threshold value for threshold strategy
    metric='test_acc',
    decoder_type='mlp'
):
    """
    Sweep through different numbers of models (K), train edge predictors, remove edges, train GNN.
    
    Args:
        gnn_cfg: GNN configuration object
        embedding: Node embeddings tensor
        dataset_name: Dataset name string
        num_models_list: List of K values to try (e.g., [2, 5, 7, 10, 15, 20])
        removal_strategy: 'top_k' or 'threshold'
        removal_threshold: Threshold value (percentage for top_k, or value for threshold)
        metric: Metric to track ('test_acc', 'test_f1', 'val_acc', etc.)
        decoder_type: Edge predictor decoder type
    
    Returns:
        baseline_result: dict with baseline metrics (original graph)
        num_models_results: list of dicts, one per K, with metrics and stats
    """
    print(f"\n{'='*80}")
    print(f"SWEEPING NUMBER OF MODELS (K)")
    print(f"{'='*80}")
    print(f"K values to test: {num_models_list}")
    print(f"Removal strategy: {removal_strategy}, Threshold: {removal_threshold}")
    print(f"Tracking metric: {metric}")
    print(f"{'='*80}\n")
    
    # Step 1: Train baseline on original graph (only once)
    print(f"\n{'='*80}")
    print(f"BASELINE: Training on Original Graph (no removal)")
    print(f"{'='*80}")
    
    # Create a temporary detector just to get the dataset
    temp_detector = NoisyEdgeDetector(
        cfg=gnn_cfg,
        features=embedding,
        num_models=1,  # Just to initialize, won't train
        decoder_type=decoder_type,
        training_strategy='with_splitting',
        negative_sampling_ratio=1.0,
    )
    
    baseline_result, baseline_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=temp_detector.dataset,  # Original dataset
        embedding=embedding,
        dataset_name=dataset_name
    )
    baseline_metric = baseline_result[metric]
    print(f"Baseline {metric}: {baseline_metric:.4f}\n")
    
    # Step 2: Sweep through different K values
    num_models_results = []
    
    for K in num_models_list:
        print(f"\n{'='*80}")
        print(f"K = {K} MODELS")
        print(f"{'='*80}")
        
        # Create detector with K models and train edge predictors
        detector = NoisyEdgeDetector(
            cfg=gnn_cfg,
            features=embedding,
            num_models=K,
            decoder_type=decoder_type,
            training_strategy='with_splitting',
            negative_sampling_ratio=1.0,
        )
        detector.train_edge_predictors()
        
        # Compute std uncertainty scores
        std_scores = detector.calculate_standard_deviation()
        
        # Remove edges based on strategy
        modified_dataset, removal_stats = create_modified_dataset(
            dataset=detector.dataset,
            uncertainty_scores=std_scores,
            threshold=removal_threshold,
            removal_strategy=removal_strategy
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
            'num_models': K,
            'removal_percentage': removal_stats['removal_percentage'],
            'removed_edges': removal_stats['removed_edges'],
            'kept_edges': removal_stats['kept_edges'],
            'metrics': modified_result,
            'metric_value': modified_metric,
            'improvement': improvement,
            'removal_stats': removal_stats,
            'std_mean': std_scores.mean().item(),
            'std_std': std_scores.std().item(),
        }
        num_models_results.append(result_entry)
        
        print(f"K={K}: {metric} = {modified_metric:.4f}, "
              f"Removed {removal_stats['removal_percentage']:.2f}%, "
              f"Improvement = {improvement:+.4f}, "
              f"Std mean = {result_entry['std_mean']:.4f}")
    
    return baseline_result, num_models_results


def plot_line_chart(
    baseline_result,
    num_models_results,
    metric='test_acc',
    dataset_name='cora',
    removal_strategy='top_k',
    removal_threshold=10.0,
    save_dir=None
):
    """
    Create line plot showing how metrics vary with number of models K.
    
    Args:
        baseline_result: dict with baseline metrics
        num_models_results: list of dicts from sweep_num_models_and_train
        metric: metric name to plot
        dataset_name: dataset name for title
        removal_strategy: removal strategy used
        removal_threshold: threshold value used
        save_dir: directory to save plot
    """
    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    num_models_list = [r['num_models'] for r in num_models_results]
    removal_percentages = [r['removal_percentage'] for r in num_models_results]
    metric_values = [r['metric_value'] for r in num_models_results]
    improvements = [r['improvement'] for r in num_models_results]
    std_means = [r['std_mean'] for r in num_models_results]
    baseline_value = baseline_result[metric]
    
    # Create figure with two subplots: main plot and improvement plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- Plot 1: Main metrics comparison ---
    ax1.plot(
        num_models_list,
        [baseline_value] * len(num_models_list),
        '--',
        label='Baseline (Original Graph)',
        linewidth=2,
        color='gray',
        alpha=0.7
    )
    ax1.plot(
        num_models_list,
        metric_values,
        '-o',
        label=f'After Edge Removal ({metric})',
        linewidth=2,
        color='steelblue',
        markersize=6
    )
    ax1.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    strategy_label = f'{removal_strategy} ({removal_threshold})'
    ax1.set_title(f'Node Classification vs Number of Models K (Std, {strategy_label}) — {dataset_name}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(num_models_list) - 1, max(num_models_list) + 1])
    
    # --- Plot 2: Improvement delta ---
    ax2.plot(
        num_models_list,
        improvements,
        '-s',
        label=f'Improvement ({metric})',
        linewidth=2,
        color='coral',
        markersize=6
    )
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Number of Models (K)', fontsize=12)
    ax2.set_ylabel(f'Improvement ({metric.replace("_", " ").title()})', fontsize=12)
    ax2.set_title('Improvement Over Baseline', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(num_models_list) - 1, max(num_models_list) + 1])
    
    plt.tight_layout()
    
    # Save plot
    metric_clean = metric.replace('_', '')
    strategy_clean = removal_strategy.replace('_', '')
    save_path = os.path.join(save_dir, f'line_plot_{metric_clean}_std_num_models_{strategy_clean}_{dataset_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved line plot: {save_path}")
    plt.close()
    
    # --- Additional: Combined single plot with all three lines ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        num_models_list,
        [baseline_value] * len(num_models_list),
        '--',
        label='Baseline (Original)',
        linewidth=2,
        color='gray',
        alpha=0.7
    )
    ax.plot(
        num_models_list,
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
        num_models_list,
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
    
    ax.set_xlabel('Number of Models (K)', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Node Classification Metrics vs Number of Models K (Std) — {dataset_name}', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(num_models_list) - 1, max(num_models_list) + 1])
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f'line_plot_combined_{metric_clean}_std_num_models_{dataset_name}.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved combined plot: {save_path2}")
    plt.close()
    
    # --- Additional: Plot std statistics vs K ---
    fig3, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        num_models_list,
        std_means,
        '-^',
        label='Mean Std Uncertainty',
        linewidth=2,
        color='purple',
        markersize=6
    )
    ax.set_xlabel('Number of Models (K)', fontsize=12)
    ax.set_ylabel('Mean Std Uncertainty', fontsize=12)
    ax.set_title(f'Std Uncertainty vs Number of Models K — {dataset_name}', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(num_models_list) - 1, max(num_models_list) + 1])
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, f'std_uncertainty_vs_K_{dataset_name}.png')
    plt.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved std uncertainty plot: {save_path3}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Line plot experiment: number of models (K) vs node classification metrics (std)'
    )
    parser.add_argument('--dataset_name', type=str, default='cora', help='Dataset name')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', help='LLM name')
    parser.add_argument('--peft_type', type=str, default='lora', help='PEFT type')
    parser.add_argument('--init_weight_approach', type=str, default='orthogonal',
                        choices=['pissa', 'orthogonal', 'guassian', 'loftq', 'eva'],
                        help='Initialization weight approach')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type')
    parser.add_argument('--num_models_list', type=int, nargs='+', default=[2, 5, 7, 10, 13, 15, 20, 30, 40, 50],
                        help='List of K values to test (default: 2 5 7 10 13 15 20 30 40 50)')
    parser.add_argument('--removal_strategy', type=str, default='top_k',
                        choices=['top_k', 'threshold'],
                        help='Edge removal strategy (default: top_k)')
    parser.add_argument('--removal_threshold', type=float, default=10.0,
                        help='Removal threshold: percentage for top_k, or value for threshold (default: 10.0)')
    parser.add_argument('--metric', type=str, default='test_acc',
                        choices=['test_acc', 'test_f1', 'val_acc', 'val_f1', 'test_weight_f1'],
                        help='Metric to plot (default: test_acc)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    save_dir = args.save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print('=' * 80)
    print('LINE PLOT EXPERIMENT: NUMBER OF MODELS (K) vs NODE CLASSIFICATION (STD)')
    print('=' * 80)
    print(f"Dataset: {args.dataset_name}, LLM: {args.llm_name}, Init: {args.init_weight_approach}")
    print(f"Decoder: {args.decoder_type}")
    print(f"K values: {args.num_models_list}")
    print(f"Removal strategy: {args.removal_strategy}, Threshold: {args.removal_threshold}")
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
    
    # Sweep number of models and train
    print(f"\n{'='*80}")
    print("SWEEPING NUMBER OF MODELS (K) AND TRAINING")
    print(f"{'='*80}")
    baseline_result, num_models_results = sweep_num_models_and_train(
        gnn_cfg=gnn_cfg,
        embedding=embedding,
        dataset_name=args.dataset_name,
        num_models_list=args.num_models_list,
        removal_strategy=args.removal_strategy,
        removal_threshold=args.removal_threshold,
        metric=args.metric,
        decoder_type=args.decoder_type
    )
    
    # Create line plots
    print(f"\n{'='*80}")
    print("CREATING LINE PLOTS")
    print(f"{'='*80}")
    plot_line_chart(
        baseline_result=baseline_result,
        num_models_results=num_models_results,
        metric=args.metric,
        dataset_name=args.dataset_name,
        removal_strategy=args.removal_strategy,
        removal_threshold=args.removal_threshold,
        save_dir=save_dir
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline ({args.metric}): {baseline_result[args.metric]:.4f}")
    print(f"\nNumber of Models Results:")
    for r in num_models_results:
        print(f"  K={r['num_models']:2d}: {r['metric_value']:.4f} "
              f"(removed {r['removal_percentage']:.2f}%, improvement: {r['improvement']:+.4f}, "
              f"std_mean: {r['std_mean']:.4f})")
    
    best_result = max(num_models_results, key=lambda x: x['metric_value'])
    print(f"\nBest result: K={best_result['num_models']} "
          f"→ {best_result['metric_value']:.4f} "
          f"(removed {best_result['removal_percentage']:.2f}%, "
          f"improvement: {best_result['improvement']:+.4f})")
    
    print(f"\n{'='*80}")
    print(f"Plots saved to: {save_dir}")
    print('=' * 80)


if __name__ == '__main__':
    main()
