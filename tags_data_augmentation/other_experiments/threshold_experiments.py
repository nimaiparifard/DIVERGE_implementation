# comprehensive experiment to show influence of threshold constraint into performance
# like before design comprehensive experiment plots and reports
# in every approaches in constructing augmented dataset

import os
import sys
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import gnn_train_and_report, gnn_train_and_report_with_modified_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.reporter import ReportResults
from LLMReasoner.TAPE.tags_data_augmentation.connect_new_nodes_to_graphs import load_augmented_cache
from LLMReasoner.TAPE.tags_data_augmentation.agumented_dataset import create_augmented_dataset
from LLMReasoner.TAPE.tags_data_augmentation.edge_predictor import edge_predictor_train_and_report


def run_experiment_for_threshold(dataset_name, llm_name, init_weight_approach,
                                  approach_name, approach_type, threshold_value,
                                  cfg, edge_predictor_model, results_dict):
    """
    Run experiment for a specific threshold value.
    
    Args:
        dataset_name: Name of the dataset
        llm_name: Name of the LLM model
        init_weight_approach: Initialization approach ('pissa', 'orthogonal', etc.)
        approach_name: Name of the approach (e.g., 'Cosine Similarity')
        approach_type: Type of approach ('cosine', 'knn', 'predictor', 'hybrid')
        threshold_value: Threshold value to test
        cfg: Configuration object
        edge_predictor_model: Pre-trained edge predictor model
        results_dict: Dictionary to store results
    """
    experiment_key = f"{approach_name}_threshold_{threshold_value}"
    
    print(f"\n{'='*100}")
    print(f"EXPERIMENT: {approach_name} with threshold={threshold_value}")
    print(f"{'='*100}\n")
    
    # Load augmented cache
    try:
        augmented_cache = load_augmented_cache(
            dataset_name=dataset_name,
            model_name=llm_name,
            pooling='mean',
            init_weight_approach=init_weight_approach
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"[SKIP] Skipping experiment")
        return None
    
    # Configure approach parameters based on type
    # Fixed top_k values for this experiment
    top_k_cosine = 5
    top_k_knn = 5
    top_k_predictor = 5
    
    if approach_type == 'cosine':
        use_cosine = True
        use_knn = False
        use_predictor = False
        threshold_cosine = threshold_value
        threshold_knn = 0.5
        threshold_predictor = 0.5
    elif approach_type == 'knn':
        use_cosine = False
        use_knn = True
        use_predictor = False
        threshold_cosine = 0.95
        threshold_knn = threshold_value
        threshold_predictor = 0.5
    elif approach_type == 'predictor':
        use_cosine = False
        use_knn = False
        use_predictor = True
        threshold_cosine = 0.95
        threshold_knn = 0.5
        threshold_predictor = threshold_value
    elif approach_type == 'hybrid':
        # For hybrid, use the same threshold for all methods
        use_cosine = True
        use_knn = True
        use_predictor = True
        # Normalize threshold for different scales
        # Cosine: 0-1 (higher is better)
        # KNN: distance (lower is better) - we'll use 1-threshold for distance
        # Predictor: 0-1 (higher is better)
        threshold_cosine = threshold_value
        threshold_knn = max(0.1, 1.0 - threshold_value)  # Inverse for distance
        threshold_predictor = threshold_value
    else:
        raise ValueError(f"Unknown approach type: {approach_type}")
    
    # Create augmented dataset
    print(f"Creating augmented dataset with threshold={threshold_value}...")
    augmented_dataset, augmented_embeddings = create_augmented_dataset(
        cfg=cfg,
        augmented_cache=augmented_cache,
        init_weight_approach=init_weight_approach,
        used_cosine_similarity=use_cosine,
        used_k_nearest_neighbors=use_knn,
        used_edge_predictor=use_predictor,
        threshold_cosine_similarity=threshold_cosine,
        threshold_k_nearest_neighbors=threshold_knn,
        threshold_edge_predictor=threshold_predictor,
        top_k_cosine_similarity=top_k_cosine,
        top_k_knearest_neighbors=top_k_knn,
        top_k_edge_predictor=top_k_predictor,
        edge_predictor_model=edge_predictor_model if use_predictor else None
    )
    
    print(f"[OK] Augmented dataset created with {augmented_dataset.y.shape[0]} total nodes")
    
    # Train GNN on augmented dataset
    print(f"Training GNN on augmented dataset...")
    augmented_dataset.x = augmented_embeddings
    augmented_results, augmented_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=augmented_dataset,
        embedding=augmented_embeddings,
        dataset_name=dataset_name
    )
    
    print(f"[OK] Test Acc: {augmented_results['test_acc']:.4f}, Test F1: {augmented_results['test_f1']:.4f}")
    
    # Store results
    results_dict[experiment_key] = {
        'approach_name': approach_name,
        'approach_type': approach_type,
        'threshold': threshold_value,
        'test_acc': augmented_results['test_acc'],
        'test_f1': augmented_results['test_f1'],
        'val_acc': augmented_results['val_acc'],
        'val_f1': augmented_results['val_f1'],
        'total_nodes': augmented_dataset.y.shape[0],
        'total_edges': augmented_dataset.edge_index.shape[1],
        'augmented_nodes': augmented_cache['embeddings'].shape[0],
        'actual_thresholds': {
            'cosine': threshold_cosine,
            'knn': threshold_knn,
            'predictor': threshold_predictor
        }
    }
    
    return results_dict[experiment_key]


def create_comprehensive_plots(results_by_approach, baseline_results, dataset_name, save_dir):
    """
    Create comprehensive plots showing the influence of threshold constraint.
    
    Args:
        results_by_approach: Dictionary with results grouped by approach
        baseline_results: Baseline results (no augmentation)
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create main figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Influence of Threshold Constraint on Performance - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    markers = ['o', 's', '^', 'D']
    
    # Plot 1: Test Accuracy vs Threshold
    ax1 = axes[0, 0]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        threshold_values = [r['threshold'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        ax1.plot(threshold_values, test_accs, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax1.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', 
               linewidth=2, label='Baseline', alpha=0.8)
    ax1.set_xlabel('Threshold Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy vs Threshold', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test F1 vs Threshold
    ax2 = axes[0, 1]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        threshold_values = [r['threshold'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        ax2.plot(threshold_values, test_f1s, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax2.axhline(y=baseline_results['test_f1'], color='red', linestyle='--', 
               linewidth=2, label='Baseline', alpha=0.8)
    ax2.set_xlabel('Threshold Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Test F1 Score vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy Improvement vs Threshold
    ax3 = axes[1, 0]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        threshold_values = [r['threshold'] for r in results]
        acc_improvements = [r['test_acc'] - baseline_results['test_acc'] for r in results]
        ax3.plot(threshold_values, acc_improvements, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Threshold Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Improvement (Δ)', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Improvement vs Threshold', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total Edges vs Threshold
    ax4 = axes[1, 1]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        threshold_values = [r['threshold'] for r in results]
        total_edges = [r['total_edges'] for r in results]
        ax4.plot(threshold_values, total_edges, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax4.set_xlabel('Threshold Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total Edges in Graph', fontsize=12, fontweight='bold')
    ax4.set_title('Graph Complexity vs Threshold', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{dataset_name}_threshold_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Threshold analysis plot saved to: {save_path}")
    plt.close()
    
    # Create detailed comparison figure for each approach
    for approach_name, results in results_by_approach.items():
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle(f'{approach_name} - Detailed Analysis of Threshold Impact\n{dataset_name.upper()}', 
                     fontsize=14, fontweight='bold')
        
        threshold_values = [r['threshold'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        total_edges = [r['total_edges'] for r in results]
        acc_improvements = [r['test_acc'] - baseline_results['test_acc'] for r in results]
        
        # Subplot 1: Accuracy with bars
        ax1 = axes2[0, 0]
        bars = ax1.bar(range(len(threshold_values)), test_accs, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', linewidth=2, label='Baseline')
        ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('Test Accuracy by Threshold Value', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(threshold_values)))
        ax1.set_xticklabels([f'{t:.2f}' for t in threshold_values], rotation=45)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, test_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Subplot 2: F1 Score with bars
        ax2 = axes2[0, 1]
        bars = ax2.bar(range(len(threshold_values)), test_f1s, alpha=0.7, color='coral', edgecolor='black')
        ax2.axhline(y=baseline_results['test_f1'], color='red', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test F1 Score', fontsize=11, fontweight='bold')
        ax2.set_title('Test F1 Score by Threshold Value', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(threshold_values)))
        ax2.set_xticklabels([f'{t:.2f}' for t in threshold_values], rotation=45)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, test_f1s):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Subplot 3: Accuracy vs Total Edges (scatter)
        ax3 = axes2[1, 0]
        # Normalize threshold values to 0-1 for colormap
        norm_thresholds = [(t - min(threshold_values)) / (max(threshold_values) - min(threshold_values)) 
                          if max(threshold_values) != min(threshold_values) else 0.5 
                          for t in threshold_values]
        scatter = ax3.scatter(total_edges, test_accs, s=150, c=norm_thresholds, 
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        ax3.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', linewidth=2, label='Baseline')
        for i, t in enumerate(threshold_values):
            ax3.annotate(f'{t:.2f}', (total_edges[i], test_accs[i]), 
                        fontsize=8, ha='center', va='bottom')
        ax3.set_xlabel('Total Edges in Graph', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
        ax3.set_title('Accuracy vs Graph Complexity', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Normalized Threshold', fontsize=9)
        
        # Subplot 4: Summary statistics
        ax4 = axes2[1, 1]
        ax4.axis('off')
        
        best_acc_idx = int(np.argmax([r['val_acc'] for r in results]))  # selected on val, test only reported
        best_f1_idx = int(np.argmax([r['val_f1'] for r in results]))  # selected on val, test only reported
        worst_acc_idx = np.argmin(test_accs)
        
        summary_text = f"Summary - {approach_name}\n\n"
        summary_text += f"{'='*45}\n"
        summary_text += f"Baseline Performance:\n"
        summary_text += f"  Accuracy: {baseline_results['test_acc']:.4f}\n"
        summary_text += f"  F1 Score: {baseline_results['test_f1']:.4f}\n\n"
        summary_text += f"Best Accuracy:\n"
        summary_text += f"  threshold = {threshold_values[best_acc_idx]:.2f}\n"
        summary_text += f"  Accuracy: {test_accs[best_acc_idx]:.4f}\n"
        summary_text += f"  Improvement: {acc_improvements[best_acc_idx]:+.4f}\n\n"
        summary_text += f"Best F1 Score:\n"
        summary_text += f"  threshold = {threshold_values[best_f1_idx]:.2f}\n"
        summary_text += f"  F1: {test_f1s[best_f1_idx]:.4f}\n"
        summary_text += f"  Improvement: {test_f1s[best_f1_idx] - baseline_results['test_f1']:+.4f}\n\n"
        summary_text += f"Worst Accuracy:\n"
        summary_text += f"  threshold = {threshold_values[worst_acc_idx]:.2f}\n"
        summary_text += f"  Accuracy: {test_accs[worst_acc_idx]:.4f}\n\n"
        summary_text += f"Average Performance:\n"
        summary_text += f"  Accuracy: {np.mean(test_accs):.4f}\n"
        summary_text += f"  F1 Score: {np.mean(test_f1s):.4f}\n"
        summary_text += f"  Avg Improvement: {np.mean(acc_improvements):+.4f}\n\n"
        summary_text += f"Std Dev:\n"
        summary_text += f"  Accuracy: {np.std(test_accs):.4f}\n"
        summary_text += f"  F1 Score: {np.std(test_f1s):.4f}\n"
        summary_text += f"{'='*45}"
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save detailed plot
        approach_filename = approach_name.lower().replace(' ', '_')
        save_path2 = os.path.join(save_dir, f'{dataset_name}_{approach_filename}_threshold_detailed.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"[OK] Detailed plot for {approach_name} saved to: {save_path2}")
        plt.close()
    
    # Create sensitivity analysis plot
    fig3, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig3.suptitle(f'Threshold Sensitivity Analysis - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold')
    
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        threshold_values = [r['threshold'] for r in results]
        acc_improvements = [r['test_acc'] - baseline_results['test_acc'] for r in results]
        
        # Calculate sensitivity (change in improvement per threshold unit)
        if len(threshold_values) > 1:
            sensitivities = []
            for j in range(len(threshold_values) - 1):
                delta_improvement = acc_improvements[j+1] - acc_improvements[j]
                delta_threshold = threshold_values[j+1] - threshold_values[j]
                sensitivity = delta_improvement / delta_threshold if delta_threshold != 0 else 0
                sensitivities.append(sensitivity)
            
            # Plot sensitivity
            mid_thresholds = [(threshold_values[j] + threshold_values[j+1]) / 2 
                             for j in range(len(threshold_values) - 1)]
            ax.plot(mid_thresholds, sensitivities, marker=markers[i], linewidth=2, markersize=8,
                   label=approach_name, color=colors[i], alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Threshold Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensitivity (ΔImprovement / ΔThreshold)', fontsize=12, fontweight='bold')
    ax.set_title('How Sensitive is Each Approach to Threshold Changes?', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, f'{dataset_name}_threshold_sensitivity.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"[OK] Sensitivity analysis plot saved to: {save_path3}")
    plt.close()


def main():
    """Main function to run threshold constraint experiments."""
    
    # Configuration
    dataset_name = 'cora'
    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'
    init_weight_approach = 'pissa'
    
    # Approaches to test with their threshold ranges
    # Note: Different approaches have different threshold meanings:
    # - Cosine: 0-1 (higher = more similar, stricter)
    # - KNN: distance (lower = closer, stricter) 
    # - Predictor: 0-1 (higher = more confident, stricter)
    approaches = [
        {
            'name': 'Cosine Similarity', 
            'type': 'cosine',
            'thresholds': [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
        },
        {
            'name': 'K-Nearest Neighbors', 
            'type': 'knn',
            'thresholds': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  # Distance thresholds
        },
        {
            'name': 'Edge Predictor', 
            'type': 'predictor',
            'thresholds': [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        },
        {
            'name': 'Full Hybrid', 
            'type': 'hybrid',
            'thresholds': [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]  # Normalized threshold
        },
    ]
    
    # Setup configuration
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    cfg.peft.init_lora_weights = init_weight_approach
    
    # Setup reporter
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "threshold_experiments")
    os.makedirs(results_dir, exist_ok=True)
    reporter = ReportResults(cfg, save_dir=results_dir, index_run=0)
    
    # Report experiment overview
    threshold_info = "\n".join([
        f"  • {a['name']}: {a['thresholds']}"
        for a in approaches
    ])
    
    reporter.report(
        "Threshold Constraint Experiment",
        f"""
        Dataset: {dataset_name}
        LLM Model: {llm_name}
        PEFT Type: {peft_type}
        Init Weight Approach: {init_weight_approach}
        
        Approaches and Threshold Ranges:
{threshold_info}
        
        Threshold Meanings:
        • Cosine Similarity: 0-1 (higher = more similar)
        • K-Nearest Neighbors: distance (lower = closer)
        • Edge Predictor: 0-1 (higher = more confident)
        • Full Hybrid: normalized 0-1 (higher = stricter)
        
        Experiment Goal:
        Investigate how threshold constraints influence the performance
        of data augmentation. This will help determine the optimal
        strictness level for connecting augmented nodes.
        """
    )
    
    # Load embeddings
    print(f"\n{'='*100}")
    print("Loading Initial Embeddings")
    print(f"{'='*100}\n")
    
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    
    if init_weight_approach == "pissa":
        embedding = get_embedding_from_data(data_pissa)
    elif init_weight_approach == "orthogonal":
        embedding = get_embedding_from_data(data_orthogonal)
    elif init_weight_approach == "guassian":
        embedding = get_embedding_from_data(data_guassian)
    elif init_weight_approach == "loftq":
        embedding = get_embedding_from_data(data_loftq)
    elif init_weight_approach == "eva":
        embedding = get_embedding_from_data(data_eva)
    
    print(f"[OK] Loaded embeddings with shape: {embedding.shape}")
    
    # Train Edge Predictor (needed for some approaches)
    print(f"\n{'='*80}")
    print("Training Edge Predictor")
    print(f"{'='*80}")
    
    edge_predictor_results, edge_predictor_model = edge_predictor_train_and_report(
        dataset_name=dataset_name,
        features=embedding,
        decoder_approach_type='mlp',
        training_strategy='without_splitting',
        negative_sampling_ratio=1.0,
        title=f"Edge Predictor - {init_weight_approach.upper()}",
        does_print_training_process=False,
        reporter=reporter
    )
    
    print(f"[OK] Edge Predictor Trained - Val Acc: {edge_predictor_results['val_acc']:.4f}")
    
    # Train Baseline GNN
    print(f"\n{'='*80}")
    print("Training Baseline GNN")
    print(f"{'='*80}")
    
    baseline_results, baseline_model = gnn_train_and_report(
        dataset_name=dataset_name,
        embedding=embedding,
        title=f"Baseline - {init_weight_approach.upper()}",
        does_print_training_process=False,
        reporter=reporter
    )
    
    print(f"[OK] Baseline - Test Acc: {baseline_results['test_acc']:.4f}, Test F1: {baseline_results['test_f1']:.4f}")
    
    # Calculate total experiments
    total_experiments = sum(len(a['thresholds']) for a in approaches)
    
    # Run experiments
    print(f"\n{'='*100}")
    print(f"Running Experiments: {len(approaches)} approaches × varying thresholds = {total_experiments} total experiments")
    print(f"{'='*100}\n")
    
    all_results = {}
    results_by_approach = {approach['name']: [] for approach in approaches}
    
    current_experiment = 0
    
    for approach in approaches:
        print(f"\n{'='*100}")
        print(f"TESTING APPROACH: {approach['name'].upper()}")
        print(f"{'='*100}\n")
        
        reporter.report_title(f"Approach: {approach['name']}")
        
        for threshold in approach['thresholds']:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] {approach['name']} with threshold={threshold}")
            
            try:
                result = run_experiment_for_threshold(
                    dataset_name=dataset_name,
                    llm_name=llm_name,
                    init_weight_approach=init_weight_approach,
                    approach_name=approach['name'],
                    approach_type=approach['type'],
                    threshold_value=threshold,
                    cfg=cfg,
                    edge_predictor_model=edge_predictor_model,
                    results_dict=all_results
                )
                
                if result is not None:
                    results_by_approach[approach['name']].append(result)
                    
                    # Report individual result
                    acc_improvement = result['test_acc'] - baseline_results['test_acc']
                    f1_improvement = result['test_f1'] - baseline_results['test_f1']
                    
                    reporter.report(
                        f"{approach['name']} - threshold={threshold}",
                        f"""
                        Configuration:
                        • Approach: {approach['name']}
                        • Threshold: {threshold}
                        • Actual Thresholds Used:
                          - Cosine: {result['actual_thresholds']['cosine']:.3f}
                          - K-NN: {result['actual_thresholds']['knn']:.3f}
                          - Predictor: {result['actual_thresholds']['predictor']:.3f}
                        
                        Results:
                        • Test Accuracy: {result['test_acc']:.4f} ({result['test_acc']:.2%})
                        • Test F1 (Macro): {result['test_f1']:.4f}
                        • Val Accuracy: {result['val_acc']:.4f}
                        • Val F1 (Macro): {result['val_f1']:.4f}
                        
                        Improvements:
                        • Accuracy Δ: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)
                        • F1 (Macro) Δ: {f1_improvement:+.4f}
                        
                        Graph Statistics:
                        • Total Nodes: {result['total_nodes']}
                        • Total Edges: {result['total_edges']}
                        • Augmented Nodes: {result['augmented_nodes']}
                        """
                    )
                    
            except Exception as e:
                print(f"\n[ERROR] Failed experiment: {e}")
                reporter.report_txt(f"ERROR in {approach['name']} with threshold={threshold}: {str(e)}")
                continue
    
    # Save results to JSON
    json_path = os.path.join(results_dir, f'{dataset_name}_threshold_results.json')
    with open(json_path, 'w') as f:
        json_results = {
            'baseline': {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                        for key, val in baseline_results.items()},
            'results_by_approach': {},
            'all_results': {}
        }
        
        for approach_name, results in results_by_approach.items():
            json_results['results_by_approach'][approach_name] = [
                {k: (float(v) if isinstance(v, (np.floating, torch.Tensor)) else v) if k != 'actual_thresholds'
                    else {k2: float(v2) for k2, v2 in v.items()}
                 for k, v in r.items()}
                for r in results
            ]
        
        for key, val in all_results.items():
            if val is not None:
                json_results['all_results'][key] = {
                    k: (float(v) if isinstance(v, (np.floating, torch.Tensor)) else v) if k != 'actual_thresholds'
                       else {k2: float(v2) for k2, v2 in v.items()}
                    for k, v in val.items()
                }
        
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Results saved to JSON: {json_path}")
    
    # Create comprehensive plots
    print(f"\n{'='*100}")
    print("Creating Comprehensive Plots")
    print(f"{'='*100}\n")
    
    plots_dir = os.path.join(results_dir, "plots")
    create_comprehensive_plots(results_by_approach, baseline_results, dataset_name, plots_dir)
    
    # Final summary report
    print(f"\n{'='*100}")
    print("GENERATING FINAL SUMMARY")
    print(f"{'='*100}\n")
    
    summary_lines = ["Final Summary - Optimal Threshold Configuration", ""]
    summary_lines.append(f"Baseline Performance:")
    summary_lines.append(f"  • Test Accuracy: {baseline_results['test_acc']:.4f}")
    summary_lines.append(f"  • Test F1 (Macro): {baseline_results['test_f1']:.4f}")
    summary_lines.append("")
    
    for approach_name, results in results_by_approach.items():
        if not results:
            continue
            
        test_accs = [r['test_acc'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        threshold_vals = [r['threshold'] for r in results]
        
        best_acc_idx = int(np.argmax([r['val_acc'] for r in results]))  # selected on val, test only reported
        best_f1_idx = int(np.argmax([r['val_f1'] for r in results]))  # selected on val, test only reported
        
        summary_lines.append(f"{approach_name}:")
        summary_lines.append(f"  Best Accuracy: threshold={threshold_vals[best_acc_idx]:.2f}, acc={test_accs[best_acc_idx]:.4f}, Δ={test_accs[best_acc_idx] - baseline_results['test_acc']:+.4f}")
        summary_lines.append(f"  Best F1 Score: threshold={threshold_vals[best_f1_idx]:.2f}, f1={test_f1s[best_f1_idx]:.4f}, Δ={test_f1s[best_f1_idx] - baseline_results['test_f1']:+.4f}")
        summary_lines.append(f"  Average Accuracy: {np.mean(test_accs):.4f} (std: {np.std(test_accs):.4f})")
        summary_lines.append(f"  Average F1 Score: {np.mean(test_f1s):.4f} (std: {np.std(test_f1s):.4f})")
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    reporter.report("Final Summary", summary_text)
    
    print(summary_text)
    
    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*100}\n")
    print(f"Results directory: {results_dir}")
    print(f"JSON results: {json_path}")
    print(f"Plots directory: {plots_dir}")


if __name__ == "__main__":
    main()
