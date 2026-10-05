# comprehensive report and plots for hybrid constructing augmented dataset
# compare and plot all possible approach for constructing augmented dataset with different approach: cosine, edge predictor, k nearest neighbor
# Comprehensive plot to show changing result accuracy and f1 with accurate plots
# Comprehensive report with reporter

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


def run_experiment_for_hybrid_approach(dataset_name, llm_name, init_weight_approach,
                                       approach_config, cfg, reporter, results_dict,
                                       edge_predictor_model=None):
    """
    Run complete experiment for a single hybrid approach configuration.
    
    Args:
        dataset_name: Name of the dataset
        llm_name: Name of the LLM model
        init_weight_approach: Initialization approach ('pissa', 'orthogonal', etc.)
        approach_config: Dictionary with approach configuration
        cfg: Configuration object
        reporter: Reporter object for logging
        results_dict: Dictionary to store results
        edge_predictor_model: Pre-trained edge predictor model (optional)
    """
    approach_name = approach_config['name']
    
    print(f"\n{'='*100}")
    print(f"RUNNING EXPERIMENT: {approach_name.upper()}")
    print(f"{'='*100}\n")
    
    reporter.report_title(f"Experiment: {approach_name}")
    
    # Load embeddings
    print(f"\n{'='*80}")
    print(f"STEP 1: Loading Embeddings ({init_weight_approach})")
    print(f"{'='*80}")
    
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
    else:
        raise ValueError(f"Unknown init weight approach: {init_weight_approach}")
    
    print(f"[OK] Loaded embeddings with shape: {embedding.shape}")
    
    # Load augmented cache
    print(f"\n{'='*80}")
    print(f"STEP 2: Loading Augmented Cache")
    print(f"{'='*80}")
    
    try:
        augmented_cache = load_augmented_cache(
            dataset_name=dataset_name,
            model_name=llm_name,
            pooling='mean',
            init_weight_approach=init_weight_approach
        )
        print(f"[OK] Loaded augmented cache with {augmented_cache['embeddings'].shape[0]} new nodes")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"[SKIP] Skipping {approach_name}")
        reporter.report_txt(f"SKIPPED: Augmented cache not found for {approach_name}")
        return None
    
    # Create augmented dataset with the specified approach
    print(f"\n{'='*80}")
    print(f"STEP 3: Creating Augmented Dataset - {approach_name}")
    print(f"{'='*80}")
    
    augmented_dataset, augmented_embeddings = create_augmented_dataset(
        cfg=cfg,
        augmented_cache=augmented_cache,
        init_weight_approach=init_weight_approach,
        used_cosine_similarity=approach_config['use_cosine'],
        used_k_nearest_neighbors=approach_config['use_knn'],
        used_edge_predictor=approach_config['use_predictor'],
        threshold_cosine_similarity=approach_config['threshold_cosine'],
        threshold_k_nearest_neighbors=approach_config['threshold_knn'],
        threshold_edge_predictor=approach_config['threshold_predictor'],
        top_k_cosine_similarity=approach_config['top_k_cosine'],
        top_k_knearest_neighbors=approach_config['top_k_knn'],
        top_k_edge_predictor=approach_config['top_k_predictor'],
        edge_predictor_model=edge_predictor_model if approach_config['use_predictor'] else None
    )
    
    print(f"[OK] Augmented dataset created with {augmented_dataset.y.shape[0]} total nodes")
    
    # Train GNN on augmented dataset
    print(f"\n{'='*80}")
    print(f"STEP 4: Training GNN on Augmented Dataset")
    print(f"{'='*80}")
    
    augmented_dataset.x = augmented_embeddings
    augmented_results, augmented_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=augmented_dataset,
        embedding=augmented_embeddings,
        dataset_name=dataset_name
    )
    
    print(f"[OK] Augmented GNN - Test Acc: {augmented_results['test_acc']:.4f}, Test F1: {augmented_results['test_f1']:.4f}")
    
    # Store results
    results_dict[approach_name] = {
        'augmented': augmented_results,
        'config': approach_config,
        'dataset_stats': {
            'original_nodes': augmented_dataset.y.shape[0] - augmented_cache['embeddings'].shape[0],
            'augmented_nodes': augmented_cache['embeddings'].shape[0],
            'total_nodes': augmented_dataset.y.shape[0],
            'total_edges': augmented_dataset.edge_index.shape[1]
        }
    }
    
    return results_dict[approach_name]


def generate_hybrid_approach_configs():
    """
    Generate all possible combinations of hybrid approaches.
    
    Returns:
        List of dictionaries with approach configurations
    """
    configs = []
    
    # Define base parameters
    threshold_cosine = 0.95
    threshold_knn = 0.5
    threshold_predictor = 0.5
    top_k_cosine = 3
    top_k_knn = 3
    top_k_predictor = 5
    
    # Configuration 1: Only Cosine Similarity
    configs.append({
        'name': 'Cosine Only',
        'use_cosine': True,
        'use_knn': False,
        'use_predictor': False,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    # Configuration 2: Only K-Nearest Neighbors
    configs.append({
        'name': 'K-NN Only',
        'use_cosine': False,
        'use_knn': True,
        'use_predictor': False,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    # Configuration 3: Only Edge Predictor
    configs.append({
        'name': 'Edge Predictor Only',
        'use_cosine': False,
        'use_knn': False,
        'use_predictor': True,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    # Configuration 4: Cosine + K-NN
    configs.append({
        'name': 'Cosine + K-NN',
        'use_cosine': True,
        'use_knn': True,
        'use_predictor': False,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    # Configuration 5: Cosine + Edge Predictor
    configs.append({
        'name': 'Cosine + Predictor',
        'use_cosine': True,
        'use_knn': False,
        'use_predictor': True,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    # Configuration 6: K-NN + Edge Predictor
    configs.append({
        'name': 'K-NN + Predictor',
        'use_cosine': False,
        'use_knn': True,
        'use_predictor': True,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    # Configuration 7: All Three (Full Hybrid)
    configs.append({
        'name': 'Full Hybrid (All)',
        'use_cosine': True,
        'use_knn': True,
        'use_predictor': True,
        'threshold_cosine': threshold_cosine,
        'threshold_knn': threshold_knn,
        'threshold_predictor': threshold_predictor,
        'top_k_cosine': top_k_cosine,
        'top_k_knn': top_k_knn,
        'top_k_predictor': top_k_predictor,
    })
    
    return configs


def create_comprehensive_plots(results_dict, baseline_results, dataset_name, save_dir):
    """
    Create comprehensive plots comparing all hybrid approaches.
    
    Args:
        results_dict: Dictionary with results for each approach
        baseline_results: Results from baseline (no augmentation)
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if not valid_results:
        print("[WARNING] No valid results to plot")
        return
    
    approaches = list(valid_results.keys())
    
    # Extract data
    augmented_acc = [valid_results[k]['augmented']['test_acc'] for k in approaches]
    augmented_f1 = [valid_results[k]['augmented']['test_f1'] for k in approaches]
    acc_improvements = [valid_results[k]['augmented']['test_acc'] - baseline_results['test_acc'] for k in approaches]
    f1_improvements = [valid_results[k]['augmented']['test_f1'] - baseline_results['test_f1'] for k in approaches]
    total_edges = [valid_results[k]['dataset_stats']['total_edges'] for k in approaches]
    
    # Create main comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Hybrid Edge Prediction Approaches Comparison - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(approaches))
    
    # Plot 1: Test Accuracy Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, augmented_acc, alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy by Approach', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(approaches, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, augmented_acc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Test F1 Comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, augmented_f1, alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
    ax2.axhline(y=baseline_results['test_f1'], color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Test F1 Score by Approach', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(approaches, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, augmented_f1):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Accuracy Improvement
    ax3 = axes[0, 2]
    colors = ['green' if x > 0 else 'red' for x in acc_improvements]
    bars = ax3.bar(x_pos, acc_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Improvement', fontsize=12, fontweight='bold')
    ax3.set_title('Test Accuracy Improvement (Δ)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(approaches, rotation=45, ha='right', fontsize=9)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, acc_improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.4f}\n({val*100:+.2f}%)', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    # Plot 4: F1 Improvement
    ax4 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in f1_improvements]
    bars = ax4.bar(x_pos, f1_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score Improvement', fontsize=12, fontweight='bold')
    ax4.set_title('Test F1 Score Improvement (Δ)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(approaches, rotation=45, ha='right', fontsize=9)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, f1_improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.4f}', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    # Plot 5: Total Edges Created
    ax5 = axes[1, 1]
    bars = ax5.bar(x_pos, total_edges, alpha=0.8, color='mediumpurple', edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Approach', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Total Edges', fontsize=12, fontweight='bold')
    ax5.set_title('Total Edges in Augmented Graph', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(approaches, rotation=45, ha='right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, total_edges):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"Summary Statistics ({dataset_name.upper()})\n\n"
    summary_text += f"{'='*50}\n"
    summary_text += f"Baseline Accuracy: {baseline_results['test_acc']:.4f}\n"
    summary_text += f"Baseline F1 Score: {baseline_results['test_f1']:.4f}\n\n"
    summary_text += f"Number of Approaches: {len(approaches)}\n\n"
    summary_text += f"Best Accuracy:\n  {approaches[np.argmax(augmented_acc)]}\n  = {max(augmented_acc):.4f}\n\n"
    summary_text += f"Best F1 Score:\n  {approaches[np.argmax(augmented_f1)]}\n  = {max(augmented_f1):.4f}\n\n"
    summary_text += f"Best Acc Improvement:\n  {approaches[np.argmax(acc_improvements)]}\n  = {max(acc_improvements):+.4f}\n\n"
    summary_text += f"Best F1 Improvement:\n  {approaches[np.argmax(f1_improvements)]}\n  = {max(f1_improvements):+.4f}\n"
    summary_text += f"{'='*50}"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{dataset_name}_hybrid_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Comprehensive plots saved to: {save_path}")
    plt.close()
    
    # Create detailed comparison plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(f'Performance vs Graph Complexity - {dataset_name.upper()}', 
                  fontsize=14, fontweight='bold')
    
    # Accuracy vs Total Edges
    ax1.scatter(total_edges, augmented_acc, s=100, alpha=0.6, c=acc_improvements, cmap='RdYlGn', edgecolors='black')
    ax1.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', linewidth=2, label='Baseline')
    for i, approach in enumerate(approaches):
        ax1.annotate(approach, (total_edges[i], augmented_acc[i]), 
                    fontsize=8, ha='center', va='bottom')
    ax1.set_xlabel('Total Edges in Graph', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Graph Complexity', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # F1 vs Total Edges
    ax2.scatter(total_edges, augmented_f1, s=100, alpha=0.6, c=f1_improvements, cmap='RdYlGn', edgecolors='black')
    ax2.axhline(y=baseline_results['test_f1'], color='red', linestyle='--', linewidth=2, label='Baseline')
    for i, approach in enumerate(approaches):
        ax2.annotate(approach, (total_edges[i], augmented_f1[i]), 
                    fontsize=8, ha='center', va='bottom')
    ax2.set_xlabel('Total Edges in Graph', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score vs Graph Complexity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f'{dataset_name}_complexity_analysis.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"[OK] Complexity analysis plot saved to: {save_path2}")
    plt.close()


def main():
    """Main function to run hybrid approach experiments."""
    
    # Configuration
    dataset_name = 'pubmed'
    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'
    init_weight_approach = 'pissa'  # Fixed for this experiment
    
    # Setup configuration
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    cfg.peft.init_lora_weights = init_weight_approach
    
    # Setup reporter
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "hybrid_experiments")
    os.makedirs(results_dir, exist_ok=True)
    reporter = ReportResults(cfg, save_dir=results_dir, index_run=0)
    
    # Report experiment overview
    reporter.report(
        "Hybrid Edge Prediction Approaches Experiment",
        f"""
        Dataset: {dataset_name}
        LLM Model: {llm_name}
        PEFT Type: {peft_type}
        Init Weight Approach: {init_weight_approach}
        
        Experiment Goal:
        Compare all possible combinations of edge prediction approaches:
        1. Cosine Similarity Only
        2. K-Nearest Neighbors Only
        3. Edge Predictor Only
        4. Cosine + K-NN
        5. Cosine + Predictor
        6. K-NN + Predictor
        7. Full Hybrid (All Three)
        
        This experiment will determine the optimal combination of methods
        for connecting augmented nodes to the existing graph.
        """
    )
    
    # Load embeddings
    print(f"\n{'='*100}")
    print("Loading Initial Embeddings and Training Edge Predictor")
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
    
    # Train Edge Predictor (needed for some configurations)
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
    
    print(f"[OK] Edge Predictor Trained - Val Acc: {edge_predictor_results['val_acc']:.4f}, Test Acc: {edge_predictor_results['test_acc']:.4f}")
    
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
    
    print(f"[OK] Baseline GNN - Test Acc: {baseline_results['test_acc']:.4f}, Test F1: {baseline_results['test_f1']:.4f}")
    
    # Generate all hybrid approach configurations
    approach_configs = generate_hybrid_approach_configs()
    
    print(f"\n{'='*100}")
    print(f"Running {len(approach_configs)} Hybrid Approach Configurations")
    print(f"{'='*100}\n")
    
    # Store results for all approaches
    all_results = {}
    
    # Run experiments for each approach
    for i, config in enumerate(approach_configs, 1):
        print(f"\n[{i}/{len(approach_configs)}] Testing: {config['name']}")
        try:
            result = run_experiment_for_hybrid_approach(
                dataset_name=dataset_name,
                llm_name=llm_name,
                init_weight_approach=init_weight_approach,
                approach_config=config,
                cfg=cfg,
                reporter=reporter,
                results_dict=all_results,
                edge_predictor_model=edge_predictor_model
            )
            
            if result is not None:
                # Report individual results
                acc_improvement = result['augmented']['test_acc'] - baseline_results['test_acc']
                f1_improvement = result['augmented']['test_f1'] - baseline_results['test_f1']
                
                reporter.report(
                    f"Results - {config['name']}",
                    f"""
                    Approach: {config['name']}
                    Methods Used:
                    • Cosine Similarity: {config['use_cosine']}
                    • K-Nearest Neighbors: {config['use_knn']}
                    • Edge Predictor: {config['use_predictor']}
                    
                    Baseline Results:
                    • Test Accuracy: {baseline_results['test_acc']:.4f} ({baseline_results['test_acc']:.2%})
                    • Test F1 (Macro): {baseline_results['test_f1']:.4f}
                    
                    Augmented Results:
                    • Test Accuracy: {result['augmented']['test_acc']:.4f} ({result['augmented']['test_acc']:.2%})
                    • Test F1 (Macro): {result['augmented']['test_f1']:.4f}
                    
                    Improvements:
                    • Test Accuracy Δ: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)
                    • Test F1 (Macro) Δ: {f1_improvement:+.4f}
                    
                    Dataset Statistics:
                    • Total Nodes: {result['dataset_stats']['total_nodes']}
                    • Total Edges: {result['dataset_stats']['total_edges']}
                    • Augmented Nodes: {result['dataset_stats']['augmented_nodes']}
                    """
                )
                
        except Exception as e:
            print(f"\n[ERROR] Failed to run experiment for {config['name']}: {e}")
            reporter.report_txt(f"ERROR in {config['name']}: {str(e)}")
            all_results[config['name']] = None
            continue
    
    # Save results to JSON
    json_path = os.path.join(results_dir, f'{dataset_name}_hybrid_results.json')
    with open(json_path, 'w') as f:
        json_results = {}
        for k, v in all_results.items():
            if v is not None:
                json_results[k] = {
                    'augmented': {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                                for key, val in v['augmented'].items()},
                    'config': v['config'],
                    'dataset_stats': v['dataset_stats']
                }
        json_results['baseline'] = {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                                    for key, val in baseline_results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Results saved to JSON: {json_path}")
    
    # Create comprehensive plots
    print(f"\n{'='*100}")
    print("Creating Comprehensive Plots")
    print(f"{'='*100}\n")
    
    plots_dir = os.path.join(results_dir, "plots")
    create_comprehensive_plots(all_results, baseline_results, dataset_name, plots_dir)
    
    # Final summary report
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        best_acc_approach = max(valid_results.items(), 
                               key=lambda x: x[1]['augmented']['val_acc'])  # select on val
        best_f1_approach = max(valid_results.items(), 
                              key=lambda x: x[1]['augmented']['val_f1'])  # select on val
        
        reporter.report(
            "Final Summary - Best Approaches",
            f"""
            Experiment completed successfully!
            
            Baseline Performance:
            • Test Accuracy: {baseline_results['test_acc']:.4f}
            • Test F1 (Macro): {baseline_results['test_f1']:.4f}
            
            Best Accuracy Approach: {best_acc_approach[0]}
            • Test Accuracy: {best_acc_approach[1]['augmented']['test_acc']:.4f}
            • Improvement: {best_acc_approach[1]['augmented']['test_acc'] - baseline_results['test_acc']:+.4f}
            
            Best F1 Approach: {best_f1_approach[0]}
            • Test F1 (Macro): {best_f1_approach[1]['augmented']['test_f1']:.4f}
            • Improvement: {best_f1_approach[1]['augmented']['test_f1'] - baseline_results['test_f1']:+.4f}
            
            Total approaches tested: {len(valid_results)}/{len(approach_configs)}
            Results saved to: {json_path}
            Plots saved to: {plots_dir}
            """
        )
    
    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*100}\n")
    print(f"Results directory: {results_dir}")
    print(f"JSON results: {json_path}")
    print(f"Plots directory: {plots_dir}")


if __name__ == "__main__":
    main()
