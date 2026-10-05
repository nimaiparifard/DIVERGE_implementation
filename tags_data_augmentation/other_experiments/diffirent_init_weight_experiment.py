# Run with different init weight embeddings
## The embedding of the generated node and existed text embedding must be from same init weight approach
# Run with all possible things for hybrid dataset creation: edge predictor + cosine + k-nearest neighbor
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


def run_experiment_for_init_weight(dataset_name, llm_name, init_weight_approach, 
                                   cfg, reporter, results_dict):
    """
    Run complete experiment for a single init weight approach.
    
    Args:
        dataset_name: Name of the dataset
        llm_name: Name of the LLM model
        init_weight_approach: Initialization approach ('pissa', 'orthogonal', etc.)
        cfg: Configuration object
        reporter: Reporter object for logging
        results_dict: Dictionary to store results
    """
    print(f"\n{'='*100}")
    print(f"RUNNING EXPERIMENT FOR INIT WEIGHT: {init_weight_approach.upper()}")
    print(f"{'='*100}\n")
    
    reporter.report_title(f"Experiment: {init_weight_approach.upper()} Init Weight")
    
    # Update config with current init weight approach
    cfg.peft.init_lora_weights = init_weight_approach
    
    # ===================== STEP 1: Load embeddings for this init approach =====================
    print(f"\n{'='*80}")
    print(f"STEP 1: Loading Initial Embeddings ({init_weight_approach})")
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
    
    # ===================== STEP 2: Train Edge Predictor =====================
    print(f"\n{'='*80}")
    print(f"STEP 2: Training Edge Predictor ({init_weight_approach})")
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
    
    # ===================== STEP 3: Load Augmented Cache =====================
    print(f"\n{'='*80}")
    print(f"STEP 3: Loading Augmented Cache ({init_weight_approach})")
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
        print(f"[SKIP] Skipping {init_weight_approach} - augmented cache not found")
        reporter.report_txt(f"SKIPPED: Augmented cache not found for {init_weight_approach}")
        return None
    
    # ===================== STEP 4: Train Baseline GNN =====================
    print(f"\n{'='*80}")
    print(f"STEP 4: Training Baseline GNN ({init_weight_approach})")
    print(f"{'='*80}")
    
    baseline_results, baseline_model = gnn_train_and_report(
        dataset_name=dataset_name,
        embedding=embedding,
        title=f"Baseline - {init_weight_approach.upper()}",
        does_print_training_process=False,
        reporter=reporter
    )
    
    print(f"[OK] Baseline GNN - Test Acc: {baseline_results['test_acc']:.4f}, Test F1: {baseline_results['test_f1']:.4f}")
    
    # ===================== STEP 5: Create Augmented Dataset with Hybrid Edge Prediction =====================
    print(f"\n{'='*80}")
    print(f"STEP 5: Creating Augmented Dataset with Hybrid Edge Prediction ({init_weight_approach})")
    print(f"{'='*80}")
    
    # Use all three methods: cosine similarity + k-nearest neighbors + edge predictor
    augmented_dataset, augmented_embeddings = create_augmented_dataset(
        cfg=cfg,
        augmented_cache=augmented_cache,
        init_weight_approach=init_weight_approach,
        used_cosine_similarity=True,
        used_k_nearest_neighbors=True,
        used_edge_predictor=True,
        threshold_cosine_similarity=0.95,
        threshold_k_nearest_neighbors=0.5,
        threshold_edge_predictor=0.5,
        top_k_cosine_similarity=3,
        top_k_knearest_neighbors=3,
        top_k_edge_predictor=5,
        edge_predictor_model=edge_predictor_model
    )
    
    print(f"[OK] Augmented dataset created with {augmented_dataset.y.shape[0]} total nodes")
    
    # ===================== STEP 6: Train GNN on Augmented Dataset =====================
    print(f"\n{'='*80}")
    print(f"STEP 6: Training GNN on Augmented Dataset ({init_weight_approach})")
    print(f"{'='*80}")
    
    augmented_dataset.x = augmented_embeddings
    augmented_results, augmented_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=augmented_dataset,
        embedding=augmented_embeddings,
        dataset_name=dataset_name
    )
    
    print(f"[OK] Augmented GNN - Test Acc: {augmented_results['test_acc']:.4f}, Test F1: {augmented_results['test_f1']:.4f}")
    
    # ===================== STEP 7: Calculate Improvements =====================
    acc_improvement = augmented_results['test_acc'] - baseline_results['test_acc']
    f1_improvement = augmented_results['test_f1'] - baseline_results['test_f1']
    
    print(f"\n{'='*80}")
    print(f"IMPROVEMENTS for {init_weight_approach.upper()}:")
    print(f"  Test Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
    print(f"  Test F1 (Macro): {f1_improvement:+.4f}")
    print(f"{'='*80}\n")
    
    # Report improvements
    reporter.report(
        f"Improvements - {init_weight_approach.upper()}",
        f"""
        Baseline Results:
        • Test Accuracy: {baseline_results['test_acc']:.4f} ({baseline_results['test_acc']:.2%})
        • Test F1 (Macro): {baseline_results['test_f1']:.4f}
        • Val Accuracy: {baseline_results['val_acc']:.4f}
        • Val F1 (Macro): {baseline_results['val_f1']:.4f}
        
        Augmented Results:
        • Test Accuracy: {augmented_results['test_acc']:.4f} ({augmented_results['test_acc']:.2%})
        • Test F1 (Macro): {augmented_results['test_f1']:.4f}
        • Val Accuracy: {augmented_results['val_acc']:.4f}
        • Val F1 (Macro): {augmented_results['val_f1']:.4f}
        
        Improvements:
        • Test Accuracy Δ: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)
        • Test F1 (Macro) Δ: {f1_improvement:+.4f}
        
        Edge Predictor Performance:
        • Val Accuracy: {edge_predictor_results['val_acc']:.4f}
        • Test Accuracy: {edge_predictor_results['test_acc']:.4f}
        • Test Loss: {edge_predictor_results['test_loss']:.4f}
        
        Dataset Statistics:
        • Original Nodes: {augmented_dataset.y.shape[0] - augmented_cache['embeddings'].shape[0]}
        • Augmented Nodes: {augmented_cache['embeddings'].shape[0]}
        • Total Nodes: {augmented_dataset.y.shape[0]}
        • Total Edges: {augmented_dataset.edge_index.shape[1]}
        """
    )
    
    # Store results
    results_dict[init_weight_approach] = {
        'baseline': baseline_results,
        'augmented': augmented_results,
        'edge_predictor': edge_predictor_results,
        'improvements': {
            'test_acc': acc_improvement,
            'test_f1': f1_improvement
        },
        'dataset_stats': {
            'original_nodes': augmented_dataset.y.shape[0] - augmented_cache['embeddings'].shape[0],
            'augmented_nodes': augmented_cache['embeddings'].shape[0],
            'total_nodes': augmented_dataset.y.shape[0],
            'total_edges': augmented_dataset.edge_index.shape[1]
        }
    }
    
    return results_dict[init_weight_approach]


def create_comprehensive_plots(results_dict, dataset_name, save_dir):
    """
    Create comprehensive plots comparing all init weight approaches.
    
    Args:
        results_dict: Dictionary with results for each init weight approach
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out None results (skipped experiments)
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if not valid_results:
        print("[WARNING] No valid results to plot")
        return
    
    approaches = list(valid_results.keys())
    
    # Extract data for plotting
    baseline_acc = [valid_results[k]['baseline']['test_acc'] for k in approaches]
    augmented_acc = [valid_results[k]['augmented']['test_acc'] for k in approaches]
    baseline_f1 = [valid_results[k]['baseline']['test_f1'] for k in approaches]
    augmented_f1 = [valid_results[k]['augmented']['test_f1'] for k in approaches]
    acc_improvements = [valid_results[k]['improvements']['test_acc'] for k in approaches]
    f1_improvements = [valid_results[k]['improvements']['test_f1'] for k in approaches]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Comprehensive Results: Different Init Weight Approaches on {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(approaches))
    width = 0.35
    
    # Plot 1: Accuracy Comparison (Baseline vs Augmented)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos - width/2, baseline_acc, width, label='Baseline', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x_pos + width/2, augmented_acc, width, label='Augmented', alpha=0.8, color='coral')
    ax1.set_xlabel('Init Weight Approach', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy: Baseline vs Augmented', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(approaches, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: F1 Score Comparison (Baseline vs Augmented)
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x_pos - width/2, baseline_f1, width, label='Baseline', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(x_pos + width/2, augmented_f1, width, label='Augmented', alpha=0.8, color='coral')
    ax2.set_xlabel('Init Weight Approach', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Test F1 Score: Baseline vs Augmented', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(approaches, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Accuracy Improvements
    ax3 = axes[0, 2]
    colors = ['green' if x > 0 else 'red' for x in acc_improvements]
    bars = ax3.bar(x_pos, acc_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Init Weight Approach', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Improvement', fontsize=12, fontweight='bold')
    ax3.set_title('Test Accuracy Improvement (Δ)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(approaches, rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, acc_improvements)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.4f}\n({val*100:+.2f}%)', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    # Plot 4: F1 Score Improvements
    ax4 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in f1_improvements]
    bars = ax4.bar(x_pos, f1_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Init Weight Approach', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score Improvement', fontsize=12, fontweight='bold')
    ax4.set_title('Test F1 Score Improvement (Δ)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(approaches, rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, f1_improvements)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.4f}', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    # Plot 5: Combined Metrics Heatmap
    ax5 = axes[1, 1]
    metrics_data = np.array([
        baseline_acc,
        augmented_acc,
        baseline_f1,
        augmented_f1
    ])
    im = ax5.imshow(metrics_data, cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(approaches, rotation=45, ha='right')
    ax5.set_yticks([0, 1, 2, 3])
    ax5.set_yticklabels(['Baseline Acc', 'Augmented Acc', 'Baseline F1', 'Augmented F1'])
    ax5.set_title('Metrics Heatmap', fontsize=13, fontweight='bold')
    # Add text annotations
    for i in range(metrics_data.shape[0]):
        for j in range(metrics_data.shape[1]):
            text = ax5.text(j, i, f'{metrics_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"Summary Statistics ({dataset_name.upper()})\n\n"
    summary_text += f"{'='*50}\n"
    summary_text += f"Number of Approaches Tested: {len(approaches)}\n\n"
    summary_text += f"Best Baseline Accuracy:\n  {approaches[np.argmax(baseline_acc)]} = {max(baseline_acc):.4f}\n\n"
    summary_text += f"Best Augmented Accuracy:\n  {approaches[np.argmax(augmented_acc)]} = {max(augmented_acc):.4f}\n\n"
    summary_text += f"Best Accuracy Improvement:\n  {approaches[np.argmax(acc_improvements)]} = {max(acc_improvements):+.4f}\n\n"
    summary_text += f"Best F1 Improvement:\n  {approaches[np.argmax(f1_improvements)]} = {max(f1_improvements):+.4f}\n\n"
    summary_text += f"Average Baseline Accuracy: {np.mean(baseline_acc):.4f}\n"
    summary_text += f"Average Augmented Accuracy: {np.mean(augmented_acc):.4f}\n\n"
    summary_text += f"Average Improvement:\n  Accuracy: {np.mean(acc_improvements):+.4f}\n  F1: {np.mean(f1_improvements):+.4f}\n"
    summary_text += f"{'='*50}"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plots
    save_path = os.path.join(save_dir, f'{dataset_name}_comprehensive_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Comprehensive plots saved to: {save_path}")
    plt.close()
    
    # Create additional plot: Line plot for trends
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(f'Performance Trends Across Init Weight Approaches - {dataset_name.upper()}', 
                  fontsize=14, fontweight='bold')
    
    # Line plot for accuracy
    ax1.plot(approaches, baseline_acc, 'o-', linewidth=2, markersize=8, 
            label='Baseline', color='steelblue')
    ax1.plot(approaches, augmented_acc, 's-', linewidth=2, markersize=8, 
            label='Augmented', color='coral')
    ax1.set_xlabel('Init Weight Approach', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Trends', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Line plot for F1
    ax2.plot(approaches, baseline_f1, 'o-', linewidth=2, markersize=8, 
            label='Baseline', color='steelblue')
    ax2.plot(approaches, augmented_f1, 's-', linewidth=2, markersize=8, 
            label='Augmented', color='coral')
    ax2.set_xlabel('Init Weight Approach', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Test F1 Score Trends', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f'{dataset_name}_trends_comparison.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"[OK] Trends plot saved to: {save_path2}")
    plt.close()


def main():
    """Main function to run experiments for all init weight approaches."""
    
    # Configuration
    dataset_name = 'pubmed'
    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'
    
    # List of init weight approaches to test
    init_weight_approaches = ['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal']
    
    # Setup configuration
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    
    # Setup reporter
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "init_weight_experiments")
    os.makedirs(results_dir, exist_ok=True)
    reporter = ReportResults(cfg, save_dir=results_dir, index_run=0)
    
    # Report experiment overview
    reporter.report(
        "Different Init Weight Approaches Experiment",
        f"""
        Dataset: {dataset_name}
        LLM Model: {llm_name}
        PEFT Type: {peft_type}
        
        Init Weight Approaches Tested: {', '.join(init_weight_approaches)}
        
        Hybrid Edge Prediction Configuration:
        • Cosine Similarity: ENABLED (threshold=0.95, top_k=3)
        • K-Nearest Neighbors: ENABLED (threshold=0.5, top_k=3)
        • Edge Predictor: ENABLED (threshold=0.5, top_k=5, decoder=mlp)
        
        Experiment Goal:
        Compare the performance of data augmentation across different LoRA
        initialization approaches. All three edge prediction methods are used
        in hybrid mode (intersection of results).
        """
    )
    
    # Store results for all approaches
    all_results = {}
    
    # Run experiments for each init weight approach
    for approach in init_weight_approaches:
        try:
            result = run_experiment_for_init_weight(
                dataset_name=dataset_name,
                llm_name=llm_name,
                init_weight_approach=approach,
                cfg=cfg,
                reporter=reporter,
                results_dict=all_results
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to run experiment for {approach}: {e}")
            reporter.report_txt(f"ERROR in {approach}: {str(e)}")
            all_results[approach] = None
            continue
    
    # Save results to JSON
    json_path = os.path.join(results_dir, f'{dataset_name}_all_results.json')
    with open(json_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for k, v in all_results.items():
            if v is not None:
                json_results[k] = {
                    'baseline': {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                                for key, val in v['baseline'].items()},
                    'augmented': {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                                 for key, val in v['augmented'].items()},
                    'edge_predictor': {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                                      for key, val in v['edge_predictor'].items()},
                    'improvements': v['improvements'],
                    'dataset_stats': v['dataset_stats']
                }
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Results saved to JSON: {json_path}")
    
    # Create comprehensive plots
    print(f"\n{'='*100}")
    print("Creating Comprehensive Plots")
    print(f"{'='*100}\n")
    
    plots_dir = os.path.join(results_dir, "plots")
    create_comprehensive_plots(all_results, dataset_name, plots_dir)
    
    # Final summary report
    reporter.report(
        "Final Summary",
        f"""
        Experiment completed successfully!
        
        Total approaches tested: {len([v for v in all_results.values() if v is not None])}
        Results saved to: {json_path}
        Plots saved to: {plots_dir}
        
        Check the comprehensive plots for detailed comparisons.
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
