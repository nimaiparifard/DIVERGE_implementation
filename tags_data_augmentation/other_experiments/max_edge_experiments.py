# comprehensive experiment to show influence of max edges constraint into performance
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


def run_experiment_for_max_edges(dataset_name, llm_name, init_weight_approach,
                                  approach_name, approach_type, top_k_value,
                                  cfg, edge_predictor_model, results_dict):
    """
    Run experiment for a specific top_k value.
    
    Args:
        dataset_name: Name of the dataset
        llm_name: Name of the LLM model
        init_weight_approach: Initialization approach ('pissa', 'orthogonal', etc.)
        approach_name: Name of the approach (e.g., 'Cosine Similarity')
        approach_type: Type of approach ('cosine', 'knn', 'predictor', 'hybrid')
        top_k_value: Maximum number of edges per new node
        cfg: Configuration object
        edge_predictor_model: Pre-trained edge predictor model
        results_dict: Dictionary to store results
    """
    experiment_key = f"{approach_name}_top_k_{top_k_value}"
    
    print(f"\n{'='*100}")
    print(f"EXPERIMENT: {approach_name} with top_k={top_k_value}")
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
    
    # Configure approach parameters
    if approach_type == 'cosine':
        use_cosine = True
        use_knn = False
        use_predictor = False
        top_k_cosine = top_k_value
        top_k_knn = 3
        top_k_predictor = 5
    elif approach_type == 'knn':
        use_cosine = False
        use_knn = True
        use_predictor = False
        top_k_cosine = 3
        top_k_knn = top_k_value
        top_k_predictor = 5
    elif approach_type == 'predictor':
        use_cosine = False
        use_knn = False
        use_predictor = True
        top_k_cosine = 3
        top_k_knn = 3
        top_k_predictor = top_k_value
    elif approach_type == 'hybrid':
        use_cosine = True
        use_knn = True
        use_predictor = True
        top_k_cosine = top_k_value
        top_k_knn = top_k_value
        top_k_predictor = top_k_value
    else:
        raise ValueError(f"Unknown approach type: {approach_type}")
    
    # Create augmented dataset
    print(f"Creating augmented dataset with top_k={top_k_value}...")
    augmented_dataset, augmented_embeddings = create_augmented_dataset(
        cfg=cfg,
        augmented_cache=augmented_cache,
        init_weight_approach=init_weight_approach,
        used_cosine_similarity=use_cosine,
        used_k_nearest_neighbors=use_knn,
        used_edge_predictor=use_predictor,
        threshold_cosine_similarity=0.95,
        threshold_k_nearest_neighbors=0.5,
        threshold_edge_predictor=0.5,
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
        'top_k': top_k_value,
        'test_acc': augmented_results['test_acc'],
        'test_f1': augmented_results['test_f1'],
        'val_acc': augmented_results['val_acc'],
        'val_f1': augmented_results['val_f1'],
        'total_nodes': augmented_dataset.y.shape[0],
        'total_edges': augmented_dataset.edge_index.shape[1],
        'augmented_nodes': augmented_cache['embeddings'].shape[0],
    }
    
    return results_dict[experiment_key]


def create_comprehensive_plots(results_by_approach, baseline_results, dataset_name, save_dir):
    """
    Create comprehensive plots showing the influence of max edges constraint.
    
    Args:
        results_by_approach: Dictionary with results grouped by approach
        baseline_results: Baseline results (no augmentation)
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create main figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Influence of Max Edges Constraint on Performance - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    markers = ['o', 's', '^', 'D']
    
    # Plot 1: Test Accuracy vs Top-K
    ax1 = axes[0, 0]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        top_k_values = [r['top_k'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        ax1.plot(top_k_values, test_accs, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax1.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', 
               linewidth=2, label='Baseline', alpha=0.8)
    ax1.set_xlabel('Max Edges per New Node (top_k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy vs Max Edges Constraint', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test F1 vs Top-K
    ax2 = axes[0, 1]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        top_k_values = [r['top_k'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        ax2.plot(top_k_values, test_f1s, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax2.axhline(y=baseline_results['test_f1'], color='red', linestyle='--', 
               linewidth=2, label='Baseline', alpha=0.8)
    ax2.set_xlabel('Max Edges per New Node (top_k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Test F1 Score vs Max Edges Constraint', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy Improvement vs Top-K
    ax3 = axes[1, 0]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        top_k_values = [r['top_k'] for r in results]
        acc_improvements = [r['test_acc'] - baseline_results['test_acc'] for r in results]
        ax3.plot(top_k_values, acc_improvements, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Max Edges per New Node (top_k)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Improvement (Δ)', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Improvement vs Max Edges Constraint', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total Edges vs Top-K
    ax4 = axes[1, 1]
    for i, (approach_name, results) in enumerate(results_by_approach.items()):
        top_k_values = [r['top_k'] for r in results]
        total_edges = [r['total_edges'] for r in results]
        ax4.plot(top_k_values, total_edges, marker=markers[i], linewidth=2, markersize=8,
                label=approach_name, color=colors[i], alpha=0.7)
    
    ax4.set_xlabel('Max Edges per New Node (top_k)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total Edges in Graph', fontsize=12, fontweight='bold')
    ax4.set_title('Graph Complexity vs Max Edges Constraint', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{dataset_name}_max_edges_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Max edges analysis plot saved to: {save_path}")
    plt.close()
    
    # Create detailed comparison figure for each approach
    for approach_name, results in results_by_approach.items():
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle(f'{approach_name} - Detailed Analysis of Max Edges Constraint\n{dataset_name.upper()}', 
                     fontsize=14, fontweight='bold')
        
        top_k_values = [r['top_k'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        total_edges = [r['total_edges'] for r in results]
        acc_improvements = [r['test_acc'] - baseline_results['test_acc'] for r in results]
        
        # Subplot 1: Accuracy with bars
        ax1 = axes2[0, 0]
        bars = ax1.bar(range(len(top_k_values)), test_accs, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', linewidth=2, label='Baseline')
        ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('Test Accuracy by Top-K Value', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(top_k_values)))
        ax1.set_xticklabels([f'k={k}' for k in top_k_values], rotation=0)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, test_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Subplot 2: F1 Score with bars
        ax2 = axes2[0, 1]
        bars = ax2.bar(range(len(top_k_values)), test_f1s, alpha=0.7, color='coral', edgecolor='black')
        ax2.axhline(y=baseline_results['test_f1'], color='red', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test F1 Score', fontsize=11, fontweight='bold')
        ax2.set_title('Test F1 Score by Top-K Value', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(top_k_values)))
        ax2.set_xticklabels([f'k={k}' for k in top_k_values], rotation=0)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, test_f1s):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Subplot 3: Accuracy vs Total Edges (scatter)
        ax3 = axes2[1, 0]
        scatter = ax3.scatter(total_edges, test_accs, s=150, c=top_k_values, 
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        ax3.axhline(y=baseline_results['test_acc'], color='red', linestyle='--', linewidth=2, label='Baseline')
        for i, k in enumerate(top_k_values):
            ax3.annotate(f'k={k}', (total_edges[i], test_accs[i]), 
                        fontsize=9, ha='center', va='bottom')
        ax3.set_xlabel('Total Edges in Graph', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
        ax3.set_title('Accuracy vs Graph Complexity', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='top_k value')
        
        # Subplot 4: Summary statistics
        ax4 = axes2[1, 1]
        ax4.axis('off')
        
        best_acc_idx = int(np.argmax([r['val_acc'] for r in results]))  # selected on val, test only reported
        best_f1_idx = int(np.argmax([r['val_f1'] for r in results]))  # selected on val, test only reported
        
        summary_text = f"Summary - {approach_name}\n\n"
        summary_text += f"{'='*45}\n"
        summary_text += f"Baseline Performance:\n"
        summary_text += f"  Accuracy: {baseline_results['test_acc']:.4f}\n"
        summary_text += f"  F1 Score: {baseline_results['test_f1']:.4f}\n\n"
        summary_text += f"Best Accuracy:\n"
        summary_text += f"  top_k = {top_k_values[best_acc_idx]}\n"
        summary_text += f"  Accuracy: {test_accs[best_acc_idx]:.4f}\n"
        summary_text += f"  Improvement: {acc_improvements[best_acc_idx]:+.4f}\n\n"
        summary_text += f"Best F1 Score:\n"
        summary_text += f"  top_k = {top_k_values[best_f1_idx]}\n"
        summary_text += f"  F1: {test_f1s[best_f1_idx]:.4f}\n"
        summary_text += f"  Improvement: {test_f1s[best_f1_idx] - baseline_results['test_f1']:+.4f}\n\n"
        summary_text += f"Average Performance:\n"
        summary_text += f"  Accuracy: {np.mean(test_accs):.4f}\n"
        summary_text += f"  F1 Score: {np.mean(test_f1s):.4f}\n"
        summary_text += f"  Avg Improvement: {np.mean(acc_improvements):+.4f}\n"
        summary_text += f"{'='*45}"
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save detailed plot
        approach_filename = approach_name.lower().replace(' ', '_')
        save_path2 = os.path.join(save_dir, f'{dataset_name}_{approach_filename}_detailed.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"[OK] Detailed plot for {approach_name} saved to: {save_path2}")
        plt.close()


def main():
    """Main function to run max edges constraint experiments."""
    
    # Configuration
    dataset_name = 'cora'
    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'
    init_weight_approach = 'pissa'
    
    # Top-K values to test
    top_k_values = [1, 3, 5, 7, 10, 15, 20]
    
    # Approaches to test
    approaches = [
        {'name': 'Cosine Similarity', 'type': 'cosine'},
        {'name': 'K-Nearest Neighbors', 'type': 'knn'},
        {'name': 'Edge Predictor', 'type': 'predictor'},
        {'name': 'Full Hybrid', 'type': 'hybrid'},
    ]
    
    # Setup configuration
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    cfg.peft.init_lora_weights = init_weight_approach
    
    # Setup reporter
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "max_edges_experiments")
    os.makedirs(results_dir, exist_ok=True)
    reporter = ReportResults(cfg, save_dir=results_dir, index_run=0)
    
    # Report experiment overview
    reporter.report(
        "Max Edges Constraint Experiment",
        f"""
        Dataset: {dataset_name}
        LLM Model: {llm_name}
        PEFT Type: {peft_type}
        Init Weight Approach: {init_weight_approach}
        
        Top-K Values to Test: {top_k_values}
        
        Approaches to Test:
        {chr(10).join([f'  • {a["name"]}' for a in approaches])}
        
        Experiment Goal:
        Investigate how the maximum edge constraint (top_k parameter) influences
        the performance of data augmentation across different edge prediction methods.
        This will help determine the optimal connectivity level for augmented nodes.
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
    
    # Run experiments
    print(f"\n{'='*100}")
    print(f"Running Experiments: {len(approaches)} approaches × {len(top_k_values)} top_k values = {len(approaches) * len(top_k_values)} total experiments")
    print(f"{'='*100}\n")
    
    all_results = {}
    results_by_approach = {approach['name']: [] for approach in approaches}
    
    total_experiments = len(approaches) * len(top_k_values)
    current_experiment = 0
    
    for approach in approaches:
        print(f"\n{'='*100}")
        print(f"TESTING APPROACH: {approach['name'].upper()}")
        print(f"{'='*100}\n")
        
        reporter.report_title(f"Approach: {approach['name']}")
        
        for top_k in top_k_values:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] {approach['name']} with top_k={top_k}")
            
            try:
                result = run_experiment_for_max_edges(
                    dataset_name=dataset_name,
                    llm_name=llm_name,
                    init_weight_approach=init_weight_approach,
                    approach_name=approach['name'],
                    approach_type=approach['type'],
                    top_k_value=top_k,
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
                        f"{approach['name']} - top_k={top_k}",
                        f"""
                        Configuration:
                        • Approach: {approach['name']}
                        • Max Edges (top_k): {top_k}
                        
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
                reporter.report_txt(f"ERROR in {approach['name']} with top_k={top_k}: {str(e)}")
                continue
    
    # Save results to JSON
    json_path = os.path.join(results_dir, f'{dataset_name}_max_edges_results.json')
    with open(json_path, 'w') as f:
        json_results = {
            'baseline': {key: float(val) if isinstance(val, (np.floating, torch.Tensor)) else val 
                        for key, val in baseline_results.items()},
            'results_by_approach': {},
            'all_results': {}
        }
        
        for approach_name, results in results_by_approach.items():
            json_results['results_by_approach'][approach_name] = [
                {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                 for k, v in r.items()}
                for r in results
            ]
        
        for key, val in all_results.items():
            if val is not None:
                json_results['all_results'][key] = {
                    k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
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
    
    summary_lines = ["Final Summary - Optimal Max Edges Configuration", ""]
    summary_lines.append(f"Baseline Performance:")
    summary_lines.append(f"  • Test Accuracy: {baseline_results['test_acc']:.4f}")
    summary_lines.append(f"  • Test F1 (Macro): {baseline_results['test_f1']:.4f}")
    summary_lines.append("")
    
    for approach_name, results in results_by_approach.items():
        if not results:
            continue
            
        test_accs = [r['test_acc'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        top_k_vals = [r['top_k'] for r in results]
        
        best_acc_idx = int(np.argmax([r['val_acc'] for r in results]))  # selected on val, test only reported
        best_f1_idx = int(np.argmax([r['val_f1'] for r in results]))  # selected on val, test only reported
        
        summary_lines.append(f"{approach_name}:")
        summary_lines.append(f"  Best Accuracy: top_k={top_k_vals[best_acc_idx]}, acc={test_accs[best_acc_idx]:.4f}, Δ={test_accs[best_acc_idx] - baseline_results['test_acc']:+.4f}")
        summary_lines.append(f"  Best F1 Score: top_k={top_k_vals[best_f1_idx]}, f1={test_f1s[best_f1_idx]:.4f}, Δ={test_f1s[best_f1_idx] - baseline_results['test_f1']:+.4f}")
        summary_lines.append(f"  Average Accuracy: {np.mean(test_accs):.4f}")
        summary_lines.append(f"  Average F1 Score: {np.mean(test_f1s):.4f}")
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
