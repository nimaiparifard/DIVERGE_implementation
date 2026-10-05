# Hyperparameter tuning for edge editing with uncertainty-based edge detection
# Tune hyperparameters for every embedding approach: pissa, eva, orthogonal, loftq, gaussian
# Find best hyperparameters for both std and entropy metrics
# Test different removal strategies: threshold and top_k
# Generate comprehensive reports and visualizations

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import *
from LLMReasoner.TAPE.structure_hybrid_enahncer.uncertainty_edge_detection import (
    NoisyEdgeDetector, create_modified_dataset, compare_results
)


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for uncertainty-based edge editing.
    Tests different embedding approaches, uncertainty metrics, and removal strategies.
    """
    
    def __init__(self, dataset_name, llm_name, peft_type='lora', num_models=10, 
                 decoder_type='mlp', results_base_dir='./results/hyperparameter_tuning'):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cora', 'citeseer', 'pubmed')
            llm_name: Name of the LLM model (e.g., 'llama_3.2_1B')
            peft_type: Type of PEFT (default: 'lora')
            num_models: Number of edge predictor models for ensemble (default: 10)
            decoder_type: Edge predictor decoder type (default: 'mlp')
            results_base_dir: Base directory for saving results
        """
        self.dataset_name = dataset_name
        self.llm_name = llm_name
        self.peft_type = peft_type
        self.num_models = num_models
        self.decoder_type = decoder_type
        self.results_base_dir = results_base_dir
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(results_base_dir, f"{dataset_name}_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define hyperparameter search space
        self.embedding_approaches = ['pissa', 'orthogonal', 'guassian', 'loftq', 'eva']
        self.uncertainty_metrics = ['std', 'entropy']
        
        # Threshold values for different metrics and strategies
        self.std_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.entropy_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        self.top_k_percentages = [5, 10, 15, 20, 25, 30, 35, 40]
        
        # Setup configuration
        self.cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
        self.gnn_cfg = set_mgnn_cfg(dataset_name)
        
        # Results storage
        self.all_results = []
        self.baseline_results = {}
        
        print(f"\n{'='*80}")
        print(f"Hyperparameter Tuner Initialized")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"LLM: {llm_name}")
        print(f"Number of models: {num_models}")
        print(f"Decoder type: {decoder_type}")
        print(f"Results directory: {self.results_dir}")
        print(f"Embedding approaches: {self.embedding_approaches}")
        print(f"Uncertainty metrics: {self.uncertainty_metrics}")
        print(f"STD thresholds: {self.std_thresholds}")
        print(f"Entropy thresholds: {self.entropy_thresholds}")
        print(f"Top-K percentages: {self.top_k_percentages}")
        print(f"{'='*80}\n")
    
    def get_embedding_for_approach(self, approach):
        """
        Get the embedding for a specific approach.
        
        Args:
            approach: Embedding approach name
            
        Returns:
            embedding: Node embeddings
        """
        # Load dataset
        dataset = load_dataset(self.cfg)
        data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(self.cfg)
        
        if approach == "pissa":
            embedding = get_embedding_from_data(data_pissa)
        elif approach == "orthogonal":
            embedding = get_embedding_from_data(data_orthogonal)
        elif approach == "guassian":
            embedding = get_embedding_from_data(data_guassian)
        elif approach == "loftq":
            embedding = get_embedding_from_data(data_loftq)
        elif approach == "eva":
            embedding = get_embedding_from_data(data_eva)
        else:
            raise ValueError(f"Unknown embedding approach: {approach}")
        
        return embedding
    
    def train_baseline(self, embedding, approach_name):
        """
        Train baseline GNN on original dataset for a given embedding approach.
        
        Args:
            embedding: Node embeddings
            approach_name: Name of the embedding approach
            
        Returns:
            baseline_results: Dictionary with baseline performance
        """
        print(f"\n{'='*80}")
        print(f"Training Baseline for {approach_name.upper()}")
        print(f"{'='*80}")
        
        # Create detector to get consistent dataset
        detector = NoisyEdgeDetector(
            cfg=self.gnn_cfg,
            features=embedding,
            num_models=self.num_models,
            decoder_type=self.decoder_type,
            training_strategy='with_splitting',
            negative_sampling_ratio=1.0
        )
        
        # Train baseline GNN
        baseline_results, _ = gnn_train_and_report_with_modified_dataset(
            modified_dataset=detector.dataset,
            embedding=embedding,
            dataset_name=self.dataset_name
        )
        
        print(f"[OK] Baseline results for {approach_name}:")
        print(f"     Test Acc: {baseline_results['test_acc']:.4f}")
        print(f"     Test F1: {baseline_results['test_f1']:.4f}")
        
        return baseline_results, detector
    
    def evaluate_configuration(self, detector, embedding, uncertainty_metric, 
                               removal_strategy, threshold_value, baseline_results):
        """
        Evaluate a single hyperparameter configuration.
        
        Args:
            detector: Trained NoisyEdgeDetector
            embedding: Node embeddings
            uncertainty_metric: 'std' or 'entropy'
            removal_strategy: 'threshold' or 'top_k'
            threshold_value: Threshold value for edge removal
            baseline_results: Baseline performance results
            
        Returns:
            result_dict: Dictionary with evaluation results
        """
        # Calculate uncertainty scores
        uncertainty_scores = detector.get_uncertainty_scores(metric=uncertainty_metric)
        
        # Create modified dataset
        modified_dataset, removal_stats = create_modified_dataset(
            dataset=detector.dataset,
            uncertainty_scores=uncertainty_scores,
            threshold=threshold_value,
            removal_strategy=removal_strategy
        )
        
        # Train GNN on modified dataset
        modified_results, _ = gnn_train_and_report_with_modified_dataset(
            modified_dataset=modified_dataset,
            embedding=embedding,
            dataset_name=self.dataset_name
        )
        
        # Calculate improvements
        test_acc_improvement = modified_results['test_acc'] - baseline_results['test_acc']
        test_f1_improvement = modified_results['test_f1'] - baseline_results['test_f1']
        val_acc_improvement = modified_results['val_acc'] - baseline_results['val_acc']
        val_f1_improvement = modified_results['val_f1'] - baseline_results['val_f1']
        
        result = {
            'uncertainty_metric': uncertainty_metric,
            'removal_strategy': removal_strategy,
            'threshold': threshold_value,
            'baseline_test_acc': baseline_results['test_acc'],
            'baseline_test_f1': baseline_results['test_f1'],
            'modified_test_acc': modified_results['test_acc'],
            'modified_test_f1': modified_results['test_f1'],
            'test_acc_improvement': test_acc_improvement,
            'test_f1_improvement': test_f1_improvement,
            'val_acc_improvement': val_acc_improvement,
            'val_f1_improvement': val_f1_improvement,
            'removed_edges': removal_stats['removed_edges'],
            'removal_percentage': removal_stats['removal_percentage']
        }
        
        return result
    
    def tune_embedding_approach(self, approach_name):
        """
        Tune hyperparameters for a specific embedding approach.
        
        Args:
            approach_name: Name of the embedding approach
            
        Returns:
            approach_results: List of results for all configurations
        """
        print(f"\n{'#'*80}")
        print(f"# TUNING EMBEDDING APPROACH: {approach_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Get embedding for this approach
        embedding = self.get_embedding_for_approach(approach_name)
        
        # Train baseline
        baseline_results, detector = self.train_baseline(embedding, approach_name)
        
        # Store baseline for this approach
        self.baseline_results[approach_name] = baseline_results
        
        # Train edge predictors (only once per embedding approach)
        print(f"\n{'='*80}")
        print(f"Training Edge Predictors for {approach_name.upper()}")
        print(f"{'='*80}")
        detector.train_edge_predictors()
        
        # Results for this approach
        approach_results = []
        
        # Iterate over uncertainty metrics
        for uncertainty_metric in self.uncertainty_metrics:
            print(f"\n{'='*80}")
            print(f"Testing Uncertainty Metric: {uncertainty_metric.upper()}")
            print(f"{'='*80}\n")
            
            # Test threshold-based removal
            print(f"Testing THRESHOLD removal strategy...")
            if uncertainty_metric == 'std':
                thresholds = self.std_thresholds
            else:  # entropy
                thresholds = self.entropy_thresholds
            
            for threshold in tqdm(thresholds, desc=f"{uncertainty_metric} thresholds"):
                try:
                    result = self.evaluate_configuration(
                        detector, embedding, uncertainty_metric,
                        'threshold', threshold, baseline_results
                    )
                    result['embedding_approach'] = approach_name
                    approach_results.append(result)
                except Exception as e:
                    print(f"  [WARNING] Failed for threshold={threshold}: {e}")
                    continue
            
            # Test top-k removal
            print(f"\nTesting TOP-K removal strategy...")
            for k_percent in tqdm(self.top_k_percentages, desc=f"{uncertainty_metric} top-k"):
                try:
                    result = self.evaluate_configuration(
                        detector, embedding, uncertainty_metric,
                        'top_k', k_percent, baseline_results
                    )
                    result['embedding_approach'] = approach_name
                    approach_results.append(result)
                except Exception as e:
                    print(f"  [WARNING] Failed for top_k={k_percent}%: {e}")
                    continue
        
        return approach_results
    
    def run_full_tuning(self):
        """
        Run hyperparameter tuning for all embedding approaches.
        """
        print(f"\n{'#'*80}")
        print(f"# STARTING FULL HYPERPARAMETER TUNING")
        print(f"{'#'*80}\n")
        
        # Iterate over all embedding approaches
        for approach in self.embedding_approaches:
            approach_results = self.tune_embedding_approach(approach)
            self.all_results.extend(approach_results)
            
            # Save intermediate results
            self.save_results()
        
        print(f"\n{'#'*80}")
        print(f"# HYPERPARAMETER TUNING COMPLETE")
        print(f"{'#'*80}\n")
        
        # Generate comprehensive report
        self.generate_report()
        self.visualize_results()
    
    def save_results(self):
        """
        Save all results to JSON file.
        """
        results_path = os.path.join(self.results_dir, 'all_results.json')
        
        # Convert to serializable format
        results_to_save = {
            'dataset': self.dataset_name,
            'llm_name': self.llm_name,
            'num_models': self.num_models,
            'decoder_type': self.decoder_type,
            'baseline_results': self.baseline_results,
            'all_results': self.all_results
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"[OK] Saved results to: {results_path}")
    
    def generate_report(self):
        """
        Generate comprehensive report with top-5 results for each embedding approach.
        """
        print(f"\n{'='*80}")
        print(f"GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*80}\n")
        
        report_path = os.path.join(self.results_dir, 'report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"LLM: {self.llm_name}\n")
            f.write(f"Number of models: {self.num_models}\n")
            f.write(f"Decoder type: {self.decoder_type}\n")
            f.write(f"Total configurations tested: {len(self.all_results)}\n\n")
            
            # Report for each embedding approach
            for approach in self.embedding_approaches:
                f.write("\n" + "="*80 + "\n")
                f.write(f"EMBEDDING APPROACH: {approach.upper()}\n")
                f.write("="*80 + "\n\n")
                
                # Filter results for this approach
                approach_results = [r for r in self.all_results if r['embedding_approach'] == approach]
                
                if not approach_results:
                    f.write("No results available for this approach.\n")
                    continue
                
                # Baseline performance
                baseline = self.baseline_results.get(approach, {})
                f.write(f"BASELINE PERFORMANCE:\n")
                f.write(f"  Test Accuracy: {baseline.get('test_acc', 0):.4f}\n")
                f.write(f"  Test F1:       {baseline.get('test_f1', 0):.4f}\n\n")
                
                # Top 5 by test accuracy improvement
                f.write("TOP 5 CONFIGURATIONS BY VALIDATION ACCURACY IMPROVEMENT (test shown, not used for ranking):\n")
                f.write("-"*80 + "\n")
                top5_acc = sorted(approach_results, key=lambda x: x['val_acc_improvement'], reverse=True)[:5]
                for i, result in enumerate(top5_acc, 1):
                    f.write(f"\n{i}. Metric: {result['uncertainty_metric']}, "
                           f"Strategy: {result['removal_strategy']}, "
                           f"Threshold: {result['threshold']}\n")
                    f.write(f"   Test Acc: {result['modified_test_acc']:.4f} "
                           f"(+{result['test_acc_improvement']:.4f})\n")
                    f.write(f"   Test F1:  {result['modified_test_f1']:.4f} "
                           f"(+{result['test_f1_improvement']:.4f})\n")
                    f.write(f"   Removed:  {result['removal_percentage']:.2f}% of edges\n")
                
                # Top 5 by test F1 improvement
                f.write("\n\nTOP 5 CONFIGURATIONS BY VALIDATION F1 IMPROVEMENT (test shown, not used for ranking):\n")
                f.write("-"*80 + "\n")
                top5_f1 = sorted(approach_results, key=lambda x: x['val_f1_improvement'], reverse=True)[:5]
                for i, result in enumerate(top5_f1, 1):
                    f.write(f"\n{i}. Metric: {result['uncertainty_metric']}, "
                           f"Strategy: {result['removal_strategy']}, "
                           f"Threshold: {result['threshold']}\n")
                    f.write(f"   Test Acc: {result['modified_test_acc']:.4f} "
                           f"(+{result['test_acc_improvement']:.4f})\n")
                    f.write(f"   Test F1:  {result['modified_test_f1']:.4f} "
                           f"(+{result['test_f1_improvement']:.4f})\n")
                    f.write(f"   Removed:  {result['removal_percentage']:.2f}% of edges\n")
                
                f.write("\n")
            
            # Overall best configurations
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL BEST CONFIGURATIONS (ACROSS ALL EMBEDDINGS)\n")
            f.write("="*80 + "\n\n")
            
            overall_top5_acc = sorted(self.all_results, key=lambda x: x['val_acc_improvement'], reverse=True)[:5]
            f.write("TOP 5 BY VALIDATION ACCURACY IMPROVEMENT (test shown, not used for ranking):\n")
            f.write("-"*80 + "\n")
            for i, result in enumerate(overall_top5_acc, 1):
                f.write(f"\n{i}. Embedding: {result['embedding_approach']}, "
                       f"Metric: {result['uncertainty_metric']}, "
                       f"Strategy: {result['removal_strategy']}, "
                       f"Threshold: {result['threshold']}\n")
                f.write(f"   Test Acc: {result['modified_test_acc']:.4f} "
                       f"(+{result['test_acc_improvement']:.4f})\n")
                f.write(f"   Test F1:  {result['modified_test_f1']:.4f} "
                       f"(+{result['test_f1_improvement']:.4f})\n")
                f.write(f"   Removed:  {result['removal_percentage']:.2f}% of edges\n")
        
        print(f"[OK] Report saved to: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print(f.read())
    
    def visualize_results(self):
        """
        Create comprehensive visualizations of tuning results.
        """
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(self.all_results)
        
        # 1. Heatmap for each embedding approach
        for approach in self.embedding_approaches:
            self.plot_heatmap_for_approach(df, approach)
        
        # 2. Comparison across embedding approaches
        self.plot_embedding_comparison(df)
        
        # 3. Removal percentage vs improvement
        self.plot_removal_vs_improvement(df)
        
        # 4. Best configurations summary
        self.plot_best_configurations_summary(df)
        
        print(f"[OK] All visualizations saved to: {self.results_dir}")
    
    def plot_heatmap_for_approach(self, df, approach):
        """
        Create heatmap showing test accuracy improvement for different configurations.
        """
        approach_df = df[df['embedding_approach'] == approach]
        
        if approach_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # STD + Threshold
        std_thresh = approach_df[
            (approach_df['uncertainty_metric'] == 'std') & 
            (approach_df['removal_strategy'] == 'threshold')
        ]
        if not std_thresh.empty:
            pivot_std_thresh = std_thresh.pivot_table(
                values='test_acc_improvement',
                index='threshold',
                aggfunc='mean'
            )
            sns.heatmap(pivot_std_thresh, annot=True, fmt='.4f', cmap='RdYlGn', 
                       center=0, ax=axes[0, 0], cbar_kws={'label': 'Acc Improvement'})
            axes[0, 0].set_title(f'{approach.upper()} - STD + Threshold', fontweight='bold', fontsize=14)
            axes[0, 0].set_ylabel('Threshold')
        
        # STD + Top-K
        std_topk = approach_df[
            (approach_df['uncertainty_metric'] == 'std') & 
            (approach_df['removal_strategy'] == 'top_k')
        ]
        if not std_topk.empty:
            pivot_std_topk = std_topk.pivot_table(
                values='test_acc_improvement',
                index='threshold',
                aggfunc='mean'
            )
            sns.heatmap(pivot_std_topk, annot=True, fmt='.4f', cmap='RdYlGn', 
                       center=0, ax=axes[0, 1], cbar_kws={'label': 'Acc Improvement'})
            axes[0, 1].set_title(f'{approach.upper()} - STD + Top-K', fontweight='bold', fontsize=14)
            axes[0, 1].set_ylabel('Top-K %')
        
        # Entropy + Threshold
        ent_thresh = approach_df[
            (approach_df['uncertainty_metric'] == 'entropy') & 
            (approach_df['removal_strategy'] == 'threshold')
        ]
        if not ent_thresh.empty:
            pivot_ent_thresh = ent_thresh.pivot_table(
                values='test_acc_improvement',
                index='threshold',
                aggfunc='mean'
            )
            sns.heatmap(pivot_ent_thresh, annot=True, fmt='.4f', cmap='RdYlGn', 
                       center=0, ax=axes[1, 0], cbar_kws={'label': 'Acc Improvement'})
            axes[1, 0].set_title(f'{approach.upper()} - Entropy + Threshold', fontweight='bold', fontsize=14)
            axes[1, 0].set_ylabel('Threshold')
        
        # Entropy + Top-K
        ent_topk = approach_df[
            (approach_df['uncertainty_metric'] == 'entropy') & 
            (approach_df['removal_strategy'] == 'top_k')
        ]
        if not ent_topk.empty:
            pivot_ent_topk = ent_topk.pivot_table(
                values='test_acc_improvement',
                index='threshold',
                aggfunc='mean'
            )
            sns.heatmap(pivot_ent_topk, annot=True, fmt='.4f', cmap='RdYlGn', 
                       center=0, ax=axes[1, 1], cbar_kws={'label': 'Acc Improvement'})
            axes[1, 1].set_title(f'{approach.upper()} - Entropy + Top-K', fontweight='bold', fontsize=14)
            axes[1, 1].set_ylabel('Top-K %')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, f'heatmap_{approach}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved heatmap for {approach}: {save_path}")
    
    def plot_embedding_comparison(self, df):
        """
        Compare best results across different embedding approaches.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get best result for each embedding approach
        best_results = []
        for approach in self.embedding_approaches:
            approach_df = df[df['embedding_approach'] == approach]
            if not approach_df.empty:
                best_acc = approach_df.nlargest(1, 'val_acc_improvement').iloc[0]
                best_results.append({
                    'embedding': approach,
                    'test_acc_improvement': best_acc['test_acc_improvement'],
                    'test_f1_improvement': best_acc['test_f1_improvement']
                })
        
        best_df = pd.DataFrame(best_results)
        
        # Plot 1: Test accuracy improvement
        axes[0].bar(best_df['embedding'], best_df['test_acc_improvement'], 
                   color='skyblue', edgecolor='black')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Embedding Approach', fontsize=12)
        axes[0].set_ylabel('Test Accuracy Improvement', fontsize=12)
        axes[0].set_title('Best Test Accuracy Improvement by Embedding', fontweight='bold', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Plot 2: Test F1 improvement
        axes[1].bar(best_df['embedding'], best_df['test_f1_improvement'], 
                   color='lightcoral', edgecolor='black')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Embedding Approach', fontsize=12)
        axes[1].set_ylabel('Test F1 Improvement', fontsize=12)
        axes[1].set_title('Best Test F1 Improvement by Embedding', fontweight='bold', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'embedding_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved embedding comparison: {save_path}")
    
    def plot_removal_vs_improvement(self, df):
        """
        Plot relationship between removal percentage and improvement.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Removal % vs Accuracy Improvement
        for approach in self.embedding_approaches:
            approach_df = df[df['embedding_approach'] == approach]
            if not approach_df.empty:
                axes[0].scatter(approach_df['removal_percentage'], 
                              approach_df['test_acc_improvement'],
                              label=approach, alpha=0.6, s=50)
        
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Removal Percentage (%)', fontsize=12)
        axes[0].set_ylabel('Test Accuracy Improvement', fontsize=12)
        axes[0].set_title('Removal % vs Accuracy Improvement', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Removal % vs F1 Improvement
        for approach in self.embedding_approaches:
            approach_df = df[df['embedding_approach'] == approach]
            if not approach_df.empty:
                axes[1].scatter(approach_df['removal_percentage'], 
                              approach_df['test_f1_improvement'],
                              label=approach, alpha=0.6, s=50)
        
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Removal Percentage (%)', fontsize=12)
        axes[1].set_ylabel('Test F1 Improvement', fontsize=12)
        axes[1].set_title('Removal % vs F1 Improvement', fontweight='bold', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'removal_vs_improvement.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved removal vs improvement plot: {save_path}")
    
    def plot_best_configurations_summary(self, df):
        """
        Create summary visualization of top-5 configurations across all embeddings.
        """
        # Get top 5 overall
        top5 = df.nlargest(5, 'val_acc_improvement')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Prepare labels
        labels = [
            f"{row['embedding_approach']}\n{row['uncertainty_metric']}\n"
            f"{row['removal_strategy']}\n{row['threshold']}"
            for _, row in top5.iterrows()
        ]
        
        # Plot 1: Test accuracy
        x = np.arange(len(labels))
        width = 0.35
        
        axes[0].bar(x - width/2, top5['baseline_test_acc'], width, 
                   label='Baseline', color='lightgray', edgecolor='black')
        axes[0].bar(x + width/2, top5['modified_test_acc'], width, 
                   label='After Removal', color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Test Accuracy', fontsize=12)
        axes[0].set_title('Top 5 Configurations - Test Accuracy', fontweight='bold', fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        # Plot 2: Test F1
        axes[1].bar(x - width/2, top5['baseline_test_f1'], width, 
                   label='Baseline', color='lightgray', edgecolor='black')
        axes[1].bar(x + width/2, top5['modified_test_f1'], width, 
                   label='After Removal', color='lightcoral', edgecolor='black')
        axes[1].set_ylabel('Test F1 Score', fontsize=12)
        axes[1].set_title('Top 5 Configurations - Test F1', fontweight='bold', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=9)
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'top5_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved top-5 summary: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for edge editing')
    parser.add_argument('--dataset_name', type=str, default='wikics',
                       help='Name of the dataset (default: cora)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                       help='Name of the LLM model (default: llama_3.2_1B)')
    parser.add_argument('--peft_type', type=str, default='lora',
                       help='Type of PEFT (default: lora)')
    parser.add_argument('--num_models', type=int, default=10,
                       help='Number of edge predictor models for ensemble (default: 10)')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                       choices=['dot_product', 'mlp', 'bilinear'],
                       help='Edge predictor decoder type (default: mlp)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING FOR EDGE EDITING")
    print("="*80)
    print(f"Dataset: {args.dataset_name}")
    print(f"LLM: {args.llm_name}")
    print(f"PEFT: {args.peft_type}")
    print(f"Number of models: {args.num_models}")
    print(f"Decoder type: {args.decoder_type}")
    print("="*80 + "\n")
    
    # Create tuner
    tuner = HyperparameterTuner(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        num_models=args.num_models,
        decoder_type=args.decoder_type
    )
    
    # Run full tuning
    tuner.run_full_tuning()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print(f"Results saved to: {tuner.results_dir}")
    print("="*80 + "\n")
