# Hyperparameter tuning for ensemble edge edge editing with uncertainty-based edge detection
# Find best hyperparameters for both std and entropy metrics
# Test different removal strategies: threshold and top_k
# Generate comprehensive reports and visualizations

import torch
import numpy as np
import os
import json
import argparse
from copy import deepcopy
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import set_mgnn_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.reporter import ReportResults
from LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_edge_predictor import EnsembleEdgePredictor


class HyperparameterTuner:
    """
    Systematic hyperparameter tuning for ensemble edge editing.
    Tests different removal strategies, top-k ratios, and uncertainty metrics.
    """
    
    def __init__(self, cfg, dataset, embeddings_dict, save_dir='./hyperparameter_tuning_results'):
        """
        Initialize hyperparameter tuner.
        
        Args:
            cfg: Configuration object
            dataset: Graph dataset
            embeddings_dict: Dictionary of embeddings
            save_dir: Directory to save tuning results
        """
        self.cfg = cfg
        self.dataset = dataset
        self.embeddings_dict = embeddings_dict
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Store results
        self.tuning_results = []
        self.best_config = None
        self.best_score = -float('inf')
        
    def define_hyperparameter_grid(self):
        """
        Define the hyperparameter search space.
        
        Returns:
            hyperparameter_grid: Dictionary of hyperparameters to tune
        """
        hyperparameter_grid = {
            'removal_strategy': ['high_uncertainty', 'low_confidence', 'entropy'],
            'top_k_ratio': [0.05, 0.1, 0.15, 0.2, 0.25],  # 5%, 10%, 15%, 20%, 25% edge removal
            'train_number': [5],  # Fixed for speed (can add [3, 5, 7] for more thorough search)
            'ensemble_approach': ['learnable_per_classes']  # Best performing approach
        }
        
        return hyperparameter_grid
    
    def run_single_configuration(self, config, run_ensemble=True):
        """
        Run a single hyperparameter configuration.
        
        Args:
            config: Dictionary with hyperparameter values
            run_ensemble: Whether to train ensemble model
            
        Returns:
            results: Dictionary with performance metrics
        """
        print(f"\n{'='*80}")
        print(f"Testing Configuration:")
        print(f"  Removal Strategy: {config['removal_strategy']}")
        print(f"  Top-k Ratio: {config['top_k_ratio']}")
        print(f"  Train Number: {config['train_number']}")
        print(f"  Ensemble Approach: {config['ensemble_approach']}")
        print(f"{'='*80}\n")
        
        try:
            # Extract dataset name properly (cfg.dataset might be an object)
            if isinstance(self.cfg.dataset, str):
                dataset_name = self.cfg.dataset
            elif hasattr(self.cfg.dataset, 'name'):
                dataset_name = self.cfg.dataset.name
            else:
                dataset_name = 'cora'  # Default fallback
            
            # Create fresh GNN config for this run
            gnn_cfg = set_mgnn_cfg(datset_name=dataset_name)
            
            # Initialize ensemble predictor
            ensemble_predictor = EnsembleEdgePredictor(gnn_cfg, self.dataset, self.device)
            
            # Step 1: Train ensemble edge predictor
            print("\n[1/5] Training ensemble edge predictor...")
            ensemble_stats = ensemble_predictor.train_ensemble_edge_predictor(
                embeddings_dict=self.embeddings_dict,
                train_number=config['train_number']
            )
            
            # Step 2: Construct modified datasets
            print("\n[2/5] Constructing modified datasets...")
            modified_datasets = ensemble_predictor.construct_ensemble_modified_dataset(
                embeddings_dict=self.embeddings_dict,
                removal_strategy=config['removal_strategy'],
                threshold=0.5,
                top_k_ratio=config['top_k_ratio']
            )
            
            # Step 3: Train GNNs on original dataset
            print("\n[3/5] Training GNNs on original dataset...")
            temp_original_dir = os.path.join(self.save_dir, f'temp_original_{config["removal_strategy"]}_{config["top_k_ratio"]}')
            original_gnn_results, original_gnn_models = ensemble_predictor.train_gnns_on_original_dataset(
                embeddings_dict=self.embeddings_dict,
                save_dir=temp_original_dir
            )
            
            # Step 4: Train GNNs on modified datasets
            print("\n[4/5] Training GNNs on modified datasets...")
            temp_modified_dir = os.path.join(self.save_dir, f'temp_modified_{config["removal_strategy"]}_{config["top_k_ratio"]}')
            modified_gnn_results, modified_gnn_models = ensemble_predictor.train_gnns_on_modified_datasets(
                embeddings_dict=self.embeddings_dict,
                save_dir=temp_modified_dir
            )
            
            # Calculate individual GNN improvements
            individual_improvements = {}
            for emb_name in original_gnn_results.keys():
                if emb_name in modified_gnn_results:
                    orig = original_gnn_results[emb_name]
                    mod = modified_gnn_results[emb_name]
                    individual_improvements[emb_name] = {
                        'val_acc_change': mod.get('val_acc', 0) - orig.get('val_acc', 0),
                        'test_acc_change': mod.get('test_acc', 0) - orig.get('test_acc', 0),
                        'test_f1_change': mod.get('test_f1', 0) - orig.get('test_f1', 0),
                    }
            
            avg_individual_val_acc_change = np.mean([v['val_acc_change'] for v in individual_improvements.values()])
            avg_individual_acc_change = np.mean([v['test_acc_change'] for v in individual_improvements.values()])
            avg_individual_f1_change = np.mean([v['test_f1_change'] for v in individual_improvements.values()])
            
            results = {
                'config': config,
                'individual_gnn_results': {
                    'original': original_gnn_results,
                    'modified': modified_gnn_results,
                    'improvements': individual_improvements,
                    'avg_val_acc_change': float(avg_individual_val_acc_change),
                    'avg_test_acc_change': float(avg_individual_acc_change),
                    'avg_test_f1_change': float(avg_individual_f1_change)
                },
                'edge_stats': {
                    'original_edges': ensemble_predictor.detector_dataset.edge_index.size(1),
                    'removed_edges': int(ensemble_predictor.detector_dataset.edge_index.size(1) * config['top_k_ratio']),
                    'remaining_edges': list(modified_datasets.values())[0].edge_index.size(1)
                }
            }
            
            # Step 5: Train ensemble (optional, can be expensive)
            if run_ensemble:
                print("\n[5/5] Training ensemble models...")
                
                # Ensemble on original dataset
                temp_original_datasets = {}
                for emb_name, embedding in self.embeddings_dict.items():
                    dataset_copy = ensemble_predictor.detector_dataset.clone()
                    dataset_copy.x = embedding
                    temp_original_datasets[emb_name] = dataset_copy
                
                original_modified_datasets = ensemble_predictor.modified_datasets
                ensemble_predictor.modified_datasets = temp_original_datasets
                
                original_ensemble_results, _, _ = ensemble_predictor.train_ensemble_node_classification(
                    embeddings_dict=self.embeddings_dict,
                    gnn_models=original_gnn_models,
                    ensemble_approach=config['ensemble_approach'],
                    does_report_training_process=False,
                    reporter=None
                )
                
                ensemble_predictor.modified_datasets = original_modified_datasets
                
                # Ensemble on modified dataset
                modified_ensemble_results, _, _ = ensemble_predictor.train_ensemble_node_classification(
                    embeddings_dict=self.embeddings_dict,
                    gnn_models=modified_gnn_models,
                    ensemble_approach=config['ensemble_approach'],
                    does_report_training_process=False,
                    reporter=None
                )
                
                # Calculate ensemble improvements
                if original_ensemble_results and modified_ensemble_results:
                    ensemble_test_acc_change = (modified_ensemble_results['test']['accuracy'] - 
                                               original_ensemble_results['test']['accuracy'])
                    ensemble_test_f1_change = (modified_ensemble_results['test']['macro_f1'] - 
                                              original_ensemble_results['test']['macro_f1'])
                    
                    results['ensemble_results'] = {
                        'original': original_ensemble_results,
                        'modified': modified_ensemble_results,
                        'test_acc_change': float(ensemble_test_acc_change),
                        'test_f1_change': float(ensemble_test_f1_change)
                    }
                else:
                    results['ensemble_results'] = None
            else:
                results['ensemble_results'] = None
                print("\n[5/5] Skipping ensemble training (use --run_ensemble to enable)")
            
            print(f"\n{'='*80}")
            print(f"Configuration Results:")
            print(f"  Individual GNN Avg Test Acc Change: {avg_individual_acc_change:+.4f}")
            print(f"  Individual GNN Avg Test F1 Change: {avg_individual_f1_change:+.4f}")
            if results['ensemble_results']:
                print(f"  Ensemble Test Acc Change: {results['ensemble_results']['test_acc_change']:+.4f}")
                print(f"  Ensemble Test F1 Change: {results['ensemble_results']['test_f1_change']:+.4f}")
            print(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            print(f"\n⚠ Error in configuration: {e}")
            import traceback
            traceback.print_exc()
            return {
                'config': config,
                'error': str(e),
                'individual_gnn_results': None,
                'ensemble_results': None
            }
    
    def run_hyperparameter_search(self, run_ensemble=True):
        """
        Run full hyperparameter search over the grid.
        
        Args:
            run_ensemble: Whether to train ensemble for each configuration
            
        Returns:
            results_df: DataFrame with all results
        """
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING FOR ENSEMBLE EDGE EDITING")
        print(f"{'='*80}\n")
        
        # Define grid
        param_grid = self.define_hyperparameter_grid()
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        configs = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"Total configurations to test: {len(configs)}")
        print(f"Dataset: {self.cfg.dataset}")
        print(f"Device: {self.device}\n")
        
        # Test each configuration
        for idx, config in enumerate(configs, 1):
            print(f"\n{'#'*80}")
            print(f"Configuration {idx}/{len(configs)}")
            print(f"{'#'*80}")
            
            results = self.run_single_configuration(config, run_ensemble)
            self.tuning_results.append(results)
            
            # Track best configuration based on individual GNN improvement on VALIDATION (never test)
            if results['individual_gnn_results'] is not None:
                score = results['individual_gnn_results']['avg_val_acc_change']
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config
                    print(f"\n🏆 New best configuration! Score: {score:.4f}")
        
        # Convert results to DataFrame
        results_df = self.create_results_dataframe()
        
        return results_df
    
    def create_results_dataframe(self):
        """
        Create a pandas DataFrame from tuning results.
        
        Returns:
            df: DataFrame with results
        """
        rows = []
        
        for result in self.tuning_results:
            config = result['config']
            
            row = {
                'removal_strategy': config['removal_strategy'],
                'top_k_ratio': config['top_k_ratio'],
                'train_number': config['train_number'],
                'ensemble_approach': config['ensemble_approach']
            }
            
            # Individual GNN results
            if result['individual_gnn_results'] is not None:
                row['ind_avg_val_acc_change'] = result['individual_gnn_results']['avg_val_acc_change']
                row['ind_avg_test_acc_change'] = result['individual_gnn_results']['avg_test_acc_change']
                row['ind_avg_test_f1_change'] = result['individual_gnn_results']['avg_test_f1_change']
            else:
                row['ind_avg_val_acc_change'] = None
                row['ind_avg_test_acc_change'] = None
                row['ind_avg_test_f1_change'] = None
            
            # Ensemble results
            if result['ensemble_results'] is not None:
                row['ens_test_acc_change'] = result['ensemble_results']['test_acc_change']
                row['ens_test_f1_change'] = result['ensemble_results']['test_f1_change']
            else:
                row['ens_test_acc_change'] = None
                row['ens_test_f1_change'] = None
            
            # Edge stats
            if 'edge_stats' in result:
                row['original_edges'] = result['edge_stats']['original_edges']
                row['removed_edges'] = result['edge_stats']['removed_edges']
                row['remaining_edges'] = result['edge_stats']['remaining_edges']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def visualize_results(self, results_df):
        """
        Create visualizations of hyperparameter tuning results.
        
        Args:
            results_df: DataFrame with tuning results
        """
        print(f"\n{'='*80}")
        print(f"Generating Visualizations...")
        print(f"{'='*80}\n")
        
        # Check if we have valid data
        if results_df.empty or 'ind_avg_test_acc_change' not in results_df.columns:
            print("⚠ No valid results to visualize. Skipping visualization.")
            return
        
        # Check if all values are None/NaN
        if results_df['ind_avg_test_acc_change'].isna().all():
            print("⚠ All results are invalid. Skipping visualization.")
            return
        
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 1. Heatmap: Removal Strategy vs Top-k Ratio (Individual GNN)
        if 'ind_avg_test_acc_change' in results_df.columns and not results_df['ind_avg_test_acc_change'].isna().all():
            pivot_ind = results_df.pivot_table(
                values='ind_avg_test_acc_change',
                index='removal_strategy',
                columns='top_k_ratio',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_ind, annot=True, fmt='.4f', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Test Accuracy Change'})
            plt.title('Individual GNN Performance: Test Accuracy Change\n(Removal Strategy vs Top-k Ratio)')
            plt.xlabel('Top-k Ratio (Edge Removal %)')
            plt.ylabel('Removal Strategy')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'heatmap_individual_gnn_accuracy.png'), dpi=300)
            plt.close()
            print("✓ Saved: heatmap_individual_gnn_accuracy.png")
        
        # 2. Heatmap: Ensemble Performance
        if 'ens_test_acc_change' in results_df.columns and results_df['ens_test_acc_change'].notna().any():
            pivot_ens = results_df.pivot_table(
                values='ens_test_acc_change',
                index='removal_strategy',
                columns='top_k_ratio',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_ens, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Test Accuracy Change'})
            plt.title('Ensemble Performance: Test Accuracy Change\n(Removal Strategy vs Top-k Ratio)')
            plt.xlabel('Top-k Ratio (Edge Removal %)')
            plt.ylabel('Removal Strategy')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'heatmap_ensemble_accuracy.png'), dpi=300)
            plt.close()
            print("✓ Saved: heatmap_ensemble_accuracy.png")
        
        # 3. Line plot: Top-k Ratio vs Performance for each strategy
        plt.figure(figsize=(12, 6))
        for strategy in results_df['removal_strategy'].unique():
            strategy_data = results_df[results_df['removal_strategy'] == strategy]
            plt.plot(strategy_data['top_k_ratio'], strategy_data['ind_avg_test_acc_change'], 
                    marker='o', label=strategy, linewidth=2)
        plt.xlabel('Top-k Ratio (Edge Removal %)')
        plt.ylabel('Test Accuracy Change')
        plt.title('Individual GNN Performance vs Edge Removal Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'line_plot_topk_vs_performance.png'), dpi=300)
        plt.close()
        print("✓ Saved: line_plot_topk_vs_performance.png")
        
        # 4. Bar plot: Best configuration for each strategy
        best_per_strategy = results_df.loc[results_df.groupby('removal_strategy')['ind_avg_val_acc_change'].idxmax()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(best_per_strategy)), best_per_strategy['ind_avg_test_acc_change'])
        plt.xticks(range(len(best_per_strategy)), best_per_strategy['removal_strategy'], rotation=45)
        plt.ylabel('Test Accuracy Change')
        plt.title('Best Performance per Removal Strategy')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Color bars
        for i, bar in enumerate(bars):
            if best_per_strategy.iloc[i]['ind_avg_test_acc_change'] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Add top-k ratio labels
        for i, (idx, row) in enumerate(best_per_strategy.iterrows()):
            plt.text(i, row['ind_avg_test_acc_change'], f"k={row['top_k_ratio']:.2f}", 
                    ha='center', va='bottom' if row['ind_avg_test_acc_change'] > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'bar_plot_best_per_strategy.png'), dpi=300)
        plt.close()
        print("✓ Saved: bar_plot_best_per_strategy.png")
        
        # 5. Comparison: Individual vs Ensemble
        if 'ens_test_acc_change' in results_df.columns and results_df['ens_test_acc_change'].notna().any():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Individual GNN
            for strategy in results_df['removal_strategy'].unique():
                strategy_data = results_df[results_df['removal_strategy'] == strategy]
                ax1.plot(strategy_data['top_k_ratio'], strategy_data['ind_avg_test_acc_change'], 
                        marker='o', label=strategy, linewidth=2)
            ax1.set_xlabel('Top-k Ratio')
            ax1.set_ylabel('Test Accuracy Change')
            ax1.set_title('Individual GNN Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Ensemble
            for strategy in results_df['removal_strategy'].unique():
                strategy_data = results_df[results_df['removal_strategy'] == strategy]
                ax2.plot(strategy_data['top_k_ratio'], strategy_data['ens_test_acc_change'], 
                        marker='s', label=strategy, linewidth=2)
            ax2.set_xlabel('Top-k Ratio')
            ax2.set_ylabel('Test Accuracy Change')
            ax2.set_title('Ensemble Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'comparison_individual_vs_ensemble.png'), dpi=300)
            plt.close()
            print("✓ Saved: comparison_individual_vs_ensemble.png")
        
        print(f"\nAll visualizations saved to: {viz_dir}\n")
    
    def generate_report(self, results_df):
        """
        Generate comprehensive text report.
        
        Args:
            results_df: DataFrame with tuning results
        """
        print(f"\n{'='*80}")
        print(f"Generating Report...")
        print(f"{'='*80}\n")
        
        report_path = os.path.join(self.save_dir, 'tuning_report.txt')
        
        # Extract dataset name properly
        if isinstance(self.cfg.dataset, str):
            dataset_name = self.cfg.dataset
        elif hasattr(self.cfg.dataset, 'name'):
            dataset_name = self.cfg.dataset.name
        else:
            dataset_name = str(self.cfg.dataset)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING REPORT\n")
            f.write("Ensemble Edge Editing with Uncertainty-Based Detection\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total Configurations Tested: {len(self.tuning_results)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("BEST CONFIGURATION\n")
            f.write("="*80 + "\n")
            if self.best_config and self.best_score != -float('inf'):
                f.write(f"Removal Strategy: {self.best_config['removal_strategy']}\n")
                f.write(f"Top-k Ratio: {self.best_config['top_k_ratio']}\n")
                f.write(f"Train Number: {self.best_config['train_number']}\n")
                f.write(f"Best Score (Avg Val Acc Change, used for selection): {self.best_score:.4f}\n\n")
            else:
                f.write("No valid configuration found. All runs failed.\n\n")
            
            f.write("="*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            # Check if we have valid data
            if results_df.empty or 'ind_avg_test_acc_change' not in results_df.columns or results_df['ind_avg_test_acc_change'].isna().all():
                f.write("No valid results available. All configurations failed.\n\n")
            else:
                # Individual GNN results
                f.write("Individual GNN Results:\n")
                f.write("-"*80 + "\n")
                f.write(f"Mean Test Acc Change: {results_df['ind_avg_test_acc_change'].mean():.4f}\n")
                f.write(f"Std Test Acc Change: {results_df['ind_avg_test_acc_change'].std():.4f}\n")
                f.write(f"Max Test Acc Change: {results_df['ind_avg_test_acc_change'].max():.4f}\n")
                f.write(f"Min Test Acc Change: {results_df['ind_avg_test_acc_change'].min():.4f}\n\n")
                
                # Ensemble results (if available)
                if 'ens_test_acc_change' in results_df.columns and results_df['ens_test_acc_change'].notna().any():
                    f.write("Ensemble Results:\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Mean Test Acc Change: {results_df['ens_test_acc_change'].mean():.4f}\n")
                    f.write(f"Std Test Acc Change: {results_df['ens_test_acc_change'].std():.4f}\n")
                    f.write(f"Max Test Acc Change: {results_df['ens_test_acc_change'].max():.4f}\n")
                    f.write(f"Min Test Acc Change: {results_df['ens_test_acc_change'].min():.4f}\n\n")
                
                f.write("="*80 + "\n")
                f.write("RESULTS BY REMOVAL STRATEGY\n")
                f.write("="*80 + "\n\n")
                
                for strategy in results_df['removal_strategy'].unique():
                    strategy_data = results_df[results_df['removal_strategy'] == strategy]
                    # Filter out NaN values
                    valid_data = strategy_data[strategy_data['ind_avg_test_acc_change'].notna()]
                    if not valid_data.empty:
                        f.write(f"{strategy.upper()}:\n")
                        f.write(f"  Mean Test Acc Change: {valid_data['ind_avg_test_acc_change'].mean():.4f}\n")
                        f.write(f"  Best Top-k Ratio (by val): {valid_data.loc[valid_data['ind_avg_val_acc_change'].idxmax(), 'top_k_ratio']:.2f}\n")
                        f.write(f"  Best Score: {valid_data['ind_avg_test_acc_change'].max():.4f}\n\n")
                    else:
                        f.write(f"{strategy.upper()}:\n")
                        f.write(f"  No valid results for this strategy.\n\n")
            
            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS TABLE\n")
            f.write("="*80 + "\n\n")
            f.write(results_df.to_string())
            f.write("\n\n")
        
        print(f"Report saved to: {report_path}\n")
    
    def save_results(self, results_df):
        """
        Save all results to files.
        
        Args:
            results_df: DataFrame with results
        """
        print(f"\n{'='*80}")
        print(f"Saving Results...")
        print(f"{'='*80}\n")
        
        # Save DataFrame as CSV
        csv_path = os.path.join(self.save_dir, 'tuning_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: tuning_results.csv")
        
        # Save full results as JSON
        json_path = os.path.join(self.save_dir, 'tuning_results_full.json')
        with open(json_path, 'w') as f:
            json.dump(self.tuning_results, f, indent=4, default=str)
        print(f"✓ Saved: tuning_results_full.json")
        
        # Save best configuration
        best_config_path = os.path.join(self.save_dir, 'best_configuration.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'best_config': self.best_config,
                'best_score': float(self.best_score)
            }, f, indent=4)
        print(f"✓ Saved: best_configuration.json")
        
        print(f"\nAll results saved to: {self.save_dir}\n")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Ensemble Edge Editing')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', help='LLM model name')
    parser.add_argument('--peft_type', type=str, default='lora', help='PEFT type')
    parser.add_argument('--save_dir', type=str, default='./hyperparameter_tuning_results', 
                       help='Directory to save tuning results')
    parser.add_argument('--run_ensemble', action='store_true', 
                       help='Train ensemble for each configuration (slow but comprehensive)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING - ENSEMBLE EDGE EDITING")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"LLM: {args.llm_name}")
    print(f"PEFT Type: {args.peft_type}")
    print(f"Run Ensemble: {args.run_ensemble}")
    print(f"{'='*80}\n")
    
    # Setup configuration and load dataset
    print("Loading configuration and dataset...")
    cfg = setup_finetuning_cfg(args.dataset, args.llm_name, args.peft_type)
    dataset = load_dataset(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract dataset name properly
    if isinstance(cfg.dataset, str):
        dataset_name = cfg.dataset
    elif hasattr(cfg.dataset, 'name'):
        dataset_name = cfg.dataset.name
    else:
        dataset_name = args.dataset
    
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}, Nodes: {dataset.num_nodes}, Edges: {dataset.edge_index.size(1)}\n")
    
    # Load all embeddings
    print("Loading embeddings from all PEFT methods...")
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    
    embeddings_dict = {
        'pissa': get_embedding_from_data(data_pissa),
        'orthogonal': get_embedding_from_data(data_orthogonal),
        'loftq': get_embedding_from_data(data_loftq),
        'eva': get_embedding_from_data(data_eva),
        'guassian': get_embedding_from_data(data_guassian)
    }
    
    print(f"Loaded {len(embeddings_dict)} embeddings:")
    for name, emb in embeddings_dict.items():
        print(f"  {name}: {emb.shape}")
    print()
    
    # Initialize tuner
    tuner = HyperparameterTuner(cfg, dataset, embeddings_dict, args.save_dir)
    
    # Run hyperparameter search
    results_df = tuner.run_hyperparameter_search(run_ensemble=args.run_ensemble)
    
    # Generate visualizations
    tuner.visualize_results(results_df)
    
    # Generate report
    tuner.generate_report(results_df)
    
    # Save all results
    tuner.save_results(results_df)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*80}")
    if tuner.best_config and tuner.best_score != -float('inf'):
        print(f"\nBest Configuration:")
        print(f"  Removal Strategy: {tuner.best_config['removal_strategy']}")
        print(f"  Top-k Ratio: {tuner.best_config['top_k_ratio']}")
        print(f"  Best Score: {tuner.best_score:.4f}")
    else:
        print(f"\n⚠ No valid configuration found. All runs failed.")
        print(f"Please check the error messages above for details.")
    print(f"\nAll results saved to: {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
