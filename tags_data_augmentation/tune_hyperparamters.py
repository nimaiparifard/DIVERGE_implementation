# in this part we want to tune hyperparamter to see best resukt for one dataset
# you report result with reporter and json in dataset_naem.txt and dataset_name.json
# saved best hyperparamters and best results
"""
    Hyperparameters consider for tuning:
        1- init_weight_approches
        2- all hybrid augmented dataset constrcuion
            2.1 cosine
            2.2 k nearest neighbors
            2.3 edge predictor
            2.4 cosine + k nearest neighbors
            2.5 cosine + edge predictor
            2.6 k nearest neighbors + edge predictor
            2.7 cosine + k nearest neighbors + edge predictor
        3- topk for cosine
        4- topk for k nearest neighbors
        5- topk for edge predictor
        6- threshold for cosine
        7- threshold for k nearest neighbors
        8- threshold for edge predictor
"""

import os
import json
import torch
import argparse
import itertools
from pathlib import Path

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import gnn_train_and_report
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.reporter import ReportResults
from LLMReasoner.TAPE.tags_data_augmentation.connect_new_nodes_to_graphs import (
    load_augmented_cache,
    train_node_classification_gnn_with_modified_data
)
from LLMReasoner.TAPE.tags_data_augmentation.agumented_dataset import create_augmented_dataset
from LLMReasoner.TAPE.tags_data_augmentation.edge_predictor import edge_predictor_train_and_report


def get_init_augmented_dataset_for_gnn(cfg, supervised=True):
    """
    Load initial datasets for GNN training with different initialization methods.

    Args:
        cfg: Configuration object with dataset and model settings
        supervised: If True, load supervised datasets; if False, load semi-supervised datasets

    Returns:
        tuple: (data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva)
    """
    base_path = 'artifacts/cache'
    data_pissa_path = os.path.join(base_path, f'llama_3.2_1B_{cfg.dataset.name}_seqcls_lora_init-pissa_pool-mean_semi_supervised_augmented.pt')
    data_orthogonal_path = os.path.join(base_path, f'llama_3.2_1B_{cfg.dataset.name}_seqcls_lora_init-orthogonal_pool-mean_semi_supervised_augmented.pt')
    data_guassian_path = os.path.join(base_path, f'llama_3.2_1B_{cfg.dataset.name}_seqcls_lora_init-gaussian_pool-mean_semi_supervised_augmented.pt')
    data_loftq_path = os.path.join(base_path, f'llama_3.2_1B_{cfg.dataset.name}_seqcls_lora_init-loftq_pool-mean_semi_supervised_augmented.pt')
    data_eva_path = os.path.join(base_path, f'llama_3.2_1B_{cfg.dataset.name}_seqcls_lora_init-eva_pool-mean_semi_supervised_augmented.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_pissa = torch.load(data_pissa_path, map_location=device)
    data_orthogonal = torch.load(data_orthogonal_path, map_location=device)
    data_guassian = torch.load(data_guassian_path, map_location=device)
    data_loftq = torch.load(data_loftq_path, map_location=device)
    data_eva = torch.load(data_eva_path, map_location=device)

    return data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva



class HyperparameterTuner:
    """Hyperparameter tuner for augmented dataset construction."""
    
    def __init__(self, dataset_name, llm_name='llama_3.2_1B', peft_type='lora'):
        self.dataset_name = dataset_name
        self.llm_name = llm_name
        self.peft_type = peft_type
        
        # Setup configuration
        self.cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
        
        # Initialize reporter
        results_dir = './results/hyperparameter_tuning'
        os.makedirs(results_dir, exist_ok=True)
        self.reporter = ReportResults(self.cfg, save_dir=results_dir, index_run=0)
        
        # Results tracking
        self.all_results = []
        self.best_result = None
        self.best_hyperparams = None
        
    def get_hybrid_configurations(self):
        """
        Get all 7 hybrid configurations for edge prediction.
        
        Returns:
            List of tuples (name, used_cosine, used_knn, used_edge_predictor)
        """
        configs = [
            ("cosine+edge_predictor", True, False, True),
        ]
        return configs
    
    def run_experiment(self, init_weight_approach, hybrid_config_name, 
                       used_cosine, used_knn, used_edge_predictor,
                       threshold_cosine, threshold_knn, threshold_edge_predictor,
                       topk_cosine, topk_knn, topk_edge_predictor,
                       edge_predictor_model=None):
        """
        Run a single experiment with given hyperparameters.
        
        Returns:
            dict: Results containing test_acc, test_f1, and hyperparameters
        """
        experiment_name = (
            f"{init_weight_approach}_{hybrid_config_name}_"
            f"tc{threshold_cosine:.2f}_tk{threshold_knn:.2f}_te{threshold_edge_predictor:.2f}_"
            f"kc{topk_cosine}_kk{topk_knn}_ke{topk_edge_predictor}"
        )
        
        print(f"\n{'='*100}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*100}")
        
        try:
            # Load augmented cache
            augmented_cache = load_augmented_cache(
                dataset_name=self.dataset_name,
                model_name=self.llm_name,
                pooling='mean',
                init_weight_approach=init_weight_approach,
            )
            
            # Create augmented dataset
            augmented_dataset, augmented_embeddings = create_augmented_dataset(
                cfg=self.cfg,
                augmented_cache=augmented_cache,
                threshold=0.5,  # Legacy parameter
                top_k=5,  # Legacy parameter
                init_weight_approach=init_weight_approach,
                used_cosine_similarity=used_cosine,
                used_k_nearest_neighbors=used_knn,
                used_edge_predictor=used_edge_predictor,
                k_cosine_similarity=5,
                k_knearest_neighbors=5,
                k_edge_predictor=5,
                threshold_cosine_similarity=threshold_cosine,
                threshold_k_nearest_neighbors=threshold_knn,
                threshold_edge_predictor=threshold_edge_predictor,
                top_k_cosine_similarity=topk_cosine,
                top_k_knearest_neighbors=topk_knn,
                top_k_edge_predictor=topk_edge_predictor,
                edge_predictor_model=edge_predictor_model,
                supervised=False
            )
            
            # Train GNN on augmented dataset
            augmented_dataset.x = augmented_embeddings
            results, model = train_node_classification_gnn_with_modified_data(
                augmented_dataset=augmented_dataset,
                augmented_embeddings=augmented_embeddings,
                cfg=self.cfg,
                title=experiment_name
            )
            
            # Package results
            experiment_result = {
                'experiment_name': experiment_name,
                'hyperparameters': {
                    'init_weight_approach': init_weight_approach,
                    'hybrid_config': hybrid_config_name,
                    'used_cosine': used_cosine,
                    'used_knn': used_knn,
                    'used_edge_predictor': used_edge_predictor,
                    'threshold_cosine': threshold_cosine,
                    'threshold_knn': threshold_knn,
                    'threshold_edge_predictor': threshold_edge_predictor,
                    'topk_cosine': topk_cosine,
                    'topk_knn': topk_knn,
                    'topk_edge_predictor': topk_edge_predictor,
                },
                'results': {
                    'test_acc': results['test_acc'],
                    'test_f1': results['test_f1'],
                    'test_weight_f1': results['test_weight_f1'],
                    'val_acc': results['val_acc'],
                    'val_f1': results['val_f1'],
                },
                'num_nodes': augmented_dataset.y.shape[0],
                'num_edges': augmented_dataset.edge_index.shape[1],
            }
            
            print(f"\n[RESULT] Test Acc: {results['test_acc']:.4f}, Test F1: {results['test_f1']:.4f}")
            
            return experiment_result
            
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def tune(self, init_weight_approaches=None, threshold_cosine_values=None,
             threshold_knn_values=None, threshold_edge_predictor_values=None,
             topk_cosine_values=None, topk_knn_values=None, topk_edge_predictor_values=None,
             train_edge_predictor=True, decoder_type='mlp'):
        """
        Run hyperparameter tuning over all combinations.
        
        Args:
            init_weight_approaches: List of init weight approaches to test
            threshold_cosine_values: List of threshold values for cosine similarity
            threshold_knn_values: List of threshold values for k-nearest neighbors
            threshold_edge_predictor_values: List of threshold values for edge predictor
            topk_cosine_values: List of top-k values for cosine similarity
            topk_knn_values: List of top-k values for k-nearest neighbors
            topk_edge_predictor_values: List of top-k values for edge predictor
            train_edge_predictor: Whether to train edge predictor
            decoder_type: Type of decoder for edge predictor ('dot_product', 'mlp', 'bilinear')
        """
        # Default values
        if init_weight_approaches is None:
            init_weight_approaches = ['pissa']
        if threshold_cosine_values is None:
            threshold_cosine_values = [0.90, 0.95]
        if threshold_knn_values is None:
            threshold_knn_values = [0.5, 1.0]
        if threshold_edge_predictor_values is None:
            threshold_edge_predictor_values = [0.5, 0.7]
        if topk_cosine_values is None:
            topk_cosine_values = [3, 5]
        if topk_knn_values is None:
            topk_knn_values = [3, 5]
        if topk_edge_predictor_values is None:
            topk_edge_predictor_values = [5, 10]
        
        # Get all hybrid configurations
        hybrid_configs = self.get_hybrid_configurations()
        
        # Report header
        self.reporter.report_title(f"Hyperparameter Tuning for {self.dataset_name.upper()}")
        self.reporter.report_txt(f"Dataset: {self.dataset_name}")
        self.reporter.report_txt(f"LLM: {self.llm_name}")
        self.reporter.report_txt(f"PEFT: {self.peft_type}")
        self.reporter.report_txt(f"\nSearching over:")
        self.reporter.report_txt(f"  - Init weight approaches: {init_weight_approaches}")
        # self.reporter.report_txt(f"  - Hybrid configs: {[name for name, _, _, _ in hybrid_configs]}")
        self.reporter.report_txt(f"  - Threshold cosine: {threshold_cosine_values}")
        self.reporter.report_txt(f"  - Threshold KNN: {threshold_knn_values}")
        self.reporter.report_txt(f"  - Threshold edge predictor: {threshold_edge_predictor_values}")
        self.reporter.report_txt(f"  - Top-K cosine: {topk_cosine_values}")
        self.reporter.report_txt(f"  - Top-K KNN: {topk_knn_values}")
        self.reporter.report_txt(f"  - Top-K edge predictor: {topk_edge_predictor_values}")
        self.reporter.report_txt(f"\n{'='*70}\n")
        
        # Baseline: Train on original dataset
        print(f"\n{'='*100}")
        print("Training Baseline (Original Dataset)")
        print(f"{'='*100}")

        from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
        data_pissa, _, _, _, _ = get_init_augmented_dataset_for_gnn(self.cfg, supervised=False)
        original_embedding = get_embedding_from_data(data_pissa)
        
        baseline_results, baseline_model = gnn_train_and_report(
            dataset_name=self.dataset_name,
            embedding=original_embedding,
            title="Baseline (Original)",
            does_print_training_process=False,
            supervised=False,
        )
        
        self.reporter.report_title("Baseline Results (Original Dataset)")
        self.reporter.report_txt(f"Test Accuracy: {baseline_results['test_acc']:.4f}")
        self.reporter.report_txt(f"Test F1 (Macro): {baseline_results['test_f1']:.4f}")
        self.reporter.report_txt(f"Test F1 (Weighted): {baseline_results['test_weight_f1']:.4f}")
        self.reporter.report_txt(f"Val Accuracy: {baseline_results['val_acc']:.4f}")
        self.reporter.report_txt(f"Val F1 (Macro): {baseline_results['val_f1']:.4f}\n")
        
        # Run hyperparameter search
        total_experiments = 0
        successful_experiments = 0
        
        for init_approach in init_weight_approaches:
            print(f"\n{'#'*100}")
            print(f"Init Weight Approach: {init_approach}")
            print(f"{'#'*100}")
            
            # Train edge predictor if needed (once per init_approach)
            edge_predictor_model = None
            if train_edge_predictor:
                print(f"\n[INFO] Training edge predictor for {init_approach}...")
                from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
                data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_augmented_dataset_for_gnn(self.cfg, supervised=False)
                
                embedding = None
                if init_approach == "pissa":
                    embedding = get_embedding_from_data(data_pissa)
                elif init_approach == "orthogonal":
                    embedding = get_embedding_from_data(data_orthogonal)
                elif init_approach == "guassian":
                    embedding = get_embedding_from_data(data_guassian)
                elif init_approach == "loftq":
                    embedding = get_embedding_from_data(data_loftq)
                elif init_approach == "eva":
                    embedding = get_embedding_from_data(data_eva)
                
                results, edge_predictor_model = edge_predictor_train_and_report(
                    dataset_name=self.dataset_name,
                    features=embedding,
                    decoder_approach_type=decoder_type,
                    training_strategy='without_splitting',
                    negative_sampling_ratio=1.0,
                    title=f"Edge Predictor ({init_approach})",
                    does_print_training_process=False
                )
            
            for hybrid_name, used_cosine, used_knn, used_edge_pred in hybrid_configs:
                # Skip configurations that don't match enabled methods
                if used_edge_pred and not train_edge_predictor:
                    continue
                
                # Generate parameter combinations for this hybrid config
                # Only iterate over parameters relevant to enabled methods
                threshold_cosine_list = threshold_cosine_values if used_cosine else [0.95]
                threshold_knn_list = threshold_knn_values if used_knn else [0.5]
                threshold_edge_pred_list = threshold_edge_predictor_values if used_edge_pred else [0.5]
                topk_cosine_list = topk_cosine_values if used_cosine else [3]
                topk_knn_list = topk_knn_values if used_knn else [3]
                topk_edge_pred_list = topk_edge_predictor_values if used_edge_pred else [5]
                
                param_combinations = itertools.product(
                    threshold_cosine_list,
                    threshold_knn_list,
                    threshold_edge_pred_list,
                    topk_cosine_list,
                    topk_knn_list,
                    topk_edge_pred_list
                )
                
                for tc, tk, te, kc, kk, ke in param_combinations:
                    total_experiments += 1
                    
                    result = self.run_experiment(
                        init_weight_approach=init_approach,
                        hybrid_config_name=hybrid_name,
                        used_cosine=used_cosine,
                        used_knn=used_knn,
                        used_edge_predictor=used_edge_pred,
                        threshold_cosine=tc,
                        threshold_knn=tk,
                        threshold_edge_predictor=te,
                        topk_cosine=kc,
                        topk_knn=kk,
                        topk_edge_predictor=ke,
                        edge_predictor_model=edge_predictor_model
                    )
                    
                    if result is not None:
                        successful_experiments += 1
                        self.all_results.append(result)
                        
                        # Update best result
                        if self.best_result is None or result['results']['val_acc'] > self.best_result['results']['val_acc']:  # select on val, never test
                            self.best_result = result
                            self.best_hyperparams = result['hyperparameters']
        
        # Report summary
        print(f"\n{'='*100}")
        print("Hyperparameter Tuning Complete!")
        print(f"{'='*100}")
        print(f"Total experiments: {total_experiments}")
        print(f"Successful experiments: {successful_experiments}")
        print(f"Failed experiments: {total_experiments - successful_experiments}")
        
        if self.best_result:
            self.reporter.report_title("Best Hyperparameters")
            self.reporter.report_txt(f"Experiment: {self.best_result['experiment_name']}")
            self.reporter.report_txt(f"\nHyperparameters:")
            for key, value in self.best_hyperparams.items():
                self.reporter.report_txt(f"  {key}: {value}")
            
            self.reporter.report_txt(f"\nResults:")
            self.reporter.report_txt(f"  Test Accuracy: {self.best_result['results']['test_acc']:.4f}")
            self.reporter.report_txt(f"  Test F1 (Macro): {self.best_result['results']['test_f1']:.4f}")
            self.reporter.report_txt(f"  Test F1 (Weighted): {self.best_result['results']['test_weight_f1']:.4f}")
            self.reporter.report_txt(f"  Val Accuracy: {self.best_result['results']['val_acc']:.4f}")
            
            improvement = self.best_result['results']['test_acc'] - baseline_results['test_acc']
            self.reporter.report_txt(f"\nImprovement over baseline:")
            self.reporter.report_txt(f"  Test Accuracy: {improvement:+.4f} ({improvement*100:+.2f}%)")
            
            print(f"\nBest Result:")
            print(f"  Experiment: {self.best_result['experiment_name']}")
            print(f"  Test Acc: {self.best_result['results']['test_acc']:.4f}")
            print(f"  Test F1: {self.best_result['results']['test_f1']:.4f}")
            print(f"  Improvement: {improvement:+.4f}")
        
        # Save all results to JSON
        self.save_results_json()
        
        return self.best_result, self.best_hyperparams
    
    def save_results_json(self):
        """Save all results to JSON file."""
        results_dir = './results/hyperparameter_tuning'
        os.makedirs(results_dir, exist_ok=True)
        json_path = os.path.join(results_dir, f"{self.dataset_name}_hyperparameter_tuning.json")
        
        output_data = {
            'dataset': self.dataset_name,
            'llm_name': self.llm_name,
            'peft_type': self.peft_type,
            'best_result': self.best_result,
            'best_hyperparameters': self.best_hyperparams,
            'all_results': self.all_results,
            'num_experiments': len(self.all_results),
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] All results saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for augmented dataset construction')
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        help='Name of the dataset (default: cora)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='Name of the LLM model (default: llama_3.2_1B)')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='Type of PEFT (default: lora)')
    parser.add_argument('--init_weight_approaches', type=str, nargs='+', 
                        default=['pissa'],
                        help='Init weight approaches to test (default: pissa)')
    parser.add_argument('--train_edge_predictor', action='store_true', default=True,
                        help='Train edge predictor (default: True)')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type (default: mlp)')
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type
    )
    
    # Define search space
    threshold_cosine_values = [0.90, 0.95, 0.97, 0.8]
    threshold_knn_values = [0.5]
    threshold_edge_predictor_values = [0.5, 0.7, 0.9]
    topk_cosine_values = [3, 5, 7, 10]
    topk_knn_values = [3]
    topk_edge_predictor_values = [5, 7, 10]
    
    # Run tuning
    best_result, best_hyperparams = tuner.tune(
        init_weight_approaches=args.init_weight_approaches,
        threshold_cosine_values=threshold_cosine_values,
        threshold_knn_values=threshold_knn_values,
        threshold_edge_predictor_values=threshold_edge_predictor_values,
        topk_cosine_values=topk_cosine_values,
        topk_knn_values=topk_knn_values,
        topk_edge_predictor_values=topk_edge_predictor_values,
        train_edge_predictor=args.train_edge_predictor,
        decoder_type=args.decoder_type
    )
    
    print("\n" + "="*100)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*100)