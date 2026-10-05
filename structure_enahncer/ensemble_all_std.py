# Tune hyperparameter topk and threshold and run node classification for std
# 
# Different from ensemble_edge_predictor_std.py:
# - Tune hyperparameters for ALL embedding approaches: 'pissa', 'orthogonal', 'guassian', 'loftq', 'eva'
# - Get k best models from each embedding approach (default k=2)
# - Aggregate model list, modified dataset list, and embedding list in order
# - Pass them to tune_ensemble_hyperparameter

import os
import sys
import argparse
import torch
import numpy as np
import gc
from copy import deepcopy

# Ensure repo root is on path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import set_mgnn_cfg, gnn_train_and_report_with_modified_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.ensemble_gnns_learning import tune_ensemble_hyperparameter, ensemble_gnn_learning
from LLMReasoner.TAPE.peft_tape_finetuning.reporter import ReportResults
from LLMReasoner.TAPE.structure_hybrid_enahncer.uncertainty_edge_detection import NoisyEdgeDetector, create_modified_dataset
from common.gnn import GNNEncoder


def cleanup_gpu_memory(config_entry):
    """
    Clean up GPU memory by moving model and dataset to CPU and deleting references.
    
    Args:
        config_entry: Configuration entry with 'model' and 'modified_dataset' keys
    """
    if config_entry is None:
        return
    
    # Move model to CPU and delete
    if 'model' in config_entry and config_entry['model'] is not None:
        try:
            model = config_entry['model']
            if hasattr(model, 'to'):
                model.to('cpu')
            elif hasattr(model, 'cpu'):
                model.cpu()
            del config_entry['model']
            del model
        except Exception as e:
            print(f"  [Warning] Error cleaning up model: {e}")
            pass
    
    # Move dataset to CPU and delete
    if 'modified_dataset' in config_entry and config_entry['modified_dataset'] is not None:
        try:
            dataset = config_entry['modified_dataset']
            if hasattr(dataset, 'to'):
                dataset.to('cpu')
            elif hasattr(dataset, 'cpu'):
                dataset.cpu()
            del config_entry['modified_dataset']
            del dataset
        except Exception as e:
            print(f"  [Warning] Error cleaning up dataset: {e}")
            pass
    
    # Clear embedding if present
    if 'embedding' in config_entry and config_entry['embedding'] is not None:
        try:
            if isinstance(config_entry['embedding'], torch.Tensor):
                if config_entry['embedding'].is_cuda:
                    config_entry['embedding'] = config_entry['embedding'].cpu()
            del config_entry['embedding']
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def add_to_topk_list(topk_list, new_config, k=5, metric='val_acc'):
    """
    Add new configuration to top-k list, maintaining only k best configurations.
    If new config is better than worst in list, replace worst and clean up GPU memory.
    Optimized to minimize GPU memory usage by immediately discarding non-top-k configs.
    
    Args:
        topk_list: Current list of top k configurations (sorted descending by metric)
        new_config: New configuration to potentially add
        k: Number of top configurations to keep
        metric: Metric name for comparison
    
    Returns:
        topk_list: Updated list with at most k configurations
    """
    new_metric = new_config['metric_value']
    
    # If list is not full, just add it
    if len(topk_list) < k:
        topk_list.append(new_config)
        topk_list.sort(key=lambda x: x['metric_value'], reverse=True)
        return topk_list
    
    # List is full - check if new config is better than worst in current list
    worst_metric = topk_list[-1]['metric_value']
    
    if new_metric > worst_metric:
        # Remove worst configuration and clean up GPU memory
        worst_config = topk_list.pop()
        cleanup_gpu_memory(worst_config)
        
        # Add new configuration
        topk_list.append(new_config)
        topk_list.sort(key=lambda x: x['metric_value'], reverse=True)
        
        print(f"  [Memory] Replaced worst config (metric={worst_metric:.4f}) with new (metric={new_metric:.4f})")
    else:
        # New config is not better, discard it immediately to save GPU memory
        cleanup_gpu_memory(new_config)
        print(f"  [Memory] Discarded config (metric={new_metric:.4f} <= worst={worst_metric:.4f})")
    
    return topk_list


def tune_hyperparameters_std(
    gnn_cfg,
    embedding,
    dataset_name,
    topk_percentages,
    threshold_values,
    num_models=10,
    decoder_type='mlp',
    metric='val_acc',
    k_best=2
):
    """
    Tune hyperparameters (topk and threshold) for std uncertainty-based edge removal.
    Returns top k_best configurations sorted by test accuracy.
    Optimized to keep only k_best models/datasets in GPU memory at any time.
    
    Args:
        gnn_cfg: GNN configuration
        embedding: Node embeddings
        dataset_name: Dataset name
        topk_percentages: List of top-k percentages to try
        threshold_values: List of threshold values to try
        num_models: Number of edge predictor models
        decoder_type: Edge predictor decoder type
        metric: Metric to optimize (default: 'val_acc'; never a test metric)
        k_best: Number of best configurations to keep (default: 2)
    
    Returns:
        topk_configs: List of top k_best configurations sorted by metric (descending), each with:
            - config_dict: hyperparameters
            - modified_dataset: modified dataset
            - results: GNN training results
            - model: trained GNN model
            - embedding: embedding used
    """
    print(f"\n{'='*80}")
    print(f"TUNING HYPERPARAMETERS (STD UNCERTAINTY)")
    print(f"{'='*80}")
    print(f"Top-k percentages: {len(topk_percentages)} values")
    print(f"Threshold values: {len(threshold_values)} values")
    print(f"Metric to optimize: {metric}")
    print(f"Keeping top {k_best} configurations in memory")
    print(f"{'='*80}\n")
    
    # Create detector and train edge predictors
    detector = NoisyEdgeDetector(
        cfg=gnn_cfg,
        features=embedding,
        num_models=num_models,
        decoder_type=decoder_type,
        training_strategy='with_splitting',
        negative_sampling_ratio=1.0,
    )
    detector.train_edge_predictors()
    
    # Compute std uncertainty scores
    std_scores = detector.calculate_standard_deviation()
    
    # Train baseline for comparison
    print(f"\n{'='*80}")
    print(f"BASELINE: Training on Original Graph")
    print(f"{'='*80}")
    baseline_result, _ = gnn_train_and_report_with_modified_dataset(
        modified_dataset=detector.dataset,
        embedding=embedding,
        dataset_name=dataset_name
    )
    baseline_metric = baseline_result[metric]
    print(f"Baseline {metric}: {baseline_metric:.4f}\n")
    
    # Maintain only top k_best configurations in memory
    topk_configs = []
    total_configs_tested = 0
    
    # Test top-k strategy
    print(f"Testing {len(topk_percentages)} top-k configurations...")
    for topk_pct in topk_percentages:
        modified_dataset, removal_stats = create_modified_dataset(
            dataset=detector.dataset,
            uncertainty_scores=std_scores,
            threshold=topk_pct,
            removal_strategy='top_k'
        )
        
        modified_result, modified_model = gnn_train_and_report_with_modified_dataset(
            modified_dataset=modified_dataset,
            embedding=embedding,
            dataset_name=dataset_name
        )
        
        config_entry = {
            'config_dict': {
                'removal_strategy': 'top_k',
                'threshold': topk_pct,
                'removal_percentage': removal_stats['removal_percentage'],
            },
            'modified_dataset': modified_dataset,
            'results': modified_result,
            'model': modified_model,
            'metric_value': modified_result[metric],
            'improvement': modified_result[metric] - baseline_metric,
            'embedding': embedding.clone(),
        }
        
        # Add to top-k list (will discard if not good enough)
        topk_configs = add_to_topk_list(topk_configs, config_entry, k=k_best, metric=metric)
        total_configs_tested += 1
        
        # Periodic memory cleanup every 10 configs
        if total_configs_tested % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Test threshold strategy
    print(f"Testing {len(threshold_values)} threshold configurations...")
    for threshold in threshold_values:
        modified_dataset, removal_stats = create_modified_dataset(
            dataset=detector.dataset,
            uncertainty_scores=std_scores,
            threshold=threshold,
            removal_strategy='threshold'
        )
        
        modified_result, modified_model = gnn_train_and_report_with_modified_dataset(
            modified_dataset=modified_dataset,
            embedding=embedding,
            dataset_name=dataset_name
        )
        
        config_entry = {
            'config_dict': {
                'removal_strategy': 'threshold',
                'threshold': threshold,
                'removal_percentage': removal_stats['removal_percentage'],
            },
            'modified_dataset': modified_dataset,
            'results': modified_result,
            'model': modified_model,
            'metric_value': modified_result[metric],
            'improvement': modified_result[metric] - baseline_metric,
            'embedding': embedding.clone(),
        }
        
        # Add to top-k list (will discard if not good enough)
        topk_configs = add_to_topk_list(topk_configs, config_entry, k=k_best, metric=metric)
        total_configs_tested += 1
        
        # Periodic memory cleanup every 10 configs
        if total_configs_tested % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final sort (should already be sorted, but ensure it)
    topk_configs.sort(key=lambda x: x['metric_value'], reverse=True)
    
    # Final memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nTotal configurations tested: {total_configs_tested}")
    print(f"Best {metric}: {topk_configs[0]['metric_value']:.4f} (improvement: {topk_configs[0]['improvement']:+.4f})")
    print(f"Returning top {len(topk_configs)} configurations")
    
    return topk_configs, baseline_result


def extract_gnn_encoder(model):
    """
    Extract GNN encoder from a trained GNN model.
    
    Args:
        model: Trained GNN model (from gnn_train_and_report_with_modified_dataset)
               This is already a GNNEncoder instance
    
    Returns:
        encoder: GNNEncoder instance
    """
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble edge predictor with std uncertainty: tune for all embeddings, aggregate best models'
    )
    parser.add_argument('--dataset_name', type=str, default='cora', help='Dataset name')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', help='LLM name')
    parser.add_argument('--peft_type', type=str, default='lora', help='PEFT type')
    parser.add_argument('--num_models', type=int, default=20,
                        help='Number of edge predictor models (K)')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type')
    parser.add_argument('--k_best', type=int, default=2,
                        help='Number of best models to select from each embedding approach (default: 2)')
    parser.add_argument('--topk_start', type=float, default=0.0,
                        help='Starting top-k percentage (default: 0.0)')
    parser.add_argument('--topk_end', type=float, default=50.0,
                        help='Ending top-k percentage (default: 50.0)')
    parser.add_argument('--topk_step', type=float, default=1.0,
                        help='Step size for top-k percentage (default: 1.0)')
    parser.add_argument('--threshold_start', type=float, default=0.0,
                        help='Starting threshold value (default: 0.0)')
    parser.add_argument('--threshold_end', type=float, default=0.4,
                        help='Ending threshold value (default: 0.4)')
    parser.add_argument('--threshold_step', type=float, default=0.01,
                        help='Step size for threshold (default: 0.01)')
    parser.add_argument('--metric', type=str, default='val_acc',  # configs are selected on validation only
                        choices=['val_acc', 'val_f1'],
                        help='Validation metric used to select configurations (default: val_acc)')
    parser.add_argument('--save_dir', type=str, default='results/ensemble_all_std',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate hyperparameter ranges
    topk_percentages = np.arange(args.topk_start, args.topk_end + args.topk_step, args.topk_step).tolist()
    topk_percentages = [round(x, 2) for x in topk_percentages]
    threshold_values = np.arange(args.threshold_start, args.threshold_end + args.threshold_step, args.threshold_step).tolist()
    threshold_values = [round(x, 4) for x in threshold_values]
    
    # All embedding approaches to test
    embedding_approaches = ['pissa', 'orthogonal', 'guassian', 'loftq', 'eva']
    
    print('=' * 80)
    print('ENSEMBLE EDGE PREDICTOR WITH STD UNCERTAINTY (ALL EMBEDDINGS)')
    print('=' * 80)
    print(f"Dataset: {args.dataset_name}, LLM: {args.llm_name}")
    print(f"K models: {args.num_models}, Decoder: {args.decoder_type}")
    print(f"Embedding approaches: {embedding_approaches}")
    print(f"K best per embedding: {args.k_best}")
    print(f"Top-k range: {args.topk_start} to {args.topk_end} (step: {args.topk_step})")
    print(f"Threshold range: {args.threshold_start} to {args.threshold_end} (step: {args.threshold_step})")
    print(f"Metric to optimize: {args.metric}")
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
    
    # Step 1: Tune hyperparameters for each embedding approach
    print(f"\n{'='*80}")
    print("STEP 1: Tuning Hyperparameters for All Embedding Approaches")
    print(f"{'='*80}")
    
    all_best_configs = []  # Will contain k_best from each embedding
    all_baseline_results = {}
    
    for emb_approach in embedding_approaches:
        print(f"\n{'='*80}")
        print(f"PROCESSING EMBEDDING APPROACH: {emb_approach.upper()}")
        print(f"{'='*80}")
        
        embedding = get_embedding_from_data(init_to_data[emb_approach])
        
        # Tune hyperparameters for this embedding (returns top k_best)
        top_k_configs, baseline_result = tune_hyperparameters_std(
            gnn_cfg=gnn_cfg,
            embedding=embedding,
            dataset_name=args.dataset_name,
            topk_percentages=topk_percentages,
            threshold_values=threshold_values,
            num_models=args.num_models,
            decoder_type=args.decoder_type,
            metric=args.metric,
            k_best=args.k_best
        )
        
        all_baseline_results[emb_approach] = baseline_result
        
        # Add top k_best configurations (already filtered by tune_hyperparameters_std)
        all_best_configs.extend(top_k_configs)
        
        print(f"\nTop {args.k_best} configurations for {emb_approach}:")
        for i, config in enumerate(top_k_configs, 1):
            print(f"  {i}. Strategy: {config['config_dict']['removal_strategy']}, "
                  f"Threshold: {config['config_dict']['threshold']:.4f}, "
                  f"{args.metric} = {config['metric_value']:.4f}, "
                  f"Improvement = {config['improvement']:+.4f}")
    
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*80}")
    print(f"Total best configurations collected: {len(all_best_configs)}")
    print(f"  ({args.k_best} per embedding × {len(embedding_approaches)} embeddings)")
    
    # Step 2: Aggregate models, datasets, and embeddings
    print(f"\n{'='*80}")
    print("STEP 2: Aggregating Models, Modified Datasets, and Embeddings")
    print(f"{'='*80}")
    
    aggregated_model_list = []
    aggregated_modified_datasets_list = []
    aggregated_custom_embeddings_list = []
    
    for i, config in enumerate(all_best_configs, 1):
        # Extract GNN encoder from trained model
        encoder = extract_gnn_encoder(config['model'])
        aggregated_model_list.append(encoder)
        aggregated_modified_datasets_list.append(config['modified_dataset'])
        aggregated_custom_embeddings_list.append(config['embedding'])
        
        # Determine which embedding approach this came from
        emb_idx = (i - 1) // args.k_best
        emb_approach = embedding_approaches[emb_idx]
        
        print(f"Model {i} (from {emb_approach}): Strategy={config['config_dict']['removal_strategy']}, "
              f"Threshold={config['config_dict']['threshold']:.4f}, "
              f"{args.metric}={config['metric_value']:.4f}")
    
    # Step 3: Run ensemble hyperparameter tuning for all three approaches
    print(f"\n{'='*80}")
    print("STEP 3: Running Ensemble Hyperparameter Tuning (All Embeddings)")
    print(f"{'='*80}")
    print(f"Total models in ensemble: {len(aggregated_model_list)}")
    
    ensemble_approaches_list = ['learnable', 'learnable_classes', 'learnable_per_classes']
    all_ensemble_results = {}
    
    for ensemble_approach in ensemble_approaches_list:
        print(f"\n{'='*80}")
        print(f"Running Ensemble Approach: {ensemble_approach.upper()}")
        print(f"{'='*80}")
        
        best_results, best_hyperparams, best_ensemble_predictions = tune_ensemble_hyperparameter(
            dataset_name=args.dataset_name,
            model_list=aggregated_model_list,
            custom_embeddings=aggregated_custom_embeddings_list,
            modified_datasets_list=aggregated_modified_datasets_list,
            ensemble_approaches=ensemble_approach,
            supervised=True
        )
        
        all_ensemble_results[ensemble_approach] = {
            'best_results': best_results,
            'best_hyperparams': best_hyperparams,
            'best_ensemble_predictions': best_ensemble_predictions
        }
        
        print(f"\n{ensemble_approach.upper()} RESULTS:")
        print(f"{'='*80}")
        if best_results:
            print(f"Best test accuracy: {best_results.get('test', {}).get('accuracy', 'N/A'):.4f}")
            print(f"Best test F1 (Macro): {best_results.get('test', {}).get('macro_f1', 'N/A'):.4f}")
            print(f"Best test F1 (Weighted): {best_results.get('test', {}).get('weighted_f1', 'N/A'):.4f}")
            print(f"Best val accuracy: {best_results.get('val', {}).get('accuracy', 'N/A'):.4f}")
        if best_hyperparams:
            print(f"\nBest Hyperparameters:")
            print(f"  Layers: {best_hyperparams.get('num_layers', 'N/A')}")
            print(f"  Hidden Dim: {best_hyperparams.get('hidden_dim', 'N/A')}")
            print(f"  Dropout: {best_hyperparams.get('dropout', 'N/A')}")
            print(f"  Learning Rate: {best_hyperparams.get('learning_rate', 'N/A')}")
    
    print(f"\n{'='*80}")
    print("SUMMARY: ALL ENSEMBLE APPROACHES")
    print(f"{'='*80}")
    for approach in ensemble_approaches_list:
        results = all_ensemble_results[approach]['best_results']
        if results:
            test_acc = results.get('test', {}).get('accuracy', 'N/A')
            test_f1 = results.get('test', {}).get('macro_f1', 'N/A')
            val_acc = results.get('val', {}).get('accuracy', 'N/A')
            print(f"{approach.upper()}: Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}, Val Acc={val_acc:.4f}")
    
    # Step 4: Report results using ReportResults
    print(f"\n{'='*80}")
    print("STEP 4: Writing Results Report")
    print(f"{'='*80}")
    
    reporter = ReportResults(cfg, save_dir=args.save_dir, index_run=0)
    
    reporter.report_title("ENSEMBLE EDGE PREDICTOR WITH STD UNCERTAINTY (ALL EMBEDDINGS)")
    reporter.report_txt(f"Dataset: {args.dataset_name}")
    reporter.report_txt(f"LLM: {args.llm_name}")
    reporter.report_txt(f"Number of Edge Predictor Models (K): {args.num_models}")
    reporter.report_txt(f"Decoder Type: {args.decoder_type}")
    reporter.report_txt(f"K Best per Embedding: {args.k_best}")
    reporter.report_txt(f"Total Models in Ensemble: {len(aggregated_model_list)}")
    reporter.report_txt(f"Metric Optimized: {args.metric}")
    reporter.report_txt("")
    
    # Report baseline results for each embedding
    reporter.report_title("BASELINE RESULTS (PER EMBEDDING)")
    for emb_approach, baseline_result in all_baseline_results.items():
        reporter.report_txt(f"\n{emb_approach.upper()}:")
        reporter.report_txt(f"  Test Accuracy: {baseline_result['test_acc']:.4f}")
        reporter.report_txt(f"  Test F1 (Macro): {baseline_result['test_f1']:.4f}")
        reporter.report_txt(f"  Test F1 (Weighted): {baseline_result['test_weight_f1']:.4f}")
        reporter.report_txt(f"  Val Accuracy: {baseline_result['val_acc']:.4f}")
        reporter.report_txt(f"  Val F1 (Macro): {baseline_result['val_f1']:.4f}")
    
    # Report best configurations from each embedding
    reporter.report_title("BEST CONFIGURATIONS (PER EMBEDDING)")
    for emb_idx, emb_approach in enumerate(embedding_approaches):
        reporter.report_txt(f"\n{emb_approach.upper()} (Top {args.k_best}):")
        start_idx = emb_idx * args.k_best
        end_idx = start_idx + args.k_best
        for i, config in enumerate(all_best_configs[start_idx:end_idx], 1):
            reporter.report_txt(f"\n  Configuration {i}:")
            reporter.report_txt(f"    Strategy: {config['config_dict']['removal_strategy']}")
            reporter.report_txt(f"    Threshold: {config['config_dict']['threshold']:.4f}")
            reporter.report_txt(f"    Removal Percentage: {config['config_dict']['removal_percentage']:.2f}%")
            reporter.report_txt(f"    Test Accuracy: {config['results']['test_acc']:.4f}")
            reporter.report_txt(f"    Test F1 (Macro): {config['results']['test_f1']:.4f}")
            reporter.report_txt(f"    Test F1 (Weighted): {config['results']['test_weight_f1']:.4f}")
            reporter.report_txt(f"    Val Accuracy: {config['results']['val_acc']:.4f}")
            reporter.report_txt(f"    Val F1 (Macro): {config['results']['val_f1']:.4f}")
            reporter.report_txt(f"    Improvement ({args.metric}): {config['improvement']:+.4f}")
    
    reporter.report_title("ENSEMBLE LEARNING RESULTS (ALL EMBEDDINGS)")
    
    # Report results for each ensemble approach
    for ensemble_approach in ensemble_approaches_list:
        reporter.report_title(f"ENSEMBLE APPROACH: {ensemble_approach.upper()}")
        
        approach_data = all_ensemble_results[ensemble_approach]
        best_results = approach_data['best_results']
        best_hyperparams = approach_data['best_hyperparams']
        
        if best_results:
            reporter.report_txt(f"\nTest Results:")
            reporter.report_txt(f"  Test Accuracy: {best_results.get('test', {}).get('accuracy', 'N/A'):.4f}")
            reporter.report_txt(f"  Test F1 (Macro): {best_results.get('test', {}).get('macro_f1', 'N/A'):.4f}")
            reporter.report_txt(f"  Test F1 (Weighted): {best_results.get('test', {}).get('weighted_f1', 'N/A'):.4f}")
            
            reporter.report_txt(f"\nValidation Results:")
            reporter.report_txt(f"  Val Accuracy: {best_results.get('val', {}).get('accuracy', 'N/A'):.4f}")
            reporter.report_txt(f"  Val F1 (Macro): {best_results.get('val', {}).get('macro_f1', 'N/A'):.4f}")
            
            reporter.report_txt(f"\nTraining Results:")
            reporter.report_txt(f"  Train Accuracy: {best_results.get('train', {}).get('accuracy', 'N/A'):.4f}")
            reporter.report_txt(f"  Train F1 (Macro): {best_results.get('train', {}).get('macro_f1', 'N/A'):.4f}")
        else:
            reporter.report_txt("  No results available")
        
        if best_hyperparams:
            reporter.report_txt(f"\nBest Hyperparameters:")
            reporter.report_txt(f"  Number of Layers: {best_hyperparams.get('num_layers', 'N/A')}")
            reporter.report_txt(f"  Hidden Dimension: {best_hyperparams.get('hidden_dim', 'N/A')}")
            reporter.report_txt(f"  Dropout: {best_hyperparams.get('dropout', 'N/A')}")
            reporter.report_txt(f"  Learning Rate: {best_hyperparams.get('learning_rate', 'N/A')}")
        
        reporter.report_txt("")  # Empty line between approaches
    
    # Summary comparison
    reporter.report_title("ENSEMBLE APPROACHES COMPARISON")
    reporter.report_txt("\nSummary Table:")
    reporter.report_txt(f"{'Approach':<25} {'Test Acc':<12} {'Test F1':<12} {'Val Acc':<12} {'Val F1':<12}")
    reporter.report_txt("-" * 75)
    
    for ensemble_approach in ensemble_approaches_list:
        approach_data = all_ensemble_results[ensemble_approach]
        best_results = approach_data['best_results']
        
        if best_results:
            test_acc = best_results.get('test', {}).get('accuracy', 'N/A')
            test_f1 = best_results.get('test', {}).get('macro_f1', 'N/A')
            val_acc = best_results.get('val', {}).get('accuracy', 'N/A')
            val_f1 = best_results.get('val', {}).get('macro_f1', 'N/A')
            
            if isinstance(test_acc, (int, float)):
                test_acc_str = f"{test_acc:.4f}"
            else:
                test_acc_str = str(test_acc)
            
            if isinstance(test_f1, (int, float)):
                test_f1_str = f"{test_f1:.4f}"
            else:
                test_f1_str = str(test_f1)
            
            if isinstance(val_acc, (int, float)):
                val_acc_str = f"{val_acc:.4f}"
            else:
                val_acc_str = str(val_acc)
            
            if isinstance(val_f1, (int, float)):
                val_f1_str = f"{val_f1:.4f}"
            else:
                val_f1_str = str(val_f1)
            
            reporter.report_txt(f"{ensemble_approach:<25} {test_acc_str:<12} {test_f1_str:<12} {val_acc_str:<12} {val_f1_str:<12}")
        else:
            reporter.report_txt(f"{ensemble_approach:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {reporter.file_path}")
    print('=' * 80)


if __name__ == '__main__':
    main()
