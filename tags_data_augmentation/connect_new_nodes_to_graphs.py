#



from gnns.gnn_mtrainer import *
from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter
from report.reporter import ReportResults
from tags_data_augmentation.edge_predictor import edge_predictor_train_and_report
import json
import os
import torch
from tags_data_augmentation.agumented_dataset import *

def load_augmented_cache(dataset_name, model_name='llama_3.2_1B', pooling='mean', init_weight_approach='pissa'):
    """
    Load cached augmented embeddings and labels.
    Tries both 'guassian' and 'gaussian' spellings when init_weight_approach is one of them.

    Args:
        dataset_name: Name of the dataset (e.g., 'cora')
        model_name: Name of the LLM model
        pooling: Pooling method used ('mean' or 'last')
        init_weight_approach: Initialization weight approach ('pissa', 'orthogonal', 'guassian', 'loftq', 'eva')

    Returns:
        cache_dict: Dictionary containing embeddings, labels, etc.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Try both: next to this package, and project root (where extract script may save when run from root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    cache_roots_to_try = [
        os.path.join(current_dir, "artifacts", "augmented_cache"),
        os.path.join(project_root, "artifacts", "augmented_cache"),
    ]
    os.makedirs(cache_roots_to_try[0], exist_ok=True)

    # Try requested spelling first, then alternate spelling for guassian/gaussian
    init_variants = [init_weight_approach]
    if init_weight_approach == "guassian":
        init_variants.append("gaussian")
    elif init_weight_approach == "gaussian":
        init_variants.append("guassian")

    for cache_root in cache_roots_to_try:
        for init_var in init_variants:
            fname = f"{model_name}_{dataset_name}_augmented_lora_init-{init_var}_pool-{pooling}.pt"
            cache_path = os.path.join(cache_root, fname)
            if os.path.exists(cache_path):
                cache_dict = torch.load(cache_path, weights_only=False)
                print(f"[OK] Loaded augmented cache from: {cache_path}")
                print(f"     - Embeddings shape: {cache_dict['embeddings'].shape}")
                print(f"     - Labels: {len(cache_dict['labels'])} samples")
                print(f"Initial weight approach: {init_weight_approach}")
                return cache_dict

    # No file found; raise with a clear message and suggested command
    raise FileNotFoundError(
        f"Augmented cache not found. Looked in:\n  - {cache_roots_to_try[0]}\n  - {cache_roots_to_try[1]}\n"
        f"Expected filename: {model_name}_{dataset_name}_augmented_lora_init-<init>_pool-{pooling}.pt (tried '{init_weight_approach}' and 'gaussian'/'guassian').\n"
        f"Generate the cache by running extract_and_cache_new_nodes_embedding.py with dataset_name='{dataset_name}', "
        f"llm_name='{model_name}', and init_weights_approaches_list including '{init_weight_approach}' (or 'gaussian'), then run it."
    )




def train_node_classification_gnn_with_modified_data(augmented_dataset, augmented_embeddings, 
                                                      cfg, title="Augmented"):
    """
    Train GNN on augmented dataset.
    
    Args:
        augmented_dataset: Dataset with augmented nodes
        augmented_embeddings: Embeddings for all nodes (original + augmented)
        cfg: Configuration object
        title: Title for reporting
    
    Returns:
        results: Dictionary with training results
        model: Trained GNN model
    """
    print(f"\n{'='*80}")
    print(f"Training GNN on Augmented Dataset - {title}")
    print(f"{'='*80}\n")
    
    # Use gnn_train_and_report_with_modified_dataset
    results, model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=augmented_dataset,
        embedding=augmented_embeddings,
        dataset_name=cfg.dataset.name,
        supervised=False,
    )
    
    return results, model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Connect new nodes to graph and train GNN')
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        help='Name of the dataset (default: cora)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', 
                        help='Name of the LLM model (default: llama_3.2_1B)')
    parser.add_argument('--peft_type', type=str, default='lora', 
                        help='Type of PEFT (default: lora)')
    parser.add_argument('--pooling', type=str, default='mean', 
                        help='Pooling method (default: mean)')
    parser.add_argument('--used_based_model', action='store_true', 
                        help='Use base model instead of LoRA adapter')
    parser.add_argument('--train_edge_predictor', action='store_true', 
                        help='Train edge predictor for analysis (Note: similarity-based approach is used for connecting new nodes)', default=True)
    parser.add_argument('--decoder_type', type=str, default='mlp', 
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type (default: mlp)')
    parser.add_argument('--split_edges', action='store_true', 
                        help='Split edges for edge predictor training (default: use all edges)')
    parser.add_argument('--edge_threshold', type=float, default=0.50,
                        help='Edge prediction threshold (default: 0.5)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Maximum edges per new node (default: 10)')
    parser.add_argument('--init_weight_approach', type=str, default='all',
                        choices=['pissa', 'orthogonal', 'guassian', 'loftq', 'eva', 'all'],
                        help='Initialization weight approach (default: all - processes all embedding types)')
    # Hybrid edge prediction arguments
    parser.add_argument('--used_cosine_similarity', action='store_true', default=True,
                        help='Use cosine similarity for edge prediction (default: True)')
    parser.add_argument('--no_cosine_similarity', dest='used_cosine_similarity', action='store_false',
                        help='Disable cosine similarity for edge prediction')
    parser.add_argument('--used_k_nearest_neighbors', action='store_true', default=False,
                        help='Use k-nearest neighbors for edge prediction (default: True)')
    parser.add_argument('--no_k_nearest_neighbors', dest='used_k_nearest_neighbors', action='store_false',
                        help='Disable k-nearest neighbors for edge prediction')
    parser.add_argument('--used_edge_predictor', action='store_true', default=True,
                        help='Use edge predictor for edge prediction (default: False)')
    parser.add_argument('--threshold_cosine_similarity', type=float, default=0.9,
                        help='Threshold for cosine similarity (default: 0.5)')
    parser.add_argument('--threshold_k_nearest_neighbors', type=float, default=0.5,
                        help='Threshold (max distance) for k-nearest neighbors (default: 0.5)')
    parser.add_argument('--threshold_edge_predictor', type=float, default=0.7,
                        help='Threshold for edge predictor (default: 0.5)')
    parser.add_argument('--top_k_cosine_similarity', type=int, default=10,
                        help='Top-K for cosine similarity (default: 10)')
    parser.add_argument('--top_k_knearest_neighbors', type=int, default=3,
                        help='Top-K for k-nearest neighbors (default: 10)')
    parser.add_argument('--top_k_edge_predictor', type=int, default=10,
                        help='Top-K for edge predictor (default: 10)')
    args = parser.parse_args()
    
    # Determine which embedding types to process
    all_embedding_types = ['pissa', 'guassian', 'loftq', 'eva', 'orthogonal']
    if args.init_weight_approach is None or args.init_weight_approach == 'all':
        embedding_types_to_process = all_embedding_types
    else:
        embedding_types_to_process = [args.init_weight_approach]
    
    # Setup configuration
    cfg = setup_finetuning_cfg(args.dataset_name, args.llm_name, args.peft_type)
    
    # Reporter for all results (single models + ensemble)
    results_dir = "./results/augmented_training"
    os.makedirs(results_dir, exist_ok=True)
    reporter = ReportResults(cfg, save_dir=results_dir, index_run=0)
    reporter.report_title(f"Augmented Training Results - {args.dataset_name.upper()}")
    reporter.report_txt(f"Dataset: {args.dataset_name} | LLM: {args.llm_name} | PEFT: {args.peft_type} | Pooling: {args.pooling}")
    reporter.report_txt(f"Edge creation: cosine={args.used_cosine_similarity}, knn={args.used_k_nearest_neighbors}, edge_predictor={args.used_edge_predictor}")
    reporter.report_txt("")
    
    # Load initial embeddings for GNN encoder (needed for baseline and edge predictor per approach)
    from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg, supervised=False)
    
    # Store results for all embedding types
    all_results = {}
    # Collect modified models, embeddings, and datasets for ensemble learning
    modified_gnn_models_list = []
    modified_embeddings_list = []
    modified_datasets_list = []

    # Process each embedding type (baseline and augmented each use the same approach's embedding)
    for init_weight_approach in embedding_types_to_process:
        print("\n" + "="*80)
        print(f"PROCESSING EMBEDDING TYPE: {init_weight_approach.upper()}")
        print("="*80)
        
        # Get embedding for this approach (used for baseline, edge predictor, and comparison)
        embedding = None
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
        
        # Step 1: Train baseline GNN on original data with this approach's embedding
        print(f"\nTraining Baseline GNN (Original Dataset - {init_weight_approach} embeddings)")
        try:
            baseline_results, baseline_model = gnn_train_and_report(
                dataset_name=args.dataset_name,
                embedding=embedding,
                title=f"Baseline ({init_weight_approach})",
                does_print_training_process=False,
                supervised=False,
            )
        except Exception as e:
            print(f"[ERROR] Baseline training failed for {init_weight_approach}: {e}")
            continue
        
        # Step 2: Train edge predictor (optional) for this embedding type
        edge_predictor_model = None
        if args.train_edge_predictor:
            print(f"\nTraining Edge Predictor for {init_weight_approach}")
            
            # Train edge predictor
            training_strategy = 'without_splitting' if not args.split_edges else 'with_splitting'
            
            print(f"  Decoder type: {args.decoder_type}")
            print(f"  Training strategy: {training_strategy}")
            
            try:
                edge_results, edge_predictor_model = edge_predictor_train_and_report(
                    dataset_name=args.dataset_name,
                    features=embedding,
                    decoder_approach_type=args.decoder_type,
                    training_strategy=training_strategy,
                    negative_sampling_ratio=1.0,
                    title=f"Edge Predictor ({init_weight_approach})",
                    does_print_training_process=False
                )
                print(f"  Edge Predictor - Val Acc: {edge_results['val_acc']:.4f}, Test Acc: {edge_results['test_acc']:.4f}")
            except Exception as e:
                print(f"  [WARNING] Edge predictor training failed: {e}")
                edge_predictor_model = None
        
        # Step 3: Load augmented cache for this embedding type
        print(f"\nLoading Augmented Cache for {init_weight_approach}")
        try:
            augmented_cache = load_augmented_cache(
                dataset_name=args.dataset_name,
                model_name=args.llm_name,
                pooling=args.pooling,
                init_weight_approach=init_weight_approach,
            )
        except FileNotFoundError as e:
            print(f"[ERROR] Could not load augmented cache for {init_weight_approach}: {e}")
            print(f"Skipping {init_weight_approach}...")
            continue
        
        # Step 4: Create augmented dataset
        print(f"\nCreating Augmented Dataset for {init_weight_approach}")
        try:
            augmented_dataset, augmented_embeddings = create_augmented_dataset(
                cfg=cfg,
                augmented_cache=augmented_cache,
                threshold=args.edge_threshold,
                top_k=args.top_k,
                init_weight_approach=init_weight_approach,
                used_cosine_similarity=args.used_cosine_similarity,
                used_k_nearest_neighbors=args.used_k_nearest_neighbors,
                used_edge_predictor=args.used_edge_predictor,
                k_cosine_similarity=5,
                k_knearest_neighbors=5,
                k_edge_predictor=5,
                threshold_cosine_similarity=args.threshold_cosine_similarity,
                threshold_k_nearest_neighbors=args.threshold_k_nearest_neighbors,
                threshold_edge_predictor=args.threshold_edge_predictor,
                top_k_cosine_similarity=args.top_k_cosine_similarity,
                top_k_knearest_neighbors=args.top_k_knearest_neighbors,
                top_k_edge_predictor=args.top_k_edge_predictor,
                edge_predictor_model=edge_predictor_model,
                supervised=False
            )
        except Exception as e:
            print(f"[ERROR] Failed to create augmented dataset for {init_weight_approach}: {e}")
            print(f"Skipping {init_weight_approach}...")
            continue
        
        # Step 5: Train GNN on augmented dataset
        print(f"\nTraining GNN on Augmented Dataset ({init_weight_approach})")
        try:
            augmented_dataset.x = augmented_embeddings
            augmented_results, augmented_model = train_node_classification_gnn_with_modified_data(
                augmented_dataset=augmented_dataset,
                augmented_embeddings=augmented_embeddings,
                cfg=cfg,
                title=f"Augmented ({init_weight_approach})"
            )
            
            # Calculate improvement
            acc_improvement = augmented_results['test_acc'] - baseline_results['test_acc']
            f1_improvement = augmented_results['test_f1'] - baseline_results['test_f1']
            
            # Store results
            all_results[init_weight_approach] = {
                'baseline': baseline_results,
                'augmented': augmented_results,
                'improvement': {
                    'test_acc': acc_improvement,
                    'test_f1': f1_improvement,
                    'test_acc_pct': acc_improvement * 100,
                    'test_f1_pct': f1_improvement * 100
                },
                'augmentation_config': {
                    'num_new_nodes': augmented_cache['embeddings'].shape[0],
                    'init_weight_approach': init_weight_approach,
                    'pooling': args.pooling,
                    'edge_creation_method': {
                        'cosine_similarity': args.used_cosine_similarity,
                        'k_nearest_neighbors': args.used_k_nearest_neighbors,
                        'edge_predictor': args.used_edge_predictor
                    },
                    'thresholds': {
                        'cosine_similarity': args.threshold_cosine_similarity,
                        'k_nearest_neighbors': args.threshold_k_nearest_neighbors,
                        'edge_predictor': args.threshold_edge_predictor
                    },
                    'top_k': {
                        'cosine_similarity': args.top_k_cosine_similarity,
                        'k_nearest_neighbors': args.top_k_knearest_neighbors,
                        'edge_predictor': args.top_k_edge_predictor
                    }
                }
            }
            
            # Collect for ensemble: modified model, modified embeddings, modified dataset
            modified_gnn_models_list.append(augmented_model)
            modified_embeddings_list.append(augmented_embeddings)
            modified_datasets_list.append(augmented_dataset)

            # Report single-model result (baseline vs augmented, improvement)
            reporter.report_title(f"Model: {init_weight_approach.upper()}")
            reporter.report_txt(f"  Baseline (original) - Test Acc: {baseline_results['test_acc']:.4f}, Test F1: {baseline_results['test_f1']:.4f}, Val Acc: {baseline_results['val_acc']:.4f}")
            reporter.report_txt(f"  Augmented           - Test Acc: {augmented_results['test_acc']:.4f}, Test F1: {augmented_results['test_f1']:.4f}, Val Acc: {augmented_results['val_acc']:.4f}")
            reporter.report_txt(f"  Improvement         - Test Acc: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%), Test F1: {f1_improvement:+.4f}")
            reporter.report_txt(f"  Augmented nodes: {augmented_cache['embeddings'].shape[0]}")
            reporter.report_txt("")

            print(f"[OK] {init_weight_approach} - Test Acc: {augmented_results['test_acc']:.4f}, Improvement: {acc_improvement:+.4f}")
            
        except Exception as e:
            print(f"[ERROR] Training failed for {init_weight_approach}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 6: Report all results (each embedding type: baseline = same approach, augmented = same approach)
    reporter.report_title("FINAL RESULTS COMPARISON - ALL EMBEDDING TYPES")
    reporter.report_txt("(Baseline and augmented both use the same embedding approach per row)")
    reporter.report_txt("")
    
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON - ALL EMBEDDING TYPES")
    print("(Baseline and augmented both use the same embedding approach per row)")
    print("="*80)
    
    # Sort by validation accuracy (the "best single model" must be chosen without test labels)
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['augmented']['val_acc'], 
                           reverse=True)
    
    reporter.report_txt("RESULTS BY EMBEDDING TYPE (Baseline vs Augmented):")
    reporter.report_txt("-"*70)
    
    print("\n" + "-"*80)
    print("RESULTS BY EMBEDDING TYPE (Baseline vs Augmented):")
    print("-"*80)
    
    for init_weight_approach, results in sorted_results:
        baseline_results = results['baseline']
        aug_results = results['augmented']
        improvement = results['improvement']
        
        line1 = f"{init_weight_approach.upper()}: Baseline Test Acc: {baseline_results['test_acc']:.4f}, F1: {baseline_results['test_f1']:.4f}"
        line2 = f"  Augmented Test Acc: {aug_results['test_acc']:.4f}, F1: {aug_results['test_f1']:.4f}"
        line3 = f"  Improvement: Test Acc {improvement['test_acc']:+.4f} ({improvement['test_acc_pct']:+.2f}%), Test F1 {improvement['test_f1']:+.4f}"
        reporter.report_txt(line1)
        reporter.report_txt(line2)
        reporter.report_txt(line3)
        reporter.report_txt("")
        
        print(f"\n{init_weight_approach.upper()}:")
        print(f"  Baseline (original) - Test Acc: {baseline_results['test_acc']:.4f}, Test F1: {baseline_results['test_f1']:.4f}")
        print(f"  Augmented           - Test Acc: {aug_results['test_acc']:.4f}, Test F1: {aug_results['test_f1']:.4f}")
        print(f"  Improvement: Test Acc {improvement['test_acc']:+.4f} ({improvement['test_acc_pct']:+.2f}%), Test F1 {improvement['test_f1']:+.4f}")
        print(f"  Val Accuracy: baseline {baseline_results['val_acc']:.4f} -> augmented {aug_results['val_acc']:.4f}")
        print(f"  Augmented Nodes: {results['augmentation_config']['num_new_nodes']}")
    
    # Summary table (each row = one approach: baseline acc, augmented acc, improvement)
    reporter.report_title("SUMMARY TABLE (Single Models)")
    reporter.report_txt(f"{'Embedding Type':<16} {'Baseline Acc':<14} {'Augmented Acc':<14} {'Improvement':<18} {'Baseline F1':<14} {'Augmented F1':<14} {'F1 Impr':<10}")
    reporter.report_txt("-"*100)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Embedding Type':<16} {'Baseline Acc':<14} {'Augmented Acc':<14} {'Improvement':<18} {'Baseline F1':<14} {'Augmented F1':<14} {'F1 Impr':<10}")
    print("-"*100)
    for init_weight_approach, results in sorted_results:
        baseline_results = results['baseline']
        aug_results = results['augmented']
        improvement = results['improvement']
        improvement_str = f"{improvement['test_acc']:+.4f} ({improvement['test_acc_pct']:+.2f}%)"
        f1_improvement_str = f"{improvement['test_f1']:+.4f}"
        row = f"{init_weight_approach.capitalize():<16} {baseline_results['test_acc']:<14.4f} {aug_results['test_acc']:<14.4f} {improvement_str:<18} {baseline_results['test_f1']:<14.4f} {aug_results['test_f1']:<14.4f} {f1_improvement_str:<10}"
        reporter.report_txt(row)
        print(row)
    
    if sorted_results:
        best_type, best_results = sorted_results[0]
        reporter.report_txt("")
        reporter.report_txt(f"Best single model: {best_type} | Test Acc: {best_results['augmented']['test_acc']:.4f} | Improvement: {best_results['improvement']['test_acc']:+.4f} ({best_results['improvement']['test_acc_pct']:+.2f}%)")
        print(f"\n[INFO] Best performing embedding type: {best_type}")
        print(f"        Test Accuracy: {best_results['augmented']['test_acc']:.4f}")
        print(f"        Improvement: {best_results['improvement']['test_acc']:+.4f} ({best_results['improvement']['test_acc_pct']:+.2f}%)")

    # Step 7: Run ensemble learning for three approaches: learnable, learnable_classes, learnable_per_classes
    ensemble_approach_names = ['learnable', 'learnable_classes', 'learnable_per_classes']
    all_ensemble_results = {}  # approach -> { results, hyperparams }
    best_ensemble_results = None
    best_ensemble_hyperparams = None
    best_ensemble_approach = None

    if len(modified_gnn_models_list) > 0:
        ensemble_save_dir = os.path.join(results_dir, "ensemble_augmented")
        os.makedirs(ensemble_save_dir, exist_ok=True)
        succeeded_types = list(all_results.keys())
        title_list = [f"Augmented ({t})" for t in succeeded_types]

        for ensemble_approach in ensemble_approach_names:
            print("\n" + "="*80)
            print(f"STEP 7: Ensemble Learning — APPROACH: {ensemble_approach.upper()}")
            print("="*80)
            print(f"Running tune_ensemble_hyperparameter with {len(modified_gnn_models_list)} modified GNN models, "
                  f"modified embeddings, and modified (augmented) datasets.")
            reporter.report_title(f"ENSEMBLE — APPROACH: {ensemble_approach.upper()}")
            reporter.report_txt(f"  (Running ensemble with weight_approach={ensemble_approach})")
            reporter.report_txt("")
            try:
                approach_results, approach_hyperparams, _ = tune_ensemble_hyperparameter(
                    dataset_name=args.dataset_name,
                    model_list=modified_gnn_models_list,
                    custom_embeddings=modified_embeddings_list,
                    modified_datasets_list=modified_datasets_list,
                    does_report_training_process=False,
                    title=f"Augmented ({ensemble_approach})",
                    reporter=None,
                    llm_name=args.llm_name,
                    peft_type=args.peft_type,
                    title_list=title_list,
                    visualize_best=False,
                    ensemble_approaches=ensemble_approach,
                    save_dir=os.path.join(ensemble_save_dir, ensemble_approach),
                    show=False,
                    predictions_list=None,
                    supervised=False,
                )
                all_ensemble_results[ensemble_approach] = {
                    'results': approach_results,
                    'hyperparameters': approach_hyperparams,
                }
                if approach_results:
                    test_metrics = approach_results.get('test', {})
                    acc = test_metrics.get('accuracy')
                    f1 = test_metrics.get('macro_f1', test_metrics.get('f1'))
                    reporter.report_txt(f"  Test Accuracy: {acc}")
                    reporter.report_txt(f"  Test F1 (macro): {f1}")
                    if approach_hyperparams:
                        reporter.report_txt(f"  Best hyperparameters: {approach_hyperparams}")
                    reporter.report_txt("")
                    print(f"\n[OK] Ensemble ({ensemble_approach}) complete. Test accuracy: {acc}")
                    if best_ensemble_results is None or (acc is not None and acc > best_ensemble_results.get('test', {}).get('accuracy', 0)):
                        best_ensemble_results = approach_results
                        best_ensemble_hyperparams = approach_hyperparams
                        best_ensemble_approach = ensemble_approach
                else:
                    reporter.report_txt("  (No valid results returned)")
                    reporter.report_txt("")
            except Exception as e:
                print(f"\n[WARNING] Ensemble ({ensemble_approach}) failed: {e}")
                reporter.report_txt(f"  FAILED: {e}")
                reporter.report_txt("")
                import traceback
                traceback.print_exc()

        if best_ensemble_approach:
            reporter.report_title("BEST ENSEMBLE OVERALL")
            reporter.report_txt(f"  Best approach: {best_ensemble_approach}")
            reporter.report_txt(f"  Test Accuracy: {best_ensemble_results.get('test', {}).get('accuracy')}")
            reporter.report_txt(f"  Test F1 (macro): {best_ensemble_results.get('test', {}).get('macro_f1') or best_ensemble_results.get('test', {}).get('f1')}")
            reporter.report_txt("")
            print(f"\n[INFO] Best ensemble approach: {best_ensemble_approach}")
    else:
        print("\n[INFO] No modified models collected; skipping ensemble learning.")

    # Save all results to JSON (single models + all ensemble approaches)
    ensemble_json = {}
    if all_ensemble_results:
        for approach, data in all_ensemble_results.items():
            r = data.get('results')
            ensemble_json[approach] = {
                'test_accuracy': r.get('test', {}).get('accuracy') if r else None,
                'test_f1_macro': (r.get('test', {}).get('macro_f1') or r.get('test', {}).get('f1')) if r else None,
                'best_hyperparameters': data.get('hyperparameters'),
            }
    results_summary = {
        'dataset': args.dataset_name,
        'llm_name': args.llm_name,
        'peft_type': args.peft_type,
        'pooling': args.pooling,
        'embedding_types': all_results,
        'best_embedding_type': sorted_results[0][0] if sorted_results else None,
        'best_improvement': sorted_results[0][1]['improvement'] if sorted_results else None,
        'ensemble': ensemble_json,
        'best_ensemble_approach': best_ensemble_approach,
        'best_ensemble': {
            'test_accuracy': best_ensemble_results.get('test', {}).get('accuracy') if best_ensemble_results else None,
            'test_f1_macro': (best_ensemble_results.get('test', {}).get('macro_f1') or best_ensemble_results.get('test', {}).get('f1')) if best_ensemble_results else None,
            'best_hyperparameters': best_ensemble_hyperparams,
        } if best_ensemble_results else None,
    }
    results_path = os.path.join(results_dir, f"{args.dataset_name}_{args.llm_name}_all_embeddings_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[OK] All results saved to: {results_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80) 