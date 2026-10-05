# Steps 3-4 of the DIVERGE tags-augmentation pipeline (see approach.txt):
# take the combined base+augmented embedding cache produced by
# cache_base_and_augmented_nodes.py (both groups encoded by the SAME LLM,
# jointly fine-tuned on base+augmented nodes) and use it to
#   3) predict edges from the new nodes into the existing graph and build
#      the augmented graph, then
#   4) run the DIVERGE ensemble on the resulting augmented graphs.
#
# This differs from connect_new_nodes_to_graphs.py, which sources base-node
# embeddings from get_init_dataset_for_gnn() (a model fine-tuned WITHOUT the
# augmented nodes) and augmented-node embeddings from a separately-run
# extract_and_cache_new_nodes_embedding.py. Here both groups always come
# from one combined cache, so they are guaranteed to be comparable when we
# predict edges between them.

import os
import json

from config import setup_finetuning_cfg
from report.reporter import ReportResults
from gnns.gnn_mtrainer import gnn_train_and_report
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter
from tags_data_augmentation.edge_predictor import edge_predictor_train_and_report
from tags_data_augmentation.agumented_dataset import create_augmented_dataset
from tags_data_augmentation.connect_new_nodes_to_graphs import train_node_classification_gnn_with_modified_data
from tags_data_augmentation.cache_base_and_augmented_nodes import load_base_and_augmented_cache


def split_base_and_augmented_cache(cache_dict, device='cuda'):
    """
    Split a combined base+augmented cache into:
      - base_embeddings: [num_base_nodes, H] tensor for the original graph
      - augmented_cache: {'embeddings': [num_augmented_nodes, H], 'labels': [num_augmented_nodes]}
        in the same shape create_augmented_dataset() expects.
    """
    embeddings = cache_dict['embeddings'].to(device)
    labels = cache_dict['labels'].to(device)
    n_base = cache_dict['num_base_nodes']

    base_embeddings = embeddings[:n_base]
    augmented_cache = {
        'embeddings': embeddings[n_base:],
        'labels': labels[n_base:],
    }
    return base_embeddings, augmented_cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Connect augmented nodes to the graph using the base+augmented cache, '
                    'train GNNs, and run the DIVERGE ensemble'
    )
    parser.add_argument('--dataset_name', type=str, default='citeseer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--init_weight_approach', type=str, default='all',
                        choices=['pissa', 'orthogonal', 'guassian', 'loftq', 'eva', 'all'])
    parser.add_argument('--train_edge_predictor', action='store_true', default=True)
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'])
    parser.add_argument('--split_edges', action='store_true')
    parser.add_argument('--used_cosine_similarity', action='store_true', default=True)
    parser.add_argument('--no_cosine_similarity', dest='used_cosine_similarity', action='store_false')
    parser.add_argument('--used_k_nearest_neighbors', action='store_true', default=False)
    parser.add_argument('--no_k_nearest_neighbors', dest='used_k_nearest_neighbors', action='store_false')
    parser.add_argument('--used_edge_predictor', action='store_true', default=True)
    parser.add_argument('--threshold_cosine_similarity', type=float, default=0.9)
    parser.add_argument('--threshold_k_nearest_neighbors', type=float, default=0.5)
    parser.add_argument('--threshold_edge_predictor', type=float, default=0.7)
    parser.add_argument('--top_k_cosine_similarity', type=int, default=10)
    parser.add_argument('--top_k_knearest_neighbors', type=int, default=3)
    parser.add_argument('--top_k_edge_predictor', type=int, default=10)
    args = parser.parse_args()

    all_embedding_types = ['pissa', 'guassian', 'loftq', 'eva', 'orthogonal']
    embedding_types_to_process = (
        all_embedding_types if args.init_weight_approach == 'all' else [args.init_weight_approach]
    )

    cfg = setup_finetuning_cfg(args.dataset_name, args.llm_name, args.peft_type)

    results_dir = "./results/augmented_training_with_augmented_cache"
    os.makedirs(results_dir, exist_ok=True)
    reporter = ReportResults(cfg, save_dir=results_dir, index_run=0)
    reporter.report_title(f"Augmented Training Results (base+augmented cache) - {args.dataset_name.upper()}")
    reporter.report_txt(f"Dataset: {args.dataset_name} | LLM: {args.llm_name} | PEFT: {args.peft_type} | Pooling: {args.pooling}")
    reporter.report_txt(f"Edge creation: cosine={args.used_cosine_similarity}, knn={args.used_k_nearest_neighbors}, edge_predictor={args.used_edge_predictor}")
    reporter.report_txt("")

    all_results = {}
    modified_gnn_models_list = []
    modified_embeddings_list = []
    modified_datasets_list = []

    for init_weight_approach in embedding_types_to_process:
        print("\n" + "=" * 80)
        print(f"PROCESSING EMBEDDING TYPE: {init_weight_approach.upper()} (base+augmented cache)")
        print("=" * 80)

        try:
            cache_dict = load_base_and_augmented_cache(
                dataset_name=args.dataset_name,
                model_name=args.llm_name,
                pooling=args.pooling,
                init_weight_approach=init_weight_approach,
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {e}\nSkipping {init_weight_approach}...")
            continue

        base_embeddings, augmented_cache = split_base_and_augmented_cache(cache_dict)

        # Step 1: baseline GNN on the ORIGINAL graph, using the same jointly-trained embeddings
        print(f"\nTraining Baseline GNN (Original Dataset - {init_weight_approach}, base+augmented embeddings)")
        try:
            baseline_results, baseline_model = gnn_train_and_report(
                dataset_name=args.dataset_name,
                embedding=base_embeddings,
                title=f"Baseline ({init_weight_approach}, base+augmented cache)",
                does_print_training_process=False,
                supervised=False,
            )
        except Exception as e:
            print(f"[ERROR] Baseline training failed for {init_weight_approach}: {e}")
            continue

        # Step 2: edge predictor trained on the same embeddings (optional)
        edge_predictor_model = None
        if args.train_edge_predictor:
            training_strategy = 'without_splitting' if not args.split_edges else 'with_splitting'
            print(f"\nTraining Edge Predictor for {init_weight_approach} "
                  f"(decoder={args.decoder_type}, strategy={training_strategy})")
            try:
                edge_results, edge_predictor_model = edge_predictor_train_and_report(
                    dataset_name=args.dataset_name,
                    features=base_embeddings,
                    decoder_approach_type=args.decoder_type,
                    training_strategy=training_strategy,
                    negative_sampling_ratio=1.0,
                    title=f"Edge Predictor ({init_weight_approach})",
                    does_print_training_process=False,
                )
                print(f"  Edge Predictor - Val Acc: {edge_results['val_acc']:.4f}, Test Acc: {edge_results['test_acc']:.4f}")
            except Exception as e:
                print(f"  [WARNING] Edge predictor training failed: {e}")
                edge_predictor_model = None

        # Step 3: build the augmented graph, reusing the SAME base embeddings (no separate reload)
        print(f"\nCreating Augmented Dataset for {init_weight_approach} (base+augmented cache)")
        try:
            augmented_dataset, augmented_embeddings = create_augmented_dataset(
                cfg=cfg,
                augmented_cache=augmented_cache,
                init_weight_approach=init_weight_approach,
                used_cosine_similarity=args.used_cosine_similarity,
                used_k_nearest_neighbors=args.used_k_nearest_neighbors,
                used_edge_predictor=args.used_edge_predictor,
                threshold_cosine_similarity=args.threshold_cosine_similarity,
                threshold_k_nearest_neighbors=args.threshold_k_nearest_neighbors,
                threshold_edge_predictor=args.threshold_edge_predictor,
                top_k_cosine_similarity=args.top_k_cosine_similarity,
                top_k_knearest_neighbors=args.top_k_knearest_neighbors,
                top_k_edge_predictor=args.top_k_edge_predictor,
                edge_predictor_model=edge_predictor_model,
                supervised=False,
                base_embeddings_from_augmented_cache=base_embeddings,
            )
        except Exception as e:
            print(f"[ERROR] Failed to create augmented dataset for {init_weight_approach}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Step 4a: train GNN on the augmented graph
        print(f"\nTraining GNN on Augmented Dataset ({init_weight_approach})")
        try:
            augmented_dataset.x = augmented_embeddings
            augmented_results, augmented_model = train_node_classification_gnn_with_modified_data(
                augmented_dataset=augmented_dataset,
                augmented_embeddings=augmented_embeddings,
                cfg=cfg,
                title=f"Augmented ({init_weight_approach}, base+augmented cache)",
            )

            acc_improvement = augmented_results['test_acc'] - baseline_results['test_acc']
            f1_improvement = augmented_results['test_f1'] - baseline_results['test_f1']

            all_results[init_weight_approach] = {
                'baseline': baseline_results,
                'augmented': augmented_results,
                'improvement': {
                    'test_acc': acc_improvement,
                    'test_f1': f1_improvement,
                    'test_acc_pct': acc_improvement * 100,
                    'test_f1_pct': f1_improvement * 100,
                },
                'augmentation_config': {
                    'num_new_nodes': augmented_cache['embeddings'].shape[0],
                    'init_weight_approach': init_weight_approach,
                    'pooling': args.pooling,
                    'edge_creation_method': {
                        'cosine_similarity': args.used_cosine_similarity,
                        'k_nearest_neighbors': args.used_k_nearest_neighbors,
                        'edge_predictor': args.used_edge_predictor,
                    },
                },
            }

            modified_gnn_models_list.append(augmented_model)
            modified_embeddings_list.append(augmented_embeddings)
            modified_datasets_list.append(augmented_dataset)

            reporter.report_title(f"Model: {init_weight_approach.upper()}")
            reporter.report_txt(f"  Baseline  - Test Acc: {baseline_results['test_acc']:.4f}, Test F1: {baseline_results['test_f1']:.4f}")
            reporter.report_txt(f"  Augmented - Test Acc: {augmented_results['test_acc']:.4f}, Test F1: {augmented_results['test_f1']:.4f}")
            reporter.report_txt(f"  Improvement - Test Acc: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%), Test F1: {f1_improvement:+.4f}")
            reporter.report_txt(f"  Augmented nodes: {augmented_cache['embeddings'].shape[0]}")
            reporter.report_txt("")

            print(f"[OK] {init_weight_approach} - Test Acc: {augmented_results['test_acc']:.4f}, Improvement: {acc_improvement:+.4f}")
        except Exception as e:
            print(f"[ERROR] Training failed for {init_weight_approach}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Results comparison across embedding types ----
    reporter.report_title("FINAL RESULTS COMPARISON (base+augmented cache) - ALL EMBEDDING TYPES")
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON (base+augmented cache) - ALL EMBEDDING TYPES")
    print("=" * 80)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['augmented']['val_acc'], reverse=True)  # rank/select on val, never test
    for init_weight_approach, results in sorted_results:
        b, a, imp = results['baseline'], results['augmented'], results['improvement']
        line = (f"{init_weight_approach.upper()}: Baseline {b['test_acc']:.4f} -> "
                f"Augmented {a['test_acc']:.4f} ({imp['test_acc']:+.4f}, {imp['test_acc_pct']:+.2f}%)")
        reporter.report_txt(line)
        print(line)

    if sorted_results:
        best_type, best_results = sorted_results[0]
        reporter.report_txt("")
        reporter.report_txt(f"Best single model: {best_type} | Test Acc: {best_results['augmented']['test_acc']:.4f} "
                            f"| Improvement: {best_results['improvement']['test_acc']:+.4f} ({best_results['improvement']['test_acc_pct']:+.2f}%)")
        print(f"\n[INFO] Best performing embedding type: {best_type} "
              f"(Test Acc: {best_results['augmented']['test_acc']:.4f})")

    # ---- Step 4b: run the DIVERGE ensemble on the augmented graphs ----
    ensemble_approach_names = ['learnable', 'learnable_classes', 'learnable_per_classes']
    all_ensemble_results = {}
    best_ensemble_results = None
    best_ensemble_hyperparams = None
    best_ensemble_approach = None

    if len(modified_gnn_models_list) > 0:
        ensemble_save_dir = os.path.join(results_dir, "ensemble_augmented")
        os.makedirs(ensemble_save_dir, exist_ok=True)
        succeeded_types = list(all_results.keys())
        title_list = [f"Augmented ({t})" for t in succeeded_types]

        for ensemble_approach in ensemble_approach_names:
            print("\n" + "=" * 80)
            print(f"DIVERGE Ensemble — APPROACH: {ensemble_approach.upper()}")
            print("=" * 80)
            reporter.report_title(f"ENSEMBLE — APPROACH: {ensemble_approach.upper()}")
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
                    reporter.report_txt("")
                    print(f"[OK] Ensemble ({ensemble_approach}) complete. Test accuracy: {acc}")
                    if best_ensemble_results is None or (acc is not None and acc > best_ensemble_results.get('test', {}).get('accuracy', 0)):
                        best_ensemble_results = approach_results
                        best_ensemble_hyperparams = approach_hyperparams
                        best_ensemble_approach = ensemble_approach
                else:
                    reporter.report_txt("  (No valid results returned)")
            except Exception as e:
                print(f"[WARNING] Ensemble ({ensemble_approach}) failed: {e}")
                reporter.report_txt(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

        if best_ensemble_approach:
            reporter.report_title("BEST ENSEMBLE OVERALL")
            reporter.report_txt(f"  Best approach: {best_ensemble_approach}")
            reporter.report_txt(f"  Test Accuracy: {best_ensemble_results.get('test', {}).get('accuracy')}")
            reporter.report_txt(f"  Test F1 (macro): {best_ensemble_results.get('test', {}).get('macro_f1') or best_ensemble_results.get('test', {}).get('f1')}")
            print(f"\n[INFO] Best ensemble approach: {best_ensemble_approach}")
    else:
        print("\n[INFO] No modified models collected; skipping ensemble learning.")

    # ---- Save all results ----
    ensemble_json = {}
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
    results_path = os.path.join(results_dir, f"{args.dataset_name}_{args.llm_name}_base_and_augmented_cache_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[OK] All results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
