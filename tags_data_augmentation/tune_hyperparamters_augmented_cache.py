# Hyperparameter tuning for the augmented-graph construction step, using the
# combined base+augmented embedding cache (cache_base_and_augmented_nodes.py)
# instead of two separately fine-tuned embedding sources.
#
# See tune_hyperparamters.py for the original (two-cache) variant and
# approach.txt for the overall pipeline description. Hyperparameters
# considered here are the same as tune_hyperparamters.py:
#     1- init_weight_approaches
#     2- hybrid edge-construction method (cosine / knn / edge predictor)
#     3- topk for cosine / knn / edge predictor
#     4- threshold for cosine / knn / edge predictor

import os
import json
import argparse
import itertools

from config import setup_finetuning_cfg
from gnns.gnn_mtrainer import gnn_train_and_report
from report.reporter import ReportResults
from tags_data_augmentation.agumented_dataset import create_augmented_dataset
from tags_data_augmentation.edge_predictor import edge_predictor_train_and_report
from tags_data_augmentation.connect_new_nodes_to_graphs import train_node_classification_gnn_with_modified_data
from tags_data_augmentation.cache_base_and_augmented_nodes import load_base_and_augmented_cache
from tags_data_augmentation.connect_new_nodes_to_graphs_with_augmented_cache import split_base_and_augmented_cache


class HyperparameterTunerAugmentedCache:
    """Hyperparameter tuner for augmented dataset construction using the base+augmented cache."""

    def __init__(self, dataset_name, llm_name='llama_3.2_1B', peft_type='lora', pooling='mean'):
        self.dataset_name = dataset_name
        self.llm_name = llm_name
        self.peft_type = peft_type
        self.pooling = pooling

        self.cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)

        results_dir = './results/hyperparameter_tuning_augmented_cache'
        os.makedirs(results_dir, exist_ok=True)
        self.reporter = ReportResults(self.cfg, save_dir=results_dir, index_run=0)

        self.all_results = []
        self.best_result = None
        self.best_hyperparams = None

    def get_hybrid_configurations(self):
        """Returns list of (name, used_cosine, used_knn, used_edge_predictor)."""
        return [("cosine+edge_predictor", True, False, True)]

    def run_experiment(self, init_weight_approach, hybrid_config_name,
                        used_cosine, used_knn, used_edge_predictor,
                        threshold_cosine, threshold_knn, threshold_edge_predictor,
                        topk_cosine, topk_knn, topk_edge_predictor,
                        base_embeddings, augmented_cache, edge_predictor_model=None):
        """Run a single experiment with given hyperparameters and the shared base+augmented embeddings."""
        experiment_name = (
            f"{init_weight_approach}_{hybrid_config_name}_"
            f"tc{threshold_cosine:.2f}_tk{threshold_knn:.2f}_te{threshold_edge_predictor:.2f}_"
            f"kc{topk_cosine}_kk{topk_knn}_ke{topk_edge_predictor}"
        )
        print(f"\n{'='*100}\nRunning Experiment: {experiment_name}\n{'='*100}")

        try:
            augmented_dataset, augmented_embeddings = create_augmented_dataset(
                cfg=self.cfg,
                augmented_cache=augmented_cache,
                init_weight_approach=init_weight_approach,
                used_cosine_similarity=used_cosine,
                used_k_nearest_neighbors=used_knn,
                used_edge_predictor=used_edge_predictor,
                threshold_cosine_similarity=threshold_cosine,
                threshold_k_nearest_neighbors=threshold_knn,
                threshold_edge_predictor=threshold_edge_predictor,
                top_k_cosine_similarity=topk_cosine,
                top_k_knearest_neighbors=topk_knn,
                top_k_edge_predictor=topk_edge_predictor,
                edge_predictor_model=edge_predictor_model,
                supervised=False,
                base_embeddings_from_augmented_cache=base_embeddings,
            )

            augmented_dataset.x = augmented_embeddings
            results, model = train_node_classification_gnn_with_modified_data(
                augmented_dataset=augmented_dataset,
                augmented_embeddings=augmented_embeddings,
                cfg=self.cfg,
                title=experiment_name,
            )

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
        """Run hyperparameter tuning over all combinations, reusing one loaded cache per init approach."""
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

        hybrid_configs = self.get_hybrid_configurations()

        self.reporter.report_title(f"Hyperparameter Tuning (base+augmented cache) for {self.dataset_name.upper()}")
        self.reporter.report_txt(f"Dataset: {self.dataset_name}")
        self.reporter.report_txt(f"LLM: {self.llm_name}")
        self.reporter.report_txt(f"PEFT: {self.peft_type}")
        self.reporter.report_txt(f"\nSearching over:")
        self.reporter.report_txt(f"  - Init weight approaches: {init_weight_approaches}")
        self.reporter.report_txt(f"  - Threshold cosine: {threshold_cosine_values}")
        self.reporter.report_txt(f"  - Threshold KNN: {threshold_knn_values}")
        self.reporter.report_txt(f"  - Threshold edge predictor: {threshold_edge_predictor_values}")
        self.reporter.report_txt(f"  - Top-K cosine: {topk_cosine_values}")
        self.reporter.report_txt(f"  - Top-K KNN: {topk_knn_values}")
        self.reporter.report_txt(f"  - Top-K edge predictor: {topk_edge_predictor_values}")
        self.reporter.report_txt(f"\n{'='*70}\n")

        total_experiments = 0
        successful_experiments = 0
        baseline_by_approach = {}

        for init_approach in init_weight_approaches:
            print(f"\n{'#'*100}\nInit Weight Approach: {init_approach}\n{'#'*100}")

            try:
                cache_dict = load_base_and_augmented_cache(
                    dataset_name=self.dataset_name,
                    model_name=self.llm_name,
                    pooling=self.pooling,
                    init_weight_approach=init_approach,
                )
            except FileNotFoundError as e:
                print(f"[ERROR] {e}\nSkipping {init_approach}...")
                continue

            # Base embeddings + augmented-node embeddings, loaded ONCE per approach and
            # reused across every hyperparameter combination below.
            base_embeddings, augmented_cache = split_base_and_augmented_cache(cache_dict)

            print(f"\nTraining Baseline (Original Dataset - {init_approach}, base+augmented embeddings)")
            try:
                baseline_results, _ = gnn_train_and_report(
                    dataset_name=self.dataset_name,
                    embedding=base_embeddings,
                    title=f"Baseline ({init_approach})",
                    does_print_training_process=False,
                    supervised=False,
                )
            except Exception as e:
                print(f"[ERROR] Baseline training failed for {init_approach}: {e}\nSkipping {init_approach}...")
                continue

            baseline_by_approach[init_approach] = baseline_results
            self.reporter.report_title(f"Baseline Results ({init_approach})")
            self.reporter.report_txt(f"Test Accuracy: {baseline_results['test_acc']:.4f}")
            self.reporter.report_txt(f"Test F1 (Macro): {baseline_results['test_f1']:.4f}")
            self.reporter.report_txt(f"Test F1 (Weighted): {baseline_results['test_weight_f1']:.4f}")
            self.reporter.report_txt(f"Val Accuracy: {baseline_results['val_acc']:.4f}\n")

            edge_predictor_model = None
            if train_edge_predictor:
                print(f"\n[INFO] Training edge predictor for {init_approach}...")
                try:
                    _, edge_predictor_model = edge_predictor_train_and_report(
                        dataset_name=self.dataset_name,
                        features=base_embeddings,
                        decoder_approach_type=decoder_type,
                        training_strategy='without_splitting',
                        negative_sampling_ratio=1.0,
                        title=f"Edge Predictor ({init_approach})",
                        does_print_training_process=False,
                    )
                except Exception as e:
                    print(f"[WARNING] Edge predictor training failed for {init_approach}: {e}")
                    edge_predictor_model = None

            for hybrid_name, used_cosine, used_knn, used_edge_pred in hybrid_configs:
                if used_edge_pred and not train_edge_predictor:
                    continue

                threshold_cosine_list = threshold_cosine_values if used_cosine else [0.95]
                threshold_knn_list = threshold_knn_values if used_knn else [0.5]
                threshold_edge_pred_list = threshold_edge_predictor_values if used_edge_pred else [0.5]
                topk_cosine_list = topk_cosine_values if used_cosine else [3]
                topk_knn_list = topk_knn_values if used_knn else [3]
                topk_edge_pred_list = topk_edge_predictor_values if used_edge_pred else [5]

                param_combinations = itertools.product(
                    threshold_cosine_list, threshold_knn_list, threshold_edge_pred_list,
                    topk_cosine_list, topk_knn_list, topk_edge_pred_list,
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
                        base_embeddings=base_embeddings,
                        augmented_cache=augmented_cache,
                        edge_predictor_model=edge_predictor_model,
                    )

                    if result is not None:
                        successful_experiments += 1
                        baseline_acc = baseline_by_approach[init_approach]['test_acc']
                        result['baseline_test_acc'] = baseline_acc
                        result['improvement_test_acc'] = result['results']['test_acc'] - baseline_acc
                        self.all_results.append(result)

                        if self.best_result is None or result['results']['val_acc'] > self.best_result['results']['val_acc']:  # select on val, never test
                            self.best_result = result
                            self.best_hyperparams = result['hyperparameters']

        print(f"\n{'='*100}\nHyperparameter Tuning Complete!\n{'='*100}")
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
            self.reporter.report_txt(f"\nImprovement over baseline: {self.best_result['improvement_test_acc']:+.4f} "
                                    f"({self.best_result['improvement_test_acc']*100:+.2f}%)")

            print(f"\nBest Result:")
            print(f"  Experiment: {self.best_result['experiment_name']}")
            print(f"  Test Acc: {self.best_result['results']['test_acc']:.4f}")
            print(f"  Test F1: {self.best_result['results']['test_f1']:.4f}")
            print(f"  Improvement: {self.best_result['improvement_test_acc']:+.4f}")

        self.save_results_json()

        return self.best_result, self.best_hyperparams

    def save_results_json(self):
        """Save all results to JSON file."""
        results_dir = './results/hyperparameter_tuning_augmented_cache'
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
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for augmented dataset construction (base+augmented cache)'
    )
    parser.add_argument('--dataset_name', type=str, default='citeseer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--init_weight_approaches', type=str, nargs='+', default=['pissa'])
    parser.add_argument('--train_edge_predictor', action='store_true', default=True)
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'])
    args = parser.parse_args()

    tuner = HyperparameterTunerAugmentedCache(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        pooling=args.pooling,
    )

    threshold_cosine_values = [0.90, 0.95, 0.97, 0.8]
    threshold_knn_values = [0.5]
    threshold_edge_predictor_values = [0.5, 0.7, 0.9]
    topk_cosine_values = [3, 5, 7, 10]
    topk_knn_values = [3]
    topk_edge_predictor_values = [5, 7, 10]

    best_result, best_hyperparams = tuner.tune(
        init_weight_approaches=args.init_weight_approaches,
        threshold_cosine_values=threshold_cosine_values,
        threshold_knn_values=threshold_knn_values,
        threshold_edge_predictor_values=threshold_edge_predictor_values,
        topk_cosine_values=topk_cosine_values,
        topk_knn_values=topk_knn_values,
        topk_edge_predictor_values=topk_edge_predictor_values,
        train_edge_predictor=args.train_edge_predictor,
        decoder_type=args.decoder_type,
    )

    print("\n" + "=" * 100)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 100)
