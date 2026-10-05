import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import torch

from common.dataloader import load_graph_dataset_for_tape
from config import setup_finetuning_cfg
from dataset.data_utils import (
  get_embedding_from_data,
  get_init_dataset_for_gnn,
  get_init_dataset_for_gnn_with_retrained_gnn_mistake,
)
from dataset.dataset_loader import load_dataset
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report
from report.reporter import ReportResults
from cosine_similarity_enhancer.cosine_similarity_enhancer import SemanticSimilarityEnhancer
from cosine_similarity_enhancer.experiments.homophily_structural_analysis import (
  HomophilyStructuralAnalysis,
)


class DIVERGEWithSemanticEnhancer:
  """
  DIVERGE pipeline with graph structural enhancement via cosine similarity.

  For each fine-tuned LLM representation:
    1. Tune the cosine similarity threshold on validation accuracy.
    2. Remove noisy edges below the best threshold.
    3. Train a GNN on the refined graph.

  Finally, combine all GNNs with ensemble hyperparameter tuning.
  """

  INIT_METHODS = [
    ('pissa', 'PISSA'),
    ('orthogonal', 'ORTHOGONAL'),
    ('loftq', 'LOFTQ'),
    ('eva', 'EVA'),
    ('guassian', 'GAUSSIAN'),
  ]

  def __init__(self):
    self.enhancer = SemanticSimilarityEnhancer()

  @staticmethod
  def _get_reports_dir() -> Path:
    return Path(__file__).resolve().parent / 'reports'

  @staticmethod
  def _get_experiment_tag(
    retrained_with_gnn_mistakes: bool,
    supervised: bool,
    use_all_nodes: bool,
  ) -> str:
    parts = ['diverge', 'cosine_similarity_enhancer']
    if retrained_with_gnn_mistakes:
      parts.append('gnn_mistakes')
    else:
      parts.append('standard_emb')
    if not supervised:
      parts.append('semi_supervised')
    if not use_all_nodes:
      parts.append('node_subset')
    return '_'.join(parts)

  @staticmethod
  def _write_summary_report(
    reporter,
    dataset_name: str,
    llm_name: str,
    peft_type: str,
    ensemble_approaches: str,
    retrained_with_gnn_mistakes: bool,
    supervised: bool,
    use_all_nodes: bool,
    original_results: dict,
    refined_results: dict,
    threshold_by_method: dict,
    ensemble_results: dict,
    best_ensemble_params: dict,
    saved_hyperparams_path: Path,
  ):
    reporter.report_title('DIVERGE WITH COSINE SIMILARITY ENHANCER - EXPERIMENT CONFIG')
    reporter.report_txt(f'Dataset: {dataset_name}')
    reporter.report_txt(f'LLM: {llm_name}')
    reporter.report_txt(f'PEFT Type: {peft_type}')
    reporter.report_txt(f'Ensemble Approach: {ensemble_approaches}')
    reporter.report_txt(
      f'Embeddings: {"retrained with GNN mistakes" if retrained_with_gnn_mistakes else "standard fine-tuned"}'
    )
    reporter.report_txt(f'Supervised: {supervised}')
    reporter.report_txt(
      f'Edge evaluation scope: {"all edges" if use_all_nodes else "node subset only"}'
    )
    reporter.report_txt(f'Saved hyperparameters: {saved_hyperparams_path}')
    reporter.report_txt('')

    reporter.report_title('STAGE 1: BASELINE GNN RESULTS (ORIGINAL GRAPH)')
    for title in original_results:
      result = original_results[title]
      reporter.report_txt(f'\n{title}:')
      reporter.report_txt(f'  Test Accuracy: {result["test_acc"]:.4f}')
      reporter.report_txt(f'  Test F1 (Macro): {result["test_f1"]:.4f}')
      reporter.report_txt(f'  Test F1 (Weighted): {result["test_weight_f1"]:.4f}')
      reporter.report_txt(f'  Val Accuracy: {result["val_acc"]:.4f}')
      reporter.report_txt(f'  Val F1 (Macro): {result["val_f1"]:.4f}')

    reporter.report_title('STAGE 2: COSINE SIMILARITY ENHANCER RESULTS (REFINED GRAPH)')
    for title in refined_results:
      reporter.report_txt(f'\n{title}:')
      reporter.report_txt(f'  Best Similarity Threshold: {threshold_by_method[title]:.4f}')
      reporter.report_txt(
        f'  Original Test Acc: {original_results[title]["test_acc"]:.4f}'
      )
      reporter.report_txt(
        f'  Refined Test Acc: {refined_results[title]["test_acc"]:.4f}'
      )
      reporter.report_txt(
        f'  Test Acc Improvement: '
        f'{(refined_results[title]["test_acc"] - original_results[title]["test_acc"]):+.4f}'
      )
      reporter.report_txt(
        f'  Original Test F1: {original_results[title]["test_f1"]:.4f}'
      )
      reporter.report_txt(f'  Refined Test F1: {refined_results[title]["test_f1"]:.4f}')
      reporter.report_txt(
        f'  Test F1 Improvement: '
        f'{(refined_results[title]["test_f1"] - original_results[title]["test_f1"]):+.4f}'
      )
      reporter.report_txt(f'  Val Accuracy: {refined_results[title]["val_acc"]:.4f}')
      reporter.report_txt(f'  Val F1 (Macro): {refined_results[title]["val_f1"]:.4f}')

    reporter.report_title('STAGE 3: ENSEMBLE RESULTS (REFINED GRAPHS)')
    reporter.report_txt(f'Best Ensemble Hyperparameters: {best_ensemble_params}')
    reporter.report_txt('')
    reporter.report_txt('Test Results:')
    reporter.report_txt(
      f'  Test Accuracy: {ensemble_results["test"]["accuracy"]:.4f}'
    )
    reporter.report_txt(
      f'  Test F1 (Macro): {ensemble_results["test"]["macro_f1"]:.4f}'
    )
    reporter.report_txt(
      f'  Test F1 (Weighted): {ensemble_results["test"]["weighted_f1"]:.4f}'
    )
    reporter.report_txt('')
    reporter.report_txt('Validation Results:')
    reporter.report_txt(
      f'  Val Accuracy: {ensemble_results["val"]["accuracy"]:.4f}'
    )
    reporter.report_txt(
      f'  Val F1 (Macro): {ensemble_results["val"]["macro_f1"]:.4f}'
    )

  @staticmethod
  def _write_homophily_report(reporter, homophily_results: dict):
    reporter.report_title('STAGE 4: HOMOPHILY / STRUCTURAL METRICS (BEFORE VS. AFTER)')
    summary = homophily_results.get('summary', {})
    reporter.report_txt(
      f"Mean delta homophily: {summary.get('mean_delta_homophily', float('nan')):+.4f}"
    )
    reporter.report_txt(
      f"Mean edges removed: {summary.get('mean_pct_edges_removed', float('nan')):.2f}%"
    )
    reporter.report_txt(
      f"Mean delta test accuracy: {summary.get('mean_delta_test_acc', float('nan')):+.4f}"
    )
    reporter.report_txt('')

    for row in homophily_results.get('rows', []):
      reporter.report_txt(f"\n{row['init_method']}:")
      reporter.report_txt(
        f"  Homophily: {row['homophily_before']:.4f} -> "
        f"{row['homophily_after']:.4f} (delta {row['delta_homophily']:+.4f})"
      )
      reporter.report_txt(f"  Cosine ESNR: {row['cosine_esnr_before']:.4f} -> "
                          f"{row['cosine_esnr_after']:.4f}")
      reporter.report_txt(
        f"  Retained/removed cosine sim: "
        f"{row['avg_cosine_sim_retained']:.4f} / {row['avg_cosine_sim_removed']:.4f}"
      )
      reporter.report_txt(f"  Edges removed: {row['pct_edges_removed']:.2f}%")
      reporter.report_txt(
        f"  Baseline vs refined test acc: "
        f"{row['baseline_test_acc']:.4f} -> {row['refined_test_acc']:.4f} "
        f"(delta {row['delta_test_acc']:+.4f})"
      )

    output_paths = homophily_results.get('output_paths', {})
    if output_paths:
      reporter.report_txt('')
      reporter.report_txt(f"Metrics CSV: {output_paths.get('metrics_csv', '')}")
      reporter.report_txt(f"Summary CSV: {output_paths.get('summary_csv', '')}")
      reporter.report_txt(f"Figures dir: {output_paths.get('figures_dir', '')}")

  @staticmethod
  def _get_best_hyperparameters_dir() -> Path:
    return Path(__file__).resolve().parent / 'best_hyperparamters'

  @classmethod
  def get_best_hyperparameters_path(
    cls,
    dataset_name: str,
    supervised: bool = True,
  ) -> Path:
    save_dir = cls._get_best_hyperparameters_dir()
    if supervised:
      return save_dir / f'{dataset_name}.json'
    return save_dir / f'{dataset_name}_semi_supervised.json'

  @classmethod
  def save_best_hyperparameters(
    cls,
    dataset_name: str,
    threshold_by_method: dict,
    llm_name: str = 'llama_3.2_1B',
    peft_type: str = 'lora',
    supervised: bool = True,
    use_all_nodes: bool = True,
    retrained_with_gnn_mistakes: bool = True,
  ) -> Path:
    """Save tuned cosine similarity thresholds for later reuse."""
    save_dir = cls._get_best_hyperparameters_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = {
      'dataset': dataset_name,
      'llm_name': llm_name,
      'peft_type': peft_type,
      'supervised': supervised,
      'use_all_nodes': use_all_nodes,
      'retrained_with_gnn_mistakes': retrained_with_gnn_mistakes,
      'similarity_thresholds': {
        method: float(threshold)
        for method, threshold in threshold_by_method.items()
      },
    }

    save_path = cls.get_best_hyperparameters_path(dataset_name, supervised=supervised)
    with open(save_path, 'w', encoding='utf-8') as f:
      json.dump(hyperparams, f, indent=4)

    print(f"\nBest enhancer hyperparameters saved to: {save_path}")
    return save_path

  @classmethod
  def load_best_hyperparameters(
    cls,
    dataset_name: str,
    supervised: bool = True,
  ) -> dict:
    """Load saved cosine similarity thresholds for a dataset."""
    save_path = cls.get_best_hyperparameters_path(dataset_name, supervised=supervised)
    if not save_path.exists():
      raise FileNotFoundError(
        f"No saved enhancer hyperparameters found for dataset '{dataset_name}' "
        f"at {save_path}"
      )
    with open(save_path, 'r', encoding='utf-8') as f:
      content = f.read().strip()
      if not content:
        raise ValueError(f"Saved hyperparameters file is empty: {save_path}")
      return json.loads(content)

  @staticmethod
  def _get_path_prefix():
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
      if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
        return '.'
      return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'

  def _load_init_datasets(self, cfg, retrained_with_gnn_mistakes: bool):
    if retrained_with_gnn_mistakes:
      print("Loading embeddings retrained with GNN mistakes...")
      return get_init_dataset_for_gnn_with_retrained_gnn_mistake(cfg)
    print("Loading standard fine-tuned embeddings...")
    return get_init_dataset_for_gnn(cfg)

  def run(
    self,
    dataset_name: str,
    llm_name: str = 'llama_3.2_1B',
    peft_type: str = 'lora',
    retrained_with_gnn_mistakes: bool = True,
    reporter_index: int = 0,
    ensemble_approaches: str = 'learnable',
    supervised: bool = True,
    use_all_nodes: bool = True,
    run_homophily_analysis: bool = True,
  ):
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    experiment_tag = self._get_experiment_tag(
      retrained_with_gnn_mistakes=retrained_with_gnn_mistakes,
      supervised=supervised,
      use_all_nodes=use_all_nodes,
    )
    reporter = ReportResults(
      cfg,
      save_dir=str(self._get_reports_dir()),
      index_run=reporter_index,
      init_approach=experiment_tag,
    )
    print(f"Experiment report file: {reporter.file_path}")

    path_prefix = self._get_path_prefix()
    graph_dataset, _, _ = load_graph_dataset_for_tape(
      dataset_name,
      'cuda:0',
      re_split=1 if supervised else 0,
      path_prefix=path_prefix,
      seed=cfg.dataset.seed,
    )
    cora_dataset = load_dataset(cfg)

    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = (
      self._load_init_datasets(cfg, retrained_with_gnn_mistakes)
    )

    data_by_key = {
      'pissa': data_pissa,
      'orthogonal': data_orthogonal,
      'loftq': data_loftq,
      'eva': data_eva,
      'guassian': data_guassian,
    }

    node_subset = None
    if not use_all_nodes:
      node_subset = list(range(cora_dataset.num_nodes))

    refined_models = []
    refined_embeddings = []
    refined_datasets = []
    original_results = {}
    refined_results = {}
    threshold_by_method = {}

    print("\n" + "=" * 80)
    print("STAGE 1: BASELINE GNN TRAINING (ORIGINAL GRAPH)")
    print("=" * 80)

    for key, title in self.INIT_METHODS:
      embeddings = get_embedding_from_data(data_by_key[key])
      original_result, _ = gnn_train_and_report(
        dataset_name,
        embeddings,
        title=title,
        does_print_training_process=False,
        reporter=reporter,
      )
      original_results[title] = original_result

    print("\n" + "=" * 80)
    print("STAGE 2: COSINE SIMILARITY THRESHOLD TUNING + REFINED GNN TRAINING")
    print("=" * 80)

    for key, title in self.INIT_METHODS:
      print(f"\n{'=' * 80}")
      print(f"Processing {title}")
      print("=" * 80)

      embeddings = get_embedding_from_data(data_by_key[key])
      best_result = self.enhancer.tune_similarity_threshold(
        dataset=cora_dataset,
        embeddings=embeddings,
        dataset_name=dataset_name,
        node_subset=node_subset,
        supervised=supervised,
        reporter=reporter,
        title=title,
      )

      threshold_by_method[title] = best_result['threshold']
      refined_results[title] = best_result['results']
      refined_models.append(best_result['model'])
      refined_embeddings.append(embeddings)
      refined_datasets.append(best_result['modified_dataset'])

    print("\n" + "=" * 80)
    print("STAGE 3: ENSEMBLE ON REFINED GRAPHS")
    print("=" * 80)

    ensemble_results, best_ensemble_params, _ = tune_ensemble_hyperparameter(
      dataset_name=dataset_name,
      model_list=refined_models,
      custom_embeddings=refined_embeddings,
      modified_datasets_list=refined_datasets,
      does_report_training_process=False,
      ensemble_approaches=ensemble_approaches,
      title="DIVERGE with Cosine Similarity Enhancer",
      reporter=reporter,
      llm_name=llm_name,
      peft_type=peft_type,
      title_list=[title for _, title in self.INIT_METHODS],
      supervised=supervised,
    )

    print("\n" + "=" * 80)
    print("DIVERGE WITH COSINE SIMILARITY ENHANCER - SUMMARY")
    print("=" * 80)

    for title in [name for _, name in self.INIT_METHODS]:
      print(f"\n{title}:")
      print(f"  Best threshold: {threshold_by_method[title]:.4f}")
      print(
        f"  Original graph  - Test Acc: {original_results[title]['test_acc']:.4f}, "
        f"Test F1: {original_results[title]['test_f1']:.4f}"
      )
      print(
        f"  Refined graph   - Test Acc: {refined_results[title]['test_acc']:.4f}, "
        f"Test F1: {refined_results[title]['test_f1']:.4f}"
      )

    print("\nEnsemble (refined graphs):")
    print(f"  Best params: {best_ensemble_params}")
    print(f"  Test Accuracy: {ensemble_results['test']['accuracy']:.4f}")
    print(f"  Test Macro-F1: {ensemble_results['test']['macro_f1']:.4f}")

    saved_hyperparams_path = self.save_best_hyperparameters(
      dataset_name=dataset_name,
      threshold_by_method=threshold_by_method,
      llm_name=llm_name,
      peft_type=peft_type,
      supervised=supervised,
      use_all_nodes=use_all_nodes,
      retrained_with_gnn_mistakes=retrained_with_gnn_mistakes,
    )

    homophily_results = None
    if run_homophily_analysis:
      print("\n" + "=" * 80)
      print("STAGE 4: HOMOPHILY / STRUCTURAL METRICS ANALYSIS")
      print("=" * 80)
      homophily_analyzer = HomophilyStructuralAnalysis()
      homophily_results = homophily_analyzer.run_from_pipeline_results(
        dataset_name=dataset_name,
        cora_dataset=cora_dataset,
        data_by_key=data_by_key,
        threshold_by_method=threshold_by_method,
        original_results=original_results,
        refined_results=refined_results,
        refined_datasets=refined_datasets,
        node_subset=node_subset,
        save_outputs=True,
      )

    self._write_summary_report(
      reporter=reporter,
      dataset_name=dataset_name,
      llm_name=llm_name,
      peft_type=peft_type,
      ensemble_approaches=ensemble_approaches,
      retrained_with_gnn_mistakes=retrained_with_gnn_mistakes,
      supervised=supervised,
      use_all_nodes=use_all_nodes,
      original_results=original_results,
      refined_results=refined_results,
      threshold_by_method=threshold_by_method,
      ensemble_results=ensemble_results,
      best_ensemble_params=best_ensemble_params,
      saved_hyperparams_path=saved_hyperparams_path,
    )
    if homophily_results is not None:
      self._write_homophily_report(reporter, homophily_results)
    print(f"\nFull experiment report saved to: {reporter.file_path}")

    return {
      'original_results': original_results,
      'refined_results': refined_results,
      'threshold_by_method': threshold_by_method,
      'ensemble_results': ensemble_results,
      'best_ensemble_params': best_ensemble_params,
      'refined_models': refined_models,
      'refined_datasets': refined_datasets,
      'refined_embeddings': refined_embeddings,
      'saved_hyperparams_path': saved_hyperparams_path,
      'report_path': reporter.file_path,
      'homophily_results': homophily_results,
    }


def main():
  parser = argparse.ArgumentParser(
    description='DIVERGE with cosine similarity graph structural enhancement'
  )
  parser.add_argument('--dataset_name', type=str, default='cora')
  parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
  parser.add_argument('--peft_type', type=str, default='lora')
  parser.add_argument(
    '--retrained_with_gnn_mistakes',
    action='store_true',
    default=True,
    help='Use embeddings retrained with GNN mistakes (default: True)',
  )
  parser.add_argument(
    '--no_retrained_with_gnn_mistakes',
    action='store_false',
    dest='retrained_with_gnn_mistakes',
    help='Use standard fine-tuned embeddings',
  )
  parser.add_argument('--reporter_index', type=int, default=0)
  parser.add_argument('--ensemble_approaches', type=str, default='learnable')
  parser.add_argument(
    '--use_all_nodes',
    action='store_true',
    default=True,
    help='Evaluate all edges for removal (default, matches Proposed Method II)',
  )
  parser.add_argument(
    '--run_homophily_analysis',
    action='store_true',
    default=True,
    help='Run homophily/structural metrics experiment after refinement (default: True)',
  )
  parser.add_argument(
    '--skip_homophily_analysis',
    action='store_false',
    dest='run_homophily_analysis',
    help='Skip homophily/structural metrics experiment',
  )
  args = parser.parse_args()

  pipeline = DIVERGEWithSemanticEnhancer()
  pipeline.run(
    dataset_name=args.dataset_name,
    llm_name=args.llm_name,
    peft_type=args.peft_type,
    retrained_with_gnn_mistakes=args.retrained_with_gnn_mistakes,
    reporter_index=args.reporter_index,
    ensemble_approaches=args.ensemble_approaches,
    use_all_nodes=args.use_all_nodes,
    run_homophily_analysis=args.run_homophily_analysis,
  )


if __name__ == '__main__':
  main()
