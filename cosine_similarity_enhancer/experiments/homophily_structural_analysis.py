"""
Homophily / structural metrics experiment (proposed experiment #3).

Computes before-vs-after refinement metrics, saves CSV tables, and generates
publication-ready figures for Q1 paper submission.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from config import setup_finetuning_cfg
from cosine_similarity_enhancer.cosine_similarity_enhancer import SemanticSimilarityEnhancer
from cosine_similarity_enhancer.experiments.structural_metrics import (
  compare_before_after_metrics,
  compute_graph_structural_metrics,
)
from dataset.data_utils import (
  get_embedding_from_data,
  get_init_dataset_for_gnn,
  get_init_dataset_for_gnn_with_retrained_gnn_mistake,
)
from dataset.dataset_loader import load_dataset


PAPER_STYLE = {
  'figure.dpi': 150,
  'savefig.dpi': 300,
  'font.family': 'serif',
  'font.size': 11,
  'axes.labelsize': 12,
  'axes.titlesize': 13,
  'legend.fontsize': 10,
  'xtick.labelsize': 10,
  'ytick.labelsize': 10,
  'axes.spines.top': False,
  'axes.spines.right': False,
}

BEFORE_COLOR = '#4C72B0'
AFTER_COLOR = '#55A868'
ACCENT_COLOR = '#C44E52'
NEUTRAL_COLOR = '#8172B3'


class HomophilyStructuralAnalysis:
  """Run homophily/structural analysis for DIVERGE + cosine similarity enhancer."""

  INIT_METHODS = [
    ('pissa', 'PISSA'),
    ('orthogonal', 'ORTHOGONAL'),
    ('loftq', 'LOFTQ'),
    ('eva', 'EVA'),
    ('guassian', 'GAUSSIAN'),
  ]

  CSV_COLUMNS = [
    'dataset',
    'init_method',
    'similarity_threshold',
    'homophily_before',
    'homophily_after',
    'delta_homophily',
    'homophily_esnr_before',
    'homophily_esnr_after',
    'delta_homophily_esnr',
    'cosine_esnr_before',
    'cosine_esnr_after',
    'delta_cosine_esnr',
    'avg_cosine_sim_retained',
    'avg_cosine_sim_removed',
    'pct_edges_removed',
    'num_edges_before',
    'num_edges_after',
    'baseline_test_acc',
    'refined_test_acc',
    'delta_test_acc',
    'baseline_test_f1',
    'refined_test_f1',
    'delta_test_f1',
  ]

  def __init__(self):
    self.enhancer = SemanticSimilarityEnhancer()
    plt.rcParams.update(PAPER_STYLE)
    sns.set_theme(style='whitegrid', context='paper')

  @staticmethod
  def get_results_dir() -> Path:
    results_dir = Path(__file__).resolve().parent.parent / 'experiment_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

  @classmethod
  def _load_init_datasets(cls, cfg, retrained_with_gnn_mistakes: bool):
    if retrained_with_gnn_mistakes:
      return get_init_dataset_for_gnn_with_retrained_gnn_mistake(cfg)
    return get_init_dataset_for_gnn(cfg)

  @staticmethod
  def _metrics_to_row(
    dataset_name: str,
    init_method: str,
    comparison: dict,
    original_result: dict | None = None,
    refined_result: dict | None = None,
  ) -> dict:
    before = comparison['before']
    after = comparison['after']
    removal = comparison['removal_stats']

    baseline_acc = original_result['test_acc'] if original_result else float('nan')
    refined_acc = refined_result['test_acc'] if refined_result else float('nan')
    baseline_f1 = original_result['test_f1'] if original_result else float('nan')
    refined_f1 = refined_result['test_f1'] if refined_result else float('nan')

    return {
      'dataset': dataset_name,
      'init_method': init_method,
      'similarity_threshold': comparison['similarity_threshold'],
      'homophily_before': before['edge_homophily_ratio'],
      'homophily_after': after['edge_homophily_ratio'],
      'delta_homophily': comparison['delta_homophily'],
      'homophily_esnr_before': before['homophily_esnr'],
      'homophily_esnr_after': after['homophily_esnr'],
      'delta_homophily_esnr': comparison['delta_homophily_esnr'],
      'cosine_esnr_before': before['cosine_esnr'],
      'cosine_esnr_after': after['cosine_esnr'],
      'delta_cosine_esnr': comparison['delta_cosine_esnr'],
      'avg_cosine_sim_retained': removal['avg_cosine_sim_retained'],
      'avg_cosine_sim_removed': removal['avg_cosine_sim_removed'],
      'pct_edges_removed': comparison['pct_edges_removed'],
      'num_edges_before': before['num_edges'],
      'num_edges_after': after['num_edges'],
      'baseline_test_acc': baseline_acc,
      'refined_test_acc': refined_acc,
      'delta_test_acc': refined_acc - baseline_acc,
      'baseline_test_f1': baseline_f1,
      'refined_test_f1': refined_f1,
      'delta_test_f1': refined_f1 - baseline_f1,
    }

  def analyze_method(
    self,
    dataset,
    embeddings,
    threshold: float,
    node_subset=None,
    init_method: str = '',
    dataset_name: str = '',
    original_result: dict | None = None,
    refined_result: dict | None = None,
    refined_dataset=None,
    verbose: bool = True,
  ) -> dict:
    """Analyze one init-method view at a fixed similarity threshold."""
    if refined_dataset is None:
      refined_dataset, _ = self.enhancer.refine_graph_by_cosine_similarity(
        dataset=dataset,
        embeddings=embeddings,
        similarity_threshold=float(threshold),
        node_subset=node_subset,
      )

    comparison = compare_before_after_metrics(
      original_dataset=dataset,
      refined_dataset=refined_dataset,
      embeddings=embeddings,
      similarity_threshold=float(threshold),
      node_subset=node_subset,
    )
    row = self._metrics_to_row(
      dataset_name=dataset_name,
      init_method=init_method,
      comparison=comparison,
      original_result=original_result,
      refined_result=refined_result,
    )

    if verbose:
      print(f"\n[{init_method}] threshold={threshold:.4f}")
      print(
        f"  Homophily: {row['homophily_before']:.4f} -> "
        f"{row['homophily_after']:.4f} (delta {row['delta_homophily']:+.4f})"
      )
      print(f"  Edges removed: {row['pct_edges_removed']:.2f}%")
      print(
        f"  Cosine sim retained/removed: "
        f"{row['avg_cosine_sim_retained']:.4f} / {row['avg_cosine_sim_removed']:.4f}"
      )
      if original_result and refined_result:
        print(
          f"  Test Acc: {row['baseline_test_acc']:.4f} -> "
          f"{row['refined_test_acc']:.4f} (delta {row['delta_test_acc']:+.4f})"
        )

    return {
      'row': row,
      'comparison': comparison,
      'refined_dataset': refined_dataset,
    }

  def run_from_pipeline_results(
    self,
    dataset_name: str,
    cora_dataset,
    data_by_key: dict,
    threshold_by_method: dict,
    original_results: dict,
    refined_results: dict,
    refined_datasets: list | None = None,
    node_subset=None,
    save_outputs: bool = True,
  ) -> dict:
    """Analyze using outputs from DIVERGEWithSemanticEnhancer.run()."""
    rows = []
    refined_by_title = {}
    if refined_datasets is not None:
      for (key, title), refined_dataset in zip(self.INIT_METHODS, refined_datasets):
        refined_by_title[title] = refined_dataset

    for key, title in self.INIT_METHODS:
      embeddings = get_embedding_from_data(data_by_key[key])
      result = self.analyze_method(
        dataset=cora_dataset,
        embeddings=embeddings,
        threshold=threshold_by_method[title],
        node_subset=node_subset,
        init_method=title,
        dataset_name=dataset_name,
        original_result=original_results.get(title),
        refined_result=refined_results.get(title),
        refined_dataset=refined_by_title.get(title),
        verbose=True,
      )
      rows.append(result['row'])

    return self._finalize_outputs(
      dataset_name=dataset_name,
      rows=rows,
      save_outputs=save_outputs,
    )

  def run_standalone(
    self,
    dataset_name: str,
    llm_name: str = 'llama_3.2_1B',
    peft_type: str = 'lora',
    retrained_with_gnn_mistakes: bool = True,
    supervised: bool = True,
    use_all_nodes: bool = True,
    threshold_by_method: dict | None = None,
    default_threshold: float = 0.5,
    original_results: dict | None = None,
    refined_results: dict | None = None,
    save_outputs: bool = True,
  ) -> dict:
    """
    Metrics-only standalone run (no GNN training).

    Uses saved thresholds when available; otherwise ``threshold_by_method`` or
  ``default_threshold``.
    """
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
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

    if threshold_by_method is None:
      threshold_by_method = self._load_thresholds_or_default(
        dataset_name=dataset_name,
        supervised=supervised,
        default_threshold=default_threshold,
      )

    node_subset = None
    if not use_all_nodes:
      node_subset = list(range(cora_dataset.num_nodes))

    rows = []
    for key, title in self.INIT_METHODS:
      embeddings = get_embedding_from_data(data_by_key[key])
      threshold = threshold_by_method.get(title, default_threshold)
      result = self.analyze_method(
        dataset=cora_dataset,
        embeddings=embeddings,
        threshold=threshold,
        node_subset=node_subset,
        init_method=title,
        dataset_name=dataset_name,
        original_result=(original_results or {}).get(title),
        refined_result=(refined_results or {}).get(title),
        verbose=True,
      )
      rows.append(result['row'])

    return self._finalize_outputs(
      dataset_name=dataset_name,
      rows=rows,
      save_outputs=save_outputs,
    )

  def _load_thresholds_or_default(
    self,
    dataset_name: str,
    supervised: bool,
    default_threshold: float,
  ) -> dict:
    from cosine_similarity_enhancer.diverge_with_cosine_similarity_enahncer import (
      DIVERGEWithSemanticEnhancer,
    )

    try:
      saved = DIVERGEWithSemanticEnhancer.load_best_hyperparameters(
        dataset_name,
        supervised=supervised,
      )
      thresholds = saved.get('similarity_thresholds', {})
      if thresholds:
        print(f"Loaded saved thresholds from {dataset_name} hyperparameters.")
        return thresholds
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
      pass

    print(
      f"No saved thresholds for {dataset_name}; "
      f"using default threshold {default_threshold:.2f} for all methods."
    )
    return {title: default_threshold for _, title in self.INIT_METHODS}

  def _finalize_outputs(
    self,
    dataset_name: str,
    rows: list[dict],
    save_outputs: bool,
  ) -> dict:
    summary = self._build_summary(rows)
    output_paths = {}
    if save_outputs:
      output_paths = self.save_results(dataset_name, rows, summary)
      self.plot_all_figures(dataset_name, rows, summary, output_paths['figures_dir'])

    return {
      'rows': rows,
      'summary': summary,
      'output_paths': output_paths,
    }

  @staticmethod
  def _build_summary(rows: list[dict]) -> dict:
    if not rows:
      return {}

    delta_hom = [r['delta_homophily'] for r in rows]
    delta_acc = [r['delta_test_acc'] for r in rows if not np.isnan(r['delta_test_acc'])]
    pct_removed = [r['pct_edges_removed'] for r in rows]

    pearson_r = float('nan')
    pearson_p = float('nan')
    if len(delta_hom) >= 2 and len(delta_acc) >= 2 and len(delta_hom) == len(delta_acc):
      if np.std(delta_hom) > 0 and np.std(delta_acc) > 0:
        pearson_r, pearson_p = pearsonr(delta_hom, delta_acc)

    removed_vs_acc_r = float('nan')
    removed_vs_acc_p = float('nan')
    valid_acc = [
      (r['pct_edges_removed'], r['delta_test_acc'])
      for r in rows
      if not np.isnan(r['delta_test_acc'])
    ]
    if len(valid_acc) >= 2:
      removed_vals, acc_vals = zip(*valid_acc)
      if np.std(removed_vals) > 0 and np.std(acc_vals) > 0:
        removed_vs_acc_r, removed_vs_acc_p = pearsonr(removed_vals, acc_vals)

    return {
      'num_methods': len(rows),
      'mean_delta_homophily': float(np.mean(delta_hom)),
      'mean_pct_edges_removed': float(np.mean(pct_removed)),
      'mean_delta_test_acc': float(np.mean(delta_acc)) if delta_acc else float('nan'),
      'pearson_r_delta_hom_vs_delta_acc': pearson_r,
      'pearson_p_delta_hom_vs_delta_acc': pearson_p,
      'pearson_r_removed_vs_delta_acc': removed_vs_acc_r,
      'pearson_p_removed_vs_delta_acc': removed_vs_acc_p,
    }

  def save_results(
    self,
    dataset_name: str,
    rows: list[dict],
    summary: dict,
  ) -> dict:
    results_dir = self.get_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = results_dir / f'homophily_structural_{dataset_name}_{timestamp}'
    figures_dir = run_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / f'homophily_structural_metrics_{dataset_name}.csv'
    summary_csv = run_dir / f'homophily_structural_summary_{dataset_name}.csv'
    summary_json = run_dir / f'homophily_structural_summary_{dataset_name}.json'

    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
      writer.writeheader()
      writer.writerows(rows)

    summary_row = {'dataset': dataset_name, **summary}
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
      writer.writeheader()
      writer.writerow(summary_row)

    with open(summary_json, 'w', encoding='utf-8') as f:
      json.dump({'dataset': dataset_name, 'rows': rows, 'summary': summary}, f, indent=2)

    latest_metrics = results_dir / f'homophily_structural_metrics_{dataset_name}_latest.csv'
    latest_summary = results_dir / f'homophily_structural_summary_{dataset_name}_latest.csv'
    with open(latest_metrics, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
      writer.writeheader()
      writer.writerows(rows)
    with open(latest_summary, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
      writer.writeheader()
      writer.writerow(summary_row)

    print(f"\nMetrics CSV: {metrics_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Latest copies: {latest_metrics}")

    return {
      'run_dir': run_dir,
      'figures_dir': figures_dir,
      'metrics_csv': metrics_csv,
      'summary_csv': summary_csv,
      'summary_json': summary_json,
      'latest_metrics_csv': latest_metrics,
      'latest_summary_csv': latest_summary,
    }

  def plot_all_figures(
    self,
    dataset_name: str,
    rows: list[dict],
    summary: dict,
    figures_dir: Path,
  ):
    self.plot_homophily_before_after(dataset_name, rows, figures_dir)
    self.plot_removed_rate_vs_delta_acc(dataset_name, rows, summary, figures_dir)
    self.plot_cosine_sim_retained_vs_removed(dataset_name, rows, figures_dir)
    self.plot_esnr_before_after(dataset_name, rows, figures_dir)
    self.plot_accuracy_comparison(dataset_name, rows, figures_dir)
    self.plot_paper_dashboard(dataset_name, rows, summary, figures_dir)

  @staticmethod
  def _save_figure(fig, figures_dir: Path, stem: str):
    for ext in ('png', 'pdf'):
      path = figures_dir / f'{stem}.{ext}'
      fig.savefig(path, bbox_inches='tight', facecolor='white')
      print(f"  Saved figure: {path}")
    plt.close(fig)

  def plot_homophily_before_after(self, dataset_name: str, rows: list[dict], figures_dir: Path):
    methods = [r['init_method'] for r in rows]
    before = [r['homophily_before'] for r in rows]
    after = [r['homophily_after'] for r in rows]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before, width, label='Before refinement', color=BEFORE_COLOR, edgecolor='white')
    ax.bar(x + width / 2, after, width, label='After refinement', color=AFTER_COLOR, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('Edge homophily ratio')
    ax.set_title(f'Edge homophily before vs. after refinement — {dataset_name}')
    ax.set_ylim(0, min(1.05, max(before + after) * 1.15 + 0.02))
    ax.legend(frameon=True, loc='lower right')
    ax.grid(axis='y', alpha=0.25)
    self._save_figure(fig, figures_dir, f'homophily_before_after_{dataset_name}')

  def plot_removed_rate_vs_delta_acc(
    self,
    dataset_name: str,
    rows: list[dict],
    summary: dict,
    figures_dir: Path,
  ):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    has_acc = any(not np.isnan(r['delta_test_acc']) for r in rows)

    for row in rows:
      if np.isnan(row['delta_test_acc']):
        continue
      ax.scatter(
        row['pct_edges_removed'],
        row['delta_test_acc'] * 100,
        s=90,
        color=NEUTRAL_COLOR,
        edgecolors='white',
        linewidths=0.8,
        zorder=3,
      )
      ax.annotate(
        row['init_method'],
        (row['pct_edges_removed'], row['delta_test_acc'] * 100),
        textcoords='offset points',
        xytext=(6, 4),
        fontsize=9,
      )

    ax.axhline(0, color='#888888', linestyle='--', linewidth=1, alpha=0.8)
    ax.set_xlabel('Edges removed (%)')
    ax.set_ylabel('Δ test accuracy (pp)')
    title = f'Removed-edge rate vs. accuracy gain — {dataset_name}'
    if has_acc and not np.isnan(summary.get('pearson_r_removed_vs_delta_acc', float('nan'))):
      title += (
        f"\nPearson r = {summary['pearson_r_removed_vs_delta_acc']:.3f} "
        f"(p = {summary['pearson_p_removed_vs_delta_acc']:.3g})"
      )
    ax.set_title(title)
    ax.grid(alpha=0.25)
    self._save_figure(fig, figures_dir, f'removed_edge_rate_vs_delta_acc_{dataset_name}')

  def plot_cosine_sim_retained_vs_removed(self, dataset_name: str, rows: list[dict], figures_dir: Path):
    plot_rows = []
    for row in rows:
      plot_rows.append({
        'Init method': row['init_method'],
        'Edge group': 'Retained',
        'Cosine similarity': row['avg_cosine_sim_retained'],
      })
      if not np.isnan(row['avg_cosine_sim_removed']):
        plot_rows.append({
          'Init method': row['init_method'],
          'Edge group': 'Removed',
          'Cosine similarity': row['avg_cosine_sim_removed'],
        })

    methods = [r['init_method'] for r in rows]
    retained = [r['avg_cosine_sim_retained'] for r in rows]
    removed = [r['avg_cosine_sim_removed'] for r in rows]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, retained, width, label='Retained edges', color=AFTER_COLOR, edgecolor='white')
    ax.bar(x + width / 2, removed, width, label='Removed edges', color=ACCENT_COLOR, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('Mean cosine similarity')
    ax.set_title(f'Cosine similarity on retained vs. removed edges — {dataset_name}')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.25)
    self._save_figure(fig, figures_dir, f'cosine_sim_retained_vs_removed_{dataset_name}')

  def plot_esnr_before_after(self, dataset_name: str, rows: list[dict], figures_dir: Path):
    methods = [r['init_method'] for r in rows]
    before = [r['cosine_esnr_before'] for r in rows]
    after = [r['cosine_esnr_after'] for r in rows]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before, width, label='Before refinement', color=BEFORE_COLOR, edgecolor='white')
    ax.bar(x + width / 2, after, width, label='After refinement', color=AFTER_COLOR, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('Cosine ESNR')
    ax.set_title(f'Cosine ESNR before vs. after refinement — {dataset_name}')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.25)
    self._save_figure(fig, figures_dir, f'cosine_esnr_before_after_{dataset_name}')

  def plot_accuracy_comparison(self, dataset_name: str, rows: list[dict], figures_dir: Path):
    if all(np.isnan(r['baseline_test_acc']) for r in rows):
      return

    methods = [r['init_method'] for r in rows]
    baseline = [r['baseline_test_acc'] for r in rows]
    refined = [r['refined_test_acc'] for r in rows]
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline, width, label='Baseline (original graph)', color=BEFORE_COLOR, edgecolor='white')
    ax.bar(x + width / 2, refined, width, label='Refined graph', color=AFTER_COLOR, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('Test accuracy')
    ax.set_title(f'Baseline vs. refined test accuracy — {dataset_name}')
    ax.legend(frameon=True, loc='lower right')
    ax.grid(axis='y', alpha=0.25)
    self._save_figure(fig, figures_dir, f'baseline_vs_refined_accuracy_{dataset_name}')

  def plot_paper_dashboard(
    self,
    dataset_name: str,
    rows: list[dict],
    summary: dict,
    figures_dir: Path,
  ):
    """Combined 2x2 figure suitable for a Q1 paper."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    methods = [r['init_method'] for r in rows]
    x = np.arange(len(methods))
    width = 0.36

    ax = axes[0, 0]
    ax.bar(x - width / 2, [r['homophily_before'] for r in rows], width, label='Before', color=BEFORE_COLOR)
    ax.bar(x + width / 2, [r['homophily_after'] for r in rows], width, label='After', color=AFTER_COLOR)
    ax.set_title('(a) Edge homophily')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Homophily ratio')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)

    ax = axes[0, 1]
    ax.bar(x - width / 2, [r['avg_cosine_sim_retained'] for r in rows], width, label='Retained', color=AFTER_COLOR)
    ax.bar(x + width / 2, [r['avg_cosine_sim_removed'] for r in rows], width, label='Removed', color=ACCENT_COLOR)
    ax.set_title('(b) Mean cosine similarity')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Cosine similarity')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)

    ax = axes[1, 0]
    for row in rows:
      if np.isnan(row['delta_test_acc']):
        continue
      ax.scatter(
        row['pct_edges_removed'],
        row['delta_test_acc'] * 100,
        s=70,
        color=NEUTRAL_COLOR,
        edgecolors='white',
      )
      ax.annotate(row['init_method'], (row['pct_edges_removed'], row['delta_test_acc'] * 100),
                  fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.axhline(0, color='#888888', linestyle='--', linewidth=0.9)
    ax.set_title('(c) Removed-edge rate vs. Δacc')
    ax.set_xlabel('Edges removed (%)')
    ax.set_ylabel('Δ test accuracy (pp)')
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    if not all(np.isnan(r['baseline_test_acc']) for r in rows):
      ax.bar(x - width / 2, [r['baseline_test_acc'] for r in rows], width, label='Baseline', color=BEFORE_COLOR)
      ax.bar(x + width / 2, [r['refined_test_acc'] for r in rows], width, label='Refined', color=AFTER_COLOR)
      ax.set_title('(d) Test accuracy')
      ax.set_ylabel('Accuracy')
      ax.legend(fontsize=8)
    else:
      ax.bar(x, [r['delta_homophily'] for r in rows], color=NEUTRAL_COLOR)
      ax.set_title('(d) Δ homophily by init method')
      ax.set_ylabel('Δ homophily')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.2)

    mean_dh = summary.get('mean_delta_homophily', float('nan'))
    mean_da = summary.get('mean_delta_test_acc', float('nan'))
    fig.suptitle(
      f'Structural refinement analysis — {dataset_name}\n'
      f'Mean Δhomophily = {mean_dh:+.4f}, '
      f'Mean Δacc = {mean_da:+.4f}' if not np.isnan(mean_da) else
      f'Structural refinement analysis — {dataset_name}\n'
      f'Mean Δhomophily = {mean_dh:+.4f}',
      fontsize=13,
      fontweight='bold',
      y=1.02,
    )
    fig.tight_layout()
    self._save_figure(fig, figures_dir, f'structural_metrics_dashboard_{dataset_name}')
