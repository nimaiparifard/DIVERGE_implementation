"""
Standalone runner for homophily / structural metrics experiment (#3).

Fast metrics-only mode (no GNN training) for validation before the full
DIVERGE pipeline. Run from repository root:

  python cosine_similarity_enhancer/experiments/run_homophily_structural_experiment.py --dataset_name cora
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from cosine_similarity_enhancer.experiments.homophily_structural_analysis import (
  HomophilyStructuralAnalysis,
)


def main():
  parser = argparse.ArgumentParser(
    description='Homophily / structural metrics before vs. after cosine refinement'
  )
  parser.add_argument('--dataset_name', type=str, default='cora')
  parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
  parser.add_argument('--peft_type', type=str, default='lora')
  parser.add_argument(
    '--retrained_with_gnn_mistakes',
    action='store_true',
    default=True,
  )
  parser.add_argument(
    '--no_retrained_with_gnn_mistakes',
    action='store_false',
    dest='retrained_with_gnn_mistakes',
  )
  parser.add_argument('--supervised', action='store_true', default=True)
  parser.add_argument('--semi_supervised', action='store_true', default=False)
  parser.add_argument('--use_all_nodes', action='store_true', default=True)
  parser.add_argument(
    '--default_threshold',
    type=float,
    default=0.5,
    help='Fallback threshold when no saved hyperparameters exist',
  )
  parser.add_argument(
    '--no_save',
    action='store_true',
    help='Skip CSV/figure export (dry run)',
  )
  args = parser.parse_args()

  supervised = not args.semi_supervised
  analysis = HomophilyStructuralAnalysis()
  results = analysis.run_standalone(
    dataset_name=args.dataset_name,
    llm_name=args.llm_name,
    peft_type=args.peft_type,
    retrained_with_gnn_mistakes=args.retrained_with_gnn_mistakes,
    supervised=supervised,
    use_all_nodes=args.use_all_nodes,
    default_threshold=args.default_threshold,
    save_outputs=not args.no_save,
  )

  print('\n' + '=' * 80)
  print('HOMOPHILY / STRUCTURAL ANALYSIS - SUMMARY')
  print('=' * 80)
  for row in results['rows']:
    print(
      f"{row['init_method']:12s} | "
      f"h: {row['homophily_before']:.4f}->{row['homophily_after']:.4f} "
      f"(d {row['delta_homophily']:+.4f}) | "
      f"removed: {row['pct_edges_removed']:.1f}%"
    )
  print('\nAggregate summary:', results['summary'])
  if results.get('output_paths'):
    print(f"\nResults directory: {results['output_paths']['run_dir']}")


if __name__ == '__main__':
  main()
