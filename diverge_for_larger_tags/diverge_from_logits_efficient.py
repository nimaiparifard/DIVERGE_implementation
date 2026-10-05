"""
diverge_from_logits.py with the memory-bounded ensemble tuner (see efficient_ensemble):
same arguments, report and CSV schema, plus --max_chunk_mb (activation budget per chunk).
"""
from functools import partial

from gnns.gnn_mtrainer import SUPPORTED_GNN_MODELS
from ensemble.ensemble_gnns_learning import ENSEMBLE_WEIGHT_APPROACHES
from experiments.diverge import APPROACHES
from diverge_for_larger_tags import logits_store as store
from diverge_for_larger_tags.diverge_from_logits import main as diverge_main
from diverge_for_larger_tags.efficient_ensemble import tune_ensemble_hyperparameter_efficient, DEFAULT_MAX_CHUNK_MB


def main(max_chunk_mb=DEFAULT_MAX_CHUNK_MB, **kwargs):
    """kwargs: as diverge_from_logits.main."""
    return diverge_main(tune_fn=partial(tune_ensemble_hyperparameter_efficient, max_chunk_mb=max_chunk_mb), **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 of DIVERGE for large TAGs (memory-efficient ensemble tuning) from saved logits")
    parser.add_argument('--dataset_name', type=str, default='computer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                         help=f"Comma-separated subset of {APPROACHES} to ensemble")
    parser.add_argument('--ensemble_approaches', type=str, default='all',
                         help=f"'all' or comma-separated subset of {ENSEMBLE_WEIGHT_APPROACHES}")
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/diverge/diverge_large_tags_results.csv')
    parser.add_argument('--gnn_model_name', type=str, default=None, choices=SUPPORTED_GNN_MODELS)
    parser.add_argument('--logits_root', type=str, default=store.DEFAULT_LOGITS_ROOT)
    parser.add_argument('--max_chunk_mb', type=float, default=DEFAULT_MAX_CHUNK_MB,
                         help='Approximate activation memory per training chunk of the ensemble heads (MB)')
    args = parser.parse_args()

    main(
        max_chunk_mb=args.max_chunk_mb,
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        approaches=[a.strip() for a in args.approaches.split(',') if a.strip()],
        ensemble_approaches=args.ensemble_approaches,
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
        gnn_model_name=args.gnn_model_name,
        logits_root=args.logits_root,
    )
