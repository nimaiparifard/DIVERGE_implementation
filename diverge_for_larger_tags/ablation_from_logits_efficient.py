"""
ablation_from_logits.py with the memory-bounded ensemble tuner (see efficient_ensemble):
same arguments, report and CSV schema, plus --max_chunk_mb (activation budget per chunk).
"""
from functools import partial

from gnns.gnn_mtrainer import SUPPORTED_GNN_MODELS
from diverge_for_larger_tags import logits_store as store
from diverge_for_larger_tags.ablation_from_logits import main as ablation_main
from diverge_for_larger_tags.efficient_ensemble import tune_ensemble_hyperparameter_efficient, DEFAULT_MAX_CHUNK_MB


def main(max_chunk_mb=DEFAULT_MAX_CHUNK_MB, **kwargs):
    """kwargs: as ablation_from_logits.main."""
    return ablation_main(tune_fn=partial(tune_ensemble_hyperparameter_efficient, max_chunk_mb=max_chunk_mb), **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="M-ablation for large TAGs (memory-efficient ensemble tuning) from saved logits")
    parser.add_argument('--dataset_name', type=str, default='computer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--ensemble_approaches', type=str, default='all',
                         help="'all' or comma-separated subset of the ensemble weight approaches")
    parser.add_argument('--m_values', type=str, default='2,3,4,5',
                         help='Comma-separated list of M values to ablate over (each in [1, 5])')
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/ablation/num_finetuned_models_large_tags_results.csv')
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
        ensemble_approaches=args.ensemble_approaches,
        m_values=[int(m.strip()) for m in args.m_values.split(',') if m.strip()],
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
        gnn_model_name=args.gnn_model_name,
        logits_root=args.logits_root,
    )
