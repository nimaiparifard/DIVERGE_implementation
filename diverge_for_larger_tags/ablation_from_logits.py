"""
M-ablation (number of fine-tuned LLMs in the DIVERGE ensemble) for large TAGs, from the
GNN logits saved by train_gnn_export_logits.py.

Same ranking / nested top-M selection / ensemble tuning / CSV schema as
experiments/number_of_finetuned_model_experiment_in_diverge_model.py (run_m_ablation),
without loading any embeddings or GNNs. Needs logits for all 5 approaches.
"""
import torch

from config import setup_finetuning_cfg
from gnns.gnn_mtrainer import SUPPORTED_GNN_MODELS, DEFAULT_GNN_MODEL
from experiments.diverge import APPROACHES, parse_ensemble_approaches
from experiments.number_of_finetuned_model_experiment_in_diverge_model import run_m_ablation
from report.reporter import ReportResults
from diverge_for_larger_tags import logits_store as store


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         ensemble_approaches='learnable', m_values=None, reporter_index=0,
         csv_path='results/ablation/num_finetuned_models_large_tags_results.csv', gnn_model_name=None,
         logits_root=store.DEFAULT_LOGITS_ROOT, tune_fn=None):
    """M-ablation for one (dataset, llm, gnn, split, seed) from saved logits.
    Args: as the original ablation's main, plus logits_root (see logits_store) and tune_fn
    (replacement ensemble tuner, see efficient_ensemble; None = tune_ensemble_hyperparameter)."""
    m_values = m_values or [2, 3, 4, 5]
    ensemble_approaches = parse_ensemble_approaches(ensemble_approaches)
    if any(m < 1 or m > len(APPROACHES) for m in m_values):
        raise ValueError(f"m_values must be within [1, {len(APPROACHES)}], got {m_values}")

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed
    gnn_tag = gnn_model_name or DEFAULT_GNN_MODEL
    logits_dir = store.run_dir(dataset_name, llm_name, peft_type, gnn_tag, supervised, resolved_seed, logits_root)

    print("=" * 80)
    print(f"ABLATION (large TAG, saved logits): Number of Finetuned LLMs (M) | dataset={dataset_name} "
          f"llm={llm_name} peft={peft_type} gnn={gnn_tag} seed={resolved_seed} supervised={supervised} M={m_values}")
    print(f"Logits: {logits_dir}")
    print("=" * 80)

    reporter = ReportResults(
        cfg,
        save_dir=f"results/train_info/{dataset_name}_num_finetuned_ablation_{llm_name}_{gnn_tag}_seed{resolved_seed}",
        index_run=reporter_index,
    )
    reporter.report_title("Ablation: Impact of Number of Finetuned LLMs (M) - from saved logits")
    for line in (f"Dataset: {dataset_name}", f"LLM: {llm_name}", f"PEFT Type: {peft_type}", f"GNN: {gnn_tag}",
                 f"Seed: {resolved_seed}", f"Supervised: {supervised}",
                 f"Ensemble Approaches: {', '.join(ensemble_approaches)}", f"M values tested: {m_values}",
                 f"Logits: {logits_dir}", ""):
        reporter.report_txt(line)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results, logits, _ = store.load_run(logits_dir, APPROACHES, device=device)

    run_info = {"llm_name": llm_name, "peft_type": peft_type, "gnn_model_name": gnn_tag,
                "seed": resolved_seed, "supervised": supervised}
    return run_m_ablation(dataset_name, results, m_values, ensemble_approaches, reporter, csv_path, run_info,
                          logits=logits, tune_fn=tune_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="M-ablation for large TAGs from the logits saved by train_gnn_export_logits.py")
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
    args = parser.parse_args()

    main(
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
