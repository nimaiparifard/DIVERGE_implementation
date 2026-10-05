"""
Stage 2 of DIVERGE for large TAGs: ensemble the GNN logits saved by
train_gnn_export_logits.py, without loading any embeddings or GNNs.

Same ensemble tuning (run_ensembles), report and CSV schema as experiments/diverge.py, so
rows are directly comparable; extra columns mark the logits pipeline. total_run_time_sec /
total_run_peak_vram_mb cover both stages (sum of the per-approach export stages + this
ensemble stage; peak = max over them).
"""
from datetime import datetime

import torch

from config import setup_finetuning_cfg
from gnns.gnn_mtrainer import SUPPORTED_GNN_MODELS, DEFAULT_GNN_MODEL
from ensemble.ensemble_gnns_learning import ENSEMBLE_WEIGHT_APPROACHES
from experiments.diverge import APPROACHES, parse_ensemble_approaches, run_ensembles, write_diverge_report
from report.reporter import ReportResults
from train_llm.efficiency import ResourceMonitor
from diverge_for_larger_tags import logits_store as store


def combined_cost(metas, ensemble_cost):
    """Whole two-stage pipeline cost: summed time, max peak VRAM."""
    costs = [m["stage_cost"] for m in metas.values()] + [ensemble_cost]
    peaks = [c["peak_vram_mb"] for c in costs if c.get("peak_vram_mb") is not None]
    deltas = [c["peak_vram_delta_mb"] for c in costs if c.get("peak_vram_delta_mb") is not None]
    return {
        "time_sec": round(sum(c["time_sec"] for c in costs), 3),
        "peak_vram_mb": max(peaks) if peaks else None,
        "peak_vram_delta_mb": max(deltas) if deltas else None,
    }


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         approaches=None, ensemble_approaches='learnable', reporter_index=0,
         csv_path='results/diverge/diverge_large_tags_results.csv', gnn_model_name=None,
         logits_root=store.DEFAULT_LOGITS_ROOT, tune_fn=None):
    """DIVERGE ensembles for one (dataset, llm, gnn, split, seed) from saved logits.
    Args: as experiments/diverge.main, plus logits_root (see logits_store) and tune_fn
    (replacement ensemble tuner, see efficient_ensemble; None = tune_ensemble_hyperparameter)."""
    approaches = approaches or APPROACHES
    gnn_tag = gnn_model_name or DEFAULT_GNN_MODEL
    ensemble_approaches = parse_ensemble_approaches(ensemble_approaches)
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed
    logits_dir = store.run_dir(dataset_name, llm_name, peft_type, gnn_tag, supervised, resolved_seed, logits_root)

    print("=" * 80)
    print(f"DIVERGE (large TAG) | stage 2: ensemble saved logits | dataset={dataset_name} llm={llm_name} "
          f"peft={peft_type} seed={resolved_seed} supervised={supervised} gnn={gnn_tag} approaches={approaches} "
          f"ensemble_approaches={ensemble_approaches}")
    print(f"Logits: {logits_dir}")
    print("=" * 80)

    split = store.split_tag(supervised)
    run_id = f"{dataset_name}_{llm_name}_{gnn_tag}_{split}_seed{resolved_seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    reporter = ReportResults(cfg, save_dir=f"results/train_info/{dataset_name}_diverge_{llm_name}_{gnn_tag}_{split}_seed{resolved_seed}",
                              index_run=reporter_index)
    reporter.report("DIVERGE Run (from saved logits)",
                    f"run_id: {run_id}\ndataset: {dataset_name}\nllm: {llm_name}\npeft: {peft_type}\n"
                    f"seed: {resolved_seed}\nsplit: {split}\napproaches: {', '.join(approaches)}\n"
                    f"ensemble approaches: {', '.join(ensemble_approaches)}\nlogits: {logits_dir}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with ResourceMonitor() as mon:
        results, logits, metas = store.load_run(logits_dir, approaches, device=device)
        print("Loaded logits: " + ", ".join(f"{n}={tuple(logits[n].shape)}" for n in approaches))
        ensembles = run_ensembles(dataset_name, None, None, ensemble_approaches, reporter, llm_name, peft_type,
                                  supervised, resolved_seed, predictions_list=[logits[n] for n in approaches],
                                  tune_fn=tune_fn)
    ensemble_cost = mon.metrics()

    run_info = {"dataset_name": dataset_name, "llm_name": llm_name, "peft_type": peft_type,
                "seed": resolved_seed, "supervised": supervised, "gnn_model_name": gnn_tag}
    extra_row = {
        "pipeline": "large_tag_logits",
        "logits_dir": logits_dir,
        "ensemble_stage_time_sec": ensemble_cost["time_sec"],
        "ensemble_stage_peak_vram_mb": ensemble_cost["peak_vram_mb"],
        "gnn_hp_source": metas[approaches[0]]["fingerprint"]["gnn_cfg"]["hp_source"],
    }
    write_diverge_report(run_id, run_info, approaches, ensemble_approaches, results, ensembles,
                         combined_cost(metas, ensemble_cost), reporter, csv_path, extra_row=extra_row)

    failed = [ea for ea in ensemble_approaches if not ensembles[ea][0]]
    if failed:
        raise RuntimeError(f"Ensemble failed for: {failed}")
    return {"results": results, "ensembles": ensembles}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 of DIVERGE for large TAGs: ensemble the logits saved by train_gnn_export_logits.py")
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
    args = parser.parse_args()

    main(
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
