"""
DIVERGE pipeline entry point (without edge editing).

For one (dataset, llm, seed) combination:
  1) Load the seed-tagged LLM embedding cache for every LoRA-init approach
     (cache/cache_embedding_with_diffrent_init_weights.py must have been run first).
  2) Train one GNN per approach.
  3) Ensemble those GNNs with every requested ensemble weight approach.
  4) Append one complete-info row per ensemble approach (dataset, llm, seed,
     per-approach + ensemble metrics and cost) to a CSV report.

The variant that additionally removes low-similarity edges around each GNN's
mistakes and retrains before ensembling is experiments/diverge_with_edge_editing.py.
"""
from datetime import datetime

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import gnn_train_and_report, append_gnn_result_csv, SUPPORTED_GNN_MODELS, DEFAULT_GNN_MODEL
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter, ENSEMBLE_WEIGHT_APPROACHES
from report.reporter import ReportResults
from train_llm.efficiency import ResourceMonitor

# Order must match dataset.data_utils.get_init_dataset_for_gnn's return tuple
APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
TITLES = {'pissa': 'PISSA', 'orthogonal': 'ORTHOGONAL', 'gaussian': 'GAUSSIAN', 'loftq': 'LOFTQ', 'eva': 'EVA'}


def parse_ensemble_approaches(value):
    """'all' or a comma-separated list (or a list) of ENSEMBLE_WEIGHT_APPROACHES."""
    if isinstance(value, str):
        value = ENSEMBLE_WEIGHT_APPROACHES if value.strip().lower() == 'all' else \
            [v.strip() for v in value.split(',') if v.strip()]
    unknown = [v for v in value if v not in ENSEMBLE_WEIGHT_APPROACHES]
    if unknown:
        raise ValueError(f"Unknown ensemble approach(es) {unknown}; choose from {ENSEMBLE_WEIGHT_APPROACHES} or 'all'")
    return list(value)


def run_ensembles(dataset_name, models, embeddings, ensemble_approaches, reporter, llm_name, peft_type,
                  supervised, seed, predictions_list=None, tune_fn=None):
    """Tune/evaluate every ensemble weight approach on the same trained GNNs.
    predictions_list: precomputed per-GNN logits; when given, models/embeddings may be None
    (see diverge_for_larger_tags/).
    tune_fn: replacement for tune_ensemble_hyperparameter (same signature); None keeps it.
    Returns {approach: (results, hyperparams)}; results is {} if that approach failed."""
    out = {}
    for ea in ensemble_approaches:
        print(f"\n>>> Ensemble | weight approach = {ea}")
        try:
            results, params, _ = (tune_fn or tune_ensemble_hyperparameter)(
                dataset_name, models, embeddings,
                does_report_training_process=False,
                ensemble_approaches=ea,
                title=f"DIVERGE | ensemble={ea} (seed={seed})",
                reporter=reporter,
                llm_name=llm_name, peft_type=peft_type,
                supervised=supervised, seed=seed,
                visualize_best=False,
                predictions_list=predictions_list,
            )
        except Exception as e:
            print(f"[Ensemble | {ea}] FAILED: {e}")
            reporter.report(f"Ensemble | {ea} FAILED", str(e))
            results, params = {}, {}
        out[ea] = (results, params)
        if results:
            print(f"[Ensemble | {ea}] test_acc={results['test']['accuracy']:.4f} "
                  f"test_f1={results['test']['macro_f1']:.4f} params={params}")
    return out


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         approaches=None, ensemble_approaches='learnable', reporter_index=0,
         csv_path='results/diverge/diverge_no_edge_editing_results.csv', gnn_model_name=None):
    """
    Run the DIVERGE pipeline (no edge editing) for one (dataset, llm, seed) combination.

    Args:
        seed: Seed the LoRA adapters / embedding caches were trained with (must match
              --seed used in train_llm/train_other_lora_init_apporaches.py and
              cache/cache_embedding_with_diffrent_init_weights.py). Also used as the
              GNN's own training seed. If None, uses configs/dataset/<name>.json's seed.
        supervised: True for the supervised embedding cache/GNN split, False for the
              semi-supervised (seed-tagged) one.
        approaches: subset of ['pissa','orthogonal','gaussian','loftq','eva'] to run
              (default: all 5). Trims runtime when sweeping many seeds/LLMs.
        ensemble_approaches: ensemble weight approach(es): a name, comma-separated names,
              a list, or 'all' (= ENSEMBLE_WEIGHT_APPROACHES). The per-approach GNNs are
              trained once; every ensemble approach is evaluated on the same models and
              gets its own CSV row (sharing run_id).
        csv_path: where to append the complete-info result row ("" / None disables it).
        gnn_model_name: GNN encoder for every per-approach GNN (one of SUPPORTED_GNN_MODELS);
              None keeps the encoder from the tuned hyperparameter JSON (SAGE by default).
    """
    approaches = approaches or APPROACHES
    gnn_tag = gnn_model_name or DEFAULT_GNN_MODEL
    ensemble_approaches = parse_ensemble_approaches(ensemble_approaches)
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    print("=" * 80)
    print(f"DIVERGE | dataset={dataset_name} llm={llm_name} peft={peft_type} "
          f"seed={resolved_seed} supervised={supervised} gnn={gnn_tag} approaches={approaches} "
          f"ensemble_approaches={ensemble_approaches}")
    print("=" * 80)

    split_tag = "supervised" if supervised else "semi_supervised"
    run_id = f"{dataset_name}_{llm_name}_{gnn_tag}_{split_tag}_seed{resolved_seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    reporter = ReportResults(cfg, save_dir=f"results/train_info/{dataset_name}_diverge_{llm_name}_{gnn_tag}_{split_tag}_seed{resolved_seed}",
                              index_run=reporter_index)
    reporter.report("DIVERGE Run", f"run_id: {run_id}\ndataset: {dataset_name}\nllm: {llm_name}\npeft: {peft_type}\n"
                                   f"seed: {resolved_seed}\nsplit: {split_tag}\napproaches: {', '.join(approaches)}\n"
                                   f"ensemble approaches: {', '.join(ensemble_approaches)}")
    run_monitor = ResourceMonitor().__enter__()  # total DIVERGE pipeline time / peak VRAM

    # 1) Load the seed-tagged embeddings for every requested init approach
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    emb = {name: get_embedding_from_data(all_data[name]) for name in approaches}
    print("Loaded embeddings: " + ", ".join(f"{name}={tuple(emb[name].shape)}" for name in approaches))

    # 2) Train one GNN per approach (each call also appends its own row to
    #    results/gnn_training/gnn_training_results.csv, see gnn_train_and_report)
    gnn_kwargs = dict(does_print_training_process=False, reporter=reporter, supervised=supervised,
                       seed=resolved_seed, llm_name=llm_name, peft_type=peft_type, gnn_model_name=gnn_model_name)
    results, models = {}, {}
    for name in approaches:
        results[name], models[name] = gnn_train_and_report(dataset_name, emb[name], title=TITLES[name], **gnn_kwargs)

    # 3) Ensemble (every requested weight approach, same GNNs)
    ensembles = run_ensembles(dataset_name, [models[n] for n in approaches], [emb[n] for n in approaches],
                              ensemble_approaches, reporter, llm_name, peft_type, supervised, resolved_seed)
    run_monitor.__exit__(None, None, None)

    # 4) Summary/cost report + complete-info CSV rows
    run_info = {"dataset_name": dataset_name, "llm_name": llm_name, "peft_type": peft_type,
                "seed": resolved_seed, "supervised": supervised, "gnn_model_name": gnn_tag}
    write_diverge_report(run_id, run_info, approaches, ensemble_approaches, results, ensembles,
                         run_monitor.metrics(), reporter, csv_path)

    return {
        "results": results,
        "ensembles": ensembles,
    }


def write_diverge_report(run_id, run_info, approaches, ensemble_approaches, results, ensembles, run_cost,
                         reporter, csv_path, extra_row=None):
    """
    Report the summary + cost tables and append one complete-info CSV row per ensemble
    approach (sharing run_id).

    Args:
        run_info: dataset_name, llm_name, peft_type, seed, supervised, gnn_model_name.
        results: {approach: per-GNN results dict from gnn_train_and_report}.
        ensembles: {ensemble approach: (results, hyperparams)} from run_ensembles.
        run_cost: {'time_sec', 'peak_vram_mb', 'peak_vram_delta_mb'} of the whole pipeline.
        extra_row: optional extra columns appended to every CSV row.
    """
    def _fmt(res, split, metric):
        return f"{res[split][metric]:.4f}" if res else "FAILED"

    lines = [f"{'model':<32}{'test acc':>12}{'test F1':>12}{'val acc':>12}"]
    for ea in ensemble_approaches:
        r = ensembles[ea][0]
        lines.append(f"{'ensemble ' + ea:<32}{_fmt(r, 'test', 'accuracy'):>12}{_fmt(r, 'test', 'macro_f1'):>12}"
                     f"{_fmt(r, 'val', 'accuracy'):>12}")
    for name in approaches:
        lines.append(f"{'GNN ' + TITLES[name]:<32}{results[name]['test_acc']:>12.4f}{results[name]['test_f1']:>12.4f}"
                     f"{results[name]['val_acc']:>12.4f}")

    # Cost table: time (s) / peak VRAM (MB)
    def _c(value, fmt="{:.2f}"):
        return fmt.format(value) if value is not None else "n/a"

    lines += ["", "Cost (time in s, peak VRAM in MB)",
              f"{'step':<52}{'time':>12}{'peak VRAM':>12}{'+VRAM':>12}"]
    for name in approaches:
        r = results[name]
        lines.append(f"{'GNN training ' + TITLES[name]:<52}{_c(r.get('train_time_sec')):>12}"
                     f"{_c(r.get('peak_vram_mb')):>12}{_c(r.get('peak_vram_delta_mb')):>12}")
    for ea in ensemble_approaches:
        cost = ensembles[ea][0].get('cost', {}) if ensembles[ea][0] else {}
        lines.append(f"{f'ensemble {ea} tuning':<52}{_c(cost.get('tuning_time_sec')):>12}"
                     f"{_c(cost.get('tuning_peak_vram_mb')):>12}{_c(cost.get('tuning_peak_vram_delta_mb')):>12}")
        lines.append(f"{f'ensemble {ea} selected fit':<52}{_c(cost.get('best_fit_time_sec')):>12}"
                     f"{_c(cost.get('best_fit_peak_vram_mb')):>12}{_c(cost.get('best_fit_peak_vram_delta_mb')):>12}")
    lines.append(f"{'TOTAL DIVERGE run':<52}{_c(run_cost['time_sec']):>12}{_c(run_cost['peak_vram_mb']):>12}"
                 f"{_c(run_cost['peak_vram_delta_mb']):>12}")
    summary = "\n".join(lines)
    print("\n" + summary)
    reporter.report(f"DIVERGE Summary - all ensemble approaches ({run_id})", summary)

    # Complete-info CSV rows: one row per ensemble approach, sharing run_id
    if csv_path:
        base_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "dataset_name": run_info["dataset_name"],
            "llm_name": run_info["llm_name"],
            "peft_type": run_info["peft_type"],
            "seed": run_info["seed"],
            "supervised": run_info["supervised"],
            "gnn_model_name": run_info["gnn_model_name"],
            "approaches": ",".join(approaches),
            "total_run_time_sec": run_cost["time_sec"],
            "total_run_peak_vram_mb": run_cost["peak_vram_mb"],
        }
        for name in approaches:
            base_row[f"{name}_test_acc"] = results[name]["test_acc"]
            base_row[f"{name}_test_f1"] = results[name]["test_f1"]
            base_row[f"{name}_val_acc"] = results[name]["val_acc"]
            base_row[f"{name}_gnn_train_time_sec"] = results[name].get("train_time_sec")
            base_row[f"{name}_gnn_epochs_run"] = results[name].get("epochs_run")
            base_row[f"{name}_gnn_peak_vram_mb"] = results[name].get("peak_vram_mb")
            base_row[f"{name}_gnn_peak_vram_delta_mb"] = results[name].get("peak_vram_delta_mb")
        for ea in ensemble_approaches:
            res, params = ensembles[ea]
            row = dict(base_row)
            row["ensemble_weight_approach"] = ea
            row["ensemble_status"] = "OK" if res else "FAILED"
            row["ensemble_test_acc"] = res["test"]["accuracy"] if res else ""
            row["ensemble_test_f1"] = res["test"]["macro_f1"] if res else ""
            row["ensemble_test_weighted_f1"] = res["test"]["weighted_f1"] if res else ""
            row["ensemble_val_acc"] = res["val"]["accuracy"] if res else ""
            row["ensemble_val_f1"] = res["val"]["macro_f1"] if res else ""
            cost = res.get("cost", {}) if res else {}
            for key in ("tuning_time_sec", "tuning_peak_vram_mb", "tuning_peak_vram_delta_mb", "n_configs",
                        "avg_config_time_sec", "best_fit_time_sec", "best_fit_peak_vram_mb",
                        "best_fit_peak_vram_delta_mb"):
                row[f"ensemble_{key}"] = cost.get(key, "")
            row["ensemble_hyperparams"] = ("n/a (uniform)" if ea == "uniform" else params) if res else ""
            row.update(extra_row or {})
            append_gnn_result_csv(row, csv_path)
        print(f"[OK] {len(ensemble_approaches)} DIVERGE result row(s) (run_id={run_id}) appended to: {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the DIVERGE pipeline (per-approach GNN -> ensemble, no edge editing) for one "
                    "dataset/llm/seed. For the edge-editing variant use experiments/diverge_with_edge_editing.py")
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches (must match --seed used in '
                              'train_llm/train_other_lora_init_apporaches.py and the caching step)')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                         help=f"Comma-separated subset of {APPROACHES} to run")
    parser.add_argument('--ensemble_approaches', type=str, default='all',
                         help=f"'all' or comma-separated subset of {ENSEMBLE_WEIGHT_APPROACHES}")
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/diverge/diverge_no_edge_editing_results.csv')
    parser.add_argument('--gnn_model_name', type=str, default=None, choices=SUPPORTED_GNN_MODELS,
                         help='GNN encoder for every per-approach GNN (default: from the tuned hyperparameter JSON)')
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
    )
