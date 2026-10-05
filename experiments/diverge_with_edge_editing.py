"""
DIVERGE pipeline WITH structural edge editing.
(experiments/diverge.py is the same pipeline without the edge-editing step.)

For one (dataset, llm, seed) combination:
  1) Load the seed-tagged LLM embedding cache for every LoRA-init approach
     (cache/cache_embedding_with_diffrent_init_weights.py must have been run first).
  2) Train one GNN per approach on the un-edited graph.
  3) Ensemble those GNNs (BEFORE edge editing).
  4) For each approach: find the GNN's mistakes, then use the LLM's own cached
     embeddings to remove low-cosine-similarity edges around those mistakes
     (searching over a similarity-threshold grid), and retrain.
  5) Ensemble the edited-graph GNNs (AFTER edge editing).
  6) Append one complete-info row (dataset, llm, seed, per-approach + ensemble
     metrics) to a CSV report.

This is the structural edge-editing approach originally prototyped in
experiments/edge_editing.py; this module reuses its helper functions but is a
leaner, --seed/--llm_name-parameterized entry point meant to be swept over in
scripts/*.bat.
"""
import os
from datetime import datetime

import numpy as np
import torch

from common.dataloader import load_graph_dataset_for_tape
from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import gnn_train_and_report, get_datasets_path, append_gnn_result_csv
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter, ENSEMBLE_WEIGHT_APPROACHES
from experiments.edge_editing import (
    load_encoder_model,
    get_precomputed_node_embeddings,
    get_best_similarity_threshold,
    print_modified_edge_result,
)
from visualize.visualize_gnn_mistakes import get_gnn_embedding_mistakes
from report.reporter import ReportResults
from train_llm.efficiency import ResourceMonitor

# Order must match dataset.data_utils.get_init_dataset_for_gnn's return tuple
APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
TITLES = {'pissa': 'PISSA', 'orthogonal': 'ORTHOGONAL', 'gaussian': 'GAUSSIAN', 'loftq': 'LOFTQ', 'eva': 'EVA'}


def _get_path_prefix():
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            return '.'
        return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'


def parse_ensemble_approaches(value):
    """'all' or a comma-separated list (or a list) of ENSEMBLE_WEIGHT_APPROACHES."""
    if isinstance(value, str):
        value = ENSEMBLE_WEIGHT_APPROACHES if value.strip().lower() == 'all' else \
            [v.strip() for v in value.split(',') if v.strip()]
    unknown = [v for v in value if v not in ENSEMBLE_WEIGHT_APPROACHES]
    if unknown:
        raise ValueError(f"Unknown ensemble approach(es) {unknown}; choose from {ENSEMBLE_WEIGHT_APPROACHES} or 'all'")
    return list(value)


def run_ensembles(stage, dataset_name, models, embeddings, ensemble_approaches, reporter, llm_name, peft_type,
                  supervised, seed):
    """Tune/evaluate every ensemble weight approach on the same trained GNNs.
    Returns {approach: (results, hyperparams)}; results is {} if that approach failed."""
    out = {}
    for ea in ensemble_approaches:
        print(f"\n>>> Ensemble {stage} edge editing | weight approach = {ea}")
        try:
            results, params, _ = tune_ensemble_hyperparameter(
                dataset_name, models, embeddings,
                does_report_training_process=False,
                ensemble_approaches=ea,
                title=f"DIVERGE {stage} Edge Editing | ensemble={ea} (seed={seed})",
                reporter=reporter,
                llm_name=llm_name, peft_type=peft_type,
                supervised=supervised, seed=seed,
                visualize_best=False,
            )
        except Exception as e:
            print(f"[Ensemble {stage} | {ea}] FAILED: {e}")
            reporter.report(f"Ensemble {stage} Edge Editing | {ea} FAILED", str(e))
            results, params = {}, {}
        out[ea] = (results, params)
        if results:
            print(f"[Ensemble {stage} | {ea}] test_acc={results['test']['accuracy']:.4f} "
                  f"test_f1={results['test']['macro_f1']:.4f} params={params}")
    return out


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         approaches=None, ensemble_approaches='learnable', similarity_threshold_steps=17,
         reporter_index=0, csv_path='results/diverge/diverge_results.csv'):
    """
    Run the full DIVERGE pipeline for one (dataset, llm, seed) combination.

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
              a list, or 'all' (= ENSEMBLE_WEIGHT_APPROACHES). The per-approach GNNs and
              edge editing run once; every ensemble approach is evaluated on the same models
              and gets its own CSV row (sharing run_id).
        similarity_threshold_steps: number of cosine-similarity thresholds to try for
              edge editing (default 17 vs. the original 42-step sweep — much faster
              for batch sweeps while still covering the same [0.1, 0.9] range).
        csv_path: where to append the complete-info result row ("" / None disables it).
    """
    approaches = approaches or APPROACHES
    ensemble_approaches = parse_ensemble_approaches(ensemble_approaches)
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    print("=" * 80)
    print(f"DIVERGE | dataset={dataset_name} llm={llm_name} peft={peft_type} "
          f"seed={resolved_seed} supervised={supervised} approaches={approaches} "
          f"ensemble_approaches={ensemble_approaches}")
    print("=" * 80)

    split_tag = "supervised" if supervised else "semi_supervised"
    run_id = f"{dataset_name}_{llm_name}_{split_tag}_seed{resolved_seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    reporter = ReportResults(cfg, save_dir=f"results/train_info/{dataset_name}_diverge_{split_tag}_seed{resolved_seed}",
                              index_run=reporter_index)
    reporter.report("DIVERGE Run", f"run_id: {run_id}\ndataset: {dataset_name}\nllm: {llm_name}\npeft: {peft_type}\n"
                                   f"seed: {resolved_seed}\nsplit: {split_tag}\napproaches: {', '.join(approaches)}\n"
                                   f"ensemble approaches: {', '.join(ensemble_approaches)}")
    run_monitor = ResourceMonitor().__enter__()  # total DIVERGE pipeline time / peak VRAM

    path_prefix = _get_path_prefix()
    re_split = 1 if supervised else 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_dataset, _, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=re_split,
                                                        path_prefix=path_prefix, seed=resolved_seed)
    cora_dataset = load_dataset(cfg)

    # 1) Load the seed-tagged embeddings for every requested init approach
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    emb = {name: get_embedding_from_data(all_data[name]) for name in approaches}
    print("Loaded embeddings: " + ", ".join(f"{name}={tuple(emb[name].shape)}" for name in approaches))

    # 2) Train one GNN per approach on the un-edited graph (each call also appends
    #    its own row to results/gnn_training/gnn_training_results.csv, see gnn_train_and_report)
    gnn_kwargs = dict(does_print_training_process=False, reporter=reporter, supervised=supervised,
                       seed=resolved_seed, llm_name=llm_name, peft_type=peft_type)
    results_before, models_before = {}, {}
    for name in approaches:
        results_before[name], models_before[name] = gnn_train_and_report(
            dataset_name, emb[name], title=TITLES[name], **gnn_kwargs)

    # 3) Ensemble BEFORE edge editing (every requested weight approach, same GNNs)
    ensembles_before = run_ensembles("Before", dataset_name, [models_before[n] for n in approaches],
                                     [emb[n] for n in approaches], ensemble_approaches, reporter,
                                     llm_name, peft_type, supervised, resolved_seed)

    # 4) Edge editing per approach: find GNN mistakes, remove low-similarity edges
    #    around them using the LLM's own cached embeddings, retrain, keep best threshold.
    tokenizer, encoder_model = load_encoder_model("intfloat/e5-large")
    threshold_range = np.linspace(0.1, 0.9, similarity_threshold_steps)
    results_after, models_after, best_thresholds, edit_stats, edit_cost = {}, {}, {}, {}, {}
    for name in approaches:
        # cost of the whole edge-editing search (mistakes + all thresholds, each retraining a GNN)
        with ResourceMonitor() as edit_monitor:
            node_embeddings = get_precomputed_node_embeddings(graph_dataset, emb[name])
            _, _, _, mistakes = get_gnn_embedding_mistakes(cora_dataset, models_before[name], emb[name], output_mask='all')
            best_acc, best_th, _, stats, best_results, best_model = get_best_similarity_threshold(
                mistakes, cora_dataset, node_embeddings, emb[name], tokenizer, encoder_model, device,
                cfg, sentence_transformer_used=False, reporter=reporter, title=TITLES[name],
                supervised=supervised, used_mistakes=True, seed=resolved_seed, llm_name=llm_name,
                threshold_range=threshold_range,
            )
        edit_cost[name] = edit_monitor.metrics()
        results_after[name] = best_results
        models_after[name] = best_model
        best_thresholds[name] = best_th
        edit_stats[name] = stats
        print_modified_edge_result(TITLES[name], results_before[name], best_results, reporter=reporter)

    # 5) Ensemble AFTER edge editing
    ensembles_after = run_ensembles("After", dataset_name, [models_after[n] for n in approaches],
                                    [emb[n] for n in approaches], ensemble_approaches, reporter,
                                    llm_name, peft_type, supervised, resolved_seed)

    # Summary table of all ensemble approaches (+ individual GNNs) in the report
    def _fmt(res, split, metric):
        return f"{res[split][metric]:.4f}" if res else "FAILED"

    lines = [f"{'model':<32}{'before acc':>12}{'before F1':>12}{'after acc':>12}{'after F1':>12}{'after val':>12}"]
    for ea in ensemble_approaches:
        rb, ra = ensembles_before[ea][0], ensembles_after[ea][0]
        lines.append(f"{'ensemble ' + ea:<32}{_fmt(rb, 'test', 'accuracy'):>12}{_fmt(rb, 'test', 'macro_f1'):>12}"
                     f"{_fmt(ra, 'test', 'accuracy'):>12}{_fmt(ra, 'test', 'macro_f1'):>12}"
                     f"{_fmt(ra, 'val', 'accuracy'):>12}")
    for name in approaches:
        lines.append(f"{'GNN ' + TITLES[name]:<32}{results_before[name]['test_acc']:>12.4f}"
                     f"{results_before[name]['test_f1']:>12.4f}{results_after[name]['test_acc']:>12.4f}"
                     f"{results_after[name]['test_f1']:>12.4f}{'':>12}")
    run_monitor.__exit__(None, None, None)
    run_cost = run_monitor.metrics()

    # Cost table: time (s) / peak VRAM (MB)
    def _c(value, fmt="{:.2f}"):
        return fmt.format(value) if value is not None else "n/a"

    lines += ["", "Cost (time in s, peak VRAM in MB)",
              f"{'step':<52}{'time':>12}{'peak VRAM':>12}{'+VRAM':>12}"]
    for name in approaches:
        r = results_before[name]
        lines.append(f"{'GNN training ' + TITLES[name]:<52}{_c(r.get('train_time_sec')):>12}"
                     f"{_c(r.get('peak_vram_mb')):>12}{_c(r.get('peak_vram_delta_mb')):>12}")
    for name in approaches:
        c = edit_cost[name]
        lines.append(f"{'edge editing ' + TITLES[name] + f' ({similarity_threshold_steps} thr)':<52}"
                     f"{_c(c['time_sec']):>12}{_c(c['peak_vram_mb']):>12}{_c(c['peak_vram_delta_mb']):>12}")
    for stage, ens in (("before", ensembles_before), ("after", ensembles_after)):
        for ea in ensemble_approaches:
            cost = ens[ea][0].get('cost', {}) if ens[ea][0] else {}
            lines.append(f"{f'ensemble {ea} ({stage}) tuning':<52}{_c(cost.get('tuning_time_sec')):>12}"
                         f"{_c(cost.get('tuning_peak_vram_mb')):>12}{_c(cost.get('tuning_peak_vram_delta_mb')):>12}")
            lines.append(f"{f'ensemble {ea} ({stage}) selected fit':<52}{_c(cost.get('best_fit_time_sec')):>12}"
                         f"{_c(cost.get('best_fit_peak_vram_mb')):>12}{_c(cost.get('best_fit_peak_vram_delta_mb')):>12}")
    lines.append(f"{'TOTAL DIVERGE run':<52}{_c(run_cost['time_sec']):>12}{_c(run_cost['peak_vram_mb']):>12}"
                 f"{_c(run_cost['peak_vram_delta_mb']):>12}")
    summary = "\n".join(lines)
    print("\n" + summary)
    reporter.report(f"DIVERGE Summary - all ensemble approaches ({run_id})", summary)

    # 6) Complete-info CSV rows: one row per ensemble approach, sharing run_id
    if csv_path:
        base_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "dataset_name": dataset_name,
            "llm_name": llm_name,
            "peft_type": peft_type,
            "seed": resolved_seed,
            "supervised": supervised,
            "approaches": ",".join(approaches),
            "total_run_time_sec": run_cost["time_sec"],
            "total_run_peak_vram_mb": run_cost["peak_vram_mb"],
        }
        for name in approaches:
            total_checked = edit_stats[name]["total_checked"]
            base_row[f"{name}_test_acc_before"] = results_before[name]["test_acc"]
            base_row[f"{name}_test_f1_before"] = results_before[name]["test_f1"]
            base_row[f"{name}_test_acc_after"] = results_after[name]["test_acc"]
            base_row[f"{name}_test_f1_after"] = results_after[name]["test_f1"]
            base_row[f"{name}_best_threshold"] = best_thresholds[name]
            base_row[f"{name}_edges_removed"] = edit_stats[name]["removed"]
            base_row[f"{name}_edges_removed_pct"] = (
                edit_stats[name]["removed"] / total_checked * 100.0 if total_checked else 0.0
            )
            base_row[f"{name}_gnn_train_time_sec"] = results_before[name].get("train_time_sec")
            base_row[f"{name}_gnn_epochs_run"] = results_before[name].get("epochs_run")
            base_row[f"{name}_gnn_peak_vram_mb"] = results_before[name].get("peak_vram_mb")
            base_row[f"{name}_gnn_peak_vram_delta_mb"] = results_before[name].get("peak_vram_delta_mb")
            base_row[f"{name}_edge_editing_time_sec"] = edit_cost[name]["time_sec"]
            base_row[f"{name}_edge_editing_peak_vram_mb"] = edit_cost[name]["peak_vram_mb"]
        for ea in ensemble_approaches:
            (rb, pb), (ra, pa) = ensembles_before[ea], ensembles_after[ea]
            row = dict(base_row)
            row["ensemble_weight_approach"] = ea
            row["ensemble_status"] = "OK" if rb and ra else "FAILED"
            for stage, res, params in (("before", rb, pb), ("after", ra, pa)):
                row[f"ensemble_{stage}_test_acc"] = res["test"]["accuracy"] if res else ""
                row[f"ensemble_{stage}_test_f1"] = res["test"]["macro_f1"] if res else ""
                row[f"ensemble_{stage}_test_weighted_f1"] = res["test"]["weighted_f1"] if res else ""
                row[f"ensemble_{stage}_val_acc"] = res["val"]["accuracy"] if res else ""
                row[f"ensemble_{stage}_val_f1"] = res["val"]["macro_f1"] if res else ""
                cost = res.get("cost", {}) if res else {}
                for key in ("tuning_time_sec", "tuning_peak_vram_mb", "tuning_peak_vram_delta_mb", "n_configs",
                            "avg_config_time_sec", "best_fit_time_sec", "best_fit_peak_vram_mb",
                            "best_fit_peak_vram_delta_mb"):
                    row[f"ensemble_{stage}_{key}"] = cost.get(key, "")
                row[f"ensemble_{stage}_hyperparams"] = ("n/a (uniform)" if ea == "uniform" else params) if res else ""
            append_gnn_result_csv(row, csv_path)
        print(f"[OK] {len(ensemble_approaches)} DIVERGE result row(s) (run_id={run_id}) appended to: {csv_path}")

    return {
        "results_before": results_before,
        "results_after": results_after,
        "ensembles_before": ensembles_before,
        "ensembles_after": ensembles_after,
        "best_thresholds": best_thresholds,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the DIVERGE pipeline WITH edge editing (per-approach GNN -> edge editing -> ensemble) "
                    "for one dataset/llm/seed")
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
    parser.add_argument('--threshold_steps', type=int, default=17,
                         help='Number of cosine-similarity thresholds to try during edge editing')
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/diverge/diverge_results.csv')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        approaches=[a.strip() for a in args.approaches.split(',') if a.strip()],
        ensemble_approaches=args.ensemble_approaches,
        similarity_threshold_steps=args.threshold_steps,
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
    )
