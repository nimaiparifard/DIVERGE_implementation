"""
Check that GNN training actually uses the hyperparameters tuned by gnns/tune_gnn_hyperparameter.py.

For every (dataset, llm, seed, gnn) combination:
  1. RESOLVE   the tuned file gnn_hyperparameters/<dataset>_<llm>_<gnn>[_semi_supervised]_seed<seed>.json
               must exist, and set_mgnn_cfg (what GNN training / DIVERGE call) must resolve to it
               with identical values -> OK / MISSING (reports the fallback used instead) / MISMATCH
  2. --retrain train a GNN through the normal GNNTrainer path (dataset reloaded from disk) with
               that cfg and check the val acc reproduces the tuned value -> REPRODUCED / DRIFT
  3. --check_training_csv
               scan results/gnn_training/gnn_training_results.csv (written by gnn_train_and_report)
               and report which hyperparameter file each past run of this combination used

Exit code 1 if any combination is MISSING / MISMATCH / DRIFT.

Example:
  python -m gnns.verify_gnn_hyperparameters --dataset_names cora --llm_names llama_3.2_1B smollm2_1.7B
         --seeds 42 43 --gnn_model_names SAGE GAT --re_split 0 --retrain --check_training_csv
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from gnns.gnn_mtrainer import (set_mgnn_cfg, GNNTrainer, append_gnn_result_csv, gnn_hyperparameter_filename,
                               get_gnn_hyperparameter_dir, SUPPORTED_GNN_MODELS, DEFAULT_GNN_MODEL)
from gnns.tune_gnn_hyperparameter import get_embedding_cache_path, REPO_ROOT, TUNING_RESULTS_DIR

# cfg attribute -> key in the tuned JSON
CHECKED_FIELDS = {
    "gnn_model_name": "gnn_model_name", "seed": "seed", "lr": "lr", "hidden_dim": "hidden_dim",
    "num_layers": "num_layers", "dropout": "dropout", "weight_decay": "weight_decay",
    "batch_norm": "batch_norm", "epochs": "epochs", "early_stop": "early_stop", "re_split": "re_split",
}
TRAINING_CSV = REPO_ROOT / "results" / "gnn_training" / "gnn_training_results.csv"


def _same(cfg_value, json_value):
    if isinstance(cfg_value, bool) or isinstance(json_value, bool):
        return bool(cfg_value) == bool(int(json_value))
    if isinstance(cfg_value, float) or isinstance(json_value, float):
        return abs(float(cfg_value) - float(json_value)) < 1e-12
    return str(cfg_value) == str(json_value)


def check_resolution(dataset_name, llm_name, seed, gnn_model_name, supervised):
    expected = get_gnn_hyperparameter_dir() / gnn_hyperparameter_filename(
        dataset_name, supervised, llm_name=llm_name, seed=seed, gnn_model_name=gnn_model_name)
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name,
                       gnn_model_name=gnn_model_name)
    row = {"expected_file": expected.name, "resolved_file": Path(cfg.hp_source).name
           if cfg.hp_source != "defaults" else "defaults"}

    if not expected.exists():
        row.update(status="MISSING", detail=f"not tuned; GNN training falls back to {row['resolved_file']}")
        return row, cfg, None

    with open(expected, 'r', encoding='utf-8') as f:
        tuned = json.load(f)
    if Path(cfg.hp_source).resolve() != expected.resolve():
        row.update(status="MISMATCH", detail=f"set_mgnn_cfg resolved {row['resolved_file']} instead")
        return row, cfg, tuned

    diffs = [f"{attr}: cfg={cfg[attr]} json={tuned[key]}" for attr, key in CHECKED_FIELDS.items()
             if key in tuned and not _same(cfg[attr], tuned[key])]
    if diffs:
        row.update(status="MISMATCH", detail="; ".join(diffs))
    else:
        row.update(status="OK", detail="set_mgnn_cfg uses the tuned hyperparameters")
    return row, cfg, tuned


def check_reproduction(cfg, tuned, dataset_name, llm_name, seed, supervised, tolerance):
    """Retrain through the normal GNNTrainer path and compare with the tuned base-seed val acc."""
    tuning = tuned.get("tuning", {})
    cache = tuning.get("embedding_cache")
    cache_path = (REPO_ROOT / cache) if cache else get_embedding_cache_path(
        dataset_name, llm_name=llm_name, peft_type=tuning.get("peft_type", "lora"),
        init_weight_approach=tuning.get("init_weight_approach", "pissa"), supervised=supervised, seed=seed)
    if not Path(cache_path).exists():
        return {"retrain_status": "SKIPPED", "retrain_detail": f"embedding cache not found: {cache_path}"}
    features = torch.load(cache_path, weights_only=False)['embeddings']

    trainer = GNNTrainer(cfg, features, track_history=False, verbose=False)
    results, _, _ = trainer.train()
    out = {"retrain_val_acc": round(results["val_acc"], 4), "retrain_test_acc": round(results["test_acc"], 4)}
    per_seed = tuning.get("val_acc_per_seed")
    if not per_seed:
        out.update(retrain_status="SKIPPED", retrain_detail="tuned file has no per-seed val acc to compare")
        return out
    expected_val = per_seed[0]  # eval seed 0 == cfg.seed
    diff = abs(results["val_acc"] - expected_val)
    out.update(tuned_val_acc=round(expected_val, 4),
               tuned_val_acc_mean=round(tuning.get("val_acc", expected_val), 4),
               retrain_status="REPRODUCED" if diff <= tolerance else "DRIFT",
               retrain_detail=f"|val diff|={diff:.4f} (tolerance {tolerance})")
    return out


def check_training_csv(dataset_name, llm_name, seed, gnn_model_name, supervised, expected_name):
    """Which hyperparameter files did past gnn_train_and_report runs of this combination use?"""
    if not TRAINING_CSV.exists():
        return {"csv_runs": 0, "csv_detail": f"{TRAINING_CSV.name} not found"}
    used = {}
    with open(TRAINING_CSV, 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r.get("dataset_name") != dataset_name or r.get("llm_name") != llm_name
                    or r.get("gnn_model_name", DEFAULT_GNN_MODEL) != gnn_model_name
                    or str(r.get("supervised")) != str(supervised)
                    or (seed is not None and r.get("embedding_seed", "") not in (str(seed), ""))):
                continue
            src = r.get("hp_source") or "unknown (run predates hp_source logging)"
            used[src] = used.get(src, 0) + 1
    n = sum(used.values())
    n_tuned = used.get(expected_name, 0)
    return {"csv_runs": n, "csv_runs_with_tuned_hp": n_tuned,
            "csv_detail": "; ".join(f"{k}: {v}" for k, v in used.items()) or "no runs yet"}


def main():
    parser = argparse.ArgumentParser(description="Verify GNN training uses the tuned hyperparameters")
    parser.add_argument('--dataset_names', nargs='+', default=['cora'])
    parser.add_argument('--llm_names', nargs='+', default=['llama_3.2_1B'])
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Seeds the hyperparameters were tuned for (omit for un-seeded files)')
    parser.add_argument('--gnn_model_names', nargs='+', default=[DEFAULT_GNN_MODEL], choices=SUPPORTED_GNN_MODELS)
    parser.add_argument('--re_split', type=int, default=0, help='0 semi-supervised, >0 supervised')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain each GNN and check the tuned val acc reproduces')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Allowed |val acc diff| (percentage points) for --retrain')
    parser.add_argument('--check_training_csv', action='store_true',
                        help=f'Report which hyperparameter file past runs in {TRAINING_CSV.name} used')
    parser.add_argument('--report_csv', type=str, default=str(TUNING_RESULTS_DIR / "verify_report.csv"))
    args = parser.parse_args()

    supervised = args.re_split > 0
    seeds = args.seeds or [None]
    rows, failed = [], 0
    for dataset_name in args.dataset_names:
        for llm_name in args.llm_names:
            for seed in seeds:
                for gnn in args.gnn_model_names:
                    print(f"\n--- {dataset_name} | {llm_name} | seed={seed} | {gnn} ---")
                    row, cfg, tuned = check_resolution(dataset_name, llm_name, seed, gnn, supervised)
                    if args.retrain and row["status"] == "OK":
                        row.update(check_reproduction(cfg, tuned, dataset_name, llm_name, seed, supervised,
                                                      args.tolerance))
                    if args.check_training_csv:
                        row.update(check_training_csv(dataset_name, llm_name, seed, gnn, supervised,
                                                      row["expected_file"]))
                    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "dataset_name": dataset_name, "llm_name": llm_name, "seed": seed,
                           "gnn_model_name": gnn, "supervised": supervised, **row}
                    rows.append(row)
                    append_gnn_result_csv(row, args.report_csv)
                    if row["status"] != "OK" or row.get("retrain_status") == "DRIFT":
                        failed += 1

    print(f"\n{'=' * 100}\nGNN hyperparameter verification ({len(rows)} combinations)\n{'=' * 100}")
    for r in rows:
        line = f"[{r['status']:<8}] {r['dataset_name']} | {r['llm_name']} | seed={r['seed']} | {r['gnn_model_name']}: {r['detail']}"
        if "retrain_status" in r:
            line += f"\n           retrain: {r['retrain_status']} {r.get('retrain_detail', '')}"
        if "csv_runs" in r:
            line += (f"\n           past GNN runs: {r.get('csv_runs_with_tuned_hp', 0)}/{r['csv_runs']} used the "
                     f"tuned file ({r['csv_detail']})")
        print(line)
    print(f"\nReport appended to: {args.report_csv}")
    if failed:
        print(f"{failed} combination(s) not using / not reproducing the tuned hyperparameters.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
