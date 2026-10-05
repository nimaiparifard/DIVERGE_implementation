"""Collect every trained LoRA adapter's results from artifacts/ into one CSV.

Scans artifacts/ for adapter folders (those containing adapter_config.json),
reads training_results.csv (+ efficiency_metrics.csv when present), tags each
row with metadata parsed from the folder name, and writes:

    results/adapters/<tag>_all_runs.csv   every row of every training_results.csv
    results/adapters/<tag>_latest.csv     last *valid* (non-NaN) row per adapter
    results/adapters/<tag>_summary.csv    mean/std test/val acc per (model, dataset, init, setting)

Usage:
    python scripts/collect_adapter_results.py                       # llama_3.2_3B only
    python scripts/collect_adapter_results.py --model all           # every model
    python scripts/collect_adapter_results.py --model gemma2_2B --dataset cora
    python scripts/collect_adapter_results.py --copy-adapters-to D:/llama3b_adapters
"""
import argparse
import csv
import os
import re
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
OUT_DIR = os.path.join(ROOT, "results", "adapters")

INIT_RE = re.compile(r"init(?:_type)?[_-]([a-z]+)")
SEED_RE = re.compile(r"_seed(\d+)")
SUFFIX_RE = re.compile(r"_(keywords|paraphrased|summary|augmented|retrained_mistakes)(?:_|$)")


def parse_dir_name(name):
    init = INIT_RE.search(name)
    seed = SEED_RE.search(name)
    suffix = SUFFIX_RE.search(name)
    return {
        "adapter_dir": name,
        "lora_init_from_dir": init.group(1) if init else "default",
        "seed": int(seed.group(1)) if seed else None,
        "semi_supervised": "semi_supervised" in name,
        "variant": suffix.group(1) if suffix else "",
    }


def find_adapters(model, dataset):
    for name in sorted(os.listdir(ARTIFACTS)):
        path = os.path.join(ARTIFACTS, name)
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            continue
        if model != "all" and not name.startswith(model + "_"):
            continue
        if dataset and f"_{dataset}_" not in name:
            continue
        yield name, path


def collect_headers():
    """Map column count -> header, from every training_results.csv in artifacts/."""
    headers = {}
    for name in os.listdir(ARTIFACTS):
        p = os.path.join(ARTIFACTS, name, "training_results.csv")
        if os.path.isfile(p):
            with open(p, newline="") as f:
                header = next(csv.reader(f), None)
            if header:
                headers.setdefault(len(header), header)
    return headers


HEADERS = None


def read_mixed_csv(path):
    """Some CSVs were appended to after the schema grew, so rows can be wider
    than the header. Match each row to a known header of the same width."""
    global HEADERS
    if HEADERS is None:
        HEADERS = collect_headers()
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    header, records = rows[0], []
    for r in rows[1:]:
        cols = header if len(r) == len(header) else HEADERS.get(len(r))
        if cols is None:
            print(f"  [unknown row width {len(r)}] {path}")
            continue
        records.append(dict(zip(cols, r)))
    df = pd.DataFrame(records)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        raw_present = ~df[col].str.strip().str.lower().isin(["", "nan", "none"])
        if converted.notna().sum() == raw_present.sum():
            df[col] = converted
    return df


def load_adapter(name, path):
    meta = parse_dir_name(name)
    meta["adapter_path"] = path
    res_csv = os.path.join(path, "training_results.csv")
    if not os.path.isfile(res_csv):
        print(f"  [no training_results.csv] {name}")
        return pd.DataFrame([meta])

    df = read_mixed_csv(res_csv)
    df["run_idx"] = range(len(df))
    df["is_valid"] = df["test_accuracy"].notna() & df["train_loss"].notna()

    eff_csv = os.path.join(path, "efficiency_metrics.csv")
    if os.path.isfile(eff_csv):
        eff = pd.read_csv(eff_csv).iloc[-1]
        for col, val in eff.items():
            if col not in df.columns:
                df[col] = val

    for k, v in meta.items():
        df[k] = v
    if "lora_init" not in df.columns:
        df["lora_init"] = meta["lora_init_from_dir"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma2_2B", help="model prefix, or 'all'")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--copy-adapters-to", default=None,
                    help="optionally copy the adapter folders into this directory")
    args = ap.parse_args()

    adapters = list(find_adapters(args.model, args.dataset))
    if not adapters:
        print(f"No adapters found for model={args.model} dataset={args.dataset}")
        return
    print(f"Found {len(adapters)} adapters:")
    for name, _ in adapters:
        print(f"  {name}")

    all_runs = pd.concat([load_adapter(n, p) for n, p in adapters], ignore_index=True, sort=False)

    front = ["adapter_dir", "model_name", "dataset_name", "lora_init", "semi_supervised", "seed",
             "variant", "run_idx", "is_valid", "timestamp", "train_accuracy", "val_accuracy",
             "test_accuracy", "train_loss", "val_loss", "test_loss"]
    front = [c for c in front if c in all_runs.columns]
    all_runs = all_runs[front + [c for c in all_runs.columns if c not in front]]

    valid = all_runs[all_runs.get("is_valid", False) == True]
    latest = valid.groupby("adapter_dir", as_index=False).tail(1)

    group_cols = [c for c in ["model_name", "dataset_name", "lora_init", "semi_supervised", "variant"]
                  if c in latest.columns]
    summary = (latest.groupby(group_cols, dropna=False)
               .agg(n_runs=("test_accuracy", "size"),
                    test_acc_mean=("test_accuracy", "mean"), test_acc_std=("test_accuracy", "std"),
                    val_acc_mean=("val_accuracy", "mean"), val_acc_std=("val_accuracy", "std"))
               .reset_index())

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model + (f"_{args.dataset}" if args.dataset else "")
    paths = {k: os.path.join(args.out_dir, f"{tag}_{k}.csv") for k in ("all_runs", "latest", "summary")}
    all_runs.to_csv(paths["all_runs"], index=False)
    latest.to_csv(paths["latest"], index=False)
    summary.to_csv(paths["summary"], index=False)

    invalid = all_runs[all_runs.get("is_valid", True) == False]
    if len(invalid):
        print(f"\nSkipped {len(invalid)} NaN/diverged row(s):")
        for _, r in invalid.iterrows():
            print(f"  {r['adapter_dir']} (run_idx={r['run_idx']}, {r.get('timestamp', '')})")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nLatest valid result per adapter:")
        print(latest[[c for c in ["adapter_dir", "lora_init", "val_accuracy", "test_accuracy"]
                      if c in latest.columns]].to_string(index=False))
        print("\nSummary:")
        print(summary.to_string(index=False))

    print("\nWrote:")
    for p in paths.values():
        print(f"  {p}")

    if args.copy_adapters_to:
        os.makedirs(args.copy_adapters_to, exist_ok=True)
        for name, path in adapters:
            dst = os.path.join(args.copy_adapters_to, name)
            shutil.copytree(path, dst, dirs_exist_ok=True)
        print(f"Copied {len(adapters)} adapter folders to {args.copy_adapters_to}")


if __name__ == "__main__":
    main()
