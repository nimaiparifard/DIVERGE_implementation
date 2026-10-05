"""
Estimate how many train_llm fine-tuning jobs can run at the same time on a
single GPU without running out of VRAM.

Uses the peak VRAM actually measured per model in
results/efficiency/lora_finetune_efficiency.csv (written by
train_llm/efficiency.py). Falls back to a manually supplied value with
--peak-vram-gb if a model has no logged runs yet.

Usage:
    python scripts/estimate_parallel_jobs.py
    python scripts/estimate_parallel_jobs.py --total-vram-gb 12 --model-name llama_3.2_1B
    python scripts/estimate_parallel_jobs.py --model-name mobilellm_600M --peak-vram-gb 2.1
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

DEFAULT_CSV = os.path.join("results", "efficiency", "lora_finetune_efficiency.csv")

# CUDA context + driver overhead each additional process pays, roughly
# independent of model size (empirically ~300-600MB on recent driver/torch
# versions). Applied once per job on top of its measured peak_vram_gb.
DEFAULT_PER_PROCESS_OVERHEAD_GB = 0.5

# Fraction of total VRAM we're willing to plan against. Keeps a buffer for
# allocator fragmentation and the fact that "peak" from one run is not a
# hard ceiling for every run (batch composition varies slightly).
DEFAULT_SAFETY_MARGIN = 0.85


def load_peak_vram_by_model(csv_path: str) -> dict[str, float]:
    """Return {model_name: max observed peak_vram_gb} from the efficiency log."""
    peaks: dict[str, float] = defaultdict(float)
    if not os.path.isfile(csv_path):
        return peaks
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row.get("model_name", "").strip()
            raw = row.get("peak_vram_gb", "")
            if not model or not raw:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if val > peaks[model]:
                peaks[model] = val
    return peaks


def max_parallel_jobs(
    total_vram_gb: float,
    peak_vram_gb: float,
    per_process_overhead_gb: float = DEFAULT_PER_PROCESS_OVERHEAD_GB,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    usable = total_vram_gb * safety_margin
    per_job = peak_vram_gb + per_process_overhead_gb
    if per_job <= 0:
        return 0
    return max(int(usable // per_job), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-vram-gb", type=float, default=12.0,
                         help="Total VRAM on the GPU (default: 12.0)")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV,
                         help=f"Path to efficiency CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--model-name", type=str, default=None,
                         help="Only estimate for this model_name; omit to list all logged models")
    parser.add_argument("--peak-vram-gb", type=float, default=None,
                         help="Manually supplied peak VRAM (GB) for one run, "
                              "used instead of / when missing from the CSV")
    parser.add_argument("--overhead-gb", type=float, default=DEFAULT_PER_PROCESS_OVERHEAD_GB,
                         help=f"Per-process CUDA context overhead in GB (default: {DEFAULT_PER_PROCESS_OVERHEAD_GB})")
    parser.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN,
                         help=f"Fraction of total VRAM to plan against (default: {DEFAULT_SAFETY_MARGIN})")
    args = parser.parse_args()

    peaks = load_peak_vram_by_model(args.csv_path)

    if args.model_name:
        peak = args.peak_vram_gb if args.peak_vram_gb is not None else peaks.get(args.model_name)
        if peak is None:
            print(f"No logged runs for '{args.model_name}' in {args.csv_path}, "
                  f"and no --peak-vram-gb given. Run it once first, or pass --peak-vram-gb manually.")
            return
        n = max_parallel_jobs(args.total_vram_gb, peak, args.overhead_gb, args.safety_margin)
        print(f"{args.model_name}: peak={peak:.2f}GB, "
              f"usable={args.total_vram_gb * args.safety_margin:.2f}GB "
              f"({args.total_vram_gb}GB * {args.safety_margin} margin) "
              f"-> up to {n} concurrent job(s)")
        return

    if not peaks:
        print(f"No data found in {args.csv_path}. Run at least one job per model first, "
              f"or use --model-name with --peak-vram-gb to estimate manually.")
        return

    print(f"Total VRAM: {args.total_vram_gb}GB | safety margin: {args.safety_margin} "
          f"| per-process overhead: {args.overhead_gb}GB\n")
    print(f"{'model_name':<20} {'peak_vram_gb':>13} {'max_parallel':>13}")
    for model, peak in sorted(peaks.items(), key=lambda kv: kv[1]):
        n = max_parallel_jobs(args.total_vram_gb, peak, args.overhead_gb, args.safety_margin)
        print(f"{model:<20} {peak:>13.2f} {n:>13}")


if __name__ == "__main__":
    main()
