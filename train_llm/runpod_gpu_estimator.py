"""
Estimate LoRA fine-tuning cost/time on RunPod GPUs for the big datasets
(arxiv, computer, instagram, ... - anything whose cached .pt file is large),
so you can decide which GPU tier to rent instead of training locally on a
12GB RTX 5070.

Reuses train_llm/computation_info.py's per-(llm, dataset) VRAM/time estimate
(measured > scaled_from_other_dataset > analytical, see that file's docstring)
as the "baseline", measured on whatever GPU is in THIS machine (detected via
torch.cuda - currently an RTX 5070), then for every GPU in RUNPOD_GPU_CATALOG:

  - checks whether the estimated peak VRAM fits in that GPU (with the same
    safety margin / per-process overhead used by scripts/estimate_parallel_jobs.py)
  - scales the baseline training time by the ratio of each GPU's dense bf16
    TFLOPS to the reference GPU's (i.e. assumes LoRA fine-tuning here is
    compute-bound and utilization-% stays roughly constant across GPU
    generations - a simplification, not a guarantee)
  - computes how many identical jobs can run in parallel on that GPU
    (train_llm.computation_info.max_parallel_jobs, same formula as
    scripts/estimate_parallel_jobs.py)
  - prices one run, and one run amortized across the parallel jobs a rented
    GPU-hour buys, using RUNPOD_GPU_CATALOG's $/hr (approximate - verify/edit
    against https://www.runpod.io/pricing before trusting the $ numbers)

Writes one full grid to CSV (overwrites each run - this is a report, not an
accumulating log like computation_info's own CSV).

Usage:
    python -m train_llm.runpod_gpu_estimator
    python -m train_llm.runpod_gpu_estimator --datasets arxiv,computer
    python -m train_llm.runpod_gpu_estimator --llm_name llama_3.2_3B --gpus "RTX 4090,A100 80GB SXM,H100 80GB SXM"
    python -m train_llm.runpod_gpu_estimator --list_gpus
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from train_llm.computation_info import (
    DEFAULT_GPU_TFLOPS,
    DEFAULT_PER_PROCESS_OVERHEAD_GB,
    DEFAULT_SAFETY_MARGIN,
    GPU_TFLOPS_TABLE,
    REPO_ROOT,
    discover_available_datasets,
    discover_available_llms,
    estimate_computation_info,
    gpu_tflops_estimate,
    max_parallel_jobs,
)

DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "computation_info" / "runpod_gpu_estimates.csv"

# =============================================================================
# RunPod GPU catalog - EDIT THESE. TFLOPS are dense bf16 tensor peak, kept on
# the same rough scale as computation_info.GPU_TFLOPS_TABLE (reused directly
# for H100/A100/4090/A6000/3090 so both files agree). $/hr are approximate
# community-cloud on-demand rates and drift constantly - re-check
# https://www.runpod.io/pricing before using the cost column for real budgeting.
# =============================================================================
RUNPOD_GPU_CATALOG = [
    {"name": "RTX 3090",        "vram_gb": 24,  "tflops_bf16": 71,  "price_usd_hr": 0.22},
    {"name": "RTX 4090",        "vram_gb": 24,  "tflops_bf16": 165, "price_usd_hr": 0.34},
    {"name": "RTX A4000",       "vram_gb": 16,  "tflops_bf16": 40,  "price_usd_hr": 0.17},
    {"name": "RTX A5000",       "vram_gb": 24,  "tflops_bf16": 55,  "price_usd_hr": 0.19},
    {"name": "RTX A6000",       "vram_gb": 48,  "tflops_bf16": 90,  "price_usd_hr": 0.33},
    {"name": "RTX 6000 Ada",    "vram_gb": 48,  "tflops_bf16": 180, "price_usd_hr": 0.77},
    {"name": "L4",               "vram_gb": 24,  "tflops_bf16": 60,  "price_usd_hr": 0.29},
    {"name": "L40",              "vram_gb": 48,  "tflops_bf16": 90,  "price_usd_hr": 0.69},
    {"name": "L40S",             "vram_gb": 48,  "tflops_bf16": 180, "price_usd_hr": 0.79},
    {"name": "A40",              "vram_gb": 48,  "tflops_bf16": 75,  "price_usd_hr": 0.35},
    {"name": "A100 40GB PCIe",  "vram_gb": 40,  "tflops_bf16": 156, "price_usd_hr": 0.99},
    {"name": "A100 80GB PCIe",  "vram_gb": 80,  "tflops_bf16": 200, "price_usd_hr": 1.19},
    {"name": "A100 80GB SXM",   "vram_gb": 80,  "tflops_bf16": 240, "price_usd_hr": 1.65},
    {"name": "H100 80GB PCIe",  "vram_gb": 80,  "tflops_bf16": 450, "price_usd_hr": 1.99},
    {"name": "H100 80GB SXM",   "vram_gb": 80,  "tflops_bf16": 550, "price_usd_hr": 2.69},
    {"name": "H200 141GB SXM",  "vram_gb": 141, "tflops_bf16": 550, "price_usd_hr": 3.59},
]

# Local/consumer dev GPUs not rentable on RunPod but useful as a reference
# point when this script runs on your own machine. Checked before
# GPU_TFLOPS_TABLE, most-specific-substring-first ("5070 Ti" before "5070").
REFERENCE_TFLOPS_OVERRIDES = [
    ("5070 Ti", 130.0), ("5070", 110.0),
    ("5080", 150.0), ("5090", 210.0),
]


def detect_reference_gpu() -> tuple[str, float | None, float]:
    """(name, total_vram_gb, dense_bf16_tflops) for whatever GPU this machine has."""
    if not torch.cuda.is_available():
        return "CPU (no CUDA detected)", None, DEFAULT_GPU_TFLOPS
    props = torch.cuda.get_device_properties(0)
    name = props.name
    vram_gb = props.total_memory / (1024 ** 3)
    for key, tflops in REFERENCE_TFLOPS_OVERRIDES:
        if key in name:
            return name, vram_gb, tflops
    for gpu in RUNPOD_GPU_CATALOG:
        if gpu["name"].split()[0] in name or gpu["name"] in name:
            return name, vram_gb, gpu["tflops_bf16"]
    for key, tflops in GPU_TFLOPS_TABLE:
        if key in name:
            return name, vram_gb, tflops
    return name, vram_gb, gpu_tflops_estimate()


# =============================================================================
# "Huge dataset" filter - selects big cached .pt files (arxiv, computer, ...)
# =============================================================================

def dataset_pt_size_mb(dataset_name: str) -> float:
    pt_path = REPO_ROOT / "datasets" / f"{dataset_name}.pt"
    return pt_path.stat().st_size / (1024 ** 2) if pt_path.exists() else 0.0


def discover_huge_datasets(min_mb: float = 100.0) -> list[str]:
    """discover_available_datasets(), filtered to .pt files >= min_mb, biggest first."""
    names = discover_available_datasets()
    sized = [(n, dataset_pt_size_mb(n)) for n in names]
    return [n for n, mb in sorted(sized, key=lambda x: -x[1]) if mb >= min_mb]


# =============================================================================
# Grid estimator
# =============================================================================

def estimate_runpod_grid(llm_names: list[str], dataset_names: list[str], peft_type: str = "lora",
                          overhead_gb: float = DEFAULT_PER_PROCESS_OVERHEAD_GB,
                          safety_margin: float = DEFAULT_SAFETY_MARGIN,
                          gpu_names: list[str] | None = None) -> tuple[list[dict], dict]:
    """
    Returns (rows, reference_info). One row per (llm, dataset, gpu).
    reference_info = {"name", "vram_gb", "tflops_bf16"} for the local GPU the
    baseline numbers were measured/estimated on.
    """
    ref_name, ref_vram_gb, ref_tflops = detect_reference_gpu()
    gpus = [g for g in RUNPOD_GPU_CATALOG if not gpu_names or g["name"] in gpu_names]
    if not gpus:
        raise SystemExit(f"No GPUs matched --gpus filter {gpu_names}. "
                          f"Known catalog names: {[g['name'] for g in RUNPOD_GPU_CATALOG]}")

    rows: list[dict] = []
    for llm_name in llm_names:
        for dataset_name in dataset_names:
            baseline = estimate_computation_info(
                llm_name, dataset_name, peft_type=peft_type,
                total_vram_gb=ref_vram_gb, overhead_gb=overhead_gb, safety_margin=safety_margin,
            )
            peak_vram_gb = baseline["peak_vram_gb_used_for_planning"]
            baseline_time_min = baseline["estimated_train_time_min"]
            has_time = isinstance(baseline_time_min, (int, float))

            for gpu in gpus:
                fits = (gpu["vram_gb"] * safety_margin) >= (peak_vram_gb + overhead_gb)

                scaled_time_min = ""
                n_parallel = 0
                cost_per_run_usd = ""
                cost_per_run_amortized_usd = ""

                if has_time:
                    scale = (ref_tflops / gpu["tflops_bf16"]) if gpu["tflops_bf16"] else 1.0
                    scaled_time_min = round(baseline_time_min * scale, 2)

                if fits:
                    n_parallel = max_parallel_jobs(gpu["vram_gb"], peak_vram_gb, overhead_gb, safety_margin)
                    if has_time and gpu["price_usd_hr"]:
                        cost_per_run_usd = round((scaled_time_min / 60.0) * gpu["price_usd_hr"], 4)
                        cost_per_run_amortized_usd = round(cost_per_run_usd / max(n_parallel, 1), 4)

                rows.append({
                    "llm_name": llm_name,
                    "dataset_name": dataset_name,
                    "peft_type": peft_type,
                    "num_train_samples_est": baseline["num_train_samples_est"],
                    "estimate_source": baseline["estimate_source"],
                    "reference_gpu": ref_name,
                    "reference_gpu_tflops_bf16": ref_tflops,
                    "reference_train_time_min": baseline_time_min,
                    "peak_vram_gb_required": peak_vram_gb,
                    "gpu_name": gpu["name"],
                    "gpu_vram_gb": gpu["vram_gb"],
                    "gpu_tflops_bf16": gpu["tflops_bf16"],
                    "gpu_price_usd_hr": gpu["price_usd_hr"],
                    "fits_vram": fits,
                    "estimated_train_time_min": scaled_time_min,
                    "estimated_train_time_hr": round(scaled_time_min / 60.0, 3) if scaled_time_min != "" else "",
                    "max_parallel_jobs": n_parallel,
                    "cost_usd_per_run": cost_per_run_usd,
                    "cost_usd_per_run_amortized": cost_per_run_amortized_usd,
                })
    return rows, {"name": ref_name, "vram_gb": ref_vram_gb, "tflops_bf16": ref_tflops}


# =============================================================================
# CSV + console reporting
# =============================================================================

def write_grid_csv(rows: list[dict], csv_path: Path | str = DEFAULT_OUTPUT_CSV) -> str:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


def print_summary(rows: list[dict]) -> None:
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_pair.setdefault((r["llm_name"], r["dataset_name"]), []).append(r)

    for (llm_name, dataset_name), pair_rows in by_pair.items():
        print(f"\n=== {llm_name} | {dataset_name} "
              f"(source={pair_rows[0]['estimate_source']}, "
              f"peak_vram~{pair_rows[0]['peak_vram_gb_required']:.2f}GB, "
              f"baseline_time~{pair_rows[0]['reference_train_time_min']}min on {pair_rows[0]['reference_gpu']}) ===")
        fitting = [r for r in pair_rows if r["fits_vram"]]
        skipped = [r for r in pair_rows if not r["fits_vram"]]
        fitting.sort(key=lambda r: (r["cost_usd_per_run_amortized"] == "", r["cost_usd_per_run_amortized"]))
        print(f"{'gpu':<18} {'time_min':>10} {'parallel':>9} {'$/run':>9} {'$/run (parallel)':>17}")
        for r in fitting:
            print(f"{r['gpu_name']:<18} {r['estimated_train_time_min']:>10} {r['max_parallel_jobs']:>9} "
                  f"{r['cost_usd_per_run']:>9} {r['cost_usd_per_run_amortized']:>17}")
        if skipped:
            print("  does not fit VRAM: " + ", ".join(r["gpu_name"] for r in skipped))


def print_gpu_catalog() -> None:
    print(f"{'name':<18} {'vram_gb':>8} {'tflops_bf16':>12} {'price_usd_hr':>13}")
    for g in RUNPOD_GPU_CATALOG:
        print(f"{g['name']:<18} {g['vram_gb']:>8} {g['tflops_bf16']:>12} {g['price_usd_hr']:>13}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate LoRA fine-tuning time/VRAM/cost on RunPod GPUs for large datasets")
    parser.add_argument('--llm_name', type=str, default=None,
                         help='Comma-separated LLM(s) to estimate (default: every discovered LLM)')
    parser.add_argument('--datasets', type=str, default=None,
                         help='Comma-separated dataset(s) (default: "huge" ones - see --min_dataset_mb)')
    parser.add_argument('--min_dataset_mb', type=float, default=100.0,
                         help='When --datasets is omitted, only include datasets whose cached .pt file is '
                              'at least this big (default: 100 MB - selects e.g. arxiv/computer/instagram)')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--gpus', type=str, default=None,
                         help='Comma-separated RunPod GPU names from the catalog (default: all). '
                              'See --list_gpus for valid names.')
    parser.add_argument('--overhead_gb', type=float, default=DEFAULT_PER_PROCESS_OVERHEAD_GB)
    parser.add_argument('--safety_margin', type=float, default=DEFAULT_SAFETY_MARGIN)
    parser.add_argument('--csv_path', type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument('--list_gpus', action='store_true', help='Print the GPU catalog and exit')
    args = parser.parse_args()

    if args.list_gpus:
        print_gpu_catalog()
        raise SystemExit(0)

    llm_list = args.llm_name.split(',') if args.llm_name else discover_available_llms()
    if args.datasets:
        dataset_list = args.datasets.split(',')
    else:
        dataset_list = discover_huge_datasets(min_mb=args.min_dataset_mb)
    gpu_list = args.gpus.split(',') if args.gpus else None

    if not llm_list:
        raise SystemExit("No LLMs discovered/specified - check configs/llm_configs/ and local_models/.")
    if not dataset_list:
        raise SystemExit(f"No datasets >= {args.min_dataset_mb}MB discovered - lower --min_dataset_mb "
                          f"or pass --datasets explicitly.")

    print(f"LLMs: {llm_list}")
    print(f"Datasets: {dataset_list}")
    print(f"Estimating {len(llm_list)} LLM(s) x {len(dataset_list)} dataset(s) "
          f"x {len(gpu_list) if gpu_list else len(RUNPOD_GPU_CATALOG)} GPU(s)...")

    rows, ref_info = estimate_runpod_grid(
        llm_list, dataset_list, peft_type=args.peft_type,
        overhead_gb=args.overhead_gb, safety_margin=args.safety_margin, gpu_names=gpu_list,
    )
    print(f"\nReference (local) GPU: {ref_info['name']}  "
          f"({ref_info['vram_gb']:.1f}GB, ~{ref_info['tflops_bf16']} TFLOPS bf16 assumed)")

    print_summary(rows)

    out_path = write_grid_csv(rows, csv_path=args.csv_path)
    print(f"\n[OK] Full grid ({len(rows)} rows) written to: {out_path}")
