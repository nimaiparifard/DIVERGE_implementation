# in this file you got llm_name and datasetname
# and tell how much dataset get vram how much model with lora get vram and total
# estimae training time for fine tuing with that model and that dataset
# and also say if want to run training parrarel with that model how much parrarel i can run to not full vram.
# then in this project means train_llm create .bat script to run with diffirent llm and dffirent dataset that exist in datasets
# report it heere in csv file.
"""
Given an LLM name and a dataset name, estimate:
  - VRAM used by the batched input data ("dataset" VRAM: input_ids/attention_mask/labels)
  - VRAM used by the model + LoRA adapter (weights, optimizer state, activations)
  - Total VRAM
  - Training time for one LoRA fine-tuning run
  - How many such runs can be launched in parallel on the GPU without running out
    of VRAM

Estimates PREFER real measured numbers from
results/efficiency/lora_finetune_efficiency.csv (written by train_llm/efficiency.py
whenever an actual training run finishes) over analytical guesses:
  - "measured":              this exact (dataset, llm) pair has already been run.
  - "scaled_from_other_dataset": this llm was run on a different dataset - its
    measured VRAM/time is reused/scaled by sample-count ratio.
  - "analytical":             no prior run of this llm exists yet - a from-scratch
    heuristic (on-disk weight size + a standard "6 * params * tokens" FLOPs
    estimate). Clearly the least accurate source; run the model once to upgrade
    later estimates to "measured".

Also:
  - Appends one row per (llm_name, dataset_name) estimated to a CSV report.
  - Can auto-discover every LLM under configs/llm_configs/ (with weights present
    under local_models/) and every dataset under configs/dataset/ (with a
    matching .pt file under datasets/), estimate the full grid, and generate a
    ready-to-run scripts/run_all_llms_all_datasets.bat that sweeps it, mirroring
    scripts/run_all_datasets_lora_init.bat's structure.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EFFICIENCY_CSV = REPO_ROOT / "results" / "efficiency" / "lora_finetune_efficiency.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "computation_info" / "computation_estimates.csv"
DEFAULT_BAT_PATH = REPO_ROOT / "scripts" / "run_all_llms_all_datasets.bat"

DTYPE_BYTES = {"bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "float32": 4, "fp32": 4}

# Mirrors scripts/estimate_parallel_jobs.py's planning constants (kept local so
# train_llm/ doesn't depend on the scripts/ folder, which isn't a real package).
DEFAULT_PER_PROCESS_OVERHEAD_GB = 0.5
DEFAULT_SAFETY_MARGIN = 0.85

# Very rough sustained-training TFLOPS by GPU name substring, used ONLY as a
# last resort when no measured run of this LLM exists anywhere yet.
GPU_TFLOPS_TABLE = [
    ("H100", 500.0), ("A100", 200.0), ("4090", 160.0), ("A6000", 90.0),
    ("4080", 120.0), ("3090", 70.0), ("3080", 60.0), ("V100", 60.0),
    ("T4", 40.0), ("2080", 40.0), ("1080", 15.0),
]
DEFAULT_GPU_TFLOPS = 20.0  # unknown/undetected GPU fallback
GPU_UTILIZATION = 0.25  # fraction of peak FLOPS actually sustained for small-batch LoRA FT


# =============================================================================
# Discovery: which LLMs / datasets are actually usable in this checkout
# =============================================================================

def discover_available_llms() -> list[str]:
    """LLMs with both a configs/llm_configs/<name>.json and local_models/<name>/ weights."""
    cfg_dir = REPO_ROOT / "configs" / "llm_configs"
    models_dir = REPO_ROOT / "local_models"
    names = []
    for f in sorted(cfg_dir.glob("*.json")):
        if (models_dir / f.stem).is_dir():
            names.append(f.stem)
    return names


def discover_available_datasets() -> list[str]:
    """Datasets with both a configs/dataset/<name>.json and a datasets/<name>.pt file."""
    cfg_dir = REPO_ROOT / "configs" / "dataset"
    data_dir = REPO_ROOT / "datasets"
    names = []
    for f in sorted(cfg_dir.glob("*.json")):
        if (data_dir / f"{f.stem}.pt").exists():
            names.append(f.stem)
    return names


# =============================================================================
# GPU introspection
# =============================================================================

def gpu_total_vram_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return None


def gpu_tflops_estimate() -> float:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_properties(0).name
        for key, tflops in GPU_TFLOPS_TABLE:
            if key in name:
                return tflops
    return DEFAULT_GPU_TFLOPS


def max_parallel_jobs(total_vram_gb: float, peak_vram_gb: float,
                       per_process_overhead_gb: float = DEFAULT_PER_PROCESS_OVERHEAD_GB,
                       safety_margin: float = DEFAULT_SAFETY_MARGIN) -> int:
    """How many concurrent jobs of `peak_vram_gb` each fit in `total_vram_gb`."""
    usable = total_vram_gb * safety_margin
    per_job = peak_vram_gb + per_process_overhead_gb
    if per_job <= 0:
        return 0
    return max(int(usable // per_job), 0)


# =============================================================================
# Model introspection (no weight loading - just file sizes + config.json)
# =============================================================================

def model_weight_bytes(llm_name: str) -> int:
    """Sum of on-disk weight file sizes (safetensors/bin) - the VRAM needed to
    hold the model's weights in their stored dtype, without loading anything."""
    model_dir = REPO_ROOT / "local_models" / llm_name
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for f in model_dir.glob(pattern):
            total += f.stat().st_size
    return total


def read_model_arch(llm_name: str) -> tuple[int | None, int | None]:
    """(hidden_size, num_hidden_layers) from local_models/<llm_name>/config.json,
    used only for the activation-memory heuristic below."""
    config_path = REPO_ROOT / "local_models" / llm_name / "config.json"
    if not config_path.exists():
        return None, None
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hidden = data.get("hidden_size") or data.get("d_model") or data.get("n_embd")
    layers = data.get("num_hidden_layers") or data.get("n_layer") or data.get("num_layers")
    return hidden, layers


def estimate_lora_trainable_params(cfg, hidden_size: int | None) -> int:
    """
    Rough LoRA trainable-parameter count: a rank-r adapter on every target
    module of every transformer layer, plus the full classification head
    (modules_to_save, e.g. "score"/"classifier").
    """
    hidden_size = hidden_size or 2048  # generic fallback if config.json lacks it
    rank = cfg.peft.rank
    num_target_modules = len(cfg.llm.ft_target_modules)
    _, num_layers = read_model_arch(cfg.llm.model_name)
    num_layers = num_layers or 24  # generic fallback
    # Linear(in, out) LoRA adapter: rank * (in + out) params; approximate in ~= out ~= hidden_size
    lora_params = num_layers * num_target_modules * rank * (2 * hidden_size)
    head_params = hidden_size * cfg.dataset.num_classes + cfg.dataset.num_classes
    return int(lora_params + head_params)


# =============================================================================
# Measured-data lookup (results/efficiency/lora_finetune_efficiency.csv)
# =============================================================================

def _as_float(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_measured_rows(csv_path: Path = EFFICIENCY_CSV) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# raw_texts count depends only on the dataset's .pt file, not on llm_name/peft_type,
# but sweeping every LLM against the same dataset would otherwise reload that (often
# large) .pt file once per LLM - cache it per dataset_name instead.
_NUM_SAMPLES_CACHE: dict[str, int] = {}


def _dataset_num_samples(cfg) -> int:
    dataset_name = cfg.dataset.name
    if dataset_name not in _NUM_SAMPLES_CACHE:
        try:
            _NUM_SAMPLES_CACHE[dataset_name] = len(load_dataset(cfg).raw_texts)
        except Exception:
            _NUM_SAMPLES_CACHE[dataset_name] = cfg.dataset.num_samples or 0
    return _NUM_SAMPLES_CACHE[dataset_name]


# =============================================================================
# Main estimator
# =============================================================================

def estimate_computation_info(llm_name: str, dataset_name: str, peft_type: str = "lora",
                               total_vram_gb: float | None = None,
                               overhead_gb: float = DEFAULT_PER_PROCESS_OVERHEAD_GB,
                               safety_margin: float = DEFAULT_SAFETY_MARGIN) -> dict:
    """
    Estimate VRAM (dataset / model+LoRA / total), training time, and max
    parallel jobs for LoRA fine-tuning `llm_name` on `dataset_name`.
    Returns a flat dict, ready to append to a CSV report.
    """
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    hidden_size, num_layers = read_model_arch(llm_name)
    dtype_bytes = DTYPE_BYTES.get(str(cfg.llm.torch_dtype).lower(), 2)

    batch_size = cfg.training_args.per_device_train_batch_size
    seq_len = cfg.tokenizer.max_length

    # --- "dataset" VRAM: the batched input tensors themselves ---
    # int64 input_ids + attention_mask (seq_len each) + int64 label per sample
    dataset_vram_bytes = batch_size * seq_len * 8 * 2 + batch_size * 8

    # --- actual training-sample count, if the cached graph/.pt file is available ---
    num_samples = _dataset_num_samples(cfg)

    # --- "model + LoRA" VRAM: weights + optimizer state (LoRA params only) + activations ---
    weight_bytes = model_weight_bytes(llm_name)
    trainable_params = estimate_lora_trainable_params(cfg, hidden_size)
    # AdamW on the trainable (LoRA) params only: dtype-native param+grad, plus fp32 grad + 2 fp32 moments
    optimizer_bytes = trainable_params * (dtype_bytes * 2 + 4 * 3)
    # Rough activation-memory heuristic (no gradient checkpointing): a handful of
    # dtype-sized [batch, seq_len, hidden] tensors kept per transformer layer for backward.
    activation_bytes = 0
    if hidden_size and num_layers:
        activation_bytes = batch_size * seq_len * hidden_size * num_layers * 12 * dtype_bytes
    model_vram_bytes = weight_bytes + optimizer_bytes + activation_bytes

    total_vram_bytes = dataset_vram_bytes + model_vram_bytes
    total_vram_gb_estimated = total_vram_bytes / (1024 ** 3)

    # --- prefer measured numbers when available ---
    measured_rows = load_measured_rows()
    exact_matches = [r for r in measured_rows
                      if r.get("dataset_name") == dataset_name and r.get("model_name") == llm_name]
    llm_matches = [r for r in measured_rows if r.get("model_name") == llm_name]

    source = "analytical"
    peak_vram_gb = total_vram_gb_estimated
    train_time_min = None

    if exact_matches:
        source = "measured"
        peaks = [v for v in (_as_float(r, "peak_vram_gb") for r in exact_matches) if v is not None]
        times = [v for v in (_as_float(r, "wall_clock_train_min") for r in exact_matches) if v is not None]
        if peaks:
            peak_vram_gb = max(peaks)
        if times:
            train_time_min = sum(times) / len(times)
    elif llm_matches:
        source = "scaled_from_other_dataset"
        ref = llm_matches[0]
        ref_peak = _as_float(ref, "peak_vram_gb")
        ref_time = _as_float(ref, "wall_clock_train_min")
        ref_dataset = ref.get("dataset_name")
        ref_num_samples = None
        if ref_dataset:
            try:
                ref_cfg = setup_finetuning_cfg(dataset_name=ref_dataset, llm_name=llm_name, peft_type=peft_type)
                ref_num_samples = _dataset_num_samples(ref_cfg)
            except Exception:
                ref_num_samples = None
        if ref_peak is not None:
            # VRAM is driven mainly by batch shape/model size, not dataset size - reuse as-is
            peak_vram_gb = ref_peak
        if ref_time is not None and ref_num_samples:
            ratio = (num_samples / ref_num_samples) if num_samples else 1.0
            train_time_min = ref_time * ratio

    if train_time_min is None:
        # Analytical FLOPs-based fallback: ~6 * params * tokens (fwd+bwd), standard scaling-law approx.
        total_params_est = (weight_bytes / dtype_bytes) if dtype_bytes else 0
        epochs = cfg.training_args.num_train_epochs
        total_tokens = num_samples * seq_len * epochs
        flops = 6 * total_params_est * total_tokens
        sustained_tflops = gpu_tflops_estimate() * GPU_UTILIZATION
        seconds = flops / (sustained_tflops * 1e12) if sustained_tflops > 0 else float("nan")
        train_time_min = seconds / 60.0

    detected_total_vram_gb = total_vram_gb if total_vram_gb is not None else gpu_total_vram_gb()
    parallel_jobs = None
    if detected_total_vram_gb:
        parallel_jobs = max_parallel_jobs(detected_total_vram_gb, peak_vram_gb, overhead_gb, safety_margin)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "llm_name": llm_name,
        "dataset_name": dataset_name,
        "peft_type": peft_type,
        "num_train_samples_est": num_samples,
        "max_seq_length": seq_len,
        "batch_size": batch_size,
        "trainable_params_est": trainable_params,
        "dataset_vram_gb": round(dataset_vram_bytes / (1024 ** 3), 6),
        "model_lora_vram_gb": round(model_vram_bytes / (1024 ** 3), 4),
        "total_vram_gb_estimated": round(total_vram_gb_estimated, 4),
        "estimate_source": source,
        "peak_vram_gb_used_for_planning": round(peak_vram_gb, 4),
        "estimated_train_time_min": round(train_time_min, 2) if train_time_min is not None else "",
        "estimated_train_time_hr": round(train_time_min / 60.0, 3) if train_time_min is not None else "",
        "gpu_total_vram_gb": round(detected_total_vram_gb, 2) if detected_total_vram_gb else "",
        "max_parallel_jobs": parallel_jobs if parallel_jobs is not None else "",
    }


# =============================================================================
# CSV reporting
# =============================================================================

def append_computation_info_csv(row: dict, csv_path: Path | str = DEFAULT_OUTPUT_CSV) -> str:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()

    fieldnames = list(row.keys())
    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header:
            fieldnames = existing_header + [k for k in row.keys() if k not in existing_header]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return str(csv_path)


# =============================================================================
# .bat generation: sweep every discovered LLM x dataset x LoRA-init combination
# =============================================================================

def generate_llm_dataset_sweep_bat(llm_names: list[str] | None = None,
                                    dataset_names: list[str] | None = None,
                                    init_approaches: list[str] | None = None,
                                    output_path: Path | str = DEFAULT_BAT_PATH,
                                    peft_type: str = "lora", split: int = 1) -> str:
    """
    Write a ready-to-run .bat script that sweeps train_llm.train_other_lora_init_apporaches
    over every combination of the given (or auto-discovered) LLMs, datasets, and
    LoRA-init approaches, mirroring scripts/run_all_datasets_lora_init.bat's
    status-tracking / history-CSV structure.
    """
    llm_names = llm_names or discover_available_llms()
    dataset_names = dataset_names or discover_available_datasets()
    init_approaches = init_approaches or ["pissa", "gaussian", "eva", "loftq", "orthogonal"]

    content = f"""@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM AUTO-GENERATED by train_llm/computation_info.py - sweeps every LLM x dataset
REM x LoRA-init approach combination discovered under configs/llm_configs/ and
REM datasets/ at generation time. Regenerate anytime with:
REM   python -m train_llm.computation_info --generate_bat
REM
REM Progress / identity of the active run is written to:
REM   results/run_status/ALLGRID_CURRENT_RUN.txt
REM   results/run_status/allgrid_run_history.csv
REM   results/run_status/ALLGRID_LAST_COMPLETED.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE={peft_type}
set SPLIT={split}

REM Disable HF Trainer logging integrations (wandb/mlflow prompts block unattended runs)
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1

REM Auto-discovered at generation time (edit freely, or regenerate with --generate_bat)
set DATASETS={' '.join(dataset_names)}
set LLM_NAMES={' '.join(llm_names)}
set INIT_APPROACHES={' '.join(init_approaches)}

REM Status / tracking files
set STATUS_DIR=results\\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\\ALLGRID_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\\allgrid_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\\ALLGRID_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\\ALLGRID_FAILED_RUNS.txt

if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,peft_type,lora_init,split,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE full LLM x dataset x LoRA-init sweep
echo Status dir: %STATUS_DIR%
echo History CSV: %HISTORY_FILE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%i in (%INIT_APPROACHES%) do (
            set "RUN_ID=dataset=%%d | llm=%%m | peft=%PEFT_TYPE% | init=%%i | split=%SPLIT%"
            set "TS=%DATE% %TIME%"

            echo ==========================================
            echo START: !RUN_ID!
            echo Time: !TS!
            echo ==========================================

            (
                echo status=RUNNING
                echo started=!TS!
                echo dataset=%%d
                echo llm_name=%%m
                echo peft_type=%PEFT_TYPE%
                echo lora_init=%%i
                echo split=%SPLIT%
                echo command=python -m train_llm.train_other_lora_init_apporaches --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --init_weight_approach %%i --split %SPLIT%
                echo pid_note=see console / Task Manager for python process
            ) > "%CURRENT_FILE%"

            python -m train_llm.train_other_lora_init_apporaches --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i" --split %SPLIT%
            set EXIT_CODE=!ERRORLEVEL!
            set "TS_END=%DATE% %TIME%"

            if !EXIT_CODE! NEQ 0 (
                echo ERROR: !RUN_ID!  exit_code=!EXIT_CODE!
                echo !TS_END!,FAILED,%%d,%%m,%PEFT_TYPE%,%%i,%SPLIT%,!EXIT_CODE!>> "%HISTORY_FILE%"
                echo !TS_END! FAILED !RUN_ID! exit=!EXIT_CODE!>> "%FAILED_FILE%"
                (
                    echo status=FAILED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo peft_type=%PEFT_TYPE%
                    echo lora_init=%%i
                    echo split=%SPLIT%
                    echo exit_code=!EXIT_CODE!
                ) > "%CURRENT_FILE%"
                echo Continuing to next run despite failure...
                echo.
            ) else (
                echo OK: !RUN_ID!
                echo !TS_END!,OK,%%d,%%m,%PEFT_TYPE%,%%i,%SPLIT%,0>> "%HISTORY_FILE%"
                (
                    echo status=COMPLETED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo peft_type=%PEFT_TYPE%
                    echo lora_init=%%i
                    echo split=%SPLIT%
                    echo exit_code=0
                ) > "%LAST_OK_FILE%"
                copy /Y "%LAST_OK_FILE%" "%CURRENT_FILE%" >nul
                echo.
            )
        )
    )
)

echo ==========================================
echo Full grid sweep finished.
echo Check:
echo   %CURRENT_FILE%
echo   %HISTORY_FILE%
echo   %LAST_OK_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued runs finished; see allgrid_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(content)
    print(f"[OK] Generated sweep .bat script: {output_path}")
    return str(output_path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate VRAM / training time / max-parallel-jobs for train_llm LoRA fine-tuning runs")
    parser.add_argument('--llm_name', type=str, default=None,
                         help='Single LLM to estimate (omit to sweep every discovered LLM)')
    parser.add_argument('--dataset_name', type=str, default=None,
                         help='Single dataset to estimate (omit to sweep every discovered dataset)')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--total_vram_gb', type=float, default=None,
                         help='Total GPU VRAM in GB (auto-detected from CUDA if omitted)')
    parser.add_argument('--overhead_gb', type=float, default=DEFAULT_PER_PROCESS_OVERHEAD_GB,
                         help=f'Per-process CUDA context overhead in GB (default: {DEFAULT_PER_PROCESS_OVERHEAD_GB})')
    parser.add_argument('--safety_margin', type=float, default=DEFAULT_SAFETY_MARGIN,
                         help=f'Fraction of total VRAM to plan against (default: {DEFAULT_SAFETY_MARGIN})')
    parser.add_argument('--csv_path', type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument('--generate_bat', action='store_true',
                         help='(Re)generate scripts/run_all_llms_all_datasets.bat sweeping every '
                              'discovered LLM x dataset x LoRA-init combination')
    parser.add_argument('--bat_path', type=str, default=str(DEFAULT_BAT_PATH))
    args = parser.parse_args()

    llm_list = [args.llm_name] if args.llm_name else discover_available_llms()
    dataset_list = [args.dataset_name] if args.dataset_name else discover_available_datasets()

    if not llm_list:
        raise SystemExit("No LLMs discovered/specified - check configs/llm_configs/ and local_models/.")
    if not dataset_list:
        raise SystemExit("No datasets discovered/specified - check configs/dataset/ and datasets/.")

    print(f"Estimating computation info for {len(llm_list)} LLM(s) x {len(dataset_list)} dataset(s)...")
    for llm_name in llm_list:
        for dataset_name in dataset_list:
            info = estimate_computation_info(
                llm_name, dataset_name, peft_type=args.peft_type,
                total_vram_gb=args.total_vram_gb, overhead_gb=args.overhead_gb,
                safety_margin=args.safety_margin,
            )
            print(
                f"[{llm_name} | {dataset_name}] "
                f"dataset_vram={info['dataset_vram_gb']:.4f}GB  "
                f"model+lora_vram={info['model_lora_vram_gb']:.3f}GB  "
                f"total~={info['total_vram_gb_estimated']:.3f}GB  "
                f"planning_vram={info['peak_vram_gb_used_for_planning']:.3f}GB ({info['estimate_source']})  "
                f"train_time~={info['estimated_train_time_min']}min  "
                f"max_parallel={info['max_parallel_jobs']}"
            )
            append_computation_info_csv(info, csv_path=args.csv_path)
    print(f"[OK] Computation-info report appended to: {args.csv_path}")

    if args.generate_bat:
        generate_llm_dataset_sweep_bat(
            llm_names=discover_available_llms(),
            dataset_names=discover_available_datasets(),
            output_path=args.bat_path,
            peft_type=args.peft_type,
        )
