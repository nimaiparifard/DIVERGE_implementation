"""
Efficiency / cost metrics for LoRA fine-tuning (paper efficiency table).

Tracks: wall-clock, GPU-hours, peak VRAM, adapter storage, inference latency.
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


def bytes_to_mb(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 ** 2)


def bytes_to_gb(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 ** 3)


def dir_size_bytes(path: str) -> int:
    total = 0
    if not path or not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def adapter_weight_size_bytes(adapter_dir: str) -> int:
    """Size of adapter weight files only (exclude tokenizer / config noise when possible)."""
    if not adapter_dir or not os.path.isdir(adapter_dir):
        return 0
    weight_exts = (".bin", ".safetensors", ".pt", ".pth")
    total = 0
    for root, _, files in os.walk(adapter_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(weight_exts) or "adapter" in lower:
                fp = os.path.join(root, name)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    # Fallback: whole directory if no weight files matched
    return total if total > 0 else dir_size_bytes(adapter_dir)


def reset_cuda_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_vram_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


_ACTIVE_MONITORS: list["ResourceMonitor"] = []


class ResourceMonitor:
    """
    Wall-clock time and peak CUDA memory of a code block (GNN training, ensemble tuning, ...):

        with ResourceMonitor() as mon:
            ...
        mon.metrics()  # {'time_sec', 'peak_vram_mb', 'peak_vram_delta_mb'}

    peak_vram_mb is the peak memory allocated by PyTorch during the block (including tensors that
    already lived on the GPU before it, e.g. embeddings/models); peak_vram_delta_mb is that peak
    minus what was allocated when the block started, i.e. the extra memory the block itself needed.
    Nesting is safe: an inner monitor resets CUDA's peak counter, so its peak is passed on to the
    enclosing monitor.
    """

    def __init__(self):
        self.time_sec = 0.0
        self.peak_bytes = None
        self.start_bytes = None
        self._child_peak = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if _ACTIVE_MONITORS:
                # keep the enclosing block's peak before resetting the counter
                parent = _ACTIVE_MONITORS[-1]
                parent._child_peak = max(parent._child_peak, int(torch.cuda.max_memory_allocated()))
            torch.cuda.reset_peak_memory_stats()
            self.start_bytes = int(torch.cuda.memory_allocated())
        _ACTIVE_MONITORS.append(self)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_bytes = max(int(torch.cuda.max_memory_allocated()), self._child_peak)
        self.time_sec = time.perf_counter() - self._t0
        _ACTIVE_MONITORS.pop()
        if _ACTIVE_MONITORS and self.peak_bytes is not None:
            parent = _ACTIVE_MONITORS[-1]
            parent._child_peak = max(parent._child_peak, self.peak_bytes)
        return False

    def metrics(self, prefix=""):
        peak = round(bytes_to_mb(self.peak_bytes), 2) if self.peak_bytes is not None else None
        delta = (round(bytes_to_mb(self.peak_bytes - self.start_bytes), 2)
                 if self.peak_bytes is not None else None)
        return {f"{prefix}time_sec": round(self.time_sec, 3), f"{prefix}peak_vram_mb": peak,
                f"{prefix}peak_vram_delta_mb": delta}


def measure_inference_latency(
    model,
    dataset,
    batch_size: int = 8,
    num_warmup: int = 3,
    num_batches: int = 20,
) -> dict[str, float]:
    """
    Measure forward-pass latency on ``dataset`` (TagsDataset-style).
    Returns ms/sample, samples/sec, and total timed samples.
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches = list(loader)
    if not batches:
        return {
            "inference_ms_per_sample": float("nan"),
            "inference_samples_per_sec": float("nan"),
            "inference_num_samples": 0.0,
            "inference_batch_size": float(batch_size),
        }

    warmup = batches[: min(num_warmup, len(batches))]
    timed = batches[: min(num_batches, len(batches))]

    with torch.no_grad():
        for batch in warmup:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            _ = model(**batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_samples = 0
        for batch in timed:
            labels = batch.pop("labels", None)
            n_samples += labels.size(0) if labels is not None else next(iter(batch.values())).size(0)
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    ms_per = (elapsed * 1000.0) / max(n_samples, 1)
    sps = n_samples / max(elapsed, 1e-12)
    return {
        "inference_ms_per_sample": ms_per,
        "inference_samples_per_sec": sps,
        "inference_num_samples": float(n_samples),
        "inference_batch_size": float(batch_size),
    }


def format_efficiency_block(metrics: dict[str, Any]) -> list[str]:
    lines = [
        f"LoRA init: {metrics.get('lora_init', 'N/A')}",
        f"Model: {metrics.get('model_name', 'N/A')}",
        f"Dataset: {metrics.get('dataset_name', 'N/A')}",
        f"PEFT: {metrics.get('peft_type', 'N/A')}",
        f"Split: {metrics.get('split', 'N/A')}",
        "",
        "--- Cost / Efficiency ---",
        f"Wall-clock train (s): {metrics.get('wall_clock_train_sec', float('nan')):.2f}",
        f"Wall-clock train (min): {metrics.get('wall_clock_train_min', float('nan')):.2f}",
        f"GPU-hours (train): {metrics.get('gpu_hours', float('nan')):.6f}",
        f"Peak VRAM allocated (MB): {metrics.get('peak_vram_mb', float('nan')):.2f}",
        f"Peak VRAM allocated (GB): {metrics.get('peak_vram_gb', float('nan')):.3f}",
        f"Adapter storage total (MB): {metrics.get('adapter_dir_mb', float('nan')):.2f}",
        f"Adapter weight files (MB): {metrics.get('adapter_weights_mb', float('nan')):.2f}",
        f"Inference latency (ms/sample): {metrics.get('inference_ms_per_sample', float('nan')):.3f}",
        f"Inference throughput (samples/s): {metrics.get('inference_samples_per_sec', float('nan')):.2f}",
        f"Inference batch size: {int(metrics.get('inference_batch_size', 0) or 0)}",
        f"Trainable params: {metrics.get('trainable_params', 'N/A')}",
        f"Total params: {metrics.get('total_params', 'N/A')}",
        f"Trainable %: {metrics.get('trainable_pct', float('nan')):.4f}",
    ]
    return lines


def append_efficiency_csv(row: dict[str, Any], csv_path: str) -> str:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    # Stable column order
    preferred = [
        "timestamp",
        "dataset_name",
        "model_name",
        "peft_type",
        "lora_init",
        "split",
        "wall_clock_train_sec",
        "wall_clock_train_min",
        "gpu_hours",
        "peak_vram_mb",
        "peak_vram_gb",
        "adapter_dir_mb",
        "adapter_weights_mb",
        "inference_ms_per_sample",
        "inference_samples_per_sec",
        "inference_batch_size",
        "inference_num_samples",
        "trainable_params",
        "total_params",
        "trainable_pct",
        "test_accuracy",
        "test_loss",
        "adapter_dir",
        "report_csv",
    ]
    fieldnames = preferred + [k for k in row.keys() if k not in preferred]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return csv_path


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_efficiency_row(
    *,
    cfg,
    split: int,
    wall_clock_train_sec: float,
    peak_vram: Optional[int],
    adapter_dir: str,
    inference_stats: dict[str, float],
    trainable_params: int,
    total_params: int,
    test_accuracy: Any = None,
    test_loss: Any = None,
) -> dict[str, Any]:
    peak_b = float(peak_vram) if peak_vram is not None else float("nan")
    adapter_total = dir_size_bytes(adapter_dir)
    adapter_w = adapter_weight_size_bytes(adapter_dir)
    trainable_pct = 100.0 * trainable_params / max(total_params, 1)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": cfg.dataset.name,
        "model_name": cfg.llm.model_name,
        "peft_type": cfg.peft.type,
        "lora_init": getattr(cfg.peft, "init_lora_weights", "N/A"),
        "split": split,
        "wall_clock_train_sec": wall_clock_train_sec,
        "wall_clock_train_min": wall_clock_train_sec / 60.0,
        "gpu_hours": wall_clock_train_sec / 3600.0,
        "peak_vram_mb": bytes_to_mb(peak_b) if peak_vram is not None else float("nan"),
        "peak_vram_gb": bytes_to_gb(peak_b) if peak_vram is not None else float("nan"),
        "adapter_dir_mb": bytes_to_mb(adapter_total),
        "adapter_weights_mb": bytes_to_mb(adapter_w),
        "inference_ms_per_sample": inference_stats.get("inference_ms_per_sample", float("nan")),
        "inference_samples_per_sec": inference_stats.get("inference_samples_per_sec", float("nan")),
        "inference_batch_size": inference_stats.get("inference_batch_size", float("nan")),
        "inference_num_samples": inference_stats.get("inference_num_samples", float("nan")),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
        "test_accuracy": test_accuracy if test_accuracy is not None else "",
        "test_loss": test_loss if test_loss is not None else "",
        "adapter_dir": adapter_dir,
    }
    return row
