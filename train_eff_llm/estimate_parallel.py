"""
Decide how many train_and_cache jobs to run at once, and at which micro-batch size.

Probes the real model (same dtype/attention/LoRA settings as train_and_cache) with
worst-case batches (every sequence at max_length, which group_by_length schedules
first) for micro_bs = 1, 2, 4, ... and records peak reserved VRAM + throughput.
Then picks the largest micro_bs that fits in the currently FREE VRAM (other
processes are accounted for) and as many parallel jobs as fit at that size,
capped by system RAM and the number of init approaches.

Writes the plan to outputs_eff/parallel_plan.json (read by run_eff_parallel.bat).

Usage:
    python -m train_eff_llm.estimate_parallel --dataset_name cora --llm_name llama_3.2_1B
"""
import argparse
import json
import os
import time

import psutil
import torch
from transformers.utils import is_flash_attn_2_available

from config import setup_finetuning_cfg
from dataset.dataset_loader import get_tokenizer
from train_eff_llm.eff_utils import bf16_supported, enable_fast_matmul, load_eff_model, pick_attn_implementation
from train_eff_llm.train_and_cache import build_lora_model

CUDA_CONTEXT_GB = 0.6      # per extra process (context + cuBLAS workspace)
SAFETY_MARGIN = 0.90       # fraction of free VRAM we plan against
RAM_PER_JOB_GB = 6.0       # model load + tokenized dataset + Python overhead


def probe(model, micro_bs, seq_len, vocab, num_classes, steps=3):
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    ids = torch.randint(10, vocab - 10, (micro_bs, seq_len), device=device)
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 labels=torch.randint(0, num_classes, (micro_bs,), device=device))
    use_bf16 = bf16_supported()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    step()  # warmup (also allocates optimizer state)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
    del opt
    return peak_gb, micro_bs * seq_len / dt


def main(dataset_name, llm_name, peft_type, num_inits, max_micro_bs, out_path, reserve_gb):
    enable_fast_matmul()
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 1024 ** 3, total_b / 1024 ** 3
    usable_gb = max(0.0, free_gb - reserve_gb) * SAFETY_MARGIN

    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    cfg.peft.init_lora_weights = "gaussian"  # memory is the same for every init; gaussian is fastest to set up
    effective_bs = cfg.training_args.per_device_train_batch_size * cfg.training_args.gradient_accumulation_steps
    seq_len = cfg.dataset.max_length

    flash = is_flash_attn_2_available()
    attn_impl = pick_attn_implementation(llm_name, "auto")
    print(f"GPU: {torch.cuda.get_device_name(0)} | total {total_gb:.1f} GB | free now {free_gb:.1f} GB "
          f"| planning against {usable_gb:.1f} GB")
    print(f"flash_attn installed: {flash} -> attention backend: {attn_impl} | bf16: {bf16_supported()}")

    tokenizer = get_tokenizer(cfg)
    base = load_eff_model(cfg, tokenizer, attn_impl)
    model = build_lora_model(base, cfg, "gaussian", False, None, None, 1)
    model.train()
    weights_gb = torch.cuda.memory_allocated() / 1024 ** 3

    results = []
    bs = 1
    while bs <= min(max_micro_bs, effective_bs):
        if effective_bs % bs == 0:
            try:
                peak_gb, tps = probe(model, bs, seq_len, base.config.vocab_size, cfg.dataset.num_classes)
                need = peak_gb + CUDA_CONTEXT_GB
                # On Windows the driver may silently spill VRAM into system RAM instead of
                # raising OOM; that shows up as peak > free VRAM and/or a throughput collapse.
                prev_tps = results[-1]["tokens_per_sec"] if results else 0
                spilling = peak_gb > free_gb or tps < 0.8 * prev_tps
                results.append(dict(micro_bs=bs, peak_gb=round(peak_gb, 2), need_per_job_gb=round(need, 2),
                                    tokens_per_sec=round(tps), spilling=spilling))
                print(f"  micro_bs={bs:>2}: peak {peak_gb:.2f} GB (+ctx -> {need:.2f} GB/job), {tps:,.0f} tok/s"
                      + ("  <- slower/over VRAM: driver is spilling to system RAM" if spilling else ""))
                if need > usable_gb or spilling:
                    break
            except torch.cuda.OutOfMemoryError:
                print(f"  micro_bs={bs:>2}: OOM")
                torch.cuda.empty_cache()
                break
        bs *= 2

    fitting = [r for r in results if r["need_per_job_gb"] <= usable_gb and not r["spilling"]]
    ram_gb = psutil.virtual_memory().available / 1024 ** 3
    ram_cap = max(1, int(ram_gb // RAM_PER_JOB_GB))
    if fitting:
        best = max(fitting, key=lambda r: r["tokens_per_sec"])
        jobs = max(1, min(num_inits, int(usable_gb // best["need_per_job_gb"]), ram_cap))
    else:
        best = dict(micro_bs=1, need_per_job_gb=float("nan"))
        jobs = 1
        print("WARNING: even micro_bs=1 at max_length does not fit the free VRAM; close other GPU apps.")

    plan = dict(dataset_name=dataset_name, llm_name=llm_name, gpu=torch.cuda.get_device_name(0),
                total_vram_gb=round(total_gb, 2), free_vram_gb=round(free_gb, 2), usable_vram_gb=round(usable_gb, 2),
                model_weights_gb=round(weights_gb, 2), available_ram_gb=round(ram_gb, 1), ram_job_cap=ram_cap,
                flash_attn_installed=flash, attn_implementation=attn_impl, bf16=bf16_supported(),
                effective_batch_size=effective_bs, micro_bs=best["micro_bs"],
                grad_accum=effective_bs // best["micro_bs"], jobs=jobs, probes=results)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print("-" * 70)
    print(f"PLAN: {jobs} parallel job(s) x micro_bs={plan['micro_bs']} (grad_accum={plan['grad_accum']}), "
          f"attn={attn_impl} -> {out_path}")
    if jobs > 1:
        print("Note: parallel jobs share one GPU's compute; expect less than linear speedup once a "
              "single job already saturates the GPU.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_name', type=str, default='cora')
    p.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    p.add_argument('--peft_type', type=str, default='lora')
    p.add_argument('--num_inits', type=int, default=5, help='Upper bound on parallel jobs')
    p.add_argument('--max_micro_bs', type=int, default=16)
    p.add_argument('--reserve_gb', type=float, default=0.5, help='VRAM to leave free for the desktop/other apps')
    p.add_argument('--out', type=str, default='outputs_eff/parallel_plan.json')
    a = p.parse_args()
    main(a.dataset_name, a.llm_name, a.peft_type, a.num_inits, a.max_micro_bs, a.out, a.reserve_gb)
