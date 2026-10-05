"""
Efficient DIVERGE step 1+2 for one (dataset, llm, LoRA init, seed):
LoRA fine-tune -> build the embedding cache in the SAME process -> save adapter.

Caching in-process (instead of reloading the adapter later) is also required
for correctness with PiSSA and LoftQ: both overwrite the base weights at init,
so reloading only the saved adapter onto the original base model gives a
different model than the one that was trained.

Usage:
    python -m train_eff_llm.train_and_cache --dataset_name cora --init_weight_approach pissa --seed 42
"""
import argparse
import os
import time

import torch
from peft import LoraConfig, LoftQConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments

from common import set_seed
from config import setup_finetuning_cfg
from dataset.dataset_loader import get_tokenizer, load_dataset
from report.reporter import ReportResults
from train_llm.efficiency import (append_efficiency_csv, build_efficiency_row, count_parameters,
                                  format_efficiency_block, peak_vram_bytes, reset_cuda_peak_stats)
from train_eff_llm.eff_utils import (bf16_supported, build_cache_fast, cache_file_name, enable_fast_matmul,
                                     get_collator, load_eff_model, measure_latency, pick_attn_implementation,
                                     prepare_eff_splits)

INIT_APPROACHES = ['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal']


def build_lora_model(model, cfg, init, eva_data_init, train_dataset, collator, micro_bs):
    lora_kwargs = dict(
        r=cfg.peft.rank,
        lora_alpha=cfg.peft.lora_alpha,
        lora_dropout=cfg.peft.lora_dropout,
        target_modules=list(cfg.llm.ft_target_modules),
        task_type=TaskType.SEQ_CLS,
        modules_to_save=list(cfg.peft.module_to_save),
        init_lora_weights=init,
        use_rslora=cfg.peft.use_rslora,
    )
    if init == "loftq":
        lora_kwargs["loftq_config"] = LoftQConfig(loftq_bits=4, loftq_iter=1)

    if init == "eva" and eva_data_init:
        from peft import EvaConfig, initialize_lora_eva_weights
        from torch.utils.data import DataLoader
        # use_label_mask=False: our labels are per-sequence class ids, not token labels
        lora_kwargs["eva_config"] = EvaConfig(use_label_mask=False)
        peft_model = get_peft_model(model, LoraConfig(**lora_kwargs), low_cpu_mem_usage=True)
        loader = DataLoader(train_dataset, batch_size=micro_bs, shuffle=True, collate_fn=collator)
        initialize_lora_eva_weights(peft_model, loader)
        return peft_model

    # Without the data-driven step PEFT's "eva" is the default LoRA init (kaiming A, zero B);
    # kept as-is to match the train_llm/ pipeline unless --eva_data_init is passed.
    return get_peft_model(model, LoraConfig(**lora_kwargs))


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', init='gaussian', seed=None, split=1,
         micro_bs=4, attn='auto', padding_side='right', pooling='mean', num_workers=0,
         adapter_root='./artifacts_eff', cache_root='./artifacts_eff/cache', eva_data_init=False,
         max_train_samples=None, skip_cache=False):
    enable_fast_matmul()
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    cfg.peft.init_lora_weights = init
    if seed is not None:
        cfg.dataset.seed = seed
    seed = cfg.dataset.seed
    set_seed(seed)

    attn_impl = pick_attn_implementation(llm_name, attn)
    use_bf16 = bf16_supported()
    effective_bs = cfg.training_args.per_device_train_batch_size * cfg.training_args.gradient_accumulation_steps
    micro_bs = max(1, min(micro_bs, effective_bs))
    grad_accum = max(1, effective_bs // micro_bs)
    split_tag = "" if split == 1 else "_semi_supervised"
    run_name = f"{dataset_name}_{llm_name}_{peft_type}_{init}_seed{seed}{split_tag}"
    adapter_dir = os.path.join(adapter_root, f"{llm_name}_{dataset_name}_seqcls_{peft_type}_init_type_{init}_seed{seed}{split_tag}")

    print("=" * 80)
    print(f"[train_eff] {run_name} | attn={attn_impl} bf16={use_bf16} micro_bs={micro_bs} "
          f"grad_accum={grad_accum} (effective={micro_bs * grad_accum}) padding_side={padding_side}")
    print("=" * 80)

    reporter = ReportResults(cfg, save_dir=f"results/train_info/train_eff/{run_name}", init_approach=init)
    reporter.report_title("Efficient LoRA fine-tuning + in-process caching")
    for k, v in dict(dataset=dataset_name, llm=llm_name, init=init, seed=seed, split=split, attn=attn_impl,
                     bf16=use_bf16, micro_bs=micro_bs, grad_accum=grad_accum, padding_side=padding_side,
                     eva_data_init=eva_data_init).items():
        reporter.report_txt(f"{k}: {v}")

    tokenizer = get_tokenizer(cfg)
    tokenizer.padding_side = padding_side
    collator = get_collator(tokenizer)
    train_ds, val_ds, test_ds, graph_data = prepare_eff_splits(cfg, tokenizer, seed, split, max_train_samples)

    base_model = load_eff_model(cfg, tokenizer, attn_impl)
    model = build_lora_model(base_model, cfg, init, eva_data_init, train_ds, collator, micro_bs)
    model.print_trainable_parameters()

    ta = cfg.training_args
    args = TrainingArguments(
        output_dir=os.path.join("./outputs_eff", run_name),
        overwrite_output_dir=True,
        num_train_epochs=ta.num_train_epochs,
        per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=min(64, micro_bs * 4),
        learning_rate=ta.learning_rate,
        max_grad_norm=ta.max_grad_norm,
        weight_decay=ta.weight_decay,
        lr_scheduler_type=ta.lr_scheduler_type,
        warmup_ratio=ta.warmup_ratio,
        logging_steps=ta.logging_steps,
        eval_strategy=ta.eval_strategy,
        eval_steps=ta.eval_steps,
        save_strategy=ta.save_strategy,
        save_steps=ta.save_steps,
        save_total_limit=1,
        metric_for_best_model=ta.metric_for_best_model,
        load_best_model_at_end=ta.load_best_model_at_end,
        bf16=use_bf16,
        tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        group_by_length=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
    )

    def compute_metrics(p):
        return {"accuracy": float((p.predictions.argmax(-1) == p.label_ids).mean())}

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, processing_class=tokenizer, compute_metrics=compute_metrics)

    reset_cuda_peak_stats()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak = peak_vram_bytes()

    test_results = trainer.evaluate(eval_dataset=test_ds)
    val_results = trainer.evaluate(eval_dataset=val_ds)
    print(f"[train_eff] train {train_sec:.1f}s | peak VRAM {peak / 1024 ** 3:.2f} GB | "
          f"val acc {val_results['eval_accuracy']:.4f} | test acc {test_results['eval_accuracy']:.4f}")

    # Free optimizer/grad memory before the cache pass
    trainer.optimizer = None
    trainer.lr_scheduler = None
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    cache_sec, cache_path, cache_test_acc = float("nan"), "", float("nan")
    if not skip_cache:
        full = load_dataset(cfg)
        max_tokens = min(cfg.llm.caching_batch_size, micro_bs * 8) * cfg.dataset.max_length
        t1 = time.perf_counter()
        cache = build_cache_fast(model, tokenizer, full.raw_texts, full.y.view(-1).tolist(), cfg.dataset.max_length,
                                 max_tokens, pooling=pooling,
                                 meta=dict(model_name=llm_name, dataset_name=dataset_name,
                                           num_classes=cfg.dataset.num_classes, init_lora_weights=init,
                                           seed=seed, split=split, padding_side=padding_side))
        cache_sec = time.perf_counter() - t1
        os.makedirs(cache_root, exist_ok=True)
        cache_path = os.path.join(cache_root, cache_file_name(cfg, init, pooling, seed, split))
        torch.save(cache, cache_path)
        # Sanity check: cached logits must reproduce the trained model's test accuracy
        test_mask = graph_data.test_mask.cpu()
        preds = cache["logits"].argmax(-1)
        cache_test_acc = float((preds[test_mask] == cache["labels"][test_mask]).float().mean())
        print(f"[train_eff] cache {cache_sec:.1f}s -> {cache_path} | test acc from cached logits {cache_test_acc:.4f}")

    latency = measure_latency(model, test_ds, collator, batch_size=8)

    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if init in ("pissa", "loftq"):
        with open(os.path.join(adapter_dir, "WARNING_base_weights_modified.txt"), "w", encoding="utf-8") as f:
            f.write(f"{init} modified the base weights during init. This adapter reloaded onto the original "
                    f"base model is NOT the trained model; use the cache built in-process: {cache_path}\n")

    trainable, total = count_parameters(model)
    row = build_efficiency_row(cfg=cfg, split=split, wall_clock_train_sec=train_sec, peak_vram=peak,
                               adapter_dir=adapter_dir, inference_stats=latency, trainable_params=trainable,
                               total_params=total, test_accuracy=test_results.get("eval_accuracy"),
                               test_loss=test_results.get("eval_loss"))
    row.update(pipeline="train_eff_llm", attn_implementation=attn_impl, bf16=use_bf16, micro_bs=micro_bs,
               grad_accum=grad_accum, padding="dynamic", padding_side=padding_side, seed=seed,
               val_accuracy=val_results.get("eval_accuracy"), cache_wall_clock_sec=cache_sec,
               cache_test_accuracy=cache_test_acc, cache_path=cache_path, eva_data_init=eva_data_init)
    append_efficiency_csv(row, os.path.join("results", "efficiency", "lora_finetune_efficiency_eff.csv"))

    reporter.report_title("Results")
    for line in format_efficiency_block(row):
        reporter.report_txt(line)
    reporter.report_txt(f"Val accuracy: {val_results.get('eval_accuracy')}")
    reporter.report_txt(f"Test accuracy: {test_results.get('eval_accuracy')}")
    reporter.report_txt(f"Cache wall-clock (s): {cache_sec:.2f} | cache test acc: {cache_test_acc:.4f}")
    reporter.report_txt(f"Cache: {cache_path}")
    reporter.report_txt(f"Adapter: {adapter_dir}")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Efficient LoRA fine-tuning + in-process embedding caching")
    p.add_argument('--dataset_name', type=str, default='cora')
    p.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    p.add_argument('--peft_type', type=str, default='lora')
    p.add_argument('--init_weight_approach', type=str, default='gaussian', choices=INIT_APPROACHES)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--split', type=int, default=1, help='1 = supervised split, 0 = semi-supervised')
    p.add_argument('--micro_bs', type=int, default=4,
                   help='Per-step batch; grad accumulation keeps the original effective batch size')
    p.add_argument('--attn', type=str, default='auto', choices=['auto', 'flash_attention_2', 'sdpa', 'eager'])
    p.add_argument('--padding_side', type=str, default='right', choices=['right', 'left'],
                   help='right keeps position ids independent of batch composition under dynamic padding')
    p.add_argument('--pooling', type=str, default='mean', choices=['mean', 'last'])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--adapter_root', type=str, default='./artifacts_eff')
    p.add_argument('--cache_root', type=str, default='./artifacts_eff/cache',
                   help='Use ./artifacts/cache to feed the GNN/ensemble code directly (overwrites old caches)')
    p.add_argument('--eva_data_init', action='store_true',
                   help='Run PEFT data-driven EVA init (otherwise "eva" == default LoRA init, as in train_llm/)')
    p.add_argument('--max_train_samples', type=int, default=None, help='Debug: cap training set size')
    p.add_argument('--skip_cache', action='store_true')
    a = p.parse_args()
    main(a.dataset_name, a.llm_name, a.peft_type, a.init_weight_approach, a.seed, a.split, a.micro_bs, a.attn,
         a.padding_side, a.pooling, a.num_workers, a.adapter_root, a.cache_root, a.eva_data_init,
         a.max_train_samples, a.skip_cache)
