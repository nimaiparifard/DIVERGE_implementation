"""
LM finetuning + node-embedding extraction for TAPE's h_orig / h_expl (Eq. 4-5).

Reuses this repo's existing LoRA/PEFT finetuning class (`train_llm.train.Train`)
as-is -- it already supports exactly the two text sources TAPE needs via its
`used_llm_responses` flag: raw title+abstract (h_orig) or the cached LLM response
text (h_expl, via `dataset.dataset_loader.load_gpt_enhanced_explanations`). We call
`Train._train()` + `.save_model()` directly rather than `.process()` to skip
plotting/efficiency-CSV side effects not needed here, then extract pooled node
embeddings with `ULTRATAG.lm_finetune.extract_pooled_embeddings` (a generic
(model, tokenizer, texts) -> embeddings helper, reused unchanged from ULTRATAG/).

Note: `cache.cache_embedding.CacheEmbedding`'s `used_llm_responses=True` path has a
pre-existing bug in this repo (`_load_llm_responses_fallback` references an
undefined `cfg`) -- we avoid it entirely by extracting embeddings ourselves.
"""
import os

import torch

from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset, load_gpt_enhanced_explanations
from report.reporter import ReportResults
from train_llm.train import Train, load_llm_with_lora_adapter
from ULTRATAG.lm_finetune import extract_pooled_embeddings

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def finetune_and_extract(
    dataset_name,
    llm_name,
    used_llm_responses,
    seed,
    run_tag,
    lr=2e-4,
    epochs=3,
    batch_size=8,
    grad_accum=2,
):
    """Returns node embeddings [N, hidden_dim] from an LM LoRA-finetuned on either
    the raw text (used_llm_responses=False, h_orig) or the cached LLM response text
    (used_llm_responses=True, h_expl)."""
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type="lora")
    cfg.dataset.seed = seed  # controls Train's internal 60/20/20 re-split (must match downstream GNN's split)

    save_dir = os.path.join(ARTIFACTS_DIR, f"{dataset_name}_{llm_name}_{run_tag}_seed{seed}")
    cfg.training_args.output_dir = save_dir
    cfg.training_args.logging_dir = os.path.join(save_dir, "logs")
    cfg.training_args.per_device_train_batch_size = batch_size
    cfg.training_args.per_device_eval_batch_size = batch_size
    cfg.training_args.gradient_accumulation_steps = grad_accum
    cfg.training_args.num_train_epochs = epochs
    cfg.training_args.learning_rate = lr
    cfg.training_args.eval_strategy = "no"
    cfg.training_args.save_strategy = "no"
    cfg.training_args.load_best_model_at_end = False

    reporter = ReportResults(cfg, save_dir=save_dir, index_run=seed)
    trainer = Train(cfg, save_dir, reporter, used_llm_responses=used_llm_responses, split=1)
    trainer._train()
    adapter_dir = trainer.save_model()

    model, tokenizer = load_llm_with_lora_adapter(adapter_dir, cfg)
    model.eval()

    if used_llm_responses:
        texts = load_gpt_enhanced_explanations(cfg)
    else:
        texts = load_dataset(cfg).raw_texts

    embeddings = extract_pooled_embeddings(model, tokenizer, texts, max_length=cfg.dataset.max_length)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings
