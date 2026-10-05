"""
Shared helpers for the efficient DIVERGE fine-tuning + caching pipeline.

Speedups vs train_llm/ (same method, same effective batch size):
  - no global padding: texts are tokenized unpadded and padded per batch
    (DataCollatorWithPadding), with length-grouped batches for training and
    length-sorted, token-budget batches for caching;
  - bf16 autocast + TF32 matmuls;
  - Flash-Attention 2 when installed, otherwise PyTorch SDPA;
  - larger micro-batch (gradient accumulation adjusted so micro_bs * accum
    equals the original per_device_train_batch_size * gradient_accumulation_steps).
"""
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers.utils import is_flash_attn_2_available

from common import load_graph_dataset_for_tape
from gnns.gnn_mtrainer import get_datasets_path
from train_llm.peft_model import get_model_path, _resolve_torch_dtype, _ensure_pad_token


def resolve_path_prefix():
    repo_root = os.path.dirname(get_datasets_path())
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            return '.'
        return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'


def enable_fast_matmul():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def bf16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def pick_attn_implementation(model_name, requested="auto"):
    """flash_attention_2 if installed, else sdpa; Gemma-2 keeps eager (as in train_llm/peft_model.py)."""
    if "gemma" in model_name.lower():
        return "eager"
    if requested != "auto":
        return requested
    return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"


def load_eff_model(cfg, tokenizer, attn_implementation):
    """Same as train_llm.peft_model.load_llm_model (non-4bit path) plus an explicit attention backend."""
    model_path = get_model_path(cfg.llm.model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Local model not found at '{model_path}'.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=_resolve_torch_dtype(cfg.llm.get("torch_dtype", "bfloat16")),
        device_map="cuda",
        local_files_only=cfg.llm.local_files_only,
        trust_remote_code=cfg.llm.trust_remote_code,
        num_labels=cfg.dataset.num_classes,
        attn_implementation=attn_implementation,
    )
    model = _ensure_pad_token(model, tokenizer=tokenizer)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model


class UnpaddedTextDataset(Dataset):
    """Token-id lists of varying length; padding happens per batch in the collator."""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": int(self.labels[idx]),
        }


def tokenize_unpadded(tokenizer, texts, labels, max_length):
    enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
    return UnpaddedTextDataset(enc["input_ids"], enc["attention_mask"], list(labels))


def get_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")


def prepare_eff_splits(cfg, tokenizer, seed, split, max_train_samples=None):
    """Same split/text source as dataset.dataset_loader.prepare_data_for_finetuning (raw texts), unpadded."""
    data, _, _ = load_graph_dataset_for_tape(cfg.dataset.name, cfg.device, re_split=split,
                                             path_prefix=resolve_path_prefix(), seed=seed)
    texts = data.raw_texts
    y = data.y.squeeze()
    splits = {}
    for name, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        if name == "train" and max_train_samples:
            idx = idx[:max_train_samples]
        splits[name] = tokenize_unpadded(tokenizer, [texts[i] for i in idx], y[idx].tolist(), cfg.dataset.max_length)
    print(f"Train size: {len(splits['train'])}, Val size: {len(splits['val'])}, Test size: {len(splits['test'])}")
    return splits["train"], splits["val"], splits["test"], data


def token_budget_batches(lengths, max_tokens, max_batch_size=256):
    """Sort by length (desc) and pack indices so padded tokens per batch stay <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_batch_size):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def cache_file_name(cfg, init_name, pooling, seed, split):
    """Matches the names dataset.data_utils.get_init_dataset_for_gnn looks for."""
    base = f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-{init_name}_pool-{pooling}"
    if split != 1:
        base += f"_seed{seed}_semi_supervised"
    return base + ".pt"


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def last_token_pool(hidden, mask):
    # Works for both padding sides: index of the right-most real token.
    positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    last_idx = (positions * mask).argmax(dim=1)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]


@torch.inference_mode()
def build_cache_fast(model, tokenizer, texts, labels, max_length, max_tokens, pooling="mean", meta=None):
    """
    Embedding cache with the same keys/order as cache/cache_embedding.py, but with
    length-sorted token-budget batches and per-batch padding.
    """
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
    lengths = [len(ids) for ids in enc["input_ids"]]
    collator = get_collator(tokenizer)
    pool = mean_pool if pooling == "mean" else last_token_pool
    use_bf16 = bf16_supported()

    n = len(lengths)
    embs, logits_all = [None] * n, [None] * n
    batches = token_budget_batches(lengths, max_tokens)
    for b_idx, idxs in enumerate(batches):
        batch = collator([{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in idxs])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**batch, output_hidden_states=True, return_dict=True)
        pooled = pool(out.hidden_states[-1], batch["attention_mask"]).float().cpu()
        logits = out.logits.float().cpu()
        for j, i in enumerate(idxs):
            embs[i] = pooled[j]
            logits_all[i] = logits[j]
        if b_idx % 50 == 0:
            print(f"  cache batch {b_idx + 1}/{len(batches)} (size={len(idxs)}, max_len={lengths[idxs[0]]})")

    logits_t = torch.stack(logits_all)
    return {
        "embeddings": torch.stack(embs),
        "labels": torch.as_tensor(labels).long(),
        "logits": logits_t,
        "probs": torch.softmax(logits_t, dim=-1),
        "indices": torch.arange(n),
        "meta": dict(meta or {}, pooling=pooling, max_length=max_length),
    }


@torch.inference_mode()
def measure_latency(model, dataset, collator, batch_size=8, num_warmup=2, num_batches=15):
    import time
    model.eval()
    device = next(model.parameters()).device
    n_total = min(len(dataset), batch_size * (num_warmup + num_batches))
    items = [dataset[i] for i in range(n_total)]
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    use_bf16 = bf16_supported()

    def run(b):
        batch = collator([{k: v for k, v in it.items() if k != "labels"} for it in b])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            model(**batch)

    for b in batches[:num_warmup]:
        run(b)
    timed = batches[num_warmup:] or batches
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for b in timed:
        run(b)
        n += len(b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "inference_ms_per_sample": elapsed * 1000.0 / max(n, 1),
        "inference_samples_per_sec": n / max(elapsed, 1e-12),
        "inference_num_samples": float(n),
        "inference_batch_size": float(batch_size),
    }
