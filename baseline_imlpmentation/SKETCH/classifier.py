"""
Long-context LM classifier trained on the SKETCH-enriched node texts.

The paper fine-tunes a long-context LM (Nomic-BERT-2048 or Llama-3-8B) directly on
"anchor text + retrieved texts" with the node label as target -- supervised node
classification without any GNN. Here any local checkpoint of this repo
(configs/llm_configs/*.json, weights under local_models/) is used as a sequence
classifier with LoRA adapters; `llama_3.2_1B` (128k context) is the default
long-context backbone. The paper uses Adam, early stopping on validation accuracy
and 2 epochs (Section 4, Appendix A.3), mirrored here.
"""
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.metrics import compute_acc_and_f1
from config import load_llm_config_json
from train_llm.peft_model import _resolve_torch_dtype, get_model_path


def load_tokenizer(llm_name):
    from transformers import AutoTokenizer

    llm_cfg = load_llm_config_json(llm_name)
    tokenizer = AutoTokenizer.from_pretrained(
        get_model_path(llm_name),
        local_files_only=llm_cfg.tokenizer.local_files_only,
        trust_remote_code=llm_cfg.tokenizer.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = llm_cfg.tokenizer.get("padding_side", "right")
    tokenizer.truncation_side = "right"  # keep the anchor text, drop the tail of the retrieved context
    return tokenizer


def model_max_positions(llm_name):
    from transformers import AutoConfig

    llm_cfg = load_llm_config_json(llm_name)
    config = AutoConfig.from_pretrained(get_model_path(llm_name), local_files_only=True,
                                        trust_remote_code=llm_cfg.llm.trust_remote_code)
    return getattr(config, "max_position_embeddings", None)


def build_lora_classifier(llm_name, num_classes, tokenizer, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                          gradient_checkpointing=True, device="cuda:0"):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    llm_cfg = load_llm_config_json(llm_name)
    model_path = get_model_path(llm_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Local model not found at '{model_path}'.")

    kwargs = dict(local_files_only=llm_cfg.llm.local_files_only, trust_remote_code=llm_cfg.llm.trust_remote_code,
                  num_labels=num_classes, torch_dtype=_resolve_torch_dtype(llm_cfg.llm.get("torch_dtype", "bfloat16")))
    if "gemma" in llm_name.lower():
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=list(llm_cfg.llm.ft_target_modules),
        modules_to_save=list(llm_cfg.llm.module_to_save),
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    # Keep the (few) trainable weights in fp32 for stable updates; the frozen base stays in its own dtype
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, node_ids):
        self.texts, self.labels, self.node_ids = texts, labels, node_ids

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, i):
        v = self.node_ids[i]
        return self.texts[v], self.labels[v]


def _make_loader(texts, labels, node_ids, tokenizer, max_length, batch_size, shuffle, seed):
    def collate(batch):
        batch_texts, batch_labels = zip(*batch)
        enc = tokenizer(list(batch_texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return enc

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(_TextDataset(texts, labels, node_ids), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=gen)


def _autocast(device):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=dtype)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gold = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with _autocast(device):
            logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu())
        gold.append(labels)
    # Eval batches use different shapes than training; release them so the next
    # training steps don't spill into shared memory on 12 GB (WDDM) cards
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(preds).numpy(), torch.cat(gold).numpy()


def train_and_evaluate(model, tokenizer, texts, labels, train_idx, val_idx, test_idx, max_length,
                       epochs=2, lr=2e-4, weight_decay=0.0, batch_size=4, grad_accum=4, eval_batch_size=8,
                       evals_per_epoch=2, patience=3, seed=0, device="cuda:0"):
    """
    Trains the LoRA classifier with early stopping on validation accuracy and
    returns metrics of the best-validation checkpoint on val and test.
    """
    train_loader = _make_loader(texts, labels, train_idx, tokenizer, max_length, batch_size, True, seed)
    val_loader = _make_loader(texts, labels, val_idx, tokenizer, max_length, eval_batch_size, False, seed)
    test_loader = _make_loader(texts, labels, test_idx, tokenizer, max_length, eval_batch_size, False, seed)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))

    eval_every = max(1, len(train_loader) // max(1, evals_per_epoch))
    best_val, best_state, bad_evals = -1.0, None, 0
    history = []

    def trainable_state():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                if any(tag in k for tag in ("lora_", "modules_to_save"))}

    stop = False
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device):
                loss = model(**batch).loss / grad_accum
            loss.backward()
            running += loss.item()
            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{running:.4f}")
                running = 0.0

            if step % eval_every == 0 or step == len(train_loader):
                pred, gold = predict(model, val_loader, device)
                val_acc = compute_acc_and_f1(pred, gold)[0]
                history.append({"epoch": epoch + 1, "step": step, "val_acc": val_acc})
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} val_acc={val_acc:.2f}")
                if val_acc > best_val:
                    best_val, best_state, bad_evals = val_acc, trainable_state(), 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        print("  early stopping")
                        stop = True
                        break
                model.train()
        if stop:
            break

    model.load_state_dict(best_state, strict=False)
    val_pred, val_gold = predict(model, val_loader, device)
    test_pred, test_gold = predict(model, test_loader, device)
    val_acc, val_macro_f1, val_weighted_f1 = compute_acc_and_f1(val_pred, val_gold)
    test_acc, test_macro_f1, test_weighted_f1 = compute_acc_and_f1(test_pred, test_gold)
    return {
        "val_acc": val_acc, "val_macro_f1": val_macro_f1, "val_weighted_f1": val_weighted_f1,
        "test_acc": test_acc, "test_macro_f1": test_macro_f1, "test_weighted_f1": test_weighted_f1,
        "history": history,
    }
