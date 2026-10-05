"""
LM-based Resilient Representation Learning (paper Section 4.2, Eq. 22-24).

Fine-tunes an LM classifier (via this repo's existing LoRA/PEFT infrastructure in
train_llm/) on the augmented texts T*, then extracts pooled node representations
H = LM(t_i*, theta*) from the fine-tuned encoder for every node, to be consumed by
the downstream dual-GNN (module 3). This reuses train_llm.peft_model (load_llm_model,
get_lora_model), train_llm.training_config, and dataset.dataset_loader's TagsDataset/
tokenizer helpers as-is; only the *source* of the texts (our augmented T* rather than
dataset.raw_texts) and output_dir/hyperparameters are new.
"""
import os

import torch
from sklearn.metrics import accuracy_score
from transformers import Trainer

from config import setup_finetuning_cfg
from dataset.dataset_loader import TagsDataset, encoding_texts_to_tokens, get_tokenizer
from train_llm.peft_model import get_lora_model, load_llm_model
from train_llm.training_config import get_training_config

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}


def finetune_lm_and_extract_embeddings(
    dataset_name,
    augmented_texts,
    labels,
    train_mask,
    val_mask,
    llm_name="llama_3.2_1B",
    peft_type="lora",
    lr=5e-5,
    epochs=3,
    batch_size=8,
    dropout=0.3,
    run_tag="run",
):
    """Returns (peft_model, tokenizer, node_embeddings[N, hidden_dim])."""
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)

    run_dir = os.path.join(ARTIFACTS_DIR, f"{dataset_name}_{llm_name}_{run_tag}")
    cfg.training_args.output_dir = run_dir
    cfg.training_args.logging_dir = os.path.join(run_dir, "logs")
    cfg.training_args.learning_rate = lr
    cfg.training_args.num_train_epochs = epochs
    cfg.training_args.per_device_train_batch_size = batch_size
    cfg.training_args.per_device_eval_batch_size = batch_size
    # configs/training_args/{dataset}.json defaults gradient_accumulation_steps to 16,
    # tuned for a per_device_train_batch_size of 1; left un-overridden it silently
    # multiplies every step here to batch_size*16 examples (measured: ~330s/step on
    # an RTX 5070 for Cora at batch_size=8, i.e. effective batch 128) -- override to 1
    # since batch_size is already an explicit, direct control here.
    cfg.training_args.gradient_accumulation_steps = 1
    cfg.training_args.eval_strategy = "no"
    cfg.training_args.save_strategy = "no"
    cfg.training_args.load_best_model_at_end = False

    tokenizer = get_tokenizer(cfg)
    base_model = load_llm_model(cfg, tokenizer=tokenizer)
    if hasattr(base_model.config, "hidden_dropout_prob"):
        base_model.config.hidden_dropout_prob = dropout
    if getattr(tokenizer, "pad_token_id", None) is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    model = get_lora_model(base_model, cfg)

    train_idx = train_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    val_idx = val_mask.nonzero(as_tuple=True)[0].cpu().tolist()

    train_texts = [augmented_texts[i] for i in train_idx]
    train_labels = [int(labels[i]) for i in train_idx]
    val_texts = [augmented_texts[i] for i in val_idx]
    val_labels = [int(labels[i]) for i in val_idx]

    train_encodings = encoding_texts_to_tokens(tokenizer, train_texts, max_length=cfg.dataset.max_length)
    val_encodings = encoding_texts_to_tokens(tokenizer, val_texts, max_length=cfg.dataset.max_length)
    train_dataset = TagsDataset(train_encodings, train_labels)
    val_dataset = TagsDataset(val_encodings, val_labels)

    training_args = get_training_config(cfg)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    embeddings = extract_pooled_embeddings(model, tokenizer, augmented_texts, max_length=cfg.dataset.max_length)
    return model, tokenizer, embeddings


@torch.no_grad()
def extract_pooled_embeddings(model, tokenizer, texts, max_length=512, batch_size=16):
    device = next(model.parameters()).device
    model.eval()
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = [t if t else "empty" for t in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        ).to(device)
        outputs = model(**encoded, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embeddings.append(pooled.float().cpu())
    return torch.cat(all_embeddings, dim=0)
