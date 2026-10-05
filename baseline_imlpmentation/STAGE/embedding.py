"""
Zero-shot LLM text embedding generation for STAGE (Zolnai-Lucas et al., 2024).

Mirrors the paper's Section 3.1 "Text Embedding Retrieval": a frozen, off-the-shelf
pre-trained LLM encodes each node's raw text attribute into a fixed-size vector via
mean-pooling of the final hidden layer. No finetuning, no LoRA.

We reuse this repo's existing local model checkpoints (configs/llm_configs/*.json,
local_models/*) instead of the paper's 7B MTEB-leaderboard embedding models, since
those checkpoints are already downloaded and sized for this project's hardware.
"""
import os

import torch

from config import load_llm_config_json
from train_llm.peft_model import get_model_path


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted


class StageTextEmbedder:
    """Frozen LLM encoder producing one pooled embedding vector per input text."""

    def __init__(self, llm_name="llama_3.2_1B", device=None, max_length=512, batch_size=16):
        from transformers import AutoModel, AutoTokenizer

        self.llm_name = llm_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        llm_cfg = load_llm_config_json(llm_name)
        model_path = get_model_path(llm_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Local model not found at '{model_path}'. STAGE reuses this repo's "
                f"local_models/ checkpoints -- place weights there or pick a different llm_name."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=llm_cfg.tokenizer.local_files_only,
            trust_remote_code=llm_cfg.tokenizer.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=llm_cfg.llm.local_files_only,
            trust_remote_code=llm_cfg.llm.trust_remote_code,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts, instruction=None, show_progress=True):
        """
        texts: list[str] -- one raw text attribute per graph node.
        instruction: optional str prefix prepended to every text to bias the
            embedding towards the downstream task (paper Table 2 / Appendix H).
        Returns a [len(texts), hidden_dim] float32 CPU tensor.
        """
        from tqdm import tqdm

        all_embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Embedding texts with {self.llm_name}")

        for start in iterator:
            batch = texts[start:start + self.batch_size]
            batch = [t if t else "empty" for t in batch]
            if instruction:
                batch = [f"{instruction}\n{t}" for t in batch]

            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            output = self.model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.float().cpu())

        return torch.cat(all_embeddings, dim=0)


def get_stage_cache_dir(cache_dir=None):
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_stage_cache_path(dataset_name, llm_name, instruction_tag="no_instruction", cache_dir=None):
    cache_dir = get_stage_cache_dir(cache_dir)
    return os.path.join(cache_dir, f"{dataset_name}_{llm_name}_{instruction_tag}_stage_embeddings.pt")


def get_or_build_stage_embeddings(
    dataset_name,
    raw_texts,
    llm_name="llama_3.2_1B",
    device=None,
    instruction=None,
    instruction_tag="no_instruction",
    cache_dir=None,
    force_rebuild=False,
):
    """Load cached STAGE text embeddings for (dataset, llm, instruction) or build+cache them."""
    cache_path = get_stage_cache_path(dataset_name, llm_name, instruction_tag, cache_dir)
    if os.path.exists(cache_path) and not force_rebuild:
        cached = torch.load(cache_path, weights_only=False)
        return cached["embeddings"]

    embedder = StageTextEmbedder(llm_name=llm_name, device=device)
    embeddings = embedder.encode(raw_texts, instruction=instruction)
    torch.save(
        {
            "embeddings": embeddings,
            "llm_name": llm_name,
            "dataset": dataset_name,
            "instruction": instruction,
        },
        cache_path,
    )

    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings
