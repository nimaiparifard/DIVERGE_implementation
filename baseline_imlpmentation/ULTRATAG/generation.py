"""
Local, offline LLM text-generation backend for UltraTAG-S.

The paper (Zhang et al., "Toward General and Robust LLM-enhanced Text-attributed
Graph Learning", ICMR 2026) uses Meta-Llama-3-8B-Instruct via an API for every
LLM-generation step (text augmentation, edge reconfiguration confidence scoring).

This repo has no instruct-tuned model downloaded locally, no local Ollama server
running, and the API keys already checked into textual_enhancer/ and
tags_data_augmentation/ are leaked secrets that must not be reused (flagged during
implementation; do not copy that pattern). Per user decision, we instead download
a small, free instruction-tuned model once (Qwen2.5-1.5B-Instruct) and run it fully
offline afterward via transformers .generate() -- this keeps the method faithful
(a real instruction-following LLM makes every judgment) while being reproducible
without any API key. Weights are cached under ULTRATAG/models_cache/ only, so
nothing outside ULTRATAG/ is touched.
"""
import json
import os
import re

import torch

MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
DEFAULT_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class InstructLLM:
    """Thin wrapper around a small local instruction-tuned model for structured generation."""

    def __init__(self, repo_id=DEFAULT_REPO_ID, device=None, max_new_tokens=200, batch_size=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.repo_id = repo_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=MODELS_CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation with a decoder-only LM

        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id, cache_dir=MODELS_CACHE_DIR, torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _generate_batch(self, prompts):
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            self.device
        )
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output_ids[:, encoded["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def generate(self, prompts, show_progress=True, desc="Generating"):
        """prompts: list[str]. Returns list[str] of raw generated text, same order/length."""
        from tqdm import tqdm

        results = []
        iterator = range(0, len(prompts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=desc)
        for i, start in enumerate(iterator):
            batch = prompts[start:start + self.batch_size]
            results.extend(self._generate_batch(batch))
            # Many consecutive .generate() calls with varying prompt lengths can
            # fragment the CUDA caching allocator over a long run (observed on a
            # 1233-batch PubMed run: per-batch time crept from ~5s to ~18s before an
            # unrecoverable OOM around batch 1183/1233, with no Python traceback).
            # Periodically releasing cached-but-unused blocks keeps allocations
            # contiguous and prevents this degrade-then-crash pattern.
            if torch.cuda.is_available() and (i + 1) % 25 == 0:
                torch.cuda.empty_cache()
        return results


def parse_json_object(raw_text, default=None):
    """Best-effort extraction of a JSON object from an LLM's raw generated text."""
    if default is None:
        default = {}
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return dict(default)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(default)
