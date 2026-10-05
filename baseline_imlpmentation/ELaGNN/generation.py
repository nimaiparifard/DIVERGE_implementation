"""
LLM service for on-demand node enhancement (t_u -> t_u^+).

The paper queries Vicuna-7B / LLaMA-2-7B-Chat with temperature 0.7. This repo
already has a small instruction-tuned model downloaded for offline generation
(Qwen2.5-1.5B-Instruct, cached under ULTRATAG/models_cache/ by
ULTRATAG.generation.InstructLLM), so that model is reused here with sampling at the
paper's temperature.

"On-demand" in practice: an enhancement is generated the first time a
(node, prompt) pair is needed and stored in ELaGNN/cache/, so later epochs, seeds
and runs with a larger budget reuse it. Only nodes chosen by the active node
selection are ever sent to the LLM, so the budget directly bounds LLM cost.
"""
import json
import os

import torch

from ELaGNN.prompts import PROMPT_CATALOG, build_prompt
from ULTRATAG.generation import DEFAULT_REPO_ID, InstructLLM

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class SamplingInstructLLM(InstructLLM):
    """InstructLLM with temperature sampling instead of greedy decoding."""

    def __init__(self, temperature=0.7, top_p=0.95, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def _generate_batch(self, prompts):
        texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                    add_generation_prompt=True) for p in prompts]
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                 max_length=1024).to(self.device)
        output_ids = self.model.generate(
            **encoded, max_new_tokens=self.max_new_tokens, do_sample=True,
            temperature=self.temperature, top_p=self.top_p, pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(output_ids[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def augmentation_cache_path(dataset_name, repo_id):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{dataset_name}_{repo_id.replace('/', '__')}_augmentations.json")


def load_augmentations(dataset_name, repo_id=DEFAULT_REPO_ID):
    """{node_id (int): {prompt_id: generated text}}"""
    path = augmentation_cache_path(dataset_name, repo_id)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _save(dataset_name, repo_id, augmentations):
    path = augmentation_cache_path(dataset_name, repo_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in augmentations.items()}, f)
    os.replace(tmp, path)


def ensure_augmentations(dataset_name, raw_texts, label_names, node_ids, prompt_ids=None,
                         repo_id=DEFAULT_REPO_ID, max_new_tokens=200, batch_size=16, temperature=0.7,
                         seed=0, save_every=40):
    """Generate every missing (node, prompt) enhancement for `node_ids`; returns the full cache dict."""
    prompt_ids = list(prompt_ids or PROMPT_CATALOG.keys())
    augmentations = load_augmentations(dataset_name, repo_id)
    todo = [(int(v), pid) for v in node_ids for pid in prompt_ids if pid not in augmentations.get(int(v), {})]
    if not todo:
        return augmentations

    print(f"LLM enhancement: {len(todo)} (node, prompt) pairs to generate with {repo_id}")
    torch.manual_seed(seed)
    llm = SamplingInstructLLM(temperature=temperature, repo_id=repo_id, max_new_tokens=max_new_tokens,
                              batch_size=batch_size)
    from tqdm import tqdm

    chunk = batch_size * save_every  # checkpoint to disk regularly so long runs can resume
    for start in tqdm(range(0, len(todo), chunk), desc="Enhancement chunks"):
        pairs = todo[start:start + chunk]
        prompts = [build_prompt(pid, dataset_name, raw_texts[v], label_names) for v, pid in pairs]
        outputs = llm.generate(prompts, show_progress=False)
        for (v, pid), out in zip(pairs, outputs):
            augmentations.setdefault(v, {})[pid] = out.strip()
        _save(dataset_name, repo_id, augmentations)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return augmentations
