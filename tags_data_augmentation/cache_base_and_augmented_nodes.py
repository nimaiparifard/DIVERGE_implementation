# Step 2 of the DIVERGE tags-augmentation pipeline (see approach.txt):
# after train_with_augmented_nodes.py fine-tunes an LLM adapter on BOTH the
# original graph nodes and the newly generated/searched nodes, this script
# runs that SAME adapter once over [base nodes] + [augmented nodes] and
# saves ONE combined embedding cache.
#
# Encoding both groups together (instead of caching them separately with
# two different scripts/models) guarantees the two embedding groups live in
# the same representation space, which connect_new_nodes_to_graphs_with_augmented_cache.py
# relies on when it predicts edges between augmented nodes and the existing graph.

import os
import torch
from typing import Literal, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset, TagsDataset, get_tokenizer
from train_llm.peft_model import load_llm_model
from train_llm.train import load_llm_with_lora_adapter
from adapter_dir import get_semi_supervised_augmented_adapter_dir
from tags_data_augmentation.agumented_dataset import load_augmented_texts


class CacheBaseAndAugmentedNodes:
    """
    Encodes the original graph's raw texts and the augmented (web-searched /
    LLM-generated) texts in a single forward pass, using the LoRA adapter
    that was jointly fine-tuned on both groups (train_with_augmented_nodes.py).

    The resulting cache stores one [num_base + num_augmented, H] embedding
    matrix plus `num_base_nodes` / `num_augmented_nodes` so downstream code
    can slice base vs. augmented embeddings back apart.
    """

    def __init__(
        self,
        cfg,
        adapter_dir: str,
        pooling: Literal["last", "mean"] = "mean",
        num_workers: int = 0,
        pin_memory: bool = True,
        used_based_model: bool = False,
    ):
        self.model_name = cfg.llm.model_name
        self.dataset_name = cfg.dataset.name
        self.pooling = pooling
        self.max_length = cfg.dataset.max_length
        self.batch_size = cfg.llm.caching_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.used_based_model = used_based_model
        self.init_lora_weights = cfg.peft.init_lora_weights

        # --- Base graph nodes ---
        base_dataset = load_dataset(cfg)
        base_texts = list(base_dataset.raw_texts)
        base_labels = base_dataset.y.tolist()
        base_dois = [None] * len(base_texts)
        self.num_base_nodes = len(base_texts)

        # --- Augmented nodes (web search + LLM generated) ---
        augmented_items = load_augmented_texts(self.dataset_name)
        aug_texts = [f"Title: {item['title']}\nAbstract: {item['abstract']}" for item in augmented_items]
        aug_labels = [int(item['label_id']) for item in augmented_items]
        aug_dois = [item.get('doi', '') for item in augmented_items]
        self.num_augmented_nodes = len(aug_texts)

        texts = base_texts + aug_texts
        self.labels = base_labels + aug_labels
        self.dois = base_dois + aug_dois

        # --- Load the jointly fine-tuned model (or the base model) ---
        if self.used_based_model:
            self.model = load_llm_model(cfg)
            self.tokenizer = get_tokenizer(cfg)
        else:
            if not os.path.exists(adapter_dir) or not os.path.exists(os.path.join(adapter_dir, 'adapter_config.json')):
                raise FileNotFoundError(
                    f"Adapter not found at: {adapter_dir}\n"
                    f"Train it first with train_with_augmented_nodes.py, or pass used_based_model=True."
                )
            self.model, self.tokenizer = load_llm_with_lora_adapter(adapter_dir=adapter_dir, cfg=cfg)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if hasattr(self.model.config, 'pad_token_id'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model.config, 'pad_token_id'):
            self.model.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None,
        )
        self.tag_dataset = TagsDataset(self.encodings, self.labels)

        # Will be filled by build_cache()
        self.cache_dict: Dict[str, Any] = {}

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        b_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[b_idx, last_idx]

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True)
        return summed / lengths.clamp(min=1)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return self._last_token_pool(hidden_states, attention_mask)
        elif self.pooling == "mean":
            return self._mean_pool(hidden_states, attention_mask)
        raise ValueError(f"Unknown pooling '{self.pooling}'")

    @torch.inference_mode()
    def build_cache(self):
        """
        Runs the model over [base nodes + augmented nodes] and builds:
          - embeddings: [num_base + num_augmented, H] float32 CPU
          - labels:     [num_base + num_augmented] int
          - logits/probs: same leading dim
          - dois: list (None for base nodes, DOI string for augmented nodes)
          - num_base_nodes / num_augmented_nodes: split point for slicing
        """
        device = next(self.model.parameters()).device
        loader = DataLoader(
            self.tag_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        embs, labs, logs, probs = [], [], [], []
        for batch in tqdm(loader, desc=f"Caching base+augmented nodes ({self.dataset_name})"):
            labels_b = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
            logits = outputs.logits
            last_hidden = outputs.hidden_states[-1]
            pooled = self._pool(last_hidden, batch["attention_mask"])

            embs.append(pooled.float().cpu())
            labs.append(labels_b.long().cpu())
            logs.append(logits.float().cpu())
            probs.append(torch.softmax(logits, dim=-1).float().cpu())

        self.cache_dict = {
            "embeddings": torch.cat(embs, dim=0),
            "labels": torch.cat(labs, dim=0),
            "logits": torch.cat(logs, dim=0),
            "probs": torch.cat(probs, dim=0),
            "dois": self.dois,
            "num_base_nodes": self.num_base_nodes,
            "num_augmented_nodes": self.num_augmented_nodes,
            "meta": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "pooling": self.pooling,
                "init_lora_weights": self.init_lora_weights,
                "used_based_model": self.used_based_model,
            },
        }
        return self.cache_dict

    def save_cache(self, save_root: str = "./artifacts/cache") -> str:
        if not self.cache_dict:
            raise RuntimeError("Cache is empty. Call build_cache() first.")

        os.makedirs(save_root, exist_ok=True)
        if self.used_based_model:
            fname = f"{self.model_name}_{self.dataset_name}_base_and_augmented_pool-{self.pooling}_based_model.pt"
        else:
            fname = f"{self.model_name}_{self.dataset_name}_base_and_augmented_lora_init-{self.init_lora_weights}_pool-{self.pooling}.pt"
        save_path = os.path.join(save_root, fname)
        torch.save(self.cache_dict, save_path)

        print(f"[OK] Saved combined base+augmented cache to: {save_path}")
        print(f"     - Base nodes:       {self.num_base_nodes}")
        print(f"     - Augmented nodes:  {self.num_augmented_nodes}")
        print(f"     - Embeddings shape: {self.cache_dict['embeddings'].shape}")
        return save_path


def load_base_and_augmented_cache(
    dataset_name, model_name='llama_3.2_1B', pooling='mean',
    init_weight_approach='pissa', cache_root=None,
):
    """Load a cache produced by CacheBaseAndAugmentedNodes.save_cache()."""
    if cache_root is not None:
        cache_roots = [cache_root]
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_roots = [
            os.path.join(current_dir, "artifacts", "cache"),
            os.path.join(os.path.dirname(current_dir), "artifacts", "cache"),
        ]

    # Try requested spelling first, then alternate spelling for guassian/gaussian
    init_variants = [init_weight_approach]
    if init_weight_approach == "guassian":
        init_variants.append("gaussian")
    elif init_weight_approach == "gaussian":
        init_variants.append("guassian")

    for root in cache_roots:
        for init_var in init_variants:
            fname = f"{model_name}_{dataset_name}_base_and_augmented_lora_init-{init_var}_pool-{pooling}.pt"
            path = os.path.join(root, fname)
            if os.path.exists(path):
                cache_dict = torch.load(path, weights_only=False)
                print(f"[OK] Loaded base+augmented cache from: {path}")
                print(f"     - Base nodes: {cache_dict['num_base_nodes']}, "
                      f"Augmented nodes: {cache_dict['num_augmented_nodes']}")
                return cache_dict

    raise FileNotFoundError(
        f"Base+augmented cache not found for dataset='{dataset_name}', model='{model_name}', "
        f"init='{init_weight_approach}', pooling='{pooling}'.\nLooked in: {cache_roots}\n"
        f"Generate it by running: python cache_base_and_augmented_nodes.py "
        f"--dataset_name {dataset_name} --llm_name {model_name} --init_weight_approaches {init_weight_approach}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Cache base graph nodes + augmented nodes together with the jointly fine-tuned LoRA adapter'
    )
    parser.add_argument('--dataset_name', type=str, default='citeseer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--init_weight_approaches', type=str, nargs='+',
                        default=['pissa', 'orthogonal', 'eva', 'loftq', 'gaussian'])
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'last'])
    parser.add_argument('--used_based_model', action='store_true')
    args = parser.parse_args()

    for init_weight in args.init_weight_approaches:
        cfg = setup_finetuning_cfg(dataset_name=args.dataset_name, llm_name=args.llm_name, peft_type=args.peft_type)
        cfg.peft.init_lora_weights = init_weight
        adapter_dir = get_semi_supervised_augmented_adapter_dir(cfg)

        print(f"\n{'='*80}\nCaching base+augmented nodes | init={init_weight}\n{'='*80}")
        print(f"[INFO] Adapter dir: {adapter_dir}")

        if not args.used_based_model and not os.path.exists(adapter_dir):
            print(f"[WARNING] Adapter not found, skipping: {adapter_dir}")
            print(f"[INFO] Train it first with train_with_augmented_nodes.py "
                  f"(init_weight_approach='{init_weight}')")
            continue

        cacher = CacheBaseAndAugmentedNodes(
            cfg=cfg,
            adapter_dir=adapter_dir,
            pooling=args.pooling,
            used_based_model=args.used_based_model,
        )
        cacher.build_cache()
        cacher.save_cache()
        print(f"[OK] Done for init approach: {init_weight}\n")
