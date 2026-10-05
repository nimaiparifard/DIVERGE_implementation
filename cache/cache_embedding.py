import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Literal, Dict, Any
from tqdm import tqdm

from dataset.dataset_loader import get_tokenizer, load_dataset, TagsDataset, load_gpt_enhanced_explanations, load_summary_explanation, load_paraphrased_explanation
from train_llm.peft_model import load_llm_model
from train_llm.train import load_llm_with_lora_adapter
from config import setup_finetuning_cfg
from adapter_dir import get_adapter_dir

class CacheEmbedding:
    """
    Cache sequence embeddings (and predictions) from a PEFT LLaMA seq-classification model.

    - Pools token-level hidden states to a single vector per sequence.
    - Supports 'last' (last non-pad token) or 'mean' pooling.
    - Stores embeddings (float32 on CPU), labels, logits, probs, and dataset indices.
    """

    def __init__(
        self,
        cfg,
        adapter_dir: str,
        pooling: Literal["last", "mean"] = "last",
        num_workers: int = 0,
        pin_memory: bool = True,
        used_llm_responses: bool = False,
        used_summary_texts=False,
        used_paraphrased_texts=False,
        used_based_model: bool = False,
    ):
        self.model_name = cfg.llm.model_name
        self.dataset_name = cfg.dataset.name
        self.pooling = pooling
        self.max_length = cfg.dataset.max_length
        self.batch_size = cfg.llm.caching_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.used_llm_responses = used_llm_responses
        self.used_summary_texts = used_summary_texts
        self.used_paraphrased_texts = used_paraphrased_texts

        # Load dataset to get num_classes first
        self.dataset = load_dataset(cfg)
        self.num_classes = int(torch.unique(self.dataset.y).numel())

        # Correct arg name is adapter_dir (not cache_dir)
        self.used_based_model = used_based_model
        if self.used_based_model:
            self.model = load_llm_model(cfg)
            self.tokenizer = get_tokenizer(cfg)
        else:
            self.model, self.tokenizer = load_llm_with_lora_adapter(
                adapter_dir=adapter_dir,
                cfg=cfg,
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        # Tokenizer safety (esp. for LLaMA causal models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CRITICAL: Set the model's pad_token_id to match the tokenizer
        if hasattr(self.model.config, 'pad_token_id'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            # For some models, we need to set it on the base model
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model.config, 'pad_token_id'):
                self.model.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Left padding is typical for causal LMs
        self.tokenizer.padding_side = "left"

        # Encode texts and build a HF-style dataset
        if used_llm_responses:
            texts = self._load_llm_responses_fallback()
        elif used_summary_texts:
            texts = load_summary_explanation(cfg)
        elif used_paraphrased_texts:
            texts = load_paraphrased_explanation(cfg)
        else:
            texts = self.dataset.raw_texts
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None,
        )
        self.labels = self.dataset.y.tolist()
        self.tag_dataset = TagsDataset(self.encodings, self.labels)

        # Will be filled by build_cache()
        self.cache_dict: Dict[str, Any] = {}

    def _load_llm_responses_fallback(self):
        # If you have your own loader for LLM responses, plug it here.
        # For now fall back to raw_texts.
        texts = load_gpt_enhanced_explanations(cfg)
        return texts


    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        attention_mask: [B, T] with 1 for real tokens; left padding means last index is the last token.
        """
        lengths = attention_mask.sum(dim=1)  # [B]
        last_idx = (lengths - 1).clamp(min=0)  # [B]
        b_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[b_idx, last_idx]  # [B, H]

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean over non-pad tokens.
        """
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        summed = (hidden_states * mask).sum(dim=1)          # [B, H]
        lengths = attention_mask.sum(dim=1, keepdim=True)   # [B, 1]
        return summed / lengths.clamp(min=1)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return self._last_token_pool(hidden_states, attention_mask)
        elif self.pooling == "mean":
            return self._mean_pool(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown pooling '{self.pooling}'")

    @torch.inference_mode()
    def build_cache(self):
        """
        Runs the model over the dataset and builds an in-memory cache:
          - embeddings: [N, H] float32 CPU
          - labels:     [N]    int
          - logits:     [N, C] float32 CPU
          - probs:      [N, C] float32 CPU
          - indices:    [N]    int (dataset order)
        """
        device = next(self.model.parameters()).device
        loader = DataLoader(
            self.tag_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        embs = []
        labs = []
        logs = []
        probs = []
        idxs = []

        n_seen = 0
        for batch_id, batch in enumerate(tqdm(loader, desc="Building cache")):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Single forward pass to get logits & hidden states
            outputs = self.model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits  # [B, num_labels]
            # hidden_states: tuple(layer_0, ..., layer_L); take last layer
            last_hidden = outputs.hidden_states[-1]  # [B, T, H]

            pooled = self._pool(last_hidden, batch["attention_mask"])  # [B, H]

            embs.append(pooled.float().cpu())
            labs.append(labels.long().cpu())
            logs.append(logits.float().cpu())
            probs.append(torch.softmax(logits, dim=-1).float().cpu())

            # Track absolute indices in original order
            bsz = labels.size(0)
            idxs.append(torch.arange(n_seen, n_seen + bsz))
            n_seen += bsz

        self.cache_dict = {
            "embeddings": torch.cat(embs, dim=0),  # [N, H]
            "labels": torch.cat(labs, dim=0),      # [N]
            "logits": torch.cat(logs, dim=0),      # [N, C]
            "probs": torch.cat(probs, dim=0),      # [N, C]
            "indices": torch.cat(idxs, dim=0),     # [N]
            "meta": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "pooling": self.pooling,
                "max_length": self.max_length,
                "num_classes": self.num_classes,
            },
        }
        return self.cache_dict


    @torch.inference_mode()
    def build_engine_cache(self):
        """
        Builds layer-wise embeddings cache for ENGINE method.
        Returns a list of tensors, one per layer (including embedding layer).
        Each tensor has shape [num_samples, hidden_dim].
        """
        device = next(self.model.parameters()).device
        loader = DataLoader(
            self.tag_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        # Initialize list of lists: one list per layer
        layers = None
        labs = []
        logs = []
        probs = []
        idxs = []

        n_seen = 0
        for batch_id, batch in enumerate(tqdm(loader, desc="Building ENGINE cache")):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Single forward pass to get logits & hidden states
            outputs = self.model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits  # [B, num_labels]
            hidden_states = outputs.hidden_states  # tuple(layer_0, ..., layer_L)
            
            # Initialize layers list on first batch
            if layers is None:
                num_layers = len(hidden_states)
                layers = [[] for _ in range(num_layers)]
            
            # Pool and store embeddings for each layer
            for i, layer_hidden in enumerate(hidden_states):
                pooled = self._pool(layer_hidden, batch["attention_mask"])  # [B, H]
                layers[i].append(pooled.float().cpu())
            
            # Store labels, logits, probs, indices
            labs.append(labels.long().cpu())
            logs.append(logits.float().cpu())
            probs.append(torch.softmax(logits, dim=-1).float().cpu())
            
            bsz = labels.size(0)
            idxs.append(torch.arange(n_seen, n_seen + bsz))
            n_seen += bsz
        
        # Concatenate all batches for each layer
        # Final output: list of [num_samples, hidden_dim] tensors
        layers_hid = [torch.cat(layer_batches, dim=0) for layer_batches in layers]
        
        # Print shapes for verification
        layers_shape = [tensor.shape for tensor in layers_hid]
        print(f"Layer-wise embeddings shapes: {layers_shape}")
        
        # Store in cache_dict
        self.cache_dict = {
            "layer_embeddings": layers_hid,  # list of [N, H] tensors
            "labels": torch.cat(labs, dim=0),      # [N]
            "logits": torch.cat(logs, dim=0),      # [N, C]
            "probs": torch.cat(probs, dim=0),      # [N, C]
            "indices": torch.cat(idxs, dim=0),     # [N]
            "meta": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "pooling": self.pooling,
                "max_length": self.max_length,
                "num_classes": self.num_classes,
                "num_layers": len(layers_hid),
            },
        }
        return self.cache_dict


    def save_engine_cache(self, save_root: str = "./artifacts/engine_cache") -> str:
        """
        Saves the ENGINE cache (layer-wise embeddings) to disk.
        """
        if not self.cache_dict or "layer_embeddings" not in self.cache_dict:
            raise RuntimeError("ENGINE cache is empty. Call build_engine_cache() first.")
        
        os.makedirs(save_root, exist_ok=True)
        if not self.used_llm_responses:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_engine_pool-{self.pooling}_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_engine_lora_pool-{self.pooling}.pt"
        else:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_engine_{self.pooling}_for_llm_responses_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_engine_{self.pooling}_for_llm_responses.pt"
        save_path = os.path.join(save_root, fname)
        torch.save(self.cache_dict, save_path)
        print(f"[OK] Saved ENGINE cache to: {save_path}")
        return save_path

    def save_cache(self, save_root: str = "./artifacts/cache") -> str:
        """
        Saves the current cache_dict to disk as a .pt file.
        """
        if not self.cache_dict:
            raise RuntimeError("Cache is empty. Call build_cache() first.")

        os.makedirs(save_root, exist_ok=True)
        if not self.used_llm_responses:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_pool-{self.pooling}_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_lora_pool-{self.pooling}.pt"
        else:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_{self.pooling}_for_llm_responses_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_{self.pooling}_for_llm_responses.pt"
        save_path = os.path.join(save_root, fname)
        torch.save(self.cache_dict, save_path)
        print(f"[OK] Saved cache to: {save_path}")
        return save_path


if __name__ == "__main__":
    # Example usage
    # adapter_dir = "./artifacts/llama_3.2_1B_cora_seqcls_with_llm_responses"  # path that contains adapter + tokenizer
    # cacher = CacheEmbedding(
    #     model_name="llama_3.2_1B",
    #     dataset_name="cora",
    #     adapter_dir=adapter_dir,
    #     batch_size=64,
    #     pooling="last",   # or "mean"
    #     used_llm_responses=True
    # )
    # cacher.build_cache()
    # cacher.save_cache()
    cfg = setup_finetuning_cfg(dataset_name='', llm_name='llama_3.2_1B', peft_type='lora')
    adapter_dir = get_adapter_dir(cfg,  used_llm_responses=False)
    cacher = CacheEmbedding(
        cfg,
        adapter_dir=adapter_dir,
        pooling="mean",   # or "mean"
        used_llm_responses=False,
        used_based_model=True
    )
    cacher.build_cache()
    cacher.save_cache()
