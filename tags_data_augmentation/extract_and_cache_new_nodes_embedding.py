# for saved augmented texts load and extract last layer and mean pooling on text and save the embedding
# load pretrain llm model 
## like from LLMReasoner.TAPE.peft_tape_finetuning.train import load_llm_with_lora_adapter
## we want to like CacheEmbedding save the embedding with pretrainre llm like in cache_embedding.py
## we extract last layer and mean pooling on text and save the embedding
# save embedding in likee cache_embedding.py 
# but some diffrence i want to know the label_id of the saved cache to use later to add it into graph
# you should read C:\Users\Nima\Desktop\Python Project\LLMNodeBed\LLMReasoner\TAPE\tags_data_augmentation\augmented_data\{dataset_name}.json
"""
the format of json file is like this:
{
    "title": "title of the paper",
    "abstract": "abstract of the paper",
    "doi": "doi of the paper",
    "label_id": "label_id of the paper",
    "label_name": "label_name of the paper"
}
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from typing import Literal, Dict, Any
from tqdm import tqdm

from train_llm.train import load_llm_with_lora_adapter
from cache.cache_embedding import CacheEmbedding
from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset, TagsDataset, get_tokenizer
from train_llm.peft_model import load_llm_model
from adapter_dir import get_semi_supervised_augmented_adapter_dir

class cacheAugmentedTexts:
    """
    Cache embeddings for augmented texts from JSON files.
    Similar to CacheEmbedding but for newly generated augmented data.
    """
    
    def __init__(
        self,
        cfg,
        adapter_dir: str,
        dataset_name: str,
        pooling: Literal["last", "mean"] = "mean",
        num_workers: int = 0,
        pin_memory: bool = True,
        used_based_model: bool = False,
        augment_data_path: str = None,
    ):
        self.model_name = cfg.llm.model_name
        self.dataset_name = dataset_name
        self.pooling = pooling
        self.max_length = cfg.dataset.max_length
        self.batch_size = cfg.llm.caching_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.used_based_model = used_based_model
        # Store the init weight approach from config
        self.init_lora_weights = cfg.peft.init_lora_weights
        self.augment_data_path = augment_data_path
        
        # Load augmented data from JSON
        self.augmented_data = self._load_augmented_data()
        
        # Get num_classes from the dataset
        dataset = load_dataset(cfg)
        self.num_classes = int(torch.unique(dataset.y).numel())
        
        # Load model with LoRA adapter or base model
        if self.used_based_model:
            self.model = load_llm_model(cfg)
            self.tokenizer = get_tokenizer(cfg)
        else:
            # Check if adapter directory exists
            if not os.path.exists(adapter_dir):
                raise FileNotFoundError(
                    f"Adapter directory not found: {adapter_dir}\n"
                    f"Please either:\n"
                    f"  1. Train the LoRA adapter first, or\n"
                    f"  2. Set used_based_model=True to use the base model without adapter"
                )
            if not os.path.exists(os.path.join(adapter_dir, 'adapter_config.json')):
                raise FileNotFoundError(
                    f"Adapter config not found in: {adapter_dir}\n"
                    f"The directory exists but doesn't contain a valid adapter.\n"
                    f"Please train the adapter first or use used_based_model=True"
                )
            self.model, self.tokenizer = load_llm_with_lora_adapter(
                adapter_dir=adapter_dir,
                cfg=cfg,
            )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Set model's pad_token_id
        if hasattr(self.model.config, 'pad_token_id'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model.config, 'pad_token_id'):
            self.model.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model.eval()
        
        # Prepare texts and labels for encoding
        texts, labels, dois = self._prepare_texts_and_labels()
        
        # Tokenize texts
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors=None,
        )
        self.labels = labels
        self.dois = dois
        self.tag_dataset = TagsDataset(self.encodings, self.labels)
        
        # Will be filled by build_cache()
        self.cache_dict: Dict[str, Any] = {}
    
    def _load_augmented_data(self):
        """Load augmented data from JSON file"""
        if not self.augment_data_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'augmented_data', f'{self.dataset_name}.json')
        else:
            json_path = self.augment_data_path

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Augmented data file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            augmented_data = json.load(f)
        
        print(f"[OK] Loaded {len(augmented_data)} augmented samples from {json_path}")
        # same leakage guard as agumented_dataset.load_augmented_texts, so cache rows stay aligned with it
        from tags_data_augmentation.agumented_dataset import drop_items_overlapping_dataset
        return drop_items_overlapping_dataset(self.dataset_name, augmented_data)
    
    def _prepare_texts_and_labels(self):
        """Prepare texts and labels from augmented data"""
        texts = []
        labels = []
        dois = []
        
        for item in self.augmented_data:
            # Combine title and abstract
            text = f"Title: {item['title']}\nAbstract: {item['abstract']}"
            texts.append(text)
            labels.append(int(item['label_id']))
            dois.append(item.get('doi', ''))
        
        return texts, labels, dois
    
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
        """Mean over non-pad tokens."""
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
        Runs the model over the augmented dataset and builds an in-memory cache:
          - embeddings: [N, H] float32 CPU
          - labels:     [N]    int (label_id)
          - logits:     [N, C] float32 CPU
          - probs:      [N, C] float32 CPU
          - indices:    [N]    int (dataset order)
          - dois:       [N]    str (DOI identifiers)
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
        for batch_id, batch in enumerate(tqdm(loader, desc="Building augmented cache")):
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
            "dois": self.dois,                     # list of DOIs
            "meta": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "pooling": self.pooling,
                "max_length": self.max_length,
                "num_classes": self.num_classes,
                "num_augmented_samples": len(self.augmented_data),
            },
        }
        return self.cache_dict
    
    def save_cache(self, save_root: str = "./artifacts/augmented_cache") -> str:
        """
        Saves the current cache_dict to disk as a .pt file.
        """
        if not self.cache_dict:
            raise RuntimeError("Cache is empty. Call build_cache() first.")
        
        os.makedirs(save_root, exist_ok=True)
        if self.used_based_model:
            fname = f"{self.model_name}_{self.dataset_name}_augmented_pool-{self.pooling}_based_model.pt"
        else:
            fname = f"{self.model_name}_{self.dataset_name}_augmented_lora_init-{self.init_lora_weights}_pool-{self.pooling}.pt"
        
        save_path = os.path.join(save_root, fname)
        torch.save(self.cache_dict, save_path)
        print(f"[OK] Saved augmented cache with init approach '{self.init_lora_weights}' to: {save_path}")
        print(f"     - Embeddings shape: {self.cache_dict['embeddings'].shape}")
        print(f"     - Labels: {len(self.cache_dict['labels'])} samples")
        print(f"     - DOIs: {len(self.cache_dict['dois'])} identifiers")
        return save_path

# def get_adapter_dir_lora_init_weights_globally(cfg, used_llm_responses=False, supervised=False):
#     """
#     Get adapter directory using global absolute path.
#     Points to: LLMReasoner/TAPE/peft_tape_finetuning/artifacts
#     """
#     # Get the project root directory (LLMNodeBed)
#     current_file = os.path.abspath(__file__)
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
#
#     # Build path to peft_tape_finetuning/artifacts
#     base_dir = os.path.join(project_root, "LLMReasoner", "TAPE", "peft_tape_finetuning", "artifacts")
#
#     if not used_llm_responses:
#         adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}")
#     else:
#         adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}")
#
#     if not supervised:
#         adapter_dir = os.path.join('../../../artifacts/', f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_semi_supervised_init_type_{cfg.peft.init_lora_weights}")
#     return adapter_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Cache embeddings for ONLY the augmented nodes (legacy/two-cache pipeline; '
                    'see cache_base_and_augmented_nodes.py for the combined base+augmented cache)'
    )
    parser.add_argument('--dataset_name', type=str, default='citeseer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--init_weight_approaches', type=str, nargs='+',
                        default=['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal'])
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'last'])
    parser.add_argument('--used_based_model', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    llm_name = args.llm_name
    init_weights_approaches_list = args.init_weight_approaches

    for approach in init_weights_approaches_list:
        cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=args.peft_type)
        cfg.peft.init_lora_weights = approach  # Change this to try different initialization approaches
        print(f"\n{'='*60}")
        print(f"Using LoRA initialization approach: {cfg.peft.init_lora_weights}")
        print(f"{'='*60}")
        
        # Get adapter directory for this specific initialization approach
        adapter_dir = get_semi_supervised_augmented_adapter_dir(cfg)
        print(f"[INFO] Using LoRA adapter from: {adapter_dir}")
        
        # Check if adapter exists, if not skip
        if not os.path.exists(adapter_dir):
            print(f"[WARNING] Adapter not found, skipping: {adapter_dir}")
            print(f"[INFO] Please train the adapter first or change dataset_name to 'instagram'")
            continue
        
        cacher = cacheAugmentedTexts(
            cfg=cfg,
            adapter_dir=adapter_dir,
            dataset_name=dataset_name,
            pooling=args.pooling,
            used_based_model=args.used_based_model
        )
        cacher.build_cache()
        cacher.save_cache()
        print(f"✓ Cache saved for init approach: {approach}\n") 