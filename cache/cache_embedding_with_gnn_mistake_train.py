
## cache embedding for the trained model with additional mistake for top 3 gnn mistake rate in retrain_llm_with_gnn_mistakes.py
## you shold first write get_adapter_dir_lora_train_weights_for_gnn_mistake in adapter_dir.py similar to get_adapter_dir_lora_init_weights
## then cache the embedding with the following code in cache_embedding_with_diffrent_init_weights.py

import os
import torch
from cache.cache_embedding import CacheEmbedding
from train_llm.peft_model import load_llm_model, get_lora_model
from dataset.dataset_loader import load_dataset, get_tokenizer
from config import setup_finetuning_cfg
from train_llm.train import Train, load_llm_with_lora_adapter, evaluate_sampled_examples
from adapter_dir import get_adapter_dir_lora_train_weights_for_gnn_mistake


class CacheEmbeddingGNNMistakeTrain(CacheEmbedding):
    def __init__(self, cfg, adapter_dir: str, pooling: str = "last", 
                 num_workers: int = 0, pin_memory: bool = True,
                 used_llm_responses: bool = False, used_based_model: bool = False):
        super().__init__(cfg, adapter_dir, pooling, num_workers, pin_memory, 
                        used_llm_responses, used_based_model=used_based_model)
        # Store the init weight approach from config
        self.init_lora_weights = cfg.peft.init_lora_weights
    
    def save_cache(self, save_root: str = "./artifacts/cache") -> str:
        """
        Override save_cache to include init_lora_weights approach and gnn_mistake suffix in filename.
        """
        if not self.cache_dict:
            raise RuntimeError("Cache is empty. Call build_cache() first.")

        os.makedirs(save_root, exist_ok=True)
        
        # Include init_lora_weights and gnn_mistake in the filename
        if not self.used_llm_responses:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_pool-{self.pooling}_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_lora_init-{self.init_lora_weights}_pool-{self.pooling}_gnn_mistake.pt"
        else:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_{self.pooling}_for_llm_responses_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_init-{self.init_lora_weights}_{self.pooling}_for_llm_responses_gnn_mistake.pt"
        
        save_path = os.path.join(save_root, fname)
        torch.save(self.cache_dict, save_path)
        print(f"[OK] Saved cache with init approach '{self.init_lora_weights}' (GNN mistake retrained) to: {save_path}")
        return save_path


# List of initialization approaches to cache
init_weights_approaches_list = ['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal']

for approach in init_weights_approaches_list:
    cfg = setup_finetuning_cfg(dataset_name='pubmed', llm_name='llama_3.2_1B', peft_type='lora')
    cfg.peft.init_lora_weights = approach
    print("=== Caching embeddings for LoRA initialization approach (GNN mistake retrained):", cfg.peft.init_lora_weights)

    # Get adapter directory for GNN mistake retrained model
    adapter_dir = get_adapter_dir_lora_train_weights_for_gnn_mistake(cfg, used_llm_responses=False)
    
    # Check if adapter directory exists before caching
    if not os.path.exists(adapter_dir):
        print(f"⚠ Warning: Adapter directory not found: {adapter_dir}")
        print(f"  Skipping {approach}. Please train the model first using retrain_llm_with_gnn_mistakes.py")
        continue
    
    # Use the custom CacheEmbeddingGNNMistakeTrain class
    cacher = CacheEmbeddingGNNMistakeTrain(
        cfg,
        adapter_dir=adapter_dir,
        pooling="mean",
        used_llm_responses=False,
        used_based_model=True
    )
    cacher.build_cache()
    cacher.save_cache()
    print(f"✓ Cache saved for GNN mistake retrained model with init approach: {approach}\n")