import os
import torch
from cache.cache_embedding import CacheEmbedding
from train_llm.peft_model import load_llm_model, get_lora_model
from dataset.dataset_loader import load_dataset, get_tokenizer
from config import setup_finetuning_cfg
from train_llm.train import Train, load_llm_with_lora_adapter, evaluate_sampled_examples
from adapter_dir import get_adapter_dir_lora_init_weights, get_adapter_dir_lora_train_weights_for_paraphrased, get_adapter_dir_lora_train_weights_for_summary, get_semi_supervised_adapter_dir


# cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
# init_weights_approaches_list =  ['pissa', 'gaussian', 'eva', 'olora', 'loftq', 'orthogonal']


class CacheEmbeddingDiffrentInitWeights(CacheEmbedding):
    def __init__(self, cfg, adapter_dir: str, pooling: str = "last", 
                 num_workers: int = 0, pin_memory: bool = True,
                 used_llm_responses: bool = False, used_summary_texts=False ,
                 used_paraphrased_texts=False, used_based_model: bool = False):
        super().__init__(cfg, adapter_dir, pooling, num_workers, pin_memory,
                        used_llm_responses, used_summary_texts, used_paraphrased_texts, used_based_model)
        # Store the init weight approach and seed from config
        self.init_lora_weights = cfg.peft.init_lora_weights
        self.seed = cfg.dataset.seed

    def save_cache(self, save_root: str = "./artifacts/cache", split=1) -> str:
        """
        Override save_cache to include init_lora_weights approach in filename.
        """
        if not self.cache_dict:
            raise RuntimeError("Cache is empty. Call build_cache() first.")

        os.makedirs(save_root, exist_ok=True)
        
        # Include init_lora_weights in the filename
        if not self.used_llm_responses:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_pool-{self.pooling}_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_lora_init-{self.init_lora_weights}_pool-{self.pooling}.pt"
        else:
            if self.used_based_model:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_{self.pooling}_for_llm_responses_based_model.pt"
            else:
                fname = f"{self.model_name}_{self.dataset_name}_seqcls_init-{self.init_lora_weights}_{self.pooling}_for_llm_responses.pt"
        if self.used_summary_texts:
            fname = f"{self.model_name}_{self.dataset_name}_seqcls_lora_init-{self.init_lora_weights}_pool-{self.pooling}_summary.pt"
        if self.used_paraphrased_texts:
            fname = f"{self.model_name}_{self.dataset_name}_seqcls_lora_init-{self.init_lora_weights}_pool-{self.pooling}_paraphrased.pt"

        if split != 1:
            from adapter_dir import lora_run_tag
            tag = lora_run_tag(self.seed, getattr(self, 'split_seed', None), getattr(self, 'init_seed', None))
            fname = f"{self.model_name}_{self.dataset_name}_seqcls_lora_init-{self.init_lora_weights}_pool-{self.pooling}{tag}_semi_supervised.pt"
        save_path = os.path.join(save_root, fname)
        torch.save(self.cache_dict, save_path)
        print(f"[OK] Saved cache with init approach '{self.init_lora_weights}' to: {save_path}")
        return save_path

def main(dataset_name, llm_name, peft_type, init_weight_approach, split,
         used_summary_texts, used_paraphrased_texts, used_llm_responses,
         used_based_model, pooling, seed=None, batch_size=None, split_seed=None, init_seed=None):
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    if batch_size is not None:
        cfg.llm.caching_batch_size = batch_size  # inference only: embeddings do not depend on it
    cfg.peft.init_lora_weights = init_weight_approach  # change this to try different initialization approaches
    if seed is not None:
        cfg.dataset.seed = seed  # override to match the seed used when the adapter was trained/saved
    print("=== Using LoRA initialization approach:", cfg.peft.init_lora_weights)
    print("=== Using seed:", cfg.dataset.seed)

    # Get adapter directory for this specific initialization approach
    if used_summary_texts:
        adapter_dir = get_adapter_dir_lora_train_weights_for_summary(cfg)
    elif used_paraphrased_texts:
        adapter_dir = get_adapter_dir_lora_train_weights_for_paraphrased(cfg)
    else:
        adapter_dir = get_adapter_dir_lora_init_weights(cfg, used_llm_responses=used_llm_responses)

    if split != 1:
        # Must match the adapter_dir naming used in train_llm/train_other_lora_init_apporaches.py
        from adapter_dir import lora_run_tag
        adapter_dir = f'{get_semi_supervised_adapter_dir(cfg)}{lora_run_tag(cfg.dataset.seed, split_seed, init_seed)}'
    print(adapter_dir)
    
    # Use the custom CacheEmbeddingDiffrentInitWeights class
    cacher = CacheEmbeddingDiffrentInitWeights(
        cfg,
        adapter_dir=adapter_dir,
        pooling=pooling,
        used_llm_responses=used_llm_responses,
        used_summary_texts=used_summary_texts,
        used_paraphrased_texts=used_paraphrased_texts,
        used_based_model=used_based_model,
    )
    cacher.split_seed = split_seed
    cacher.init_seed = init_seed
    cacher.build_cache()
    cacher.save_cache(split=split)
    print(f"✓ Cache saved for init approach: {init_weight_approach}\n")


# init_weights_approaches_list = ['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal']  # Add more as needed
if __name__ == "__main__":
    ## argument definition
    import argparse
    
    parser = argparse.ArgumentParser(description='Cache embeddings with different LoRA initialization approaches')
    parser.add_argument('--dataset_name', type=str, default='pubmed',
                        help='Dataset name (e.g., cora, citeseer, pubmed, wikics)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='LLM model name')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='PEFT type (e.g., lora)')
    parser.add_argument('--init_weight_approach', type=str, default='gaussian',
                        choices=['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal'],
                        help='LoRA initialization approach')
    parser.add_argument('--split', type=int, default=1,
                        help='Split value for semi-supervised learning')
    parser.add_argument('--used_summary_texts', action='store_true',
                        help='Use summary texts')
    parser.add_argument('--used_paraphrased_texts', action='store_true',
                        help='Use paraphrased texts')
    parser.add_argument('--used_llm_responses', action='store_true',
                        help='Use LLM responses')
    parser.add_argument('--used_based_model', action='store_true',
                        help='Use base model')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'last', 'cls'],
                        help='Pooling strategy for embeddings')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed of the trained adapter to load (must match the --seed used when training it)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override caching_batch_size from configs/llm_configs/<llm>.json (bigger GPU -> bigger batch)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Split seed of the adapter if it was trained with --split_seed (default: same as --seed)')
    parser.add_argument('--init_seed', type=int, default=None,
                        help='LoRA init seed of the adapter if it was trained with --init_seed')

    args = parser.parse_args()

    # Run main function with parsed arguments
    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        init_weight_approach=args.init_weight_approach,
        split=args.split,
        used_summary_texts=args.used_summary_texts,
        used_paraphrased_texts=args.used_paraphrased_texts,
        used_llm_responses=args.used_llm_responses,
        used_based_model=args.used_based_model,
        pooling=args.pooling,
        seed=args.seed,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        init_seed=args.init_seed,
    )
