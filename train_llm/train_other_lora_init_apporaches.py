import os
from train_llm.peft_model import load_llm_model, get_lora_model
from dataset.dataset_loader import load_dataset, get_tokenizer
from config import setup_finetuning_cfg
from train_llm.train import Train, load_llm_with_lora_adapter, evaluate_sampled_examples
from adapter_dir import get_semi_supervised_adapter_dir, lora_run_tag
from report.reporter import ReportResults

def main(database_name, llm_name, peft_type, init_weight_approach, split, seed=None, split_seed=None, init_seed=None):
    cfg = setup_finetuning_cfg(dataset_name=database_name, llm_name=llm_name, peft_type=peft_type)
    init_weights_approaches_list =  ['pissa', 'gaussian', 'eva',  'loftq', 'orthogonal']
    # init_weights_approaches_list = ['orthogonal']
    cfg.peft.init_lora_weights = init_weight_approach  # change this to try different initialization approaches
    if seed is not None:
        cfg.dataset.seed = seed  # override dataset/training seed for reproducibility sweeps
    print("=== Using LoRA initialization approach:", cfg.peft.init_lora_weights)
    print("=== Using seed:", cfg.dataset.seed)
    tag = lora_run_tag(cfg.dataset.seed, split_seed, init_seed)
    if tag != f'_seed{cfg.dataset.seed}':
        # Decoupled run: the split comes from split_seed, the LoRA A/B draw from init_seed (if
        # given), everything else (head init, data order, dropout) from the LoRA seed, including
        # the HF Trainer seed.
        cfg.training_args.seed = int(cfg.dataset.seed)
        print("=== Using split seed:", split_seed)
        if init_seed is not None:
            print("=== Using LoRA init seed:", init_seed)
    # Train with LLM responses
    adapter_dir = f'{get_semi_supervised_adapter_dir(cfg)}{tag}'
    # Per-run HF Trainer checkpoint dir: with save_strategy=best + load_best_model_at_end a shared
    # ./outputs/ would let parallel runs overwrite / delete / reload each other's checkpoints.
    run_name = os.path.basename(adapter_dir.rstrip('/\\'))
    cfg.training_args.output_dir = os.path.join('outputs', run_name)
    cfg.training_args.logging_dir = os.path.join('logs', run_name)
    save_dir = f'results/train_info/train_1/{database_name}_{llm_name}_{peft_type}_{init_weight_approach}{tag}_semi_supervised_adapter'
    reporter = ReportResults(cfg, save_dir=save_dir, init_approach=init_weight_approach)
    train = Train(cfg, adapter_dir, reporter, split=split, split_seed=split_seed, init_seed=init_seed)
    train.process()
    if train.efficiency_metrics:
        print(
            f"[Efficiency summary] {llm_name} | {init_weight_approach} | "
            f"GPU-h={train.efficiency_metrics['gpu_hours']:.6f} | "
            f"peakVRAM={train.efficiency_metrics['peak_vram_gb']:.3f}GB | "
            f"adapter={train.efficiency_metrics['adapter_weights_mb']:.2f}MB | "
            f"infer={train.efficiency_metrics['inference_ms_per_sample']:.3f}ms/sample"
        )

if __name__ == "__main__":
    ## argument definition
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with different LoRA initialization approaches')
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name (e.g., cora, citeseer, pubmed, wikics)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        choices=[
                            'llama_3.2_1B',
                            'deberta_v3_large',
                            'qwen2.5_1.5B',
                            'smollm2_1.7B',
                            'opt_1.3B',
                            'gemma2_2B',
                            'olmo_1B',
                            'mobilellm_600M',
                        ],
                        help='LLM model name (must match configs/llm_configs/<name>.json and local_models/<name>)')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='PEFT type (e.g., lora)')
    parser.add_argument('--init_weight_approach', type=str, default='gaussian',
                        choices=['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal'],
                        help='LoRA initialization approach')
    parser.add_argument('--split', type=int, default=0,)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for training/data split (overrides configs/dataset/<name>.json seed if set)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Seed of the train/val/test split if it should differ from --seed (default: same as --seed). '
                             'Adapters are then saved as ..._seed<seed>_split<split_seed>')
    parser.add_argument('--init_seed', type=int, default=None,
                        help='Seed of the LoRA A/B initialization if it should not follow --seed (head init, data '
                             'order and dropout still follow --seed). Adapters are then saved as '
                             '..._seed<seed>_split<split_seed>_init<init_seed>')

    args = parser.parse_args()

    # Run main function with parsed arguments
    main(
        database_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        init_weight_approach=args.init_weight_approach,
        split=args.split,
        seed=args.seed,
        split_seed=args.split_seed,
        init_seed=args.init_seed,
    )
