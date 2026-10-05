from yacs.config import CfgNode as CN
import json
import os

def setup_finetuning_cfg(dataset_name: str = 'cora', llm_name: str = 'llama_3.2_1B', peft_type: str = 'lora') -> CN:
    ### dataset name
    cfg = CN()
    dataset_data = load_dataset_config_json(dataset_name)
    cfg.dataset = CN()
    cfg.dataset.name = dataset_data.name
    cfg.dataset.seed = dataset_data.seed
    cfg.dataset.llm_responses_name = dataset_data.llm_responses_name
    cfg.dataset.enhance_features_name = dataset_data.enhance_features_name
    cfg.dataset.max_length = dataset_data.get('max_length', 512)
    cfg.dataset.num_classes = dataset_data.num_classes
    cfg.dataset.num_samples = dataset_data.get('num_samples', None)

    ### device
    cfg.device = "cuda:0"
    
    ## llm name and tokenizer settings
    cfg.llm = CN()
    llm_data = load_llm_config_json(llm_name)
    cfg.llm.model_name = llm_data.llm.model_name
    cfg.llm.hf_repo_id = llm_data.llm.get('hf_repo_id', None)
    cfg.llm.local_files_only = llm_data.llm.local_files_only
    cfg.llm.trust_remote_code = llm_data.llm.trust_remote_code
    cfg.llm.architecture = llm_data.llm.get('architecture', 'decoder')
    cfg.llm.use_4bit = llm_data.llm.get('use_4bit', False)
    cfg.llm.torch_dtype = llm_data.llm.get('torch_dtype', 'bfloat16')
    cfg.llm.ft_target_modules = llm_data.llm.get('ft_target_modules', ["q_proj", "v_proj"])
    cfg.llm.caching_batch_size = llm_data.llm.get('caching_batch_size', 64)

    cfg.tokenizer = CN()
    cfg.tokenizer.model_name = llm_data.tokenizer.model_name
    cfg.tokenizer.local_files_only = llm_data.tokenizer.local_files_only
    cfg.tokenizer.trust_remote_code = llm_data.tokenizer.trust_remote_code
    cfg.tokenizer.max_length = llm_data.tokenizer.max_length
    cfg.tokenizer.padding_side = llm_data.tokenizer.padding_side
    cfg.tokenizer.padding = llm_data.tokenizer.padding
    cfg.tokenizer.truncation = llm_data.tokenizer.truncation

    cfg.peft = CN()
    peft_data = load_peft_config_json(peft_type)
    cfg.peft.type = peft_data.type
    cfg.peft.rank = peft_data.rank
    cfg.peft.lora_alpha = peft_data.lora_alpha
    cfg.peft.lora_dropout = peft_data.lora_dropout
    # Encoder models (e.g. DeBERTa) use "classifier"; causal LMs use "score"
    default_head = ["classifier"] if cfg.llm.architecture == "encoder" else ["score"]
    cfg.peft.module_to_save = llm_data.llm.get(
        'module_to_save',
        peft_data.get('module_to_save', default_head),
    )
    cfg.peft.init_lora_weights = peft_data.get('init_lora_weights', 'pissa')
    cfg.peft.use_rslora = peft_data.get('use_rslora', True)

    cfg.training_args = CN()
    training_data = load_training_config_json(dataset_name)
    cfg.training_args.output_dir = training_data.output_dir
    cfg.training_args.overwrite_output_dir = training_data.overwrite_output_dir
    cfg.training_args.num_train_epochs = training_data.num_train_epochs
    cfg.training_args.per_device_train_batch_size = training_data.per_device_train_batch_size
    cfg.training_args.per_device_eval_batch_size = training_data.per_device_eval_batch_size
    cfg.training_args.learning_rate = training_data.learning_rate
    cfg.training_args.gradient_accumulation_steps = training_data.gradient_accumulation_steps   # effective batch size = 16
    cfg.training_args.max_grad_norm = training_data.max_grad_norm             # enable grad clipping
    cfg.training_args.weight_decay = training_data.weight_decay              # for LoRA, often better as 0
    cfg.training_args.lr_scheduler_type = training_data.lr_scheduler_type # or "linear"
    cfg.training_args.warmup_ratio = training_data.warmup_ratio          # smaller warmup for short runs
    cfg.training_args.logging_dir = training_data.logging_dir            # directory for storing logs
    cfg.training_args.logging_steps = training_data.logging_steps                 # more frequent logging
    cfg.training_args.eval_strategy = training_data.eval_strategy           # evaluate every few steps
    cfg.training_args.eval_steps = training_data.eval_steps                   # evaluate every 50 steps
    cfg.training_args.save_strategy = training_data.save_strategy           # save every few steps
    cfg.training_args.save_steps = training_data.save_steps                   # save every 100 steps
    cfg.training_args.metric_for_best_model = training_data.metric_for_best_model
    cfg.training_args.load_best_model_at_end = training_data.load_best_model_at_end
    cfg.training_args.fp16 = training_data.fp16                      # disable mixed precision
    cfg.training_args.bf16 = training_data.bf16                      # disable bfloat16
    cfg.dataloader_num_workers = training_data.dataloader_num_workers                     # avoid multiprocessing issues
    cfg.remove_unused_columns = training_data.remove_unused_columns                   # keep all columns
    cfg.report_to = training_data.report_to                                # disable wandb/tensorboard
    cfg.dataloader_pin_memory = training_data.dataloader_pin_memory                   # disable pin memory for CPU
    cfg.use_cpu = training_data.use_cpu                                  # explicitly use CPU

    return cfg

def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_dataset_config_json(dataset_name: str):
    dataset_config_path = os.path.join(get_current_dir(), 'configs', 'dataset', f'{dataset_name}.json')
    with open(dataset_config_path, 'r') as f:
        data = json.load(f)
    return CN(data)

def load_llm_config_json(llm_name: str):
    llm_config_path = os.path.join(get_current_dir(), 'configs', 'llm_configs', f'{llm_name}.json')
    with open(llm_config_path, 'r') as f:
        data = json.load(f)
    return CN(data)

def load_peft_config_json(peft_type: str):
    peft_config_path = os.path.join(get_current_dir(), 'configs', 'peft_configs', f'{peft_type}.json')
    with open(peft_config_path, 'r') as f:
        data = json.load(f)
    return CN(data)

def load_training_config_json(dataset_name: str):
    training_config_path = os.path.join(get_current_dir(), 'configs', 'training_args', f'{dataset_name}.json')
    with open(training_config_path, 'r') as f:
        data = json.load(f)
    return CN(data)




# if __name__ == "__main__":
#     test_cfg = setup_finetuning_cfg()
#     print("Test config:")
#
#     print("=" * 50)
#     print("Dataset Config:")
#     print("Dataset Info:", test_cfg.dataset.name, test_cfg.dataset.llm_responses_name, test_cfg.dataset.enhance_features_name, '\n')
#
#
#     print("=" * 50)
#     print("LLM Config:")
#     print("Model Name:", test_cfg.llm.model_name)
#     print("FT Target Modules:", test_cfg.llm.ft_target_modules, '\n')
#
#     print("=" * 50)
#     print("PEFT Config:")
#     print("PEFT Type:", test_cfg.peft.type)
#
#     print("=" * 50)
#     print("Training Args Config:")
#     print("Num Train Epochs:", test_cfg.training_args.num_train_epochs)
#
