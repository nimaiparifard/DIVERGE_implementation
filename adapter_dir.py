import os

def get_adapter_dir(cfg, used_llm_responses: bool = False) -> str:
    base_dir = "./artifacts"
    if not used_llm_responses:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}")
    else:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}")
    return adapter_dir

def lora_run_tag(seed, split_seed=None, init_seed=None):
    """Suffix of adapter dirs / embedding caches: '_seed{s}', or '_seed{s}_split{t}' when the
    split seed differs from the LoRA seed (so existing single-seed artifacts keep their names).
    With an init seed (LoRA A/B draw fixed independently of the training seed) the suffix is
    always '_seed{s}_split{t}_init{i}'."""
    if init_seed is not None:
        split = seed if split_seed is None else split_seed
        return f'_seed{seed}_split{split}_init{init_seed}'
    if split_seed is None or int(split_seed) == int(seed):
        return f'_seed{seed}'
    return f'_seed{seed}_split{split_seed}'


def get_semi_supervised_adapter_dir(cfg) -> str:
    base_dir = "./artifacts"
    adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_semi_supervised_init_type_{cfg.peft.init_lora_weights}")
    return adapter_dir

def get_semi_supervised_augmented_adapter_dir(cfg) -> str:
    base_dir = "./artifacts"
    adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_semi_supervised_init_type_{cfg.peft.init_lora_weights}_augmented")
    return adapter_dir

def get_adapter_dir_lora_init_weights(cfg, used_llm_responses: bool = False) -> str:
    base_dir = "./artifacts"
    if not used_llm_responses:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}")
    else:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}")
    return adapter_dir

def get_adapter_dir_lora_train_weights_for_summary(cfg, used_llm_responses: bool = False) -> str:
    base_dir = "./artifacts"
    if not used_llm_responses:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}_summary")
    else:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}_summary")
    return adapter_dir

def get_adapter_dir_lora_train_weights_for_paraphrased(cfg, used_llm_responses: bool = False) -> str:
    base_dir = "./artifacts"
    if not used_llm_responses:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}_paraphrased")
    else:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}_paraphrased")
    return adapter_dir

def get_adapter_dir_lora_train_weights_for_keywords(cfg, used_llm_responses: bool = False) -> str:
    base_dir = "./artifacts"
    if not used_llm_responses:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}_keywords")
    else:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}_init_type_{cfg.peft.init_lora_weights}_keywords")
    return adapter_dir

def get_adapter_dir_lora_train_weights_for_gnn_mistake(cfg, used_llm_responses: bool = False) -> str:
    base_dir = "./artifacts"
    if not used_llm_responses:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init_{cfg.peft.init_lora_weights}_retrained_mistakes")
    else:
        adapter_dir = os.path.join(base_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_with_llm_responses_{cfg.peft.type}_init_{cfg.peft.init_lora_weights}_retrained_mistakes")
    return adapter_dir