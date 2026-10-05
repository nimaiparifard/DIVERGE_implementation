import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, LoftQConfig, prepare_model_for_kbit_training


SUPPORTED_LLM_NAMES = [
    "llama_3.2_1B",
    "deberta_v3_large",
    "qwen2.5_1.5B",
    "smollm2_1.7B",
    "opt_1.3B",
    "gemma2_2B",
    "olmo_1B",
    "mobilellm_600M",
]


def get_model_path(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    local_model_dir_path = os.path.join(repo_root, "local_models")
    return os.path.join(local_model_dir_path, model_name)


def _resolve_torch_dtype(dtype_name):
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(dtype_name).lower(), torch.bfloat16)


def _ensure_pad_token(model, tokenizer=None):
    """Align model pad_token_id with tokenizer (required for left-padded causal LMs)."""
    pad_token_id = None
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif getattr(model.config, "pad_token_id", None) is not None:
        pad_token_id = model.config.pad_token_id
    elif getattr(model.config, "eos_token_id", None) is not None:
        pad_token_id = model.config.eos_token_id

    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
        if hasattr(model, "config") and hasattr(model.config, "problem_type"):
            # keep default single_label_classification when labels are class ids
            pass
    return model


def load_llm_model(cfg, tokenizer=None):
    model_name = cfg.llm.model_name
    model_path = get_model_path(model_name)
    if not os.path.isdir(model_path):
        hf_repo = cfg.llm.get("hf_repo_id", None)
        hint = f" Download it with: huggingface-cli download {hf_repo} --local-dir local_models/{model_name}" if hf_repo else ""
        raise FileNotFoundError(
            f"Local model not found at '{model_path}'. Place weights under local_models/{model_name}.{hint}"
        )

    use_4bit = cfg.llm.get("use_4bit", False)
    # PiSSA/OLoRA/CorDA run SVD/QR on the base weights and LoftQ quantizes them itself,
    # so PEFT requires an unquantized (fp32/fp16/bf16) base model for these inits.
    init_lora_weights = str(getattr(cfg.peft, "init_lora_weights", "")).lower()
    if use_4bit and init_lora_weights.startswith(("pissa", "olora", "corda", "loftq")):
        print(f"[load_llm_model] init '{init_lora_weights}' requires a full-precision base model; disabling 4-bit loading.")
        use_4bit = False
    torch_dtype = _resolve_torch_dtype(cfg.llm.get("torch_dtype", "bfloat16"))
    common_kwargs = dict(
        local_files_only=cfg.llm.local_files_only,
        trust_remote_code=cfg.llm.trust_remote_code,
        num_labels=cfg.dataset.num_classes,
    )

    # Gemma-2 sequence classification is more stable with eager attention on some stacks
    if "gemma" in model_name.lower():
        common_kwargs["attn_implementation"] = "eager"

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            **common_kwargs,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cuda",
            **common_kwargs,
        )

    model = _ensure_pad_token(model, tokenizer=tokenizer)
    return model


def get_lora_model(model, cfg):
    loftq_config = None
    init_lora_weights = cfg.peft.init_lora_weights
    if init_lora_weights == "loftq":
        loftq_config = LoftQConfig(
            loftq_bits=4,
            loftq_iter=1,
        )

    modules_to_save = list(cfg.peft.module_to_save)
    target_modules = list(cfg.llm.ft_target_modules)

    lora_kwargs = dict(
        r=cfg.peft.rank,
        lora_alpha=cfg.peft.lora_alpha,
        lora_dropout=cfg.peft.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=modules_to_save,
        init_lora_weights=init_lora_weights,
        use_rslora=cfg.peft.use_rslora,
    )
    if loftq_config is not None:
        lora_kwargs["loftq_config"] = loftq_config

    # PEFT's orthogonal init casts A/B to the *base* weight dtype, which is uint8 for bitsandbytes
    # 4-bit layers ("Only Tensors of floating point ... can require gradients"). Orthogonal init doesn't
    # depend on the base weights, so keep the model quantized and apply the same init ourselves.
    manual_orthogonal = init_lora_weights == "orthogonal" and getattr(model, "is_loaded_in_4bit", False)
    if manual_orthogonal:
        lora_kwargs["init_lora_weights"] = True

    lora_config = LoraConfig(**lora_kwargs)
    trainable_model = get_peft_model(model, lora_config)
    if manual_orthogonal:
        _orthogonal_init_lora(trainable_model)
    return trainable_model


@torch.no_grad()
def _orthogonal_init_lora(peft_model, adapter_name="default"):
    """
    Same algorithm as peft.tuners.lora.layer.LoraLayer.orthogonal_init (A and B built from the odd /
    even rows of a random orthogonal r x r matrix), but keeps each LoRA matrix in its own float dtype
    instead of the (quantized) base layer's. The saved adapter_config keeps init_lora_weights=True so
    reloading the adapter onto the 4-bit model later doesn't re-trigger PEFT's orthogonal init.
    """
    n_layers = 0
    for module in peft_model.modules():
        lora_A = getattr(module, "lora_A", None)
        if not isinstance(lora_A, torch.nn.ModuleDict) or adapter_name not in lora_A:
            continue
        rank = module.r[adapter_name]
        if rank % 2 != 0:
            raise ValueError(f"Orthogonal initialization requires the LoRA rank to be even, got {rank} instead.")
        A = module.lora_A[adapter_name].weight   # (r, in_features)
        B = module.lora_B[adapter_name].weight   # (out_features, r)
        Q, _ = torch.linalg.qr(torch.randn(rank, rank))
        q_odd, q_even = Q[0::2, :], Q[1::2, :]
        new_A = torch.randn(A.shape[1], rank // 2).mm(q_odd).T / 10.0
        new_B = torch.randn(rank // 2, B.shape[0]).T.mm(q_even) / 10.0
        A.copy_(new_A.to(device=A.device, dtype=A.dtype))
        B.copy_(new_B.to(device=B.device, dtype=B.dtype))
        n_layers += 1
    print(f"[get_lora_model] Orthogonal LoRA init applied manually to {n_layers} 4-bit layers.")


if __name__ == "__main__":
    from config import setup_finetuning_cfg

    cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
    base_model = load_llm_model(cfg)
    model = get_lora_model(base_model, cfg)
    for name, modules in base_model.named_modules():
        print(name, modules)
