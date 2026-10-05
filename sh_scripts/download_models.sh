#!/usr/bin/env bash
# =============================================================================
# Download base LLMs into local_models/<name>/ (the path train_llm/peft_model.py loads from).
# The HF repo id comes from configs/llm_configs/<name>.json -> llm.hf_repo_id.
#
#   export HF_TOKEN=hf_xxx            # REQUIRED for gated models (Llama-3.2, Gemma-2):
#                                     # accept the license on their HF pages first.
#   bash sh_scripts/download_models.sh
#   MODELS="gemma2_2B" bash sh_scripts/download_models.sh
#
# Env overrides:
#   MODELS="llama_3.2_1B smollm2_1.7B gemma2_2B"   (default = the 3 models of the LoRA-init sweep)
#   FORCE=1                                         re-download even if the folder looks complete
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODELS="${MODELS:-llama_3.2_1B smollm2_1.7B gemma2_2B}"
FORCE="${FORCE:-0}"

hr; log "Downloading models: $MODELS"; hr
[[ -z "${HF_TOKEN:-}" ]] && log "NOTE: HF_TOKEN not set - gated models (llama/gemma) will fail. Colab: add it under Secrets or run: export HF_TOKEN=..."

MODELS="$MODELS" FORCE="$FORCE" "$PYTHON" - <<'PY'
import json, os, sys, glob
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

failed = []
for name in os.environ["MODELS"].split():
    cfg_path = os.path.join("configs", "llm_configs", f"{name}.json")
    if not os.path.isfile(cfg_path):
        print(f"[SKIP] {name}: no {cfg_path}"); failed.append(name); continue
    repo = json.load(open(cfg_path))["llm"].get("hf_repo_id")
    if not repo:
        print(f"[SKIP] {name}: no llm.hf_repo_id in {cfg_path}"); failed.append(name); continue
    dest = os.path.join("local_models", name)
    complete = os.path.isfile(os.path.join(dest, "config.json")) and (
        glob.glob(os.path.join(dest, "*.safetensors")) or glob.glob(os.path.join(dest, "*.bin")))
    if complete and os.environ.get("FORCE") != "1":
        print(f"[OK] {name}: already in {dest} (FORCE=1 to re-download)"); continue
    print(f"[..] {name}: {repo} -> {dest}")
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            token=os.environ.get("HF_TOKEN"),
            # weights + tokenizer + remote code; skip duplicate formats (original/*.pth, gguf, onnx, flax, tf)
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "tokenizer*", "*.py"],
            ignore_patterns=["original/*", "*.gguf", "onnx/*", "*.onnx", "*.msgpack", "*.h5"],
        )
        if not glob.glob(os.path.join(dest, "*.safetensors")):
            # a few repos only ship pytorch_model.bin
            snapshot_download(repo_id=repo, local_dir=dest, token=os.environ.get("HF_TOKEN"),
                              allow_patterns=["*.bin"], ignore_patterns=["original/*"])
        print(f"[OK] {name}")
    except GatedRepoError:
        print(f"[FAIL] {name}: {repo} is gated. Accept the license at https://huggingface.co/{repo} "
              f"with the account that owns HF_TOKEN, then rerun.")
        failed.append(name)
    except RepositoryNotFoundError:
        print(f"[FAIL] {name}: repo {repo} not found (or no access with this token)."); failed.append(name)

if failed:
    print("\nFAILED:", " ".join(failed)); sys.exit(1)
print("\nAll models ready under local_models/")
PY
du -sh local_models/* 2>/dev/null || true
