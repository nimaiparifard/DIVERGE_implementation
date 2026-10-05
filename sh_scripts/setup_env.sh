#!/usr/bin/env bash
# =============================================================================
# Install the Python environment for DIVERGE on Linux (Colab / RunPod / any CUDA box).
#
#   bash sh_scripts/setup_env.sh
#
# Env overrides:
#   TORCH_VERSION=2.7.1   torch to install (default = same as the Windows dev machine).
#                         TORCH_VERSION=keep -> keep the preinstalled torch (Colab/RunPod image).
#   CUDA_TAG=cu128        CUDA build of torch / PyG wheels (cu118, cu121, cu124, cu126, cu128, cpu)
#   INSTALL_OPTIONAL=1    also install umap-learn + statsmodels
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
CUDA_TAG="${CUDA_TAG:-cu128}"
INSTALL_OPTIONAL="${INSTALL_OPTIONAL:-0}"

# torch -> matching torchvision / torchaudio (a mismatched torchvision breaks `import transformers`)
declare -A TV=( [2.4.1]=0.19.1 [2.5.1]=0.20.1 [2.6.0]=0.21.0 [2.7.0]=0.22.0 [2.7.1]=0.22.1 [2.8.0]=0.23.0 )

hr; log "DIVERGE environment setup  (repo: $REPO_ROOT)"; hr
"$PYTHON" --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || log "WARNING: nvidia-smi not found (no GPU?)"

"$PYTHON" -m pip install --upgrade pip wheel setuptools -q

# ---- 1) torch ---------------------------------------------------------------
if [[ "$TORCH_VERSION" == "keep" ]]; then
    log "Keeping preinstalled torch"
else
    CUR="$("$PYTHON" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
    if [[ "$CUR" == "${TORCH_VERSION}+${CUDA_TAG}" ]]; then
        log "torch ${CUR} already installed"
    else
        log "Installing torch==${TORCH_VERSION} (${CUDA_TAG}), current: ${CUR}"
        EXTRA=()
        if [[ -n "${TV[$TORCH_VERSION]:-}" ]]; then
            EXTRA=("torchvision==${TV[$TORCH_VERSION]}" "torchaudio==${TORCH_VERSION}")
        fi
        "$PYTHON" -m pip install "torch==${TORCH_VERSION}" ${EXTRA[@]+"${EXTRA[@]}"} \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    fi
fi

# ---- 2) PyG wheels matching the installed torch -----------------------------
read -r TORCH_FULL TORCH_CUDA < <("$PYTHON" - <<'PY'
import torch
v = torch.__version__.split('+')[0]
cuda = 'cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', '')
print(v, cuda)
PY
)
PYG_TORCH="$(echo "$TORCH_FULL" | awk -F. '{print $1"."$2".0"}')"   # PyG indexes by X.Y.0
PYG_URL="https://data.pyg.org/whl/torch-${PYG_TORCH}+${TORCH_CUDA}.html"
log "torch ${TORCH_FULL} / ${TORCH_CUDA} -> PyG wheel index ${PYG_URL}"

if curl -sfL "$PYG_URL" | grep -q "torch_sparse"; then
    "$PYTHON" -m pip install torch_scatter torch_sparse -f "$PYG_URL"
else
    log "WARNING: no prebuilt torch_sparse wheel for torch ${PYG_TORCH}+${TORCH_CUDA}."
    log "         Building from source (10-20 min). Faster: rerun with TORCH_VERSION=2.7.1 CUDA_TAG=cu128"
    "$PYTHON" -m pip install --no-build-isolation torch_scatter torch_sparse
fi
"$PYTHON" -m pip install torch-geometric

# ---- 3) project requirements ------------------------------------------------
"$PYTHON" -m pip install -r requirements.txt
# Colab preinstalls torchao; peft imports it when present and fails on version mismatch. Not used here.
"$PYTHON" -m pip uninstall -y -q torchao 2>/dev/null || true
if [[ "$INSTALL_OPTIONAL" == "1" ]]; then
    "$PYTHON" -m pip install umap-learn statsmodels
fi

# ---- 4) sanity check --------------------------------------------------------
hr; log "Sanity check"; hr
"$PYTHON" - <<'PY'
import torch, torch_geometric, torch_sparse, transformers, peft, accelerate, bitsandbytes, numpy
print(f"torch {torch.__version__} | cuda available={torch.cuda.is_available()} | "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'} | "
      f"bf16={torch.cuda.is_available() and torch.cuda.is_bf16_supported()}")
print(f"torch_geometric {torch_geometric.__version__} | torch_sparse {torch_sparse.__version__}")
print(f"transformers {transformers.__version__} | peft {peft.__version__} | accelerate {accelerate.__version__} | "
      f"bitsandbytes {bitsandbytes.__version__} | numpy {numpy.__version__}")
import common, config, train_llm.peft_model  # project imports resolve
print("[OK] project imports")
import re
pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", open("requirements.txt").read(), re.M))
got = {"transformers": transformers.__version__, "peft": peft.__version__,
       "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__}
bad = {k: (v, pins[k]) for k, v in got.items() if k in pins and v != pins[k]}
if bad:
    raise SystemExit(f"[FAIL] version mismatch (installed, pinned): {bad} -> rerun this script")
print("[OK] pinned LLM stack versions")
if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
    print("[WARN] this GPU has no bfloat16 (e.g. Tesla T4): configs use bfloat16 -> slow/unstable. Prefer L4/A100.")
PY
log "Setup done. Next: bash sh_scripts/download_models.sh && bash sh_scripts/download_datasets.sh"
