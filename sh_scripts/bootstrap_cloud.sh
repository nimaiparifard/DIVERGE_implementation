#!/usr/bin/env bash
# =============================================================================
# One-shot setup on a fresh Colab / RunPod machine (run from inside the repo):
#   env install -> base models -> datasets -> (optional) restore earlier outputs
#
#   export HF_TOKEN=hf_xxx
#   bash sh_scripts/bootstrap_cloud.sh
#
#   # continue where an earlier session / another machine stopped:
#   RESTORE_FROM=/content/drive/MyDrive/DIVERGE_sync bash sh_scripts/bootstrap_cloud.sh
#
# Env overrides: everything setup_env.sh / download_models.sh / download_datasets.sh accept
#   (TORCH_VERSION, CUDA_TAG, MODELS, DATASETS, TAPE, EXTRAS_TAR), plus
#   RESTORE_FROM   tar.gz from export_outputs.sh, or a SYNC_DIR folder
#   SKIP_SETUP=1   skip pip installs (e.g. RunPod pod restarted on the same volume)
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

# Normalise line endings in case the repo was copied from Windows without git
sed -i 's/\r$//' "$HERE"/*.sh 2>/dev/null || true

if [[ -d /content ]]; then log "Platform: Google Colab"
elif [[ -d /workspace ]]; then log "Platform: RunPod (keep the repo under /workspace - it is the persistent volume)"
fi

[[ "${SKIP_SETUP:-0}" == "1" ]] || bash "$HERE/setup_env.sh"
bash "$HERE/download_models.sh"
bash "$HERE/download_datasets.sh"
if [[ -n "${RESTORE_FROM:-}" ]]; then
    bash "$HERE/restore_outputs.sh" "$RESTORE_FROM"
fi

hr
log "Ready. Example:"
echo "  SYNC_DIR=/content/drive/MyDrive/DIVERGE_sync DATASETS=cora LLM_NAMES=gemma2_2B SEEDS=42 \\"
echo "      bash sh_scripts/run_all_datasets_lora_init.sh"
hr
