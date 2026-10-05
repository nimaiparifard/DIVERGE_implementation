#!/usr/bin/env bash
# =============================================================================
# Run on YOUR machine (Git Bash on Windows works): packs the dataset folders that are
# NOT on HF (xxwu/LLMNodeBed) so you can upload them to Google Drive / RunPod once.
#
#   bash sh_scripts/pack_local_data.sh        -> exports/datasets_extras.tar.gz
#
# On the cloud box:
#   EXTRAS_TAR=/content/drive/MyDrive/datasets_extras.tar.gz bash sh_scripts/download_datasets.sh
#
# Only needed for experiments that use GPT explanations / enhanced texts / precomputed
# embeddings. The LoRA-init -> cache -> DIVERGE pipeline does not need it.
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXTRAS="${EXTRAS:-gpt_4o PubMed enhanced_texts e5-large roberta Qwen-3B Node2Vec}"
mkdir -p exports
paths=()
for e in $EXTRAS; do
    [[ -e "datasets/$e" ]] && paths+=("datasets/$e") || log "skip datasets/$e (not found)"
done
[[ ${#paths[@]} -eq 0 ]] && { log "Nothing to pack"; exit 1; }
log "Packing: ${paths[*]}"
tar -czf exports/datasets_extras.tar.gz "${paths[@]}"
ls -lh exports/datasets_extras.tar.gz
