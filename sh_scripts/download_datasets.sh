#!/usr/bin/env bash
# =============================================================================
# Download the graph datasets (datasets/<name>.pt) from HF: xxwu/LLMNodeBed
#   https://huggingface.co/datasets/xxwu/LLMNodeBed/tree/main
#
#   bash sh_scripts/download_datasets.sh                       # default dataset list
#   DATASETS="cora pubmed" bash sh_scripts/download_datasets.sh
#
# Env overrides:
#   DATASETS="..."     which <name>.pt to fetch (default: every dataset with a configs/dataset/*.json
#                      that exists on HF: cora citeseer pubmed wikics instagram reddit computer photo history arxiv)
#   TAPE=1             also fetch LLM explanations TAPE/{Mistral-7B,gpt-4o-mini}/<name>.json into
#                      datasets/Mistral-7B/ and datasets/gpt-4o-mini/ (LLMNodeBed layout)
#   EXTRAS_TAR=path    extract a datasets_extras.tar.gz made locally with sh_scripts/pack_local_data.sh
#                      (gpt_4o/, PubMed/, enhanced_texts/, e5-large/, ... - NOT on HF)
#   FORCE=1            re-download existing files
#
# The LoRA-init -> cache -> DIVERGE pipeline only needs datasets/<name>.pt.
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASETS="${DATASETS:-cora citeseer pubmed wikics instagram reddit computer photo history arxiv}"
TAPE="${TAPE:-0}"
FORCE="${FORCE:-0}"
EXTRAS_TAR="${EXTRAS_TAR:-}"

mkdir -p datasets
hr; log "Downloading datasets: $DATASETS"; hr

DATASETS="$DATASETS" TAPE="$TAPE" FORCE="$FORCE" "$PYTHON" - <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

REPO = "xxwu/LLMNodeBed"
force = os.environ.get("FORCE") == "1"
failed = []

def fetch(remote, local):
    if os.path.isfile(local) and not force:
        print(f"[OK] {local} (exists)"); return
    try:
        p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=remote)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        shutil.copyfile(p, local)
        print(f"[OK] {remote} -> {local} ({os.path.getsize(local)/1e6:.1f} MB)")
    except Exception as e:
        print(f"[FAIL] {remote}: {e}"); failed.append(remote)

for d in os.environ["DATASETS"].split():
    fetch(f"{d}.pt", os.path.join("datasets", f"{d}.pt"))
    if os.environ.get("TAPE") == "1":
        for llm in ("Mistral-7B", "gpt-4o-mini"):
            fetch(f"TAPE/{llm}/{d}.json", os.path.join("datasets", llm, f"{d}.json"))

if failed:
    print("\nFAILED:", " ".join(failed)); sys.exit(1)
PY

if [[ -n "$EXTRAS_TAR" ]]; then
    log "Extracting extras $EXTRAS_TAR -> datasets/"
    tar -xzf "$EXTRAS_TAR" -C .
fi

# quick integrity check: every .pt must torch.load
DATASETS="$DATASETS" "$PYTHON" - <<'PY'
import os, torch
for d in os.environ["DATASETS"].split():
    p = os.path.join("datasets", f"{d}.pt")
    g = torch.load(p, weights_only=False, map_location="cpu")
    print(f"[OK] {d:10s} nodes={g.num_nodes:>8} edges={g.edge_index.shape[1]:>9} "
          f"classes={len(getattr(g, 'label_name', [])) or int(g.y.max()) + 1}")
PY
log "Datasets ready in datasets/"
