#!/usr/bin/env bash
# =============================================================================
# Pack trained LoRA adapters + embedding caches + results/ into ONE tar.gz so the run
# (with its seed) can be continued on another machine (Colab <-> RunPod <-> your PC).
#
#   bash sh_scripts/export_outputs.sh                                  # everything
#   DATASETS="cora" SEEDS="42" bash sh_scripts/export_outputs.sh       # only cora, seed 42
#   COPY_TO=/content/drive/MyDrive/DIVERGE_exports bash sh_scripts/export_outputs.sh
#
# Env filters (space separated; default "*" = all):
#   DATASETS  LLM_NAMES  INIT_APPROACHES  SEEDS     PEFT_TYPE [lora]
#   INCLUDE_RESULTS [1]  also pack results/ (reports, efficiency + history CSVs)
#   INCLUDE_CACHE   [1]  also pack artifacts/cache/*_seed<seed>_semi_supervised.pt
#   COPY_TO         []   copy the tar.gz there as well (Drive / network volume)
#
# Output: exports/diverge_outputs_<host>_<timestamp>.tar.gz
# Restore anywhere with:  bash sh_scripts/restore_outputs.sh <that file>
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
shopt -s nullglob

DATASETS="${DATASETS:-*}"; LLM_NAMES="${LLM_NAMES:-*}"; INIT_APPROACHES="${INIT_APPROACHES:-*}"; SEEDS="${SEEDS:-*}"
PEFT_TYPE="${PEFT_TYPE:-lora}"
INCLUDE_RESULTS="${INCLUDE_RESULTS:-1}"; INCLUDE_CACHE="${INCLUDE_CACHE:-1}"
COPY_TO="${COPY_TO:-}"

STAMP="$(date '+%Y%m%d-%H%M%S')"
NAME="diverge_outputs_$(hostname)_${STAMP}"
mkdir -p exports "$STATUS_DIR/exports"
MANIFEST="$STATUS_DIR/exports/${NAME}.MANIFEST.txt"

set -f  # keep "*" literal while splitting the filter lists
read -r -a DS <<< "$DATASETS"; read -r -a LS <<< "$LLM_NAMES"
read -r -a IS <<< "$INIT_APPROACHES"; read -r -a SS <<< "$SEEDS"
set +f

paths=()
for d in "${DS[@]}"; do for m in "${LS[@]}"; do for i in "${IS[@]}"; do for s in "${SS[@]}"; do
    for p in artifacts/${m}_${d}_seqcls_${PEFT_TYPE}_semi_supervised_init_type_${i}_seed${s}; do
        [[ -d "$p" ]] && paths+=("$p")
    done
    if [[ "$INCLUDE_CACHE" == "1" ]]; then
        for p in artifacts/cache/${m}_${d}_seqcls_lora_init-${i}_pool-*_seed${s}_semi_supervised.pt; do paths+=("$p"); done
    fi
done; done; done; done
# de-duplicate (overlapping "*" filters)
mapfile -t paths < <(printf '%s\n' "${paths[@]+"${paths[@]}"}" | awk 'NF && !seen[$0]++')

# Manifest: what is inside + which seed/split each adapter was trained with + env
{
    echo "export=$NAME"
    echo "created=$(ts)"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a) dirty=$([[ -n "$(git status --porcelain 2>/dev/null | head -1)" ]] && echo yes || echo no)"
    echo "env: $(env_fingerprint)"
    echo "filters: DATASETS=[$DATASETS] LLM_NAMES=[$LLM_NAMES] INIT_APPROACHES=[$INIT_APPROACHES] SEEDS=[$SEEDS]"
    echo "--- adapters (from each adapter's training_results.csv) ---"
    "$PYTHON" - "${paths[@]+"${paths[@]}"}" <<'PY'
import csv, os, sys
for p in sys.argv[1:]:
    f = os.path.join(p, "training_results.csv")
    if os.path.isdir(p) and os.path.isfile(f):
        r = list(csv.DictReader(open(f, encoding="utf-8")))[-1]
        print(f"{os.path.basename(p)}  split={r.get('split')}  test_acc={r.get('test_accuracy')}  trained={r.get('timestamp')}")
    elif os.path.isdir(p):
        print(f"{os.path.basename(p)}  (no training_results.csv - training may not have finished!)")
PY
    echo "--- files ---"
    printf '%s\n' "${paths[@]+"${paths[@]}"}"
} > "$MANIFEST"

[[ "$INCLUDE_RESULTS" == "1" && -d results ]] && paths+=("results")
if [[ ${#paths[@]} -eq 0 ]]; then log "Nothing matched the filters."; exit 1; fi
paths+=("$MANIFEST")

OUT="exports/${NAME}.tar.gz"
log "Packing ${#paths[@]} entries -> $OUT"
tar --exclude='checkpoint-*' -czf "$OUT" "${paths[@]}"
ls -lh "$OUT"
cat "$MANIFEST"

if [[ -n "$COPY_TO" ]]; then
    mkdir -p "$COPY_TO" && cp -f "$OUT" "$COPY_TO/" && log "Copied to $COPY_TO/$(basename "$OUT")"
fi
