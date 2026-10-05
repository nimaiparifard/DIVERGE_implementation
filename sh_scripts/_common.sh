#!/usr/bin/env bash
# Shared helpers, sourced by every sh_scripts/*.sh (not meant to be run directly).

# Repo root = parent of sh_scripts/, so scripts work from any working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python interpreter (override with PYTHON=python3.11 etc.)
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then PYTHON=python3; fi

# Disable HF Trainer logging integrations (wandb/mlflow prompts block unattended runs)
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
# Unicode-safe console/log output (ensemble code prints non-ASCII symbols)
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Seeds exported from another machine (scripts/export_seeds.py); an explicit SEEDS=... still wins
if [[ -z "${SEEDS:-}" && -f seed_snapshot/seeds.env ]]; then
    source seed_snapshot/seeds.env
fi

STATUS_DIR="results/run_status"
mkdir -p "$STATUS_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
hr() { echo "=========================================="; }

# write_kv FILE key=value ...   (overwrites FILE, one key=value per line)
write_kv() {
    local file="$1"; shift
    printf '%s\n' "$@" > "$file"
}

# Name of the LoRA adapter dir written by train_llm/train_other_lora_init_apporaches.py
#   adapter_dir <llm> <dataset> <init> <seed> [peft]
adapter_dir() {
    echo "artifacts/${1}_${2}_seqcls_${5:-lora}_semi_supervised_init_type_${3}_seed${4}"
}

# Name of the embedding cache written by cache/cache_embedding_with_diffrent_init_weights.py (split != 1)
#   cache_file <llm> <dataset> <init> <seed> <pooling>
cache_file() {
    echo "artifacts/cache/${1}_${2}_seqcls_lora_init-${3}_pool-${5}_seed${4}_semi_supervised.pt"
}

# merge_tree SRC DST
# Copy every file of SRC into DST without destroying what DST already has:
#   - *.csv present on both sides -> rows of SRC appended to DST, exact duplicate rows dropped
#     (if the headers differ, SRC rows go to <name>.from_<host>.csv instead)
#   - other files present on both sides -> kept as in DST (OVERWRITE=1 replaces them)
#   - HF Trainer checkpoint-*/ folders are skipped
merge_tree() {
    local src="$1" dst="$2" f s d tmp
    mkdir -p "$dst"
    (cd "$src" && find . -type f -not -path '*/checkpoint-*/*' -print0) |
    while IFS= read -r -d '' f; do
        s="$src/${f#./}"; d="$dst/${f#./}"
        mkdir -p "$(dirname "$d")"
        if [[ ! -e "$d" ]]; then
            cp -p "$s" "$d"
        elif [[ "$d" == *.csv ]]; then
            if [[ "$(head -n1 "$s")" == "$(head -n1 "$d")" ]]; then
                tmp="$d.merge.$$"
                { cat "$d"; tail -n +2 "$s"; } | awk 'NR==1{print;next} !seen[$0]++' > "$tmp" && mv "$tmp" "$d"
            else
                cp -p "$s" "${d%.csv}.from_$(hostname).csv"
                echo "  [csv header differs] $f -> ${d%.csv}.from_$(hostname).csv"
            fi
        elif [[ "${OVERWRITE:-0}" == "1" ]]; then
            cp -p "$s" "$d"
        fi
    done
}

# sync_outputs: if SYNC_DIR is set (e.g. /content/drive/MyDrive/DIVERGE_sync on Colab,
# /workspace/DIVERGE_sync on RunPod), merge artifacts/ + results/ into it so a
# disconnect never loses finished runs. Restore later with sh_scripts/restore_outputs.sh $SYNC_DIR
sync_outputs() {
    [[ -z "${SYNC_DIR:-}" ]] && return 0
    local p
    for p in artifacts results; do
        [[ -d "$p" ]] && merge_tree "$REPO_ROOT/$p" "$SYNC_DIR/$p"
    done
    log "synced artifacts/ + results/ -> $SYNC_DIR"
}

# Echo the environment facts that matter for reproducing a seed on another machine
env_fingerprint() {
    "$PYTHON" - <<'PY' 2>/dev/null || true
import platform, torch, transformers, peft, numpy
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"host={platform.node()} os={platform.system()} python={platform.python_version()} gpu={gpu} "
      f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} numpy={numpy.__version__}")
PY
}
