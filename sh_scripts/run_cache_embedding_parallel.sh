#!/usr/bin/env bash
# =============================================================================
# PARALLEL version of sh_scripts/run_cache_embedding_all_datasets.sh
# Caches the LLM embeddings of every trained (dataset, llm, init, seed) adapter, running
# several cache jobs on the same GPU at once (see sh_scripts/_parallel.sh for the pool).
#
#   bash sh_scripts/run_cache_embedding_parallel.sh
#   MAX_JOBS=6 DATASETS="cora pubmed" LLM_NAMES="gemma2_2B" bash sh_scripts/run_cache_embedding_parallel.sh
#
# Env overrides (defaults in brackets):
#   DATASETS [cora]  LLM_NAMES [llama_3.2_1B smollm2_1.7B gemma2_2B]
#   INIT_APPROACHES [pissa gaussian eva loftq orthogonal]  SEEDS [42 43 44 45]
#   SPLIT [0]  POOLING [mean]  PEFT_TYPE [lora]
#   BATCH_SIZE []      caching batch size for every job ("" = caching_batch_size of the llm config;
#                      on a 24GB card 16-32 is fine). Embeddings do not depend on it.
#   MAX_JOBS [4]  MIN_FREE_GB [4]  STAGGER_SEC [20]  OOM_RETRIES [2]  GPU_ID [0]
#   SKIP_EXISTING [1]  SKIP_MISSING_ADAPTER [1]  SYNC_DIR []
#
# Output:  artifacts/cache/*.pt  (same files as the sequential script, so both can be mixed)
# Status:  results/run_status/cache_run_history.csv, CACHE_FAILED_RUNS.txt, PARALLEL_CACHE_STATUS.txt
# Logs:    results/run_status/logs/cache_<dataset>_<llm>_<init>_seed<seed>.log
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PEFT_TYPE="${PEFT_TYPE:-lora}"
SPLIT="${SPLIT:-0}"
POOLING="${POOLING:-mean}"
DATASETS="${DATASETS:-cora}"
LLM_NAMES="${LLM_NAMES:-llama_3.2_1B smollm2_1.7B gemma2_2B}"
INIT_APPROACHES="${INIT_APPROACHES:-pissa gaussian eva loftq orthogonal}"
SEEDS="${SEEDS:-42 43 44 45}"
BATCH_SIZE="${BATCH_SIZE:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_MISSING_ADAPTER="${SKIP_MISSING_ADAPTER:-1}"
MAX_JOBS="${MAX_JOBS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
STAGGER_SEC="${STAGGER_SEC:-20}"
OOM_RETRIES="${OOM_RETRIES:-2}"
source "$(dirname "${BASH_SOURCE[0]}")/_parallel.sh"

HISTORY_FILE="$STATUS_DIR/cache_run_history.csv"
FAILED_FILE="$STATUS_DIR/CACHE_FAILED_RUNS.txt"
LOG_DIR="$STATUS_DIR/logs"; mkdir -p "$LOG_DIR"
[[ -f "$HISTORY_FILE" ]] || echo "timestamp,status,dataset,llm_name,peft_type,lora_init,split,seed,exit_code" > "$HISTORY_FILE"

hr; echo "DIVERGE embedding-cache sweep  (PARALLEL, max $MAX_JOBS jobs)"
echo "datasets=[$DATASETS] llms=[$LLM_NAMES] inits=[$INIT_APPROACHES] seeds=[$SEEDS] split=$SPLIT pooling=$POOLING batch=${BATCH_SIZE:-config}"
echo "$(env_fingerprint)"; hr

n_skip=0
# seed loop outermost: the first jobs of the pool are different LLMs/inits -> mixed VRAM footprint
for s in $SEEDS; do
  for d in $DATASETS; do
    for m in $LLM_NAMES; do
      for i in $INIT_APPROACHES; do
        RUN_ID="dataset=$d | llm=$m | init=$i | seed=$s"
        CFILE="$(cache_file "$m" "$d" "$i" "$s" "$POOLING")"
        ADIR="$(adapter_dir "$m" "$d" "$i" "$s" "$PEFT_TYPE")"
        if [[ "$SKIP_EXISTING" == "1" && "$SPLIT" != "1" && -f "$CFILE" ]]; then
            log "SKIP (cache exists): $RUN_ID"; n_skip=$((n_skip + 1)); continue
        fi
        if [[ "$SKIP_MISSING_ADAPTER" == "1" && ! -f "$ADIR/adapter_model.safetensors" ]]; then
            log "SKIP (no adapter at $ADIR): $RUN_ID"; n_skip=$((n_skip + 1)); continue
        fi
        CMD=("$PYTHON" -m cache.cache_embedding_with_diffrent_init_weights --dataset_name "$d" --llm_name "$m"
             --peft_type "$PEFT_TYPE" --init_weight_approach "$i" --split "$SPLIT" --pooling "$POOLING" --seed "$s")
        [[ -n "$BATCH_SIZE" ]] && CMD+=(--batch_size "$BATCH_SIZE")
        JOB_NAMES+=("$RUN_ID")
        JOB_META+=("$d,$m,$PEFT_TYPE,$i,$SPLIT,$s")
        JOB_CMDS+=("$(printf '%q ' "${CMD[@]}")")
        JOB_LOGS+=("$LOG_DIR/cache_${d}_${m}_${i}_seed${s}.log")
        JOB_CLEAN+=("")
      done
    done
  done
done

if (( ${#JOB_CMDS[@]} == 0 )); then
    hr; echo "Nothing to do (skipped=$n_skip)."; hr; exit 0
fi

pool_run cache

hr; echo "Parallel cache sweep finished: ok=$POOL_OK failed=$POOL_FAIL skipped=$n_skip   (artifacts/cache/)"
[[ "$POOL_FAIL" -gt 0 ]] && echo "Failed runs: $FAILED_FILE"
hr
sync_outputs
