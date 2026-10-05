#!/usr/bin/env bash
# =============================================================================
# Linux port of scripts/run_cache_embedding_all_datasets.bat
# Cache LLM embeddings for every trained (dataset, llm, init, seed) adapter.
# Run AFTER sh_scripts/run_all_datasets_lora_init.sh (or after restoring those adapters).
#
#   bash sh_scripts/run_cache_embedding_all_datasets.sh
#   DATASETS="cora" LLM_NAMES="gemma2_2B" SEEDS="42" bash sh_scripts/run_cache_embedding_all_datasets.sh
#
# Env overrides (defaults in brackets):
#   DATASETS [cora]  LLM_NAMES [llama_3.2_1B smollm2_1.7B gemma2_2B]
#   INIT_APPROACHES [pissa gaussian eva loftq orthogonal]  SEEDS [42 43 44 45]
#   SPLIT [0]  POOLING [mean]  PEFT_TYPE [lora]
#   SKIP_EXISTING [1]  skip combos whose cache .pt already exists
#   SKIP_MISSING_ADAPTER [1]  skip (instead of fail) combos whose adapter was never trained
#   SYNC_DIR   []      merge artifacts/ + results/ into this dir after every finished run
#
# SPLIT=0 is required for seed-tagged caches: with SPLIT=1 the cache script looks for a
# non-seeded adapter dir that train_other_lora_init_apporaches.py never writes.
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
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_MISSING_ADAPTER="${SKIP_MISSING_ADAPTER:-1}"

CURRENT_FILE="$STATUS_DIR/CACHE_CURRENT_RUN.txt"
HISTORY_FILE="$STATUS_DIR/cache_run_history.csv"
LAST_OK_FILE="$STATUS_DIR/CACHE_LAST_COMPLETED.txt"
FAILED_FILE="$STATUS_DIR/CACHE_FAILED_RUNS.txt"
LOG_DIR="$STATUS_DIR/logs"; mkdir -p "$LOG_DIR"
[[ -f "$HISTORY_FILE" ]] || echo "timestamp,status,dataset,llm_name,peft_type,lora_init,split,seed,exit_code" > "$HISTORY_FILE"

hr; echo "DIVERGE embedding-cache multi-model sweep"
echo "datasets=[$DATASETS] llms=[$LLM_NAMES] inits=[$INIT_APPROACHES] seeds=[$SEEDS] split=$SPLIT pooling=$POOLING"
echo "$(env_fingerprint)"; hr

n_ok=0; n_fail=0; n_skip=0
for d in $DATASETS; do
  for m in $LLM_NAMES; do
    for i in $INIT_APPROACHES; do
      for s in $SEEDS; do
        RUN_ID="dataset=$d | llm=$m | peft=$PEFT_TYPE | init=$i | split=$SPLIT | seed=$s"
        CFILE="$(cache_file "$m" "$d" "$i" "$s" "$POOLING")"
        ADIR="$(adapter_dir "$m" "$d" "$i" "$s" "$PEFT_TYPE")"
        if [[ "$SKIP_EXISTING" == "1" && "$SPLIT" != "1" && -f "$CFILE" ]]; then
            log "SKIP (cache exists): $RUN_ID"; n_skip=$((n_skip+1)); continue
        fi
        if [[ "$SKIP_MISSING_ADAPTER" == "1" && ! -f "$ADIR/adapter_model.safetensors" ]]; then
            log "SKIP (no adapter at $ADIR): $RUN_ID"; n_skip=$((n_skip+1)); continue
        fi

        CMD=("$PYTHON" -m cache.cache_embedding_with_diffrent_init_weights --dataset_name "$d" --llm_name "$m"
             --peft_type "$PEFT_TYPE" --init_weight_approach "$i" --split "$SPLIT" --pooling "$POOLING" --seed "$s")
        TS="$(ts)"
        hr; echo "START: $RUN_ID"; echo "Time: $TS"; hr
        write_kv "$CURRENT_FILE" status=RUNNING "started=$TS" "dataset=$d" "llm_name=$m" "peft_type=$PEFT_TYPE" \
            "lora_init=$i" "split=$SPLIT" "seed=$s" "command=${CMD[*]}" "host=$(hostname)"

        "${CMD[@]}" 2>&1 | tee "$LOG_DIR/cache_${d}_${m}_${i}_seed${s}.log"
        EXIT_CODE=${PIPESTATUS[0]}
        TS_END="$(ts)"

        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "ERROR: $RUN_ID  exit_code=$EXIT_CODE"
            echo "$TS_END,FAILED,$d,$m,$PEFT_TYPE,$i,$SPLIT,$s,$EXIT_CODE" >> "$HISTORY_FILE"
            echo "$TS_END FAILED $RUN_ID exit=$EXIT_CODE" >> "$FAILED_FILE"
            write_kv "$CURRENT_FILE" status=FAILED "started=$TS" "finished=$TS_END" "dataset=$d" "llm_name=$m" \
                "peft_type=$PEFT_TYPE" "lora_init=$i" "split=$SPLIT" "seed=$s" "exit_code=$EXIT_CODE"
            echo "Continuing to next run despite failure..."; n_fail=$((n_fail+1))
        else
            echo "OK: $RUN_ID"
            echo "$TS_END,OK,$d,$m,$PEFT_TYPE,$i,$SPLIT,$s,0" >> "$HISTORY_FILE"
            write_kv "$LAST_OK_FILE" status=COMPLETED "started=$TS" "finished=$TS_END" "dataset=$d" "llm_name=$m" \
                "peft_type=$PEFT_TYPE" "lora_init=$i" "split=$SPLIT" "seed=$s" "exit_code=0"
            cp -f "$LAST_OK_FILE" "$CURRENT_FILE"; n_ok=$((n_ok+1))
            sync_outputs
        fi
        echo
      done
    done
  done
done

hr; echo "Cache sweep finished: ok=$n_ok failed=$n_fail skipped=$n_skip   (artifacts/cache/)"; hr
write_kv "$CURRENT_FILE" status=IDLE "finished=$(ts)" "note=all queued cache runs finished; see cache_run_history.csv"
sync_outputs
