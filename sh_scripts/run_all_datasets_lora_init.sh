#!/usr/bin/env bash
# =============================================================================
# Linux port of scripts/run_all_datasets_lora_init.bat
# LoRA-init fine-tuning across datasets x LLMs x init approaches x seeds.
#
#   bash sh_scripts/run_all_datasets_lora_init.sh
#   DATASETS="cora pubmed" LLM_NAMES="gemma2_2B" SEEDS="42" bash sh_scripts/run_all_datasets_lora_init.sh
#
# Env overrides (defaults in brackets):
#   DATASETS [cora]  LLM_NAMES [llama_3.2_1B smollm2_1.7B gemma2_2B]
#   INIT_APPROACHES [pissa gaussian eva loftq orthogonal]  SEEDS [42 43 44 45]
#   SPLIT [1]  PEFT_TYPE [lora]
#   SKIP_EXISTING [1]  skip a combo whose adapter dir already holds a finished adapter
#                      (adapter_model.safetensors + training_results.csv) -> safe to re-run / resume
#   SYNC_DIR   []      after every finished run, merge artifacts/ + results/ into this dir
#                      (Colab: /content/drive/MyDrive/DIVERGE_sync  RunPod: /workspace/DIVERGE_sync)
#
# Each run writes artifacts/<llm>_<dataset>_seqcls_lora_semi_supervised_init_type_<init>_seed<seed>/
# Status: results/run_status/{CURRENT_RUN.txt,run_history.csv,LAST_COMPLETED.txt,FAILED_RUNS.txt}
# Logs:   results/run_status/logs/lora_<dataset>_<llm>_<init>_seed<seed>.log
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PEFT_TYPE="${PEFT_TYPE:-lora}"
# NOTE: the cache / DIVERGE scripts default to SPLIT=0. With SPLIT=1 the LLM is trained on a
# seeded 60/20/20 re-split, with SPLIT=0 on the dataset's own split - keep them consistent.
SPLIT="${SPLIT:-1}"
DATASETS="${DATASETS:-cora}"
LLM_NAMES="${LLM_NAMES:-llama_3.2_1B smollm2_1.7B gemma2_2B}"
INIT_APPROACHES="${INIT_APPROACHES:-pissa gaussian eva loftq orthogonal}"
SEEDS="${SEEDS:-42 43 44 45}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CURRENT_FILE="$STATUS_DIR/CURRENT_RUN.txt"
HISTORY_FILE="$STATUS_DIR/run_history.csv"
LAST_OK_FILE="$STATUS_DIR/LAST_COMPLETED.txt"
FAILED_FILE="$STATUS_DIR/FAILED_RUNS.txt"
LOG_DIR="$STATUS_DIR/logs"; mkdir -p "$LOG_DIR"
[[ -f "$HISTORY_FILE" ]] || echo "timestamp,status,dataset,llm_name,peft_type,lora_init,split,seed,exit_code" > "$HISTORY_FILE"

hr; echo "DIVERGE LoRA-init multi-model sweep"
echo "datasets=[$DATASETS] llms=[$LLM_NAMES] inits=[$INIT_APPROACHES] seeds=[$SEEDS] split=$SPLIT"
echo "$(env_fingerprint)"
echo "History CSV: $HISTORY_FILE   SYNC_DIR: ${SYNC_DIR:-<none>}"; hr

n_ok=0; n_fail=0; n_skip=0
for d in $DATASETS; do
  for m in $LLM_NAMES; do
    for i in $INIT_APPROACHES; do
      for s in $SEEDS; do
        RUN_ID="dataset=$d | llm=$m | peft=$PEFT_TYPE | init=$i | split=$SPLIT | seed=$s"
        ADIR="$(adapter_dir "$m" "$d" "$i" "$s" "$PEFT_TYPE")"
        if [[ "$SKIP_EXISTING" == "1" && -f "$ADIR/adapter_model.safetensors" && -f "$ADIR/training_results.csv" ]]; then
            log "SKIP (adapter exists): $RUN_ID"; n_skip=$((n_skip+1)); continue
        fi

        CMD=("$PYTHON" -m train_llm.train_other_lora_init_apporaches --dataset_name "$d" --llm_name "$m"
             --peft_type "$PEFT_TYPE" --init_weight_approach "$i" --split "$SPLIT" --seed "$s")
        TS="$(ts)"
        hr; echo "START: $RUN_ID"; echo "Time: $TS"; hr
        write_kv "$CURRENT_FILE" status=RUNNING "started=$TS" "dataset=$d" "llm_name=$m" "peft_type=$PEFT_TYPE" \
            "lora_init=$i" "split=$SPLIT" "seed=$s" "command=${CMD[*]}" "host=$(hostname)"

        "${CMD[@]}" 2>&1 | tee "$LOG_DIR/lora_${d}_${m}_${i}_seed${s}.log"
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

hr; echo "Sweep finished: ok=$n_ok failed=$n_fail skipped=$n_skip"
echo "Check: $HISTORY_FILE  $FAILED_FILE  results/efficiency/lora_finetune_efficiency.csv"; hr
write_kv "$CURRENT_FILE" status=IDLE "finished=$(ts)" "note=all queued runs finished; see run_history.csv"
sync_outputs
