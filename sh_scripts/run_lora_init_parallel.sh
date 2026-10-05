#!/usr/bin/env bash
# =============================================================================
# PARALLEL version of sh_scripts/run_all_datasets_lora_init.sh (Linux counterpart of
# scripts/run_lora_init_parallel.bat): several LoRA fine-tuning runs on the same GPU at once.
# Training uses per_device_train_batch_size=1, so one run leaves most of a 4090 idle
# (~4GB peak VRAM for llama_3.2_1B) - 3-5 runs at once finish the sweep several times faster.
#
#   bash sh_scripts/run_lora_init_parallel.sh
#   MAX_JOBS=5 DATASETS="cora" LLM_NAMES="llama_3.2_1B" SEEDS="42 43" bash sh_scripts/run_lora_init_parallel.sh
#
# Env overrides (defaults in brackets):
#   DATASETS [cora]  LLM_NAMES [llama_3.2_1B smollm2_1.7B gemma2_2B]
#   INIT_APPROACHES [pissa gaussian eva loftq orthogonal]  SEEDS [42 43 44 45]
#   SPLIT [1]  PEFT_TYPE [lora]
#   MAX_JOBS [4]  MIN_FREE_GB [6]  STAGGER_SEC [60]  OOM_RETRIES [2]  GPU_ID [0]
#   KEEP_CHECKPOINTS [0]  1 = keep outputs/<run>/checkpoint-* (the adapter is saved in artifacts/ anyway)
#   SKIP_EXISTING [1]  SYNC_DIR []
#
# Each run has its own HF Trainer dir outputs/<adapter name>/ (set in
# train_llm/train_other_lora_init_apporaches.py), so parallel runs never touch each other's checkpoints.
# Status: results/run_status/run_history.csv, FAILED_RUNS.txt, PARALLEL_TRAIN_STATUS.txt
# Logs:   results/run_status/logs/lora_<dataset>_<llm>_<init>_seed<seed>.log
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PEFT_TYPE="${PEFT_TYPE:-lora}"
SPLIT="${SPLIT:-1}"
DATASETS="${DATASETS:-computer}"
LLM_NAMES="${LLM_NAMES:-llama_3.2_1B smollm2_1.7B gemma2_2B}"
INIT_APPROACHES="${INIT_APPROACHES:-pissa gaussian eva loftq orthogonal}"
SEEDS="${SEEDS:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
MAX_JOBS="${MAX_JOBS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
STAGGER_SEC="${STAGGER_SEC:-60}"
OOM_RETRIES="${OOM_RETRIES:-2}"
source "$(dirname "${BASH_SOURCE[0]}")/_parallel.sh"

HISTORY_FILE="$STATUS_DIR/run_history.csv"
FAILED_FILE="$STATUS_DIR/FAILED_RUNS.txt"
LOG_DIR="$STATUS_DIR/logs"; mkdir -p "$LOG_DIR"
[[ -f "$HISTORY_FILE" ]] || echo "timestamp,status,dataset,llm_name,peft_type,lora_init,split,seed,exit_code" > "$HISTORY_FILE"

hr; echo "DIVERGE LoRA-init sweep  (PARALLEL, max $MAX_JOBS jobs)"
echo "datasets=[$DATASETS] llms=[$LLM_NAMES] inits=[$INIT_APPROACHES] seeds=[$SEEDS] split=$SPLIT"
echo "$(env_fingerprint)"; hr

n_skip=0
for s in $SEEDS; do
  for d in $DATASETS; do
    for m in $LLM_NAMES; do
      for i in $INIT_APPROACHES; do
        RUN_ID="dataset=$d | llm=$m | init=$i | seed=$s"
        ADIR="$(adapter_dir "$m" "$d" "$i" "$s" "$PEFT_TYPE")"
        if [[ "$SKIP_EXISTING" == "1" && -f "$ADIR/adapter_model.safetensors" && -f "$ADIR/training_results.csv" ]]; then
            log "SKIP (adapter exists): $RUN_ID"; n_skip=$((n_skip + 1)); continue
        fi
        CMD=("$PYTHON" -m train_llm.train_other_lora_init_apporaches --dataset_name "$d" --llm_name "$m"
             --peft_type "$PEFT_TYPE" --init_weight_approach "$i" --split "$SPLIT" --seed "$s")
        JOB_NAMES+=("$RUN_ID")
        JOB_META+=("$d,$m,$PEFT_TYPE,$i,$SPLIT,$s")
        JOB_CMDS+=("$(printf '%q ' "${CMD[@]}")")
        JOB_LOGS+=("$LOG_DIR/lora_${d}_${m}_${i}_seed${s}.log")
        if [[ "$KEEP_CHECKPOINTS" == "1" ]]; then
            JOB_CLEAN+=("")
        else
            JOB_CLEAN+=("rm -rf $(printf '%q' "outputs/$(basename "$ADIR")")")
        fi
      done
    done
  done
done

if (( ${#JOB_CMDS[@]} == 0 )); then
    hr; echo "Nothing to do (skipped=$n_skip)."; hr; exit 0
fi

pool_run train

hr; echo "Parallel training sweep finished: ok=$POOL_OK failed=$POOL_FAIL skipped=$n_skip"
echo "Check: $HISTORY_FILE  $FAILED_FILE  results/efficiency/lora_finetune_efficiency.csv"
echo "NOTE: wall_clock / gpu_hours in the efficiency CSV are inflated for runs that shared the GPU."
hr
sync_outputs
