#!/usr/bin/env bash
# =============================================================================
# Linux port of scripts/run_diverge.bat
# Full DIVERGE pipeline (experiments/diverge_with_edge_editing.py) across datasets x LLMs x seeds:
# per-approach GNN training -> mistake-driven edge editing -> ensemble.
# Needs, for every (dataset, llm, seed): trained adapters + their embedding caches.
#
#   bash sh_scripts/run_diverge.sh
#   DATASETS="cora" LLM_NAMES="gemma2_2B" SEEDS="42" bash sh_scripts/run_diverge.sh
#
# Env overrides (defaults in brackets):
#   DATASETS [cora]  LLM_NAMES [llama_3.2_1B smollm2_1.7B gemma2_2B]  SEEDS [42 43 44 45]
#   SPLIT [0]  PEFT_TYPE [lora]  APPROACHES [pissa,orthogonal,gaussian,loftq,eva]
#   ENSEMBLE_APPROACHES [uniform,learnable,learnable_classes,learnable_per_classes]
#   THRESHOLD_STEPS [17]  DIVERGE_CSV [results/diverge/diverge_results.csv]  POOLING [mean]
#   SKIP_EXISTING [1]  skip combos already marked OK in diverge_run_history.csv
#   SKIP_MISSING_CACHE [1]  skip combos missing any of their embedding caches
#   SYNC_DIR   []      merge artifacts/ + results/ into this dir after every finished run
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PEFT_TYPE="${PEFT_TYPE:-lora}"
SPLIT="${SPLIT:-0}"
POOLING="${POOLING:-mean}"
ENSEMBLE_APPROACHES="${ENSEMBLE_APPROACHES:-uniform,learnable,learnable_classes,learnable_per_classes}"
THRESHOLD_STEPS="${THRESHOLD_STEPS:-17}"
APPROACHES="${APPROACHES:-pissa,orthogonal,gaussian,loftq,eva}"
DIVERGE_CSV="${DIVERGE_CSV:-results/diverge/diverge_results.csv}"
DATASETS="${DATASETS:-cora}"
LLM_NAMES="${LLM_NAMES:-llama_3.2_1B smollm2_1.7B gemma2_2B}"
SEEDS="${SEEDS:-42 43 44 45}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_MISSING_CACHE="${SKIP_MISSING_CACHE:-1}"

CURRENT_FILE="$STATUS_DIR/DIVERGE_CURRENT_RUN.txt"
HISTORY_FILE="$STATUS_DIR/diverge_run_history.csv"
LAST_OK_FILE="$STATUS_DIR/DIVERGE_LAST_COMPLETED.txt"
FAILED_FILE="$STATUS_DIR/DIVERGE_FAILED_RUNS.txt"
LOG_DIR="$STATUS_DIR/logs"; mkdir -p "$LOG_DIR"
[[ -f "$HISTORY_FILE" ]] || echo "timestamp,status,dataset,llm_name,peft_type,split,seed,exit_code" > "$HISTORY_FILE"

hr; echo "DIVERGE pipeline multi-model / multi-seed sweep"
echo "datasets=[$DATASETS] llms=[$LLM_NAMES] seeds=[$SEEDS] split=$SPLIT approaches=$APPROACHES"
echo "$(env_fingerprint)"
echo "Result CSV: $DIVERGE_CSV"; hr

n_ok=0; n_fail=0; n_skip=0
for d in $DATASETS; do
  for m in $LLM_NAMES; do
    for s in $SEEDS; do
      RUN_ID="dataset=$d | llm=$m | peft=$PEFT_TYPE | split=$SPLIT | seed=$s | ensembles=$ENSEMBLE_APPROACHES"
      if [[ "$SKIP_EXISTING" == "1" ]] && grep -q ",OK,$d,$m,$PEFT_TYPE,$SPLIT,$s,0\$" "$HISTORY_FILE"; then
          log "SKIP (already OK in history): $RUN_ID"; n_skip=$((n_skip+1)); continue
      fi
      if [[ "$SKIP_MISSING_CACHE" == "1" && "$SPLIT" != "1" ]]; then
          missing=""
          for a in ${APPROACHES//,/ }; do
              [[ -f "$(cache_file "$m" "$d" "$a" "$s" "$POOLING")" ]] || missing+=" $a"
          done
          if [[ -n "$missing" ]]; then
              log "SKIP (missing embedding cache for:$missing): $RUN_ID"; n_skip=$((n_skip+1)); continue
          fi
      fi

      CMD=("$PYTHON" -m experiments.diverge_with_edge_editing --dataset_name "$d" --llm_name "$m" --peft_type "$PEFT_TYPE"
           --seed "$s" --split "$SPLIT" --approaches "$APPROACHES" --ensemble_approaches "$ENSEMBLE_APPROACHES"
           --threshold_steps "$THRESHOLD_STEPS" --csv_path "$DIVERGE_CSV")
      TS="$(ts)"
      hr; echo "START: $RUN_ID"; echo "Time: $TS"; hr
      write_kv "$CURRENT_FILE" status=RUNNING "started=$TS" "dataset=$d" "llm_name=$m" "peft_type=$PEFT_TYPE" \
          "split=$SPLIT" "seed=$s" "ensemble_approaches=$ENSEMBLE_APPROACHES" "command=${CMD[*]}" "host=$(hostname)"

      "${CMD[@]}" 2>&1 | tee "$LOG_DIR/diverge_${d}_${m}_seed${s}.log"
      EXIT_CODE=${PIPESTATUS[0]}
      TS_END="$(ts)"

      if [[ $EXIT_CODE -ne 0 ]]; then
          echo "ERROR: $RUN_ID  exit_code=$EXIT_CODE"
          echo "$TS_END,FAILED,$d,$m,$PEFT_TYPE,$SPLIT,$s,$EXIT_CODE" >> "$HISTORY_FILE"
          echo "$TS_END FAILED $RUN_ID exit=$EXIT_CODE" >> "$FAILED_FILE"
          write_kv "$CURRENT_FILE" status=FAILED "started=$TS" "finished=$TS_END" "dataset=$d" "llm_name=$m" \
              "peft_type=$PEFT_TYPE" "split=$SPLIT" "seed=$s" "exit_code=$EXIT_CODE"
          echo "Continuing to next run despite failure..."; n_fail=$((n_fail+1))
      else
          echo "OK: $RUN_ID"
          echo "$TS_END,OK,$d,$m,$PEFT_TYPE,$SPLIT,$s,0" >> "$HISTORY_FILE"
          write_kv "$LAST_OK_FILE" status=COMPLETED "started=$TS" "finished=$TS_END" "dataset=$d" "llm_name=$m" \
              "peft_type=$PEFT_TYPE" "split=$SPLIT" "seed=$s" "exit_code=0"
          cp -f "$LAST_OK_FILE" "$CURRENT_FILE"; n_ok=$((n_ok+1))
          sync_outputs
      fi
      echo
    done
  done
done

hr; echo "DIVERGE sweep finished: ok=$n_ok failed=$n_fail skipped=$n_skip"
echo "Check: $HISTORY_FILE  $DIVERGE_CSV  results/gnn_training/gnn_training_results.csv"; hr
write_kv "$CURRENT_FILE" status=IDLE "finished=$(ts)" "note=all queued DIVERGE runs finished; see diverge_run_history.csv"
sync_outputs
