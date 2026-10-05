#!/usr/bin/env bash
# Q2 controlled pools for Cora and CiteSeer only (subset of scripts/run_q2_experiment.bat, same
# settings). LLaMA pools first (faster). Finished adapters / caches are skipped, so the script can be
# restarted; a failed step is logged and the loop continues.
# Usage (Git Bash, from the repo root):  bash scripts/run_q2_cora_citeseer.sh
cd "$(dirname "$0")/.." || exit 1

LLMS="llama_3.2_1B gemma2_2B"
DATASETS="cora citeseer"
INITS="pissa orthogonal gaussian loftq eva"
TRAIN_SEEDS="42 43 44 45"
SPLIT_SEED=42
INIT_SEED=0
PEFT_TYPE=lora
CACHE=artifacts/cache
LOG_DIR=results/diversity_q2/logs
mkdir -p "$LOG_DIR"

export WANDB_DISABLED=true WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1 PYTHONIOENCODING=utf-8

for l in $LLMS; do for d in $DATASETS; do
    for i in $INITS; do for s in $TRAIN_SEEDS; do
        tag="_seed${s}_split${SPLIT_SEED}_init${INIT_SEED}"
        if [ ! -f "artifacts/${l}_${d}_seqcls_${PEFT_TYPE}_semi_supervised_init_type_${i}${tag}/adapter_model.safetensors" ]; then
            echo "$(date '+%F %T') [train] $d $l $i seed $s"
            python -m train_llm.train_other_lora_init_apporaches --dataset_name "$d" --llm_name "$l" \
                --peft_type "$PEFT_TYPE" --init_weight_approach "$i" --split 1 --seed "$s" \
                --split_seed "$SPLIT_SEED" --init_seed "$INIT_SEED" > "$LOG_DIR/train_${d}_${l}_${i}_${s}.log" 2>&1 \
                || echo "$(date '+%F %T') [FAILED train] $d $l $i $s"
        fi
        if [ ! -f "$CACHE/${l}_${d}_seqcls_${PEFT_TYPE}_init-${i}_pool-mean${tag}_semi_supervised.pt" ]; then
            echo "$(date '+%F %T') [cache] $d $l $i seed $s"
            python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name "$d" --llm_name "$l" \
                --peft_type "$PEFT_TYPE" --init_weight_approach "$i" --split 0 --pooling mean --seed "$s" \
                --split_seed "$SPLIT_SEED" --init_seed "$INIT_SEED" > "$LOG_DIR/cache_${d}_${l}_${i}_${s}.log" 2>&1 \
                || echo "$(date '+%F %T') [FAILED cache] $d $l $i $s"
        fi
    done; done
    echo "$(date '+%F %T') [analyse] $d $l"
    python -m experiments.diversity_experiment.init_vs_training_noise --dataset_name "$d" --llm_name "$l" \
        --peft_type "$PEFT_TYPE" --seeds 42,43,44,45 --split_seed "$SPLIT_SEED" --init_seed "$INIT_SEED" \
        --knn --no_plots > "$LOG_DIR/analyse_${d}_${l}.log" 2>&1 \
        || echo "$(date '+%F %T') [FAILED analyse] $d $l"
done; done

python -m experiments.diversity_experiment.aggregate_q2 > "$LOG_DIR/aggregate.log" 2>&1
echo "$(date '+%F %T') Done. See results/diversity_q2/q2_across_pools.txt"
