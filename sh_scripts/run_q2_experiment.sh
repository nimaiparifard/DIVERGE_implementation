#!/usr/bin/env bash
# Q2 experiment (R1-C2, R3-C2), Linux/cloud version of scripts/run_q2_experiment.bat.
# Controlled pool per (dataset, llm): 5 inits x training seeds on one split (SPLIT_SEED) and one
# LoRA init draw (INIT_SEED); train -> cache -> analyse, then the across-pool claim.
# Finished adapters / caches are skipped. Output: results/diversity_q2/
set -u
cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"cora citeseer pubmed wikics instagram"}
LLMS=${LLMS:-"gemma2_2B llama_3.2_1B"}
INITS=${INITS:-"pissa orthogonal gaussian loftq eva"}
TRAIN_SEEDS=${TRAIN_SEEDS:-"42 43 44 45"}
SPLIT_SEED=${SPLIT_SEED:-42}
INIT_SEED=${INIT_SEED:-0}
PEFT_TYPE=lora
EXTRA_ARGS=${EXTRA_ARGS:-"--knn"}
CACHE=artifacts/cache
export WANDB_DISABLED=true WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1 PYTHONIOENCODING=utf-8

for d in $DATASETS; do for l in $LLMS; do
    for i in $INITS; do for s in $TRAIN_SEEDS; do
        tag="_seed${s}_split${SPLIT_SEED}_init${INIT_SEED}"
        if [ ! -f "artifacts/${l}_${d}_seqcls_${PEFT_TYPE}_semi_supervised_init_type_${i}${tag}/adapter_model.safetensors" ]; then
            echo "[train] $d $l $i seed $s"
            python -m train_llm.train_other_lora_init_apporaches --dataset_name "$d" --llm_name "$l" \
                --peft_type $PEFT_TYPE --init_weight_approach "$i" --split 1 --seed "$s" \
                --split_seed "$SPLIT_SEED" --init_seed "$INIT_SEED"
        fi
        if [ ! -f "$CACHE/${l}_${d}_seqcls_${PEFT_TYPE}_init-${i}_pool-mean${tag}_semi_supervised.pt" ]; then
            echo "[cache] $d $l $i seed $s"
            python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name "$d" --llm_name "$l" \
                --peft_type $PEFT_TYPE --init_weight_approach "$i" --split 0 --pooling mean --seed "$s" \
                --split_seed "$SPLIT_SEED" --init_seed "$INIT_SEED"
        fi
    done; done
    python -m experiments.diversity_experiment.init_vs_training_noise --dataset_name "$d" --llm_name "$l" \
        --peft_type $PEFT_TYPE --seeds "$(echo $TRAIN_SEEDS | tr ' ' ',')" --split_seed "$SPLIT_SEED" \
        --init_seed "$INIT_SEED" $EXTRA_ARGS
done; done

python -m experiments.diversity_experiment.aggregate_q2
echo "Done. See results/diversity_q2/q2_across_pools.txt"
