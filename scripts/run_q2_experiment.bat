@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Q2 experiment (R1-C2, R3-C2): is the diversity between LoRA initializations larger
REM than the diversity between training seeds of the SAME initialization?
REM
REM Controlled pool per (dataset, llm): 5 inits x training seeds, all fine-tuned on the
REM same split (SPLIT_SEED) from the same LoRA init draw (INIT_SEED); only the training
REM seed (head init, data order, dropout) varies. The GNN uses the same split.
REM   1) train   train_llm.train_other_lora_init_apporaches --split 1 --seed s
REM              --split_seed SPLIT_SEED --init_seed INIT_SEED
REM   2) cache   cache.cache_embedding_with_diffrent_init_weights (same seeds)
REM   3) analyse experiments.diversity_experiment.init_vs_training_noise
REM   4) claim   experiments.diversity_experiment.aggregate_q2 (sign test + Wilcoxon over pools)
REM Finished adapters / caches are skipped, so the script can be restarted.
REM Output: results\diversity_q2\
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM 10 pools = 5 datasets x 2 LLMs (fixed before running; do not drop pools that fail)
set DATASETS=cora citeseer pubmed wikics instagram
set LLMS=gemma2_2B llama_3.2_1B
set INITS=pissa orthogonal gaussian loftq eva
set TRAIN_SEEDS=42 43 44 45
set SPLIT_SEED=42
set INIT_SEED=0
set PEFT_TYPE=lora
REM e.g. "--exclude_collapsed" or "--gnn_model_name SAGE"
set EXTRA_ARGS=--knn

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
set CACHE=artifacts\cache

for %%d in (%DATASETS%) do for %%l in (%LLMS%) do (
    for %%i in (%INITS%) do for %%s in (%TRAIN_SEEDS%) do (
        set "TAG=_seed%%s_split%SPLIT_SEED%_init%INIT_SEED%"
        if not exist "artifacts\%%l_%%d_seqcls_%PEFT_TYPE%_semi_supervised_init_type_%%i!TAG!\adapter_model.safetensors" (
            echo [train] %%d %%l %%i seed %%s
            python -m train_llm.train_other_lora_init_apporaches --dataset_name %%d --llm_name %%l --peft_type %PEFT_TYPE% --init_weight_approach %%i --split 1 --seed %%s --split_seed %SPLIT_SEED% --init_seed %INIT_SEED%
        )
        if not exist "%CACHE%\%%l_%%d_seqcls_%PEFT_TYPE%_init-%%i_pool-mean!TAG!_semi_supervised.pt" (
            echo [cache] %%d %%l %%i seed %%s
            python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name %%d --llm_name %%l --peft_type %PEFT_TYPE% --init_weight_approach %%i --split 0 --pooling mean --seed %%s --split_seed %SPLIT_SEED% --init_seed %INIT_SEED%
        )
    )
    python -m experiments.diversity_experiment.init_vs_training_noise --dataset_name %%d --llm_name %%l --peft_type %PEFT_TYPE% --seeds 42,43,44,45 --split_seed %SPLIT_SEED% --init_seed %INIT_SEED% %EXTRA_ARGS%
)

python -m experiments.diversity_experiment.aggregate_q2

echo Done. See results\diversity_q2\q2_across_pools.txt
pause
endlocal
