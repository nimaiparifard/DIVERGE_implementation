@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Seed-control ensemble (reviewers R2-C2, R4-C1, R1-C7b): DIVERGE's 5 initializations
REM vs 5 LoRA seeds of ONE initialization, both fine-tuned on the same split.
REM   1) trains CONTROL_INIT with LoRA seeds split+100..split+500 on every split seed
REM      (train_llm/train_other_lora_init_apporaches.py --split_seed)
REM   2) caches their embeddings (and any missing seed-tagged DIVERGE cache)
REM   3) runs experiments/seed_control_ensemble.py
REM Finished adapters / caches are skipped, so the script can be restarted.
REM Result: results/seed_control/<dataset>_<llm>_<gnn>_<init>_report.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set DATASETS=cora citeseer
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora
set SPLIT_SEEDS=42 43 44 45
set CONTROL_INIT=gaussian
set OFFSETS=100 200 300 400 500
set DIVERGE_INITS=pissa orthogonal gaussian loftq eva

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8

REM cache file prefix uses cfg.llm.model_name, which equals LLM_NAME for the local configs
set CACHE=artifacts\cache

for %%d in (%DATASETS%) do (
    REM --- DIVERGE members (LoRA seed = split seed): cache any missing seed-tagged embedding
    for %%s in (%SPLIT_SEEDS%) do for %%i in (%DIVERGE_INITS%) do (
        if not exist "%CACHE%\%LLM_NAME%_%%d_seqcls_%PEFT_TYPE%_init-%%i_pool-mean_seed%%s_semi_supervised.pt" (
            echo [cache] %%d %%i seed %%s
            python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name %%d --llm_name %LLM_NAME% --peft_type %PEFT_TYPE% --init_weight_approach %%i --split 0 --pooling mean --seed %%s
        )
    )
    REM --- control members: same split, different LoRA seeds
    for %%s in (%SPLIT_SEEDS%) do for %%o in (%OFFSETS%) do (
        set /a LSEED=%%s+%%o
        set "CFILE=%CACHE%\%LLM_NAME%_%%d_seqcls_%PEFT_TYPE%_init-%CONTROL_INIT%_pool-mean_seed!LSEED!_split%%s_semi_supervised.pt"
        if not exist "!CFILE!" (
            echo [train] %%d %CONTROL_INIT% lora seed !LSEED! split %%s
            python -m train_llm.train_other_lora_init_apporaches --dataset_name %%d --llm_name %LLM_NAME% --peft_type %PEFT_TYPE% --init_weight_approach %CONTROL_INIT% --split 1 --seed !LSEED! --split_seed %%s
            python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name %%d --llm_name %LLM_NAME% --peft_type %PEFT_TYPE% --init_weight_approach %CONTROL_INIT% --split 0 --pooling mean --seed !LSEED! --split_seed %%s
        )
    )
    python -m experiments.seed_control_ensemble --dataset_name %%d --llm_name %LLM_NAME% --peft_type %PEFT_TYPE% --control_init %CONTROL_INIT%
)

echo Done. See results\seed_control\
pause
endlocal
