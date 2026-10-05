@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Initialization-diversity vs seed-variance experiments (reviewer R1-C2 / R2-C1)
REM across datasets x LLMs. Every run uses the whole init x seed pool at once
REM (e.g. 5 inits x 4 seeds = 20 members) for one (dataset, llm):
REM
REM   1) experiments/seed_vs_init_diversity.py     seed noise vs init effect (CKA,
REM                                                disagreement, PERMANOVA)
REM   2) experiments/error_consistency_analysis.py are init differences systematic?
REM                                                (error overlap, rescues, per-class ANOVA)
REM   3) experiments/complementarity_analysis.py   complementarity + ensemble gain,
REM                                                multi-init vs seed-only ensembles
REM   4) experiments/lora_subspace_analysis.py     adapter weight subspaces, spectral
REM                                                position, stage-chain Mantel tests
REM   5) experiments/knn_neighborhood_analysis.py  local k-NN neighbourhood agreement
REM
REM Requires, for every (dataset, llm, init, seed):
REM   1) scripts/run_all_datasets_lora_init.bat       (trains the LoRA adapters)
REM   2) scripts/run_cache_embedding_all_datasets.bat (caches seed-tagged embeddings)
REM Members without a seed-tagged cache are skipped and listed in each report.
REM
REM The first experiment of a (dataset, llm) trains one GNN per member (fixed
REM GNN seed) and caches it in results/diversity/gnn_pool/; the others reuse it.
REM
REM Outputs: results/diversity/<experiment>/  (CSVs, heatmaps, <tag>_report.txt)
REM Progress: results/run_status/DIVERSITY_CURRENT_RUN.txt
REM           results/run_status/diversity_run_history.csv
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
REM SPLIT=0 = semi-supervised seed-tagged caches (what run_cache_embedding_all_datasets.bat writes)
set SPLIT=1
REM LoRA seeds forming the pool (comma-separated, must match the trained/cached adapters)
set SEEDS=42,43,44,45
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
REM Leakage-free setup (default): every member's LoRA was fine-tuned on split SPLIT_SEED
REM (trained with --split_seed, e.g. by scripts/run_q2_experiment.bat) and every GNN is
REM trained and tested on the same split (GNN_SEED = SPLIT_SEED), so no member has seen a
REM test label. INIT_SEED: the --init_seed the members were trained with (empty = none).
REM Old members (LoRA seed s fine-tuned on split s): set SPLIT_SEED= and GNN_SEED=-1
REM (own split), or add --allow_leakage to EXTRA_ARGS to reproduce the old leaky reports.
set SPLIT_SEED=42
set INIT_SEED=0
set GNN_SEED=42
set LORA_ARGS=
if not "%SPLIT_SEED%"=="" set LORA_ARGS=--split_seed %SPLIT_SEED%
if not "%SPLIT_SEED%"=="" if not "%INIT_SEED%"=="" set LORA_ARGS=!LORA_ARGS! --init_seed %INIT_SEED%
set N_PERM=2000
set OUT_DIR=results\diversity
REM Extra flags for every experiment, e.g. "--gnn_model_name GCN" or "--retrain_gnn"
set EXTRA_ARGS=

REM Which experiments to run (1 = on, 0 = off)
set RUN_SEED_VS_INIT=1
set RUN_ERROR_CONSISTENCY=1
set RUN_COMPLEMENTARITY=1
set RUN_LORA_SUBSPACE=1
set RUN_KNN=1

set PYTHONIOENCODING=utf-8

REM Datasets
set DATASETS=cora

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=llama_3.2_1B smollm2_1.7B gemma2_2B

set EXPERIMENTS=
if "%RUN_SEED_VS_INIT%"=="1" set EXPERIMENTS=!EXPERIMENTS! seed_vs_init_diversity
if "%RUN_ERROR_CONSISTENCY%"=="1" set EXPERIMENTS=!EXPERIMENTS! error_consistency_analysis
if "%RUN_COMPLEMENTARITY%"=="1" set EXPERIMENTS=!EXPERIMENTS! complementarity_analysis
if "%RUN_LORA_SUBSPACE%"=="1" set EXPERIMENTS=!EXPERIMENTS! lora_subspace_analysis
if "%RUN_KNN%"=="1" set EXPERIMENTS=!EXPERIMENTS! knn_neighborhood_analysis

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\DIVERSITY_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\diversity_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\DIVERSITY_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\DIVERSITY_FAILED_RUNS.txt

REM Create history CSV header if missing
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,experiment,dataset,llm_name,peft_type,split,seeds,gnn_seed,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE init-diversity vs seed-variance experiments
echo Experiments:%EXPERIMENTS%
echo Status dir: %STATUS_DIR%
echo History CSV: %HISTORY_FILE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%e in (%EXPERIMENTS%) do (
            set "RUN_ID=experiment=%%e | dataset=%%d | llm=%%m | peft=%PEFT_TYPE% | split=%SPLIT% | seeds=%SEEDS% | gnn_seed=%GNN_SEED%"
            set "TS=!DATE! !TIME!"

            echo ==========================================
            echo START: !RUN_ID!
            echo Time: !TS!
            echo ==========================================

            (
                echo status=RUNNING
                echo started=!TS!
                echo experiment=%%e
                echo dataset=%%d
                echo llm_name=%%m
                echo seeds=%SEEDS%
                echo command=python -m experiments.%%e --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seeds %SEEDS% --approaches %APPROACHES% --split %SPLIT% --gnn_seed %GNN_SEED% %LORA_ARGS% --n_perm %N_PERM% --out_dir %OUT_DIR% %EXTRA_ARGS%
            ) > "%CURRENT_FILE%"

            python -m experiments.%%e --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --seeds "%SEEDS%" --approaches "%APPROACHES%" --split %SPLIT% --gnn_seed %GNN_SEED% %LORA_ARGS% --n_perm %N_PERM% --out_dir "%OUT_DIR%" %EXTRA_ARGS%
            set EXIT_CODE=!ERRORLEVEL!
            set "TS_END=!DATE! !TIME!"

            if !EXIT_CODE! NEQ 0 (
                echo ERROR: !RUN_ID!  exit_code=!EXIT_CODE!
                REM redirect first: "...,0>> file" would be parsed as a redirect of handle 0
                >> "%HISTORY_FILE%" echo !TS_END!,FAILED,%%e,%%d,%%m,%PEFT_TYPE%,%SPLIT%,"%SEEDS%",%GNN_SEED%,!EXIT_CODE!
                >> "%FAILED_FILE%" echo !TS_END! FAILED !RUN_ID! exit=!EXIT_CODE!
                echo Continuing to next run despite failure...
                echo.
            ) else (
                echo OK: !RUN_ID!
                >> "%HISTORY_FILE%" echo !TS_END!,OK,%%e,%%d,%%m,%PEFT_TYPE%,%SPLIT%,"%SEEDS%",%GNN_SEED%,0
                (
                    echo status=COMPLETED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo experiment=%%e
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo exit_code=0
                ) > "%LAST_OK_FILE%"
                echo.
            )
        )
    )
)

echo ==========================================
echo Diversity experiments finished.
echo Check:
echo   %OUT_DIR%\seed_vs_init\seed_vs_init_summary.csv
echo   %OUT_DIR%\error_consistency\error_consistency_summary.csv
echo   %OUT_DIR%\complementarity\complementarity_summary.csv
echo   %OUT_DIR%\lora_subspace\lora_subspace_summary.csv
echo   %OUT_DIR%\knn_neighborhood\knn_summary.csv
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued diversity runs finished; see diversity_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
