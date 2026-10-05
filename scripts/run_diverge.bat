@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the full DIVERGE pipeline (experiments/diverge.py, no edge editing) across
REM datasets x LLMs x seeds: per-approach GNN training -> ensemble.
REM Requires that, for every
REM (dataset, llm, seed) combination, you already ran:
REM   1) scripts/run_all_datasets_lora_init.bat   (trains the LoRA adapters)
REM   2) scripts/run_cache_embedding_all_datasets.bat (caches their embeddings)
REM
REM Every run appends one complete-info row PER ENSEMBLE APPROACH to:
REM   results/diverge/diverge_no_edge_editing_results.csv
REM (rows of the same run share run_id; columns: per-approach accuracy & F1,
REM  ensemble_weight_approach, ensemble test/val accuracy & F1
REM  - see experiments/diverge.py)
REM The per-run report (all ensemble tuning results + a summary table) is in:
REM   results/train_info/<dataset>_diverge_<split>_seed<seed>/
REM
REM Progress / identity of the active run is also written to:
REM   results/run_status/DIVERGE_CURRENT_RUN.txt
REM   results/run_status/diverge_run_history.csv
REM   results/run_status/DIVERGE_LAST_COMPLETED.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
REM SPLIT=0 (semi-supervised) so --seed actually selects a different, seed-tagged
REM embedding cache per run (see cache/cache_embedding_with_diffrent_init_weights.py).
REM Under SPLIT=1 the embedding cache is seed-independent; only the GNN train/val/test
REM re-split would vary across seeds. Set SPLIT=1 below if that's what you want instead.
set SPLIT=1
REM Ensemble weight approaches (comma-separated, or "all"). GNN training
REM runs once per (dataset, llm, seed); every approach reuses those models.
set ENSEMBLE_APPROACHES=uniform,learnable,learnable_classes,learnable_per_classes
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
set DIVERGE_CSV=results\diverge\diverge_no_edge_editing_results.csv

REM Disable HF Trainer logging integrations (wandb/mlflow prompts block unattended runs)
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
REM Unicode-safe console/log output (ensemble code prints non-ASCII symbols)
set PYTHONIOENCODING=utf-8

REM Datasets
set DATASETS=cora

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=llama_3.2_1B

REM Seeds (must match the seeds used to train/cache the adapters being loaded)
set SEEDS=42

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\DIVERGE_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\diverge_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\DIVERGE_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\DIVERGE_FAILED_RUNS.txt

REM Create history CSV header if missing
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,peft_type,split,seed,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE pipeline multi-model / multi-seed sweep
echo Status dir: %STATUS_DIR%
echo Current run file: %CURRENT_FILE%
echo History CSV: %HISTORY_FILE%
echo Result CSV: %DIVERGE_CSV%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%s in (%SEEDS%) do (
            set "RUN_ID=dataset=%%d | llm=%%m | peft=%PEFT_TYPE% | split=%SPLIT% | seed=%%s | ensembles=%ENSEMBLE_APPROACHES%"
            set "TS=%DATE% %TIME%"

            echo ==========================================
            echo START: !RUN_ID!
            echo Time: !TS!
            echo ==========================================

            REM Write identity of the active run (overwrite)
            (
                echo status=RUNNING
                echo started=!TS!
                echo dataset=%%d
                echo llm_name=%%m
                echo peft_type=%PEFT_TYPE%
                echo split=%SPLIT%
                echo seed=%%s
                echo ensemble_approaches=%ENSEMBLE_APPROACHES%
                echo command=python -m experiments.diverge --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --split %SPLIT% --approaches %APPROACHES% --ensemble_approaches %ENSEMBLE_APPROACHES% --csv_path %DIVERGE_CSV%
                echo pid_note=see console / Task Manager for python process
            ) > "%CURRENT_FILE%"

            python -m experiments.diverge --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --seed %%s --split %SPLIT% --approaches "%APPROACHES%" --ensemble_approaches "%ENSEMBLE_APPROACHES%" --csv_path "%DIVERGE_CSV%"
            set EXIT_CODE=!ERRORLEVEL!
            set "TS_END=%DATE% %TIME%"

            if !EXIT_CODE! NEQ 0 (
                echo ERROR: !RUN_ID!  exit_code=!EXIT_CODE!
                echo !TS_END!,FAILED,%%d,%%m,%PEFT_TYPE%,%SPLIT%,%%s,!EXIT_CODE!>> "%HISTORY_FILE%"
                echo !TS_END! FAILED !RUN_ID! exit=!EXIT_CODE!>> "%FAILED_FILE%"
                (
                    echo status=FAILED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo peft_type=%PEFT_TYPE%
                    echo split=%SPLIT%
                    echo seed=%%s
                    echo exit_code=!EXIT_CODE!
                ) > "%CURRENT_FILE%"
                echo Continuing to next run despite failure...
                echo.
            ) else (
                echo OK: !RUN_ID!
                echo !TS_END!,OK,%%d,%%m,%PEFT_TYPE%,%SPLIT%,%%s,0>> "%HISTORY_FILE%"
                (
                    echo status=COMPLETED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo peft_type=%PEFT_TYPE%
                    echo split=%SPLIT%
                    echo seed=%%s
                    echo exit_code=0
                ) > "%LAST_OK_FILE%"
                copy /Y "%LAST_OK_FILE%" "%CURRENT_FILE%" >nul
                echo.
            )
        )
    )
)

echo ==========================================
echo DIVERGE sweep finished.
echo Check:
echo   %CURRENT_FILE%
echo   %HISTORY_FILE%
echo   %LAST_OK_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo   %DIVERGE_CSV%
echo   results\gnn_training\gnn_training_results.csv
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued DIVERGE runs finished; see diverge_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
