@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the CKA embedding-diversity analysis (experiments/cka.py) across
REM datasets x LLMs x seeds. Each run computes the pairwise linear-CKA matrix
REM between the cached embeddings of every LoRA-init approach for one
REM (dataset, llm, seed). Requires that, for every combination, you already ran:
REM   1) scripts/run_all_datasets_lora_init.bat       (trains the LoRA adapters)
REM   2) scripts/run_cache_embedding_all_datasets.bat (caches their embeddings)
REM
REM Outputs:
REM   results/cka/cka_pairwise_results.csv   (one row per ordered approach pair)
REM   results/cka/cka_summary_results.csv    (one row per run: mean off-diag CKA, ...)
REM   results/cka/<dataset>_<llm>_<peft>_seed<seed>_cka_heatmap.png
REM   results/train_info/<dataset>_cka_seed<seed>/  (per-run text report)
REM
REM Progress / identity of the active run is also written to:
REM   results/run_status/CKA_CURRENT_RUN.txt
REM   results/run_status/cka_run_history.csv
REM   results/run_status/CKA_LAST_COMPLETED.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
REM SPLIT=0 (semi-supervised) matches run_cache_embedding_all_datasets.bat, where
REM --seed selects a seed-tagged embedding cache. Under SPLIT=1 the cache is
REM seed-independent, so every seed would give the same CKA matrix.
set SPLIT=0
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
set CKA_PAIRWISE_CSV=results\cka\cka_pairwise_results.csv
set CKA_SUMMARY_CSV=results\cka\cka_summary_results.csv

set PYTHONIOENCODING=utf-8

REM Datasets
set DATASETS=cora

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=llama_3.2_1B smollm2_1.7B gemma2_2B

REM Seeds (must match the seeds used to train/cache the adapters being loaded)
set SEEDS=42 43 44 45

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\CKA_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\cka_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\CKA_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\CKA_FAILED_RUNS.txt

REM Create history CSV header if missing
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,peft_type,split,seed,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE CKA multi-model sweep
echo Status dir: %STATUS_DIR%
echo Current run file: %CURRENT_FILE%
echo History CSV: %HISTORY_FILE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%s in (%SEEDS%) do (
            set "RUN_ID=dataset=%%d | llm=%%m | peft=%PEFT_TYPE% | split=%SPLIT% | seed=%%s"
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
                echo command=python -m experiments.cka --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --split %SPLIT% --approaches %APPROACHES% --csv_path %CKA_PAIRWISE_CSV% --summary_csv_path %CKA_SUMMARY_CSV%
            ) > "%CURRENT_FILE%"

            python -m experiments.cka --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --seed %%s --split %SPLIT% --approaches "%APPROACHES%" --csv_path "%CKA_PAIRWISE_CSV%" --summary_csv_path "%CKA_SUMMARY_CSV%"
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
echo CKA sweep finished.
echo Check:
echo   %CKA_PAIRWISE_CSV%
echo   %CKA_SUMMARY_CSV%
echo   results\cka\*_cka_heatmap.png
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued CKA runs finished; see cka_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
