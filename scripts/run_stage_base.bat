@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM STAGE baseline (base LLM, no adapter) over datasets x LLMs:  STAGE/run_stage_base.py
REM   1) base-model embeddings of the raw text - the same artifacts/cache file
REM      TAPE (scripts/run_tape_base.bat) builds; built here if missing
REM   2) per seed: SAGE, RevGAT, GCN, GIN on them, logits averaged (STAGE)
REM
REM Outputs:
REM   %CSV%           one row per (dataset, llm, seed), every GNN + the ensemble
REM   %SUMMARY%       mean/std over seeds per (dataset, llm)
REM   results/run_status/stage_base_history.csv    OK/FAILED per call
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM ---------------------------- sweep grid -------------------------------------
set DATASETS=arxiv
REM LLMs (configs/llm_configs/<name>.json + local_models/<name>)
set LLM_NAMES=llama_3.2_1B
REM Ensemble members
set GNN_MODELS=SAGE RevGAT GCN GIN
set SEEDS=42 43 44 45
REM 1 = supervised split (DIVERGE default), 0 = semi-supervised
set SPLIT=1
REM Optional embedding batch size override (empty = llm config)
set BATCH_SIZE=

set CSV=results\stage\stage_base_results.csv
set SUMMARY=results\stage\stage_base_summary.csv

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8

set BATCH_FLAG=
if not "%BATCH_SIZE%"=="" set BATCH_FLAG=--batch_size %BATCH_SIZE%

REM ---------------------------- status tracking --------------------------------
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set HISTORY_FILE=%STATUS_DIR%\stage_base_history.csv
set FAILED_FILE=%STATUS_DIR%\STAGE_BASE_FAILED_RUNS.txt
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,gnn_models,seeds,split,exit_code> "%HISTORY_FILE%"
)
set /a N_OK=0
set /a N_FAIL=0

echo ==========================================
echo STAGE (base model): datasets=[%DATASETS%] llms=[%LLM_NAMES%]
echo                     gnns=[%GNN_MODELS%] seeds=[%SEEDS%] split=%SPLIT%
echo Results: %CSV%
echo ==========================================

for %%d in (%DATASETS%) do for %%m in (%LLM_NAMES%) do (
    echo.
    echo ==========================================
    echo START: dataset=%%d ^| llm=%%m
    echo Time: !DATE! !TIME!
    echo ==========================================
    python -m STAGE.run_stage_base --dataset_name %%d --llm_name %%m --gnn_models %GNN_MODELS% --seeds %SEEDS% --split %SPLIT% --csv_path "%CSV%" --summary_path "%SUMMARY%" %BATCH_FLAG%
    set EXIT_CODE=!ERRORLEVEL!
    set "TS=!DATE! !TIME!"
    if !EXIT_CODE! NEQ 0 (
        echo ERROR: dataset=%%d llm=%%m exit_code=!EXIT_CODE!
        >> "%HISTORY_FILE%" echo !TS!,FAILED,%%d,%%m,%GNN_MODELS%,%SEEDS%,%SPLIT%,!EXIT_CODE!
        >> "%FAILED_FILE%" echo !TS! FAILED dataset=%%d llm=%%m exit=!EXIT_CODE!
        set /a N_FAIL+=1
        echo Continuing to next run despite failure...
    ) else (
        echo OK: dataset=%%d llm=%%m
        >> "%HISTORY_FILE%" echo !TS!,OK,%%d,%%m,%GNN_MODELS%,%SEEDS%,%SPLIT%,0
        set /a N_OK+=1
    )
)

echo.
echo ==========================================
echo STAGE sweep finished: ok=!N_OK! failed=!N_FAIL!
echo Check:
echo   %CSV%
echo   %SUMMARY%
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
pause
endlocal
