@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM TAPE baseline (base LLM, no adapter) over datasets x LLMs:  TAPE/run_tape_base.py
REM   1) base-model embeddings of the raw text and of the GPT-4o augmented text
REM      (artifacts/gpt_4o/<dataset>.json), built once and cached in artifacts/cache
REM   2) per seed and GNN: one GNN on each embedding, logits averaged (TAPE)
REM
REM Embeddings do not depend on the seed; seeds only change GNN init + split.
REM One python call per (dataset, llm) runs every GNN x seed, so the LLM is loaded once.
REM
REM This is the TAPE comparator of the significance tests (experiments/significance_vs_baselines.py):
REM GCN only, the paper datasets, seeds 42-45, plus the split each DIVERGE primary run uses
REM (EXTRA_RUNS: ArXiv adapters were fine-tuned on split 50).
REM PubMed is not run: there are no GPT-4o responses for it (artifacts/gpt_4o/pubmed.json).
REM
REM Outputs:
REM   %CSV%           one row per (dataset, llm, gnn, seed)
REM   %SUMMARY%       mean/std over seeds per (dataset, llm, gnn)
REM   %PREDS_DIR%     per-node predictions per (dataset, llm, gnn, seed), for paired tests
REM   results/run_status/tape_base_history.csv     OK/FAILED per call
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM ---------------------------- sweep grid -------------------------------------
set DATASETS=cora citeseer wikics arxiv
REM LLMs (configs/llm_configs/<name>.json + local_models/<name>)
set LLM_NAMES=llama_3.2_1B
REM GNN trained on each view: SAGE GCN GIN RevGAT
set GNN_MODELS=GCN
set SEEDS=42 43 44 45
REM Extra dataset:seed runs (split of the DIVERGE primary run when it is not in SEEDS)
set EXTRA_RUNS=arxiv:50
REM 1 = supervised split (DIVERGE default), 0 = semi-supervised
set SPLIT=1
REM Optional embedding batch size override (empty = llm config)
set BATCH_SIZE=

set CSV=results\tape\tape_base_results.csv
set SUMMARY=results\tape\tape_base_summary.csv
set PREDS_DIR=results\tape\preds

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8

set BATCH_FLAG=
if not "%BATCH_SIZE%"=="" set BATCH_FLAG=--batch_size %BATCH_SIZE%

REM ---------------------------- status tracking --------------------------------
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set HISTORY_FILE=%STATUS_DIR%\tape_base_history.csv
set FAILED_FILE=%STATUS_DIR%\TAPE_BASE_FAILED_RUNS.txt
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,gnn_models,seeds,split,exit_code> "%HISTORY_FILE%"
)
set /a N_OK=0
set /a N_FAIL=0

echo ==========================================
echo TAPE (base model): datasets=[%DATASETS%] llms=[%LLM_NAMES%]
echo                    gnns=[%GNN_MODELS%] seeds=[%SEEDS%] extra=[%EXTRA_RUNS%] split=%SPLIT%
echo Results: %CSV%
echo ==========================================

for %%d in (%DATASETS%) do for %%m in (%LLM_NAMES%) do call :run_one %%d %%m "%SEEDS%"
for %%r in (%EXTRA_RUNS%) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%r") do (
        for %%m in (%LLM_NAMES%) do call :run_one %%a %%m "%%b"
    )
)

echo.
echo ==========================================
echo TAPE sweep finished: ok=!N_OK! failed=!N_FAIL!
echo Check:
echo   %CSV%
echo   %SUMMARY%
echo   %PREDS_DIR%
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
pause
endlocal
exit /b 0

REM ---------------------------- one (dataset, llm, seeds) call -----------------
:run_one
set "RUN_DS=%~1"
set "RUN_LLM=%~2"
set "RUN_SEEDS=%~3"
echo.
echo ==========================================
echo START: dataset=%RUN_DS% ^| llm=%RUN_LLM% ^| seeds=%RUN_SEEDS%
echo Time: !DATE! !TIME!
echo ==========================================
python -m TAPE.run_tape_base --dataset_name %RUN_DS% --llm_name %RUN_LLM% --gnn_models %GNN_MODELS% --seeds %RUN_SEEDS% --split %SPLIT% --csv_path "%CSV%" --summary_path "%SUMMARY%" --preds_dir "%PREDS_DIR%" %BATCH_FLAG%
set EXIT_CODE=!ERRORLEVEL!
set "TS=!DATE! !TIME!"
if !EXIT_CODE! NEQ 0 (
    echo ERROR: dataset=%RUN_DS% llm=%RUN_LLM% exit_code=!EXIT_CODE!
    >> "%HISTORY_FILE%" echo !TS!,FAILED,%RUN_DS%,%RUN_LLM%,%GNN_MODELS%,%RUN_SEEDS%,%SPLIT%,!EXIT_CODE!
    >> "%FAILED_FILE%" echo !TS! FAILED dataset=%RUN_DS% llm=%RUN_LLM% seeds=%RUN_SEEDS% exit=!EXIT_CODE!
    set /a N_FAIL+=1
    echo Continuing to next run despite failure...
) else (
    echo OK: dataset=%RUN_DS% llm=%RUN_LLM%
    >> "%HISTORY_FILE%" echo !TS!,OK,%RUN_DS%,%RUN_LLM%,%GNN_MODELS%,%RUN_SEEDS%,%SPLIT%,0
    set /a N_OK+=1
)
exit /b 0
