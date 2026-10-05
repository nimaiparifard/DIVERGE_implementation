@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the LLM-BP pipeline (LLM_BP/run_llm_bp.py) from Windows without needing to
REM remember the "python -m LLM_BP.run_llm_bp" invocation or `cd` to the repo root.
REM
REM Usage:
REM   LLM_BP\run_llm_bp.bat
REM       -> runs the default sweep: cora citeseer pubmed wikics, llama_3.2_1B, seeds 0 1 2 3
REM
REM   LLM_BP\run_llm_bp.bat --dataset cora --encoding vanilla --seeds 0 1 2 3
REM       -> any arguments are forwarded as-is to run_llm_bp.py (single run)
REM
REM Logs are written to LLM_BP\results\run_logs\, JSON summaries to LLM_BP\results\.
REM =============================================================================

REM run_llm_bp.bat lives in LLM_BP\, repo root is one level up
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Avoid interactive prompts / network calls that would block an unattended run
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1

set LOG_DIR=LLM_BP\results\run_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not "%~1"=="" (
    REM Arguments supplied: forward them verbatim to a single run_llm_bp.py call
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\run_!TS!.log"

    echo ==========================================
    echo LLM-BP run  (args: %*)
    echo Log: !LOG_FILE!
    echo ==========================================

    python -m LLM_BP.run_llm_bp %* > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo.
        echo ERROR: LLM-BP run failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo.
        echo OK: LLM-BP run finished. Log: !LOG_FILE!
    )

    goto :end
)

REM No arguments: default sweep matching LLM_BP\README.md
set LLM_NAME=llama_3.2_1B
set DATASETS=cora citeseer pubmed wikics
set SEEDS=0 1 2 3

echo ==========================================
echo LLM-BP default sweep
echo LLM:      %LLM_NAME%
echo Datasets: %DATASETS%
echo Seeds:    %SEEDS%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\%%d_!TS!.log"

    echo ------------------------------------------
    echo START dataset=%%d llm=%LLM_NAME%
    echo Log: !LOG_FILE!
    echo ------------------------------------------

    python -m LLM_BP.run_llm_bp --dataset %%d --llm_name %LLM_NAME% --seeds %SEEDS% > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo ERROR: dataset=%%d failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo OK: dataset=%%d finished. Results: LLM_BP\results\%%d_%LLM_NAME%_task_adaptive_last_repo_tuned.json
    )
    echo.
)

echo ==========================================
echo LLM-BP sweep finished. Summaries in LLM_BP\results\*.json
echo Full logs in %LOG_DIR%
echo ==========================================

:end
endlocal
pause
