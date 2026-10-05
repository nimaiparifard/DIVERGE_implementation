@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the SKETCH pipeline (SKETCH/run_sketch.py) from Windows.
REM
REM Usage:
REM   SKETCH\run_sketch.bat
REM       -> default sweep: cora + pubmed, llama_3.2_1B, strategy=sketch, seeds 42 43 44 45
REM
REM   SKETCH\run_sketch.bat --dataset cora --strategy original --seeds 42
REM       -> any arguments are forwarded as-is to run_sketch.py (single run)
REM
REM Logs are written to SKETCH\results\run_logs\, JSON summaries to SKETCH\results\.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1

set LOG_DIR=SKETCH\results\run_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not "%~1"=="" (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\run_!TS!.log"

    echo ==========================================
    echo SKETCH run  (args: %*)
    echo Log: !LOG_FILE!
    echo ==========================================

    python -m SKETCH.run_sketch %* > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo ERROR: SKETCH run failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo OK: SKETCH run finished. Log: !LOG_FILE!
    )
    goto :end
)

set LLM_NAME=llama_3.2_1B
set DATASETS=cora pubmed
set SEEDS=42 43 44 45

echo ==========================================
echo SKETCH default sweep
echo LLM:      %LLM_NAME%
echo Datasets: %DATASETS%
echo Seeds:    %SEEDS%
echo ==========================================

for %%d in (%DATASETS%) do (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\%%d_!TS!.log"

    echo ------------------------------------------
    echo START dataset=%%d llm=%LLM_NAME%
    echo Log: !LOG_FILE!
    echo ------------------------------------------

    python -m SKETCH.run_sketch --dataset %%d --llm_name %LLM_NAME% --seeds %SEEDS% > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo ERROR: dataset=%%d failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo OK: dataset=%%d finished.
    )
    echo.
)

:end
endlocal
pause
