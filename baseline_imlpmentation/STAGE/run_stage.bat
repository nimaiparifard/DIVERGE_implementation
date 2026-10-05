@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the STAGE pipeline (STAGE/run_stage.py) from Windows without needing to
REM remember the "python -m STAGE.run_stage" invocation or `cd` to the repo root.
REM
REM Usage:
REM   STAGE\run_stage.bat
REM       -> runs the default sweep: cora + pubmed, llama_3.2_1B, seeds 0 1 2 3
REM
REM   STAGE\run_stage.bat --dataset cora --llm_name qwen2.5_1.5B --seeds 0 1 2 3
REM       -> any arguments are forwarded as-is to run_stage.py (single run)
REM
REM Logs are written to STAGE\results\run_logs\, JSON summaries to STAGE\results\.
REM =============================================================================

REM run_stage.bat lives in STAGE\, repo root is one level up
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Avoid interactive prompts / network calls that would block an unattended run
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1

set LOG_DIR=STAGE\results\run_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not "%~1"=="" (
    REM Arguments supplied: forward them verbatim to a single run_stage.py call
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\run_!TS!.log"

    echo ==========================================
    echo STAGE run  (args: %*)
    echo Log: !LOG_FILE!
    echo ==========================================

    python -m STAGE.run_stage %* > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo.
        echo ERROR: STAGE run failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo.
        echo OK: STAGE run finished. Log: !LOG_FILE!
    )

    goto :end
)

REM No arguments: default sweep matching STAGE\README.md ("Results so far" table)
set LLM_NAME=llama_3.2_1B
set DATASETS=cora pubmed
set SEEDS=0 1 2 3

echo ==========================================
echo STAGE default sweep
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

    python -m STAGE.run_stage --dataset %%d --llm_name %LLM_NAME% --seeds %SEEDS% > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo ERROR: dataset=%%d failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo OK: dataset=%%d finished. Results: STAGE\results\%%d_%LLM_NAME%_no_instruction.json
    )
    echo.
)

echo ==========================================
echo STAGE sweep finished. Summaries in STAGE\results\*.json
echo Full logs in %LOG_DIR%
echo ==========================================

:end
endlocal
pause
