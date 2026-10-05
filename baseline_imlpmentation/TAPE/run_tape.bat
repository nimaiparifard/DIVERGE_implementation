@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the TAPE pipeline (TAPE/run_tape.py) from Windows without needing to
REM remember the "python -m TAPE.run_tape" invocation or `cd` to the repo root.
REM
REM Usage:
REM   TAPE\run_tape.bat
REM       -> runs the default sweep: cora + pubmed, seeds 42 43 44 45
REM
REM   TAPE\run_tape.bat --dataset cora --seeds 42 43
REM       -> any arguments are forwarded as-is to run_tape.py (single dataset run)
REM
REM Logs are written to TAPE\results\run_logs\, JSON summaries to TAPE\results\.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONUNBUFFERED=1

set LOG_DIR=TAPE\results\run_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not "%~1"=="" (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\run_!TS!.log"

    echo ==========================================
    echo TAPE run  (args: %*)
    echo Log: !LOG_FILE!
    echo ==========================================

    python -m TAPE.run_tape %* > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo.
        echo ERROR: TAPE run failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo.
        echo OK: TAPE run finished. Log: !LOG_FILE!
    )

    goto :end
)

set DATASETS=cora pubmed
set SEEDS=42 43 44 45

echo ==========================================
echo TAPE default sweep
echo Datasets: %DATASETS%
echo Seeds:    %SEEDS%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\%%d_!TS!.log"

    echo ------------------------------------------
    echo START dataset=%%d
    echo Log: !LOG_FILE!
    echo ------------------------------------------

    python -m TAPE.run_tape --dataset %%d --seeds %SEEDS% > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo ERROR: dataset=%%d failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo OK: dataset=%%d finished. Results: TAPE\results\%%d_llama_3.2_1B.json
    )
    echo.
)

echo ==========================================
echo TAPE sweep finished. Summaries in TAPE\results\*.json
echo Full logs in %LOG_DIR%
echo ==========================================

:end
endlocal
pause
