@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the UltraTAG-S pipeline (ULTRATAG/run_ultratag.py) from Windows without
REM needing to remember the "python -m ULTRATAG.run_ultratag" invocation or `cd`
REM to the repo root.
REM
REM Usage:
REM   ULTRATAG\run_ultratag.bat
REM       -> runs the default sweep: cora + pubmed, ratios 0.0 0.8, seeds 42 43 44
REM
REM   ULTRATAG\run_ultratag.bat --dataset cora --ratios 0.0 0.8 --seeds 42 43
REM       -> any arguments are forwarded as-is to run_ultratag.py (single dataset run)
REM
REM Logs are written to ULTRATAG\results\run_logs\, JSON summaries to ULTRATAG\results\.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONUNBUFFERED=1

set LOG_DIR=ULTRATAG\results\run_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not "%~1"=="" (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\run_!TS!.log"

    echo ==========================================
    echo UltraTAG-S run  (args: %*)
    echo Log: !LOG_FILE!
    echo ==========================================

    python -m ULTRATAG.run_ultratag %* > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo.
        echo ERROR: UltraTAG-S run failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo.
        echo OK: UltraTAG-S run finished. Log: !LOG_FILE!
    )

    goto :end
)

set DATASETS=cora pubmed
set RATIOS=0.0 0.8
set SEEDS=42 43 44

echo ==========================================
echo UltraTAG-S default sweep
echo Datasets: %DATASETS%
echo Ratios:   %RATIOS%
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

    python -m ULTRATAG.run_ultratag --dataset %%d --ratios %RATIOS% --seeds %SEEDS% > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"

    if !EXIT_CODE! NEQ 0 (
        echo ERROR: dataset=%%d failed with exit code !EXIT_CODE!. See !LOG_FILE!
    ) else (
        echo OK: dataset=%%d finished. Results: ULTRATAG\results\%%d_llama_3.2_1B.json
    )
    echo.
)

echo ==========================================
echo UltraTAG-S sweep finished. Summaries in ULTRATAG\results\*.json
echo Full logs in %LOG_DIR%
echo ==========================================

:end
endlocal
pause
