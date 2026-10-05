@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run E-LLaGNN (ELaGNN/run_elagnn.py) from Windows.
REM
REM Usage:
REM   ELaGNN\run_elagnn.bat
REM       -> default sweep on cora: budgets 0 / 0.05 / 0.10 / 0.25 / 0.50 x layers 2 / 4 / 8
REM          (paper Table 5), pagerank selection, seeds 42-46
REM
REM   ELaGNN\run_elagnn.bat --dataset pubmed --budget 0.25 --heuristic text_length
REM       -> any arguments are forwarded as-is to run_elagnn.py (single run)
REM
REM LLM enhancements are cached in ELaGNN\cache\, so each (node, prompt) pair is
REM generated only once across the whole sweep.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set WANDB_DISABLED=true
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1

set LOG_DIR=ELaGNN\results\run_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not "%~1"=="" (
    set "TS=%DATE:/=-%_%TIME::=-%"
    set "TS=!TS: =0!"
    set "LOG_FILE=%LOG_DIR%\run_!TS!.log"
    echo E-LLaGNN run  (args: %*)  Log: !LOG_FILE!
    python -m ELaGNN.run_elagnn %* > "!LOG_FILE!" 2>&1
    set EXIT_CODE=!ERRORLEVEL!
    type "!LOG_FILE!"
    if !EXIT_CODE! NEQ 0 (echo ERROR: run failed with exit code !EXIT_CODE!.) else (echo OK: run finished.)
    goto :end
)

set DATASET=cora
set SEEDS=42 43 44 45 46

REM Largest budget first: its enhancements cover every smaller pagerank budget
for %%b in (0.5 0.25 0.1 0.05 0) do (
    for %%l in (2 4 8) do (
        set "LOG_FILE=%LOG_DIR%\%DATASET%_b%%b_L%%l.log"
        echo ------------------------------------------
        echo START dataset=%DATASET% budget=%%b layers=%%l
        python -m ELaGNN.run_elagnn --dataset %DATASET% --budget %%b --n_layers %%l --seeds %SEEDS% > "!LOG_FILE!" 2>&1
        if !ERRORLEVEL! NEQ 0 (echo ERROR: see !LOG_FILE!) else (findstr /C:"Test acc" "!LOG_FILE!")
    )
)

:end
endlocal
pause
