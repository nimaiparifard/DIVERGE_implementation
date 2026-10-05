@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run LoRA-init fine-tuning across datasets x LLMs x init approaches.
REM Progress / identity of the active run is written to:
REM   results/run_status/CURRENT_RUN.txt   (what is running now)
REM   results/run_status/run_history.csv   (append log of every run)
REM   results/run_status/LAST_COMPLETED.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
set SPLIT=1

REM Disable HF Trainer logging integrations (wandb/mlflow prompts block unattended runs)
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1

REM Datasets
set DATASETS=pubmed 

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=gemma2_2B

REM LoRA weight initialization approaches
set INIT_APPROACHES=orthogonal

REM Random seeds to sweep (each seed gets its own adapter dir / results)
set SEEDS=42 

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\FAILED_RUNS.txt

REM Create history CSV header if missing
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,peft_type,lora_init,split,seed,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE LoRA-init multi-model sweep
echo Status dir: %STATUS_DIR%
echo Current run file: %CURRENT_FILE%
echo History CSV: %HISTORY_FILE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%i in (%INIT_APPROACHES%) do (
            for %%s in (%SEEDS%) do (
                set "RUN_ID=dataset=%%d | llm=%%m | peft=%PEFT_TYPE% | init=%%i | split=%SPLIT% | seed=%%s"
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
                    echo lora_init=%%i
                    echo split=%SPLIT%
                    echo seed=%%s
                    echo command=python -m train_llm.train_other_lora_init_apporaches --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --init_weight_approach %%i --split %SPLIT% --seed %%s
                    echo pid_note=see console / Task Manager for python process
                ) > "%CURRENT_FILE%"

                python -m train_llm.train_other_lora_init_apporaches --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i" --split %SPLIT% --seed %%s
                set EXIT_CODE=!ERRORLEVEL!
                set "TS_END=%DATE% %TIME%"

                if !EXIT_CODE! NEQ 0 (
                    echo ERROR: !RUN_ID!  exit_code=!EXIT_CODE!
                    echo !TS_END!,FAILED,%%d,%%m,%PEFT_TYPE%,%%i,%SPLIT%,%%s,!EXIT_CODE!>> "%HISTORY_FILE%"
                    echo !TS_END! FAILED !RUN_ID! exit=!EXIT_CODE!>> "%FAILED_FILE%"
                    (
                        echo status=FAILED
                        echo started=!TS!
                        echo finished=!TS_END!
                        echo dataset=%%d
                        echo llm_name=%%m
                        echo peft_type=%PEFT_TYPE%
                        echo lora_init=%%i
                        echo split=%SPLIT%
                        echo seed=%%s
                        echo exit_code=!EXIT_CODE!
                    ) > "%CURRENT_FILE%"
                    echo Continuing to next run despite failure...
                    echo.
                ) else (
                    echo OK: !RUN_ID!
                    echo !TS_END!,OK,%%d,%%m,%PEFT_TYPE%,%%i,%SPLIT%,%%s,0>> "%HISTORY_FILE%"
                    (
                        echo status=COMPLETED
                        echo started=!TS!
                        echo finished=!TS_END!
                        echo dataset=%%d
                        echo llm_name=%%m
                        echo peft_type=%PEFT_TYPE%
                        echo lora_init=%%i
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
)

echo ==========================================
echo Sweep finished.
echo Check:
echo   %CURRENT_FILE%
echo   %HISTORY_FILE%
echo   %LAST_OK_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo   results\efficiency\lora_finetune_efficiency.csv
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued runs finished; see run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
