@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Cache LLM embeddings across datasets x LLMs x LoRA-init approaches x seeds.
REM Run this AFTER scripts/run_all_datasets_lora_init.bat has trained the
REM matching LoRA adapters for the same (dataset, llm, init, seed) combinations.
REM
REM SPLIT=0 (semi-supervised) is the default because the seed only changes the
REM cached embedding filename in that branch (see
REM cache/cache_embedding_with_diffrent_init_weights.py) - the SPLIT=1
REM (supervised) cache is seed-independent, so sweeping seeds there just
REM re-writes the same file. Set SPLIT=1 below if you need the supervised cache.
REM
REM Progress / identity of the active run is written to:
REM   results/run_status/CACHE_CURRENT_RUN.txt   (what is running now)
REM   results/run_status/cache_run_history.csv   (append log of every run)
REM   results/run_status/CACHE_LAST_COMPLETED.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
set SPLIT=0
set POOLING=mean

REM Disable HF Trainer logging integrations (wandb/mlflow prompts block unattended runs)
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1

REM Datasets
set DATASETS=arxiv

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=llama_3.2_1B 

REM LoRA weight initialization approaches
set INIT_APPROACHES=pissa gaussian eva loftq orthogonal

REM Seeds to cache embeddings for (must match seeds used when training the adapters)
set SEEDS=42

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\CACHE_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\cache_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\CACHE_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\CACHE_FAILED_RUNS.txt

REM Create history CSV header if missing
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,peft_type,lora_init,split,seed,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE embedding-cache multi-model sweep
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
                    echo command=python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --init_weight_approach %%i --split %SPLIT% --pooling %POOLING% --seed %%s
                    echo pid_note=see console / Task Manager for python process
                ) > "%CURRENT_FILE%"

                python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i" --split %SPLIT% --pooling %POOLING% --seed %%s
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
echo Cache sweep finished.
echo Check:
echo   %CURRENT_FILE%
echo   %HISTORY_FILE%
echo   %LAST_OK_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo   artifacts\cache\
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued cache runs finished; see cache_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
