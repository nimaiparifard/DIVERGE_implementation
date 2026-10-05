@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM DIVERGE tags-data-augmentation pipeline (see tags_data_augmentation\approach.txt)
REM
REM   1) create_new_data.py                              - web-search / LLM-generate new
REM                                                          (pseudo-labeled) nodes for each dataset
REM   2) train_with_augmented_nodes.py                    - fine-tune an LLM adapter jointly on
REM                                                          base graph nodes + augmented nodes
REM   3) cache_base_and_augmented_nodes.py                - cache base+augmented embeddings with
REM                                                          that SAME fine-tuned adapter
REM   4) connect_new_nodes_to_graphs_with_augmented_cache.py
REM                                                        - predict edges, build the augmented graph,
REM                                                          train GNNs, run the DIVERGE ensemble
REM   5) tune_hyperparamters_augmented_cache.py           - (optional, slow) sweep edge-construction
REM                                                          hyperparameters
REM
REM A legacy two-cache pipeline (extract_and_cache_new_nodes_embedding.py +
REM connect_new_nodes_to_graphs.py + tune_hyperparamters.py, which source base-node
REM and augmented-node embeddings from two SEPARATELY fine-tuned models instead of
REM one combined cache) is also wired up below, disabled by default.
REM
REM Per-step status is appended to results\run_status\tags_augmentation_history.csv
REM so a failed step doesn't stop the whole sweep.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Disable HF Trainer logging integrations (wandb/mlflow prompts block unattended runs)
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1

REM Required by Step 1 (create_new_data.py) when using --llm_type openai.
REM Uncomment and fill in, or set it once in your user/system environment instead.
REM set OPENAI_API_KEY=your_key_here

REM ============================== CONFIG ======================================
REM All datasets available under datasets\: arxiv citeseer computer cora history
REM instagram photo pubmed reddit wikics. Only citeseer/cora/pubmed currently have
REM augmented_data\*.json and labels\*.json prepared - add others as you build them.
set DATASETS=citeseer cora pubmed

set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora
set POOLING=mean
set INIT_APPROACHES=pissa orthogonal eva loftq gaussian

set WEB_SEARCH_RATIO=0.9
set TARGET_PAPERS_PER_LABEL=1500

REM Step toggles (true/false)
set RUN_STEP1_GENERATE_DATA=true
set RUN_STEP2_FINETUNE=true
set RUN_STEP3_CACHE=true
set RUN_STEP4_CONNECT_ENSEMBLE=true
set RUN_STEP5_TUNE_HYPERPARAMS=false
set RUN_LEGACY_PIPELINE=false
REM =============================================================================

set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set HISTORY_FILE=%STATUS_DIR%\tags_augmentation_history.csv
if not exist "%HISTORY_FILE%" (
    echo timestamp,step,dataset,status,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE tags-data-augmentation pipeline
echo Datasets: %DATASETS%
echo Init approaches: %INIT_APPROACHES%
echo History CSV: %HISTORY_FILE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (

    echo ==========================================
    echo DATASET: %%d
    echo ==========================================

    if /I "%RUN_STEP1_GENERATE_DATA%"=="true" (
        echo [1/5] create_new_data.py - dataset=%%d
        python -m tags_data_augmentation.create_new_data --dataset_name "%%d" --web_search_ratio %WEB_SEARCH_RATIO% --target_papers_per_label %TARGET_PAPERS_PER_LABEL%
        call :log_result step1_generate_data %%d !ERRORLEVEL!
    )

    if /I "%RUN_STEP2_FINETUNE%"=="true" (
        echo [2/5] train_with_augmented_nodes.py - dataset=%%d
        python -m tags_data_augmentation.train_with_augmented_nodes --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approaches %INIT_APPROACHES%
        call :log_result step2_finetune %%d !ERRORLEVEL!
    )

    if /I "%RUN_STEP3_CACHE%"=="true" (
        echo [3/5] cache_base_and_augmented_nodes.py - dataset=%%d
        python -m tags_data_augmentation.cache_base_and_augmented_nodes --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --pooling "%POOLING%" --init_weight_approaches %INIT_APPROACHES%
        call :log_result step3_cache %%d !ERRORLEVEL!
    )

    if /I "%RUN_STEP4_CONNECT_ENSEMBLE%"=="true" (
        echo [4/5] connect_new_nodes_to_graphs_with_augmented_cache.py - dataset=%%d
        python -m tags_data_augmentation.connect_new_nodes_to_graphs_with_augmented_cache --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --pooling "%POOLING%" --init_weight_approach all
        call :log_result step4_connect_ensemble %%d !ERRORLEVEL!
    )

    if /I "%RUN_STEP5_TUNE_HYPERPARAMS%"=="true" (
        echo [5/5] tune_hyperparamters_augmented_cache.py - dataset=%%d
        python -m tags_data_augmentation.tune_hyperparamters_augmented_cache --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --pooling "%POOLING%" --init_weight_approaches %INIT_APPROACHES%
        call :log_result step5_tune_hyperparams %%d !ERRORLEVEL!
    )

    if /I "%RUN_LEGACY_PIPELINE%"=="true" (
        echo [legacy 1/3] extract_and_cache_new_nodes_embedding.py - dataset=%%d
        python -m tags_data_augmentation.extract_and_cache_new_nodes_embedding --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --pooling "%POOLING%" --init_weight_approaches %INIT_APPROACHES%
        call :log_result legacy_extract_cache %%d !ERRORLEVEL!

        echo [legacy 2/3] connect_new_nodes_to_graphs.py - dataset=%%d
        python -m tags_data_augmentation.connect_new_nodes_to_graphs --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --pooling "%POOLING%" --init_weight_approach all
        call :log_result legacy_connect_ensemble %%d !ERRORLEVEL!

        echo [legacy 3/3] tune_hyperparamters.py - dataset=%%d
        python -m tags_data_augmentation.tune_hyperparamters --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approaches %INIT_APPROACHES%
        call :log_result legacy_tune_hyperparams %%d !ERRORLEVEL!
    )

    echo.
)

echo ==========================================
echo Pipeline finished. Check %HISTORY_FILE% for per-step status.
echo ==========================================
pause
endlocal
goto :eof

:log_result
set "STEP_NAME=%~1"
set "DS=%~2"
set "CODE=%~3"
set "TS=%DATE% %TIME%"
if "%CODE%"=="0" (
    echo   OK: %STEP_NAME% ^(%DS%^)
    echo %TS%,%STEP_NAME%,%DS%,OK,0>> "%HISTORY_FILE%"
) else (
    echo   ERROR: %STEP_NAME% ^(%DS%^) exit_code=%CODE%
    echo %TS%,%STEP_NAME%,%DS%,FAILED,%CODE%>> "%HISTORY_FILE%"
)
exit /b 0
