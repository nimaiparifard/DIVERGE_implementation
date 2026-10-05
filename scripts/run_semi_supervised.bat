@echo off
REM Script to run training with different LoRA initialization approaches across multiple datasets

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Change to project root (parent of scripts/)
cd /d "%SCRIPT_DIR%.."

REM Default values
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora

REM List of datasets to process
set DATASETS=citeseer

REM List of initialization approaches to test
set INIT_APPROACHES=gaussian eva loftq orthogonal pissa

REM Outer loop: iterate through each dataset
for %%d in (%DATASETS%) do (
    echo ==========================================
    echo Processing dataset: %%d
    echo ==========================================

    REM Inner loop: iterate through each initialization approach
    for %%i in (%INIT_APPROACHES%) do (
        echo ------------------------------------------
        echo Dataset: %%d - Init Approach: %%i
        echo ------------------------------------------

@REM         REM Step 1: Train the model
@REM         echo [Step 1/2] Training model...
@REM         python -m train_llm.train_other_lora_init_apporaches --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i" --split 0
@REM
@REM         REM Check if the training command was successful
@REM         if errorlevel 1 (
@REM             echo Error training dataset %%d with init approach %%i
@REM             exit /b 1
@REM         )
@REM
@REM         echo [Step 2/2] Caching embeddings...
@REM         REM Step 2: Cache embeddings for semi-supervised model
@REM         python -m cache.cache_embedding_with_diffrent_init_weights --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i" --split 0 --pooling mean
@REM
@REM         REM Check if the cache command was successful
@REM         if errorlevel 1 (
@REM             echo Error caching embeddings for dataset %%d with init approach %%i
@REM             exit /b 1
@REM         )
@REM
@REM         echo Successfully completed training and caching for %%d with %%i
@REM         echo.
@REM     )
@REM
@REM     echo Successfully completed all approaches for %%d
@REM     echo.

    REM Step 3: Tune GNN hyperparameters for this dataset
    echo ------------------------------------------
    echo [Step 3] Tuning GNN hyperparameters for dataset: %%d
    echo ------------------------------------------
    python -m gnns.tune_gnn_hyperparameter --dataset_name "%%d" --re_split 0

    REM Check if the hyperparameter tuning command was successful
    if errorlevel 1 (
        echo Error tuning hyperparameters for dataset %%d
        exit /b 1
    )
    
@REM     REM Step 4: Run semi-supervised training
@REM     echo ------------------------------------------
@REM     echo [Step 4] Running semi-supervised training for dataset: %%d
@REM     echo ------------------------------------------
@REM     python -m experiments.semi_supervised_running --dataset_name "%%d"
@REM
@REM     REM Check if the semi-supervised training command was successful
@REM     if errorlevel 1 (
@REM         echo Error running semi-supervised training for dataset %%d
@REM
@REM     echo Successfully completed hyperparameter tuning for %%d
@REM     echo.
)

echo ==========================================
echo All datasets and initialization approaches processed successfully!
echo Training, cache generation, and hyperparameter tuning completed for all combinations.
echo ==========================================
pause
