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
set DATASETS=arxiv

REM List of initialization approaches to test
set INIT_APPROACHES=orthogonal

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
        
        python -m train_llm.train_other_lora_init_apporaches --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        
        REM Check if the command was successful
        if errorlevel 1 (
            echo Error processing dataset %%d with init approach %%i
            exit /b 1
        )
        
        echo Successfully completed %%d with %%i
        echo.
    )
    
    echo Successfully completed all approaches for %%d
    echo.
)

echo ==========================================
echo All datasets and initialization approaches processed successfully!
echo ==========================================
pause
