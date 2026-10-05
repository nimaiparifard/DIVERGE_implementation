@echo off
REM Script to run ensemble edge editing analysis for multiple datasets

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Change to project root (parent of scripts/)
cd /d "%SCRIPT_DIR%.."

REM Default values
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora
set RETRAINED_WITH_GNN_MISTAKES=True
set REPORTER_INDEX=0
set ENSEMBLE_APPROACHES=learnable

REM List of datasets to process
set DATASETS=arxiv

REM Loop through each dataset
for %%d in (%DATASETS%) do (
    echo ==========================================
    echo Processing dataset: %%d
    echo ==========================================
    
    python -m experiments.edge_editing --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --retrained_with_gnn_mistakes "%RETRAINED_WITH_GNN_MISTAKES%" --reporter_index "%REPORTER_INDEX%" --ensemble_approaches "%ENSEMBLE_APPROACHES%"
    
    REM Check if the command was successful
    if errorlevel 1 (
        echo Error processing %%d
        exit /b 1
    )
    
    echo Successfully completed %%d
    echo.
)

echo ==========================================
echo All datasets processed successfully!
echo ==========================================
pause
