@echo off
REM Script to run tune_edge_edditing_hyperparamters.py for different datasets
REM This script runs hyperparameter tuning for edge editing

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Change to project root (3 levels up from structure_hybrid_enahncer)
cd /d "%SCRIPT_DIR%..\..\.."

REM Default values
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora

REM List of datasets to process (modify as needed)
set DATASETS=cora citeseer pubmed

echo ==========================================
echo TUNE EDGE EDITING HYPERPARAMETERS
echo ==========================================
echo Datasets: %DATASETS%
echo LLM: %LLM_NAME%
echo PEFT Type: %PEFT_TYPE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    echo ------------------------------------------
    echo Processing dataset: %%d
    echo ------------------------------------------
    python -m LLMReasoner.TAPE.structure_hybrid_enahncer.tune_edge_edditing_hyperparamters --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
    if errorlevel 1 (
        echo ERROR: tune_edge_edditing_hyperparamters failed for %%d
    ) else (
        echo SUCCESS: tune_edge_edditing_hyperparamters completed for %%d
    )
    echo.
)

echo ==========================================
echo Hyperparameter tuning completed!
echo ==========================================
pause
