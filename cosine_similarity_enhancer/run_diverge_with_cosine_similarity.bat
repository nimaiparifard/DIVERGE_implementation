@echo off
REM Run DIVERGE with cosine similarity graph enhancement for multiple datasets

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora
set REPORTER_INDEX=0
set ENSEMBLE_APPROACHES=learnable

set DATASETS=cora pubmed citeseer wikics

for %%d in (%DATASETS%) do (
    echo ==========================================
    echo Processing dataset: %%d
    echo ==========================================

    python "%SCRIPT_DIR%diverge_with_cosine_similarity_enahncer.py" ^
        --dataset_name "%%d" ^
        --llm_name "%LLM_NAME%" ^
        --peft_type "%PEFT_TYPE%" ^
        --retrained_with_gnn_mistakes ^
        --reporter_index "%REPORTER_INDEX%" ^
        --ensemble_approaches "%ENSEMBLE_APPROACHES%" ^
        --use_all_nodes

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
