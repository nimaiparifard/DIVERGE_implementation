@echo off
REM Script to run ensemble scripts for different datasets and init_weight_approaches
REM This script runs:
REM   - ensemble_edge_predictor_std.py
REM   - ensemble_edge_predictor_entropy.py
REM   - ensemble_all_std.py
REM   - ensemble_all_entropy.py
REM   - ensemble_all_hybrid.py

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Change to project root (3 levels up from structure_hybrid_enahncer)
cd /d "%SCRIPT_DIR%..\..\.."

REM Default values
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora

REM List of datasets to process (modify as needed)
REM Small-scale: cora citeseer wikics instagram
REM Middle: pubmed reddit
REM Large: photo history computer
REM Single: arxiv
set DATASETS=cora citeseer pubmed

REM List of initialization approaches to test
set INIT_APPROACHES=pissa orthogonal guassian loftq eva

echo ==========================================
echo ENSEMBLE EDGE PREDICTOR EXPERIMENTS
echo ==========================================
echo Datasets: %DATASETS%
echo Init Approaches: %INIT_APPROACHES%
echo LLM: %LLM_NAME%
echo PEFT Type: %PEFT_TYPE%
echo ==========================================
echo.

REM 1. ensemble_edge_predictor_std.py
echo ------------------------------------------
echo [1/5] Running ensemble_edge_predictor_std.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_edge_predictor_std --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: ensemble_edge_predictor_std failed for %%d with %%i
        ) else (
            echo   SUCCESS: ensemble_edge_predictor_std completed for %%d with %%i
        )
        echo.
    )
)

REM 2. ensemble_edge_predictor_entropy.py
echo ------------------------------------------
echo [2/5] Running ensemble_edge_predictor_entropy.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_edge_predictor_entropy --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: ensemble_edge_predictor_entropy failed for %%d with %%i
        ) else (
            echo   SUCCESS: ensemble_edge_predictor_entropy completed for %%d with %%i
        )
        echo.
    )
)

REM 3. ensemble_all_std.py (runs for all embeddings automatically)
echo ------------------------------------------
echo [3/5] Running ensemble_all_std.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    echo   Dataset: %%d
    python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_all_std --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
    if errorlevel 1 (
        echo   ERROR: ensemble_all_std failed for %%d
    ) else (
        echo   SUCCESS: ensemble_all_std completed for %%d
    )
    echo.
)

REM 4. ensemble_all_entropy.py (runs for all embeddings automatically)
echo ------------------------------------------
echo [4/5] Running ensemble_all_entropy.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    echo   Dataset: %%d
    python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_all_entropy --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
    if errorlevel 1 (
        echo   ERROR: ensemble_all_entropy failed for %%d
    ) else (
        echo   SUCCESS: ensemble_all_entropy completed for %%d
    )
    echo.
)

REM 5. ensemble_all_hybrid.py (runs for all embeddings automatically)
echo ------------------------------------------
echo [5/5] Running ensemble_all_hybrid.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    echo   Dataset: %%d
    python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_all_hybrid --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
    if errorlevel 1 (
        echo   ERROR: ensemble_all_hybrid failed for %%d
    ) else (
        echo   SUCCESS: ensemble_all_hybrid completed for %%d
    )
    echo.
)

echo ==========================================
echo All ensemble experiments completed!
echo ==========================================
pause
