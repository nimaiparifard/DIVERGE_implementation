@echo off
REM Script to run random edge denoising baseline for different datasets and init_weight_approaches
REM Runs: LLMReasoner.TAPE.poor_TAGs_enhacner_apporeches.random_model_edge_denoising
REM (Randomly removes a percentage of edges, no edge predictor; compares GNN node classification.)

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Change to project root (3 levels up from structure_hybrid_enahncer)
cd /d "%SCRIPT_DIR%..\..\.."

REM Default values
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora
set REMOVE_PERCENTAGE=20
set SEED=42

REM List of datasets to process (modify as needed)
set DATASETS=cora citeseer pubmed wikics

REM List of initialization approaches to test
set INIT_APPROACHES=pissa orthogonal guassian loftq eva

echo ==========================================
echo RANDOM EDGE DENOISING EXPERIMENTS
echo ==========================================
echo Datasets: %DATASETS%
echo Init Approaches: %INIT_APPROACHES%
echo LLM: %LLM_NAME%
echo PEFT Type: %PEFT_TYPE%
echo Remove percentage: %REMOVE_PERCENTAGE%%
echo Seed: %SEED%
echo ==========================================
echo.

echo ------------------------------------------
echo Running random_model_edge_denoising.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.random_model_edge_denoising --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i" --remove_percentage %REMOVE_PERCENTAGE% --seed %SEED%
        if errorlevel 1 (
            echo   ERROR: random edge denoising failed for %%d with %%i
        ) else (
            echo   SUCCESS: random edge denoising completed for %%d with %%i
        )
        echo.
    )
)

echo ==========================================
echo All random edge denoising experiments completed!
echo ==========================================
pause
