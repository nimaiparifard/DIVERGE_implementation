@echo off
REM Script to run line plot experiment scripts for different datasets and init_weight_approaches
REM This script runs:
REM   - STD experiments: std_topk, std_threshold, std_num_models
REM   - ENTROPY experiments: entropy_topk, entropy_threshold, entropy_num_models

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Change to project root (3 levels up from structure_hybrid_enahncer)
cd /d "%SCRIPT_DIR%..\..\.."

REM Default values
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora

REM List of datasets to process (modify as needed)
set DATASETS=cora citeseer pubmed wikics

REM List of initialization approaches to test
set INIT_APPROACHES=pissa orthogonal guassian loftq eva

echo ==========================================
echo LINE PLOT EXPERIMENTS
echo ==========================================
echo Datasets: %DATASETS%
echo Init Approaches: %INIT_APPROACHES%
echo LLM: %LLM_NAME%
echo PEFT Type: %PEFT_TYPE%
echo ==========================================
echo.

REM ==========================================
REM STD UNCERTAINTY EXPERIMENTS
REM ==========================================
echo ==========================================
echo STD UNCERTAINTY EXPERIMENTS
echo ==========================================

REM 1. line_plot_node_classifcation_metrics_std_topk.py
echo ------------------------------------------
echo [1/3 STD] Running line_plot_node_classifcation_metrics_std_topk.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.experiments.line_plot_node_classifcation_metrics_std_topk --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: std_topk failed for %%d with %%i
        ) else (
            echo   SUCCESS: std_topk completed for %%d with %%i
        )
        echo.
    )
)

REM 2. line_plot_node_classifcation_metrics_std_threshhold.py
echo ------------------------------------------
echo [2/3 STD] Running line_plot_node_classifcation_metrics_std_threshhold.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.experiments.line_plot_node_classifcation_metrics_std_threshhold --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: std_threshold failed for %%d with %%i
        ) else (
            echo   SUCCESS: std_threshold completed for %%d with %%i
        )
        echo.
    )
)

REM 3. line_plot_node_classifcation_metrics_std_num_models.py
echo ------------------------------------------
echo [3/3 STD] Running line_plot_node_classifcation_metrics_std_num_models.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.experiments.line_plot_node_classifcation_metrics_std_num_models --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: std_num_models failed for %%d with %%i
        ) else (
            echo   SUCCESS: std_num_models completed for %%d with %%i
        )
        echo.
    )
)

REM ==========================================
REM ENTROPY UNCERTAINTY EXPERIMENTS
REM ==========================================
echo ==========================================
echo ENTROPY UNCERTAINTY EXPERIMENTS
echo ==========================================

REM 4. line_plot_node_classifcation_metrics_entropy_topk.py
echo ------------------------------------------
echo [1/3 ENTROPY] Running line_plot_node_classifcation_metrics_entropy_topk.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.experiments.line_plot_node_classifcation_metrics_entropy_topk --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: entropy_topk failed for %%d with %%i
        ) else (
            echo   SUCCESS: entropy_topk completed for %%d with %%i
        )
        echo.
    )
)

REM 5. line_plot_node_classifcation_metrics_entropy_threshhold.py
echo ------------------------------------------
echo [2/3 ENTROPY] Running line_plot_node_classifcation_metrics_entropy_threshhold.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.experiments.line_plot_node_classifcation_metrics_entropy_threshhold --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: entropy_threshold failed for %%d with %%i
        ) else (
            echo   SUCCESS: entropy_threshold completed for %%d with %%i
        )
        echo.
    )
)

REM 6. line_plot_node_classifcation_metrics_entropy_num_models.py
echo ------------------------------------------
echo [3/3 ENTROPY] Running line_plot_node_classifcation_metrics_entropy_num_models.py
echo ------------------------------------------
for %%d in (%DATASETS%) do (
    for %%i in (%INIT_APPROACHES%) do (
        echo   Dataset: %%d - Init: %%i
        python -m LLMReasoner.TAPE.structure_hybrid_enahncer.experiments.line_plot_node_classifcation_metrics_entropy_num_models --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
        if errorlevel 1 (
            echo   ERROR: entropy_num_models failed for %%d with %%i
        ) else (
            echo   SUCCESS: entropy_num_models completed for %%d with %%i
        )
        echo.
    )
)

echo ==========================================
echo All line plot experiments completed!
echo ==========================================
pause
