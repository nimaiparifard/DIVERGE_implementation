@echo off
REM Script to run all ensemble and experiment scripts for different datasets and init_weight_approaches
REM This script runs:
REM   1. Ensemble scripts (ensemble_edge_predictor_std, ensemble_edge_predictor_entropy, ensemble_all_std, ensemble_all_entropy, ensemble_all_hybrid)
REM   2. Tune edge editing hyperparameters
REM   3. Experiment scripts (line plots for std/entropy with topk/threshold/num_models)
REM
REM Usage: Modify DATASETS and INIT_APPROACHES variables below, then run this script
REM
REM Note: This script can take a very long time to complete. Consider running individual
REM       batch files (run_ensemble_experiments.bat, run_line_plot_experiments.bat, etc.)
REM       for specific experiment types instead.

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
set DATASETS=cora citeseer

REM List of initialization approaches to test
set INIT_APPROACHES=pissa orthogonal guassian loftq eva

@REM echo ==========================================
@REM echo RUNNING ALL EXPERIMENTS
@REM echo ==========================================
@REM echo Datasets: %DATASETS%
@REM echo Init Approaches: %INIT_APPROACHES%
@REM echo LLM: %LLM_NAME%
@REM echo PEFT Type: %PEFT_TYPE%
@REM echo ==========================================
@REM echo.
@REM echo WARNING: This will run many experiments and may take a very long time!
@REM echo Press Ctrl+C to cancel, or wait 5 seconds to continue...
@REM timeout /t 5 /nobreak >nul
@REM echo.
@REM
@REM REM ==========================================
@REM REM SECTION 1: Ensemble Edge Predictor Scripts
@REM REM ==========================================
@REM echo ==========================================
@REM echo SECTION 1: Ensemble Edge Predictor Scripts
@REM echo ==========================================
@REM echo.
@REM
@REM REM 1.1: ensemble_edge_predictor_std.py
@REM echo ------------------------------------------
@REM echo [1/5] Running ensemble_edge_predictor_std.py
@REM echo ------------------------------------------
@REM for %%d in (%DATASETS%) do (
@REM     for %%i in (%INIT_APPROACHES%) do (
@REM         echo   Dataset: %%d - Init: %%i
@REM         python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_edge_predictor_std --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
@REM         if errorlevel 1 (
@REM             echo   ERROR: ensemble_edge_predictor_std failed for %%d with %%i
@REM         ) else (
@REM             echo   SUCCESS: ensemble_edge_predictor_std completed for %%d with %%i
@REM         )
@REM         echo.
@REM     )
@REM )
@REM
@REM REM 1.2: ensemble_edge_predictor_entropy.py
@REM echo ------------------------------------------
@REM echo [2/5] Running ensemble_edge_predictor_entropy.py
@REM echo ------------------------------------------
@REM for %%d in (%DATASETS%) do (
@REM     for %%i in (%INIT_APPROACHES%) do (
@REM         echo   Dataset: %%d - Init: %%i
@REM         python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_edge_predictor_entropy --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%%i"
@REM         if errorlevel 1 (
@REM             echo   ERROR: ensemble_edge_predictor_entropy failed for %%d with %%i
@REM         ) else (
@REM             echo   SUCCESS: ensemble_edge_predictor_entropy completed for %%d with %%i
@REM         )
@REM         echo.
@REM     )
@REM )
@REM
@REM REM 1.3: ensemble_all_std.py (runs for all embeddings automatically)
@REM echo ------------------------------------------
@REM echo [3/5] Running ensemble_all_std.py
@REM echo ------------------------------------------
@REM for %%d in (%DATASETS%) do (
@REM     echo   Dataset: %%d
@REM     python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_all_std --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
@REM     if errorlevel 1 (
@REM         echo   ERROR: ensemble_all_std failed for %%d
@REM     ) else (
@REM         echo   SUCCESS: ensemble_all_std completed for %%d
@REM     )
@REM     echo.
@REM )
@REM
@REM REM 1.4: ensemble_all_entropy.py (runs for all embeddings automatically)
@REM echo ------------------------------------------
@REM echo [4/5] Running ensemble_all_entropy.py
@REM echo ------------------------------------------
@REM for %%d in (%DATASETS%) do (
@REM     echo   Dataset: %%d
@REM     python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_all_entropy --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
@REM     if errorlevel 1 (
@REM         echo   ERROR: ensemble_all_entropy failed for %%d
@REM     ) else (
@REM         echo   SUCCESS: ensemble_all_entropy completed for %%d
@REM     )
@REM     echo.
@REM )
@REM
@REM REM 1.5: ensemble_all_hybrid.py (runs for all embeddings automatically)
@REM echo ------------------------------------------
@REM echo [5/5] Running ensemble_all_hybrid.py
@REM echo ------------------------------------------
@REM for %%d in (%DATASETS%) do (
@REM     echo   Dataset: %%d
@REM     python -m LLMReasoner.TAPE.structure_hybrid_enahncer.ensemble_all_hybrid --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
@REM     if errorlevel 1 (
@REM         echo   ERROR: ensemble_all_hybrid failed for %%d
@REM     ) else (
@REM         echo   SUCCESS: ensemble_all_hybrid completed for %%d
@REM     )
@REM     echo.
@REM )
@REM
@REM REM ==========================================
@REM REM SECTION 2: Tune Edge Editing Hyperparameters
@REM REM ==========================================
@REM echo ==========================================
@REM echo SECTION 2: Tune Edge Editing Hyperparameters
@REM echo ==========================================
@REM echo.
@REM for %%d in (%DATASETS%) do (
@REM     echo   Dataset: %%d
@REM     python -m LLMReasoner.TAPE.structure_hybrid_enahncer.tune_edge_edditing_hyperparamters --dataset_name "%%d" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%"
@REM     if errorlevel 1 (
@REM         echo   ERROR: tune_edge_edditing_hyperparamters failed for %%d
@REM     ) else (
@REM         echo   SUCCESS: tune_edge_edditing_hyperparamters completed for %%d
@REM     )
@REM     echo.
@REM )

REM ==========================================
REM SECTION 3: Experiment Scripts - STD
REM ==========================================
echo ==========================================
echo SECTION 3: Experiment Scripts - STD Uncertainty
echo ==========================================
echo.

REM 3.1: line_plot_node_classifcation_metrics_std_topk.py
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

REM 3.2: line_plot_node_classifcation_metrics_std_threshhold.py
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

REM 3.3: line_plot_node_classifcation_metrics_std_num_models.py
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
REM SECTION 4: Experiment Scripts - ENTROPY
REM ==========================================
echo ==========================================
echo SECTION 4: Experiment Scripts - ENTROPY Uncertainty
echo ==========================================
echo.

REM 4.1: line_plot_node_classifcation_metrics_entropy_topk.py
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

REM 4.2: line_plot_node_classifcation_metrics_entropy_threshhold.py
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

REM 4.3: line_plot_node_classifcation_metrics_entropy_num_models.py
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
echo All experiments completed!
echo ==========================================
pause
