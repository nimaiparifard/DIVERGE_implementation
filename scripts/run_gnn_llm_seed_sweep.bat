@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Full DIVERGE sweep over datasets x LLM encoders x GNN encoders x seeds:
REM   STAGE 1  tune GNN hyperparameters        (gnns/tune_gnn_hyperparameter.py)
REM   STAGE 2  M-ablation (number of finetuned LLMs, M=2..5)
REM            (experiments/number_of_finetuned_model_experiment_in_diverge_model.py)
REM   STAGE 3  DIVERGE pipeline, no edge editing (experiments/diverge.py)
REM
REM Stage 1 writes gnn_hyperparameters/<dataset>_<llm>_<gnn>[_semi_supervised]_seed<seed>.json,
REM which stages 2 and 3 pick up automatically for the same (dataset, llm, gnn, seed).
REM
REM Requires, for every (dataset, llm, seed): trained LoRA adapters + cached embeddings
REM   (scripts/run_all_datasets_lora_init.bat, scripts/run_cache_embedding_all_datasets.bat)
REM
REM Outputs:
REM   Stage 1 : results/gnn_tuning/tuning_summary.csv
REM   Stage 2 : %ABLATION_CSV%
REM   Stage 3 : %DIVERGE_CSV%
REM   Every run (all stages): results/run_status/gnn_llm_seed_sweep_history.csv
REM   Failures              : results/run_status/GNN_LLM_SEED_SWEEP_FAILED_RUNS.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM ---------------------------- sweep grid -------------------------------------
set DATASETS=computer
REM LLM encoders (must have adapters + embedding caches for every seed below)
set LLM_NAMES=llama_3.2_1B smollm2_1.7B
REM GNN encoders: SAGE GCN  GIN 
set GNN_MODELS=SAGE GCN GIN 
REM Seeds (must match the seeds the LoRA adapters / embedding caches were made with)
set SEEDS=42

REM ---------------------------- stages to run (1 = on, 0 = off) ----------------
set RUN_TUNE=1
set RUN_ABLATION=1
set RUN_DIVERGE=1

REM ---------------------------- shared settings --------------------------------
set PEFT_TYPE=lora
REM 1 = supervised split, 0 = semi-supervised (seed-tagged) split
set SPLIT=1

REM Stage 1: GNN tuning (see python -m gnns.tune_gnn_hyperparameter --help)
set TUNE_INIT_APPROACH=eva
set SEARCH=random
set N_TRIALS=512
set RUNG0_EPOCHS=50
set TOP_K=16
set N_EVAL_SEEDS=3
set PRUNE_RATIO=0.9
REM 1 = re-tune combinations that already have a tuned hyperparameter file
set RETUNE=1
set SKIP_FLAG=--skip_existing
if "%RETUNE%"=="1" set SKIP_FLAG=

REM Stage 2: M-ablation (SAGE only, independent of GNN_MODELS)
set ABLATION_GNN_MODELS=SAGE
set M_VALUES=2,3,4,5
set ABLATION_ENSEMBLE=uniform,learnable,learnable_classes,learnable_per_classes
set ABLATION_CSV=results\ablation\num_finetuned_models_gnn_llm_seed_results.csv

REM Stage 3: DIVERGE
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
set DIVERGE_ENSEMBLES=uniform,learnable,learnable_classes,learnable_per_classes
set DIVERGE_CSV=results\diverge\diverge_gnn_llm_seed_results.csv

REM Disable HF Trainer logging integrations / keep console output unicode-safe
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8

REM ---------------------------- status tracking --------------------------------
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set HISTORY_FILE=%STATUS_DIR%\gnn_llm_seed_sweep_history.csv
set FAILED_FILE=%STATUS_DIR%\GNN_LLM_SEED_SWEEP_FAILED_RUNS.txt
if not exist "%HISTORY_FILE%" (
    echo timestamp,stage,status,dataset,llm_name,gnn_model,seed,split,exit_code> "%HISTORY_FILE%"
)
set /a N_OK=0
set /a N_FAIL=0

echo ==========================================
echo DIVERGE sweep: datasets=[%DATASETS%] llms=[%LLM_NAMES%]
echo                gnns=[%GNN_MODELS%] ablation_gnns=[%ABLATION_GNN_MODELS%] seeds=[%SEEDS%] split=%SPLIT%
echo Stages: tune=%RUN_TUNE% ablation=%RUN_ABLATION% diverge=%RUN_DIVERGE%
echo History CSV: %HISTORY_FILE%
echo ==========================================

REM ============================ STAGE 1: tune GNNs =============================
if "%RUN_TUNE%"=="1" (
    for %%d in (%DATASETS%) do for %%m in (%LLM_NAMES%) do for %%s in (%SEEDS%) do for %%g in (%GNN_MODELS%) do (
        call :run_step TUNE %%d %%m %%g %%s python -m gnns.tune_gnn_hyperparameter --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --re_split %SPLIT% --init_weight_approach %TUNE_INIT_APPROACH% --search %SEARCH% --n_trials %N_TRIALS% --rung0_epochs %RUNG0_EPOCHS% --top_k %TOP_K% --n_eval_seeds %N_EVAL_SEEDS% --prune_ratio %PRUNE_RATIO% --no_plot %SKIP_FLAG%
    )
)

REM ============================ STAGE 2: M-ablation ============================
if "%RUN_ABLATION%"=="1" (
    for %%d in (%DATASETS%) do for %%m in (%LLM_NAMES%) do for %%s in (%SEEDS%) do for %%g in (%ABLATION_GNN_MODELS%) do (
        call :run_step ABLATION %%d %%m %%g %%s python -m experiments.number_of_finetuned_model_experiment_in_diverge_model --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --split %SPLIT% --ensemble_approaches "%ABLATION_ENSEMBLE%" --m_values "%M_VALUES%" --csv_path "%ABLATION_CSV%"
    )
)

REM ============================ STAGE 3: DIVERGE ===============================
if "%RUN_DIVERGE%"=="1" (
    for %%d in (%DATASETS%) do for %%m in (%LLM_NAMES%) do for %%s in (%SEEDS%) do for %%g in (%GNN_MODELS%) do (
        call :run_step DIVERGE %%d %%m %%g %%s python -m experiments.diverge --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --split %SPLIT% --approaches "%APPROACHES%" --ensemble_approaches "%DIVERGE_ENSEMBLES%" --csv_path "%DIVERGE_CSV%"
    )
)

echo.
echo ==========================================
echo Sweep finished: ok=!N_OK! failed=!N_FAIL!
echo Check:
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo   results\gnn_tuning\tuning_summary.csv
echo   %ABLATION_CSV%
echo   %DIVERGE_CSV%
echo ==========================================
pause
endlocal
goto :eof


REM -----------------------------------------------------------------------------
REM :run_step STAGE DATASET LLM GNN SEED <command...>
REM Runs the command, records OK/FAILED in the history CSV, and keeps going on failure.
REM -----------------------------------------------------------------------------
:run_step
set "STEP_STAGE=%~1"
set "STEP_D=%~2"
set "STEP_M=%~3"
set "STEP_G=%~4"
set "STEP_S=%~5"
shift & shift & shift & shift & shift
set "STEP_CMD="
:collect_args
if "%~1"=="" goto run_cmd
set "STEP_CMD=!STEP_CMD! %1"
shift
goto collect_args
:run_cmd
echo.
echo ==========================================
echo [!STEP_STAGE!] START: dataset=!STEP_D! ^| llm=!STEP_M! ^| gnn=!STEP_G! ^| seed=!STEP_S!
echo Time: !DATE! !TIME!
echo ==========================================
%STEP_CMD%
set STEP_EXIT=!ERRORLEVEL!
set "STEP_TS=!DATE! !TIME!"
if !STEP_EXIT! NEQ 0 (
    echo [!STEP_STAGE!] ERROR: dataset=!STEP_D! llm=!STEP_M! gnn=!STEP_G! seed=!STEP_S! exit_code=!STEP_EXIT!
    >> "%HISTORY_FILE%" echo !STEP_TS!,!STEP_STAGE!,FAILED,!STEP_D!,!STEP_M!,!STEP_G!,!STEP_S!,%SPLIT%,!STEP_EXIT!
    >> "%FAILED_FILE%" echo !STEP_TS! [!STEP_STAGE!] FAILED dataset=!STEP_D! llm=!STEP_M! gnn=!STEP_G! seed=!STEP_S! exit=!STEP_EXIT!
    set /a N_FAIL+=1
    echo Continuing to next run despite failure...
) else (
    echo [!STEP_STAGE!] OK: dataset=!STEP_D! llm=!STEP_M! gnn=!STEP_G! seed=!STEP_S!
    >> "%HISTORY_FILE%" echo !STEP_TS!,!STEP_STAGE!,OK,!STEP_D!,!STEP_M!,!STEP_G!,!STEP_S!,%SPLIT%,0
    set /a N_OK+=1
)
goto :eof
