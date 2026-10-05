@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM DIVERGE sweep for LARGE TAGs (computer, arxiv, ...): same grid/stages as
REM scripts/run_gnn_llm_seed_sweep.bat, but GNNs are trained one init approach at a
REM time and the ensembles run on saved logits, so the 5 embedding caches never sit
REM on the GPU together.
REM
REM   STAGE 1  tune GNN hyperparameters        (gnns/tune_gnn_hyperparameter.py)
REM   STAGE 2  M-ablation:  EXPORT -> ABLATION (diverge_for_larger_tags/ablation_from_logits.py)
REM   STAGE 3  DIVERGE:     EXPORT -> DIVERGE  (diverge_for_larger_tags/diverge_from_logits.py)
REM
REM EXPORT = diverge_for_larger_tags/train_gnn_export_logits.py: train one GNN per init
REM approach and save its logits (+ weights). It skips approaches whose logits are
REM up to date (same tuned hyperparameters + embedding cache), so stages 2 and 3 share
REM one export, and re-tuning in stage 1 triggers retraining automatically.
REM If EXPORT fails for a combination, its ABLATION/DIVERGE step is skipped.
REM
REM Requires, for every (dataset, llm, seed): trained LoRA adapters + cached embeddings
REM   (scripts/run_all_datasets_lora_init.bat, scripts/run_cache_embedding_all_datasets.bat)
REM
REM Outputs:
REM   Logits/weights : %LOGITS_ROOT%\<dataset>\<llm>_<peft>_<gnn>_<split>_seed<seed>\
REM   Stage 1 : results/gnn_tuning/tuning_summary.csv
REM   Stage 2 : %ABLATION_CSV%
REM   Stage 3 : %DIVERGE_CSV%
REM   Every run (all stages): results/run_status/diverge_large_tags_history.csv
REM   Failures              : results/run_status/DIVERGE_LARGE_TAGS_FAILED_RUNS.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM ---------------------------- sweep grid -------------------------------------
set DATASETS=arxiv
REM LLM encoders (must have adapters + embedding caches for every seed below)
set LLM_NAMES=llama_3.2_1B
REM GNN encoders: SAGE GCN GIN
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

REM Export: where logits go, and whether to also keep the GNN weights
set LOGITS_ROOT=artifacts\diverge_logits
REM 1 = also save each GNN's best-epoch weights next to its logits
set SAVE_WEIGHTS=1
REM 1 = retrain every GNN even if up-to-date logits exist
set FORCE_RETRAIN=0
set EXPORT_FLAGS=
if "%SAVE_WEIGHTS%"=="1" set EXPORT_FLAGS=!EXPORT_FLAGS! --save_weights
if "%FORCE_RETRAIN%"=="1" set EXPORT_FLAGS=!EXPORT_FLAGS! --overwrite

REM Stage 2: M-ablation (SAGE only, independent of GNN_MODELS; needs all 5 approaches)
set ABLATION_GNN_MODELS=SAGE
set M_VALUES=2,3,4,5
set ABLATION_ENSEMBLE=uniform,learnable,learnable_classes,learnable_per_classes
set ABLATION_CSV=results\ablation\num_finetuned_models_large_tags_results.csv

REM Stage 3: DIVERGE
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
set DIVERGE_ENSEMBLES=uniform,learnable,learnable_classes,learnable_per_classes
set DIVERGE_CSV=results\diverge\diverge_large_tags_results.csv

REM Disable HF Trainer logging integrations / keep console output unicode-safe
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8

REM ---------------------------- status tracking --------------------------------
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set HISTORY_FILE=%STATUS_DIR%\diverge_large_tags_history.csv
set FAILED_FILE=%STATUS_DIR%\DIVERGE_LARGE_TAGS_FAILED_RUNS.txt
if not exist "%HISTORY_FILE%" (
    echo timestamp,stage,status,dataset,llm_name,gnn_model,seed,split,exit_code> "%HISTORY_FILE%"
)
set /a N_OK=0
set /a N_FAIL=0
set /a N_SKIP=0

echo ==========================================
echo DIVERGE large-TAG sweep: datasets=[%DATASETS%] llms=[%LLM_NAMES%]
echo                gnns=[%GNN_MODELS%] ablation_gnns=[%ABLATION_GNN_MODELS%] seeds=[%SEEDS%] split=%SPLIT%
echo Stages: tune=%RUN_TUNE% ablation=%RUN_ABLATION% diverge=%RUN_DIVERGE%
echo Logits: %LOGITS_ROOT%  save_weights=%SAVE_WEIGHTS% force_retrain=%FORCE_RETRAIN%
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
        call :run_step EXPORT %%d %%m %%g %%s python -m diverge_for_larger_tags.train_gnn_export_logits --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --split %SPLIT% --logits_root "%LOGITS_ROOT%" !EXPORT_FLAGS!
        if !STEP_EXIT! EQU 0 (
            call :run_step ABLATION %%d %%m %%g %%s python -m diverge_for_larger_tags.ablation_from_logits --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --split %SPLIT% --ensemble_approaches "%ABLATION_ENSEMBLE%" --m_values "%M_VALUES%" --csv_path "%ABLATION_CSV%" --logits_root "%LOGITS_ROOT%"
        ) else (
            call :skip_step ABLATION %%d %%m %%g %%s
        )
    )
)

REM ============================ STAGE 3: DIVERGE ===============================
if "%RUN_DIVERGE%"=="1" (
    for %%d in (%DATASETS%) do for %%m in (%LLM_NAMES%) do for %%s in (%SEEDS%) do for %%g in (%GNN_MODELS%) do (
        call :run_step EXPORT %%d %%m %%g %%s python -m diverge_for_larger_tags.train_gnn_export_logits --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --split %SPLIT% --approaches "%APPROACHES%" --logits_root "%LOGITS_ROOT%" !EXPORT_FLAGS!
        if !STEP_EXIT! EQU 0 (
            call :run_step DIVERGE %%d %%m %%g %%s python -m diverge_for_larger_tags.diverge_from_logits --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --gnn_model_name %%g --split %SPLIT% --approaches "%APPROACHES%" --ensemble_approaches "%DIVERGE_ENSEMBLES%" --csv_path "%DIVERGE_CSV%" --logits_root "%LOGITS_ROOT%"
        ) else (
            call :skip_step DIVERGE %%d %%m %%g %%s
        )
    )
)

echo.
echo ==========================================
echo Sweep finished: ok=!N_OK! failed=!N_FAIL! skipped=!N_SKIP!
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
REM Leaves the exit code in STEP_EXIT.
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


REM -----------------------------------------------------------------------------
REM :skip_step STAGE DATASET LLM GNN SEED
REM Records a step skipped because its EXPORT step failed.
REM -----------------------------------------------------------------------------
:skip_step
set "STEP_TS=!DATE! !TIME!"
echo [%~1] SKIPPED ^(export failed^): dataset=%~2 llm=%~3 gnn=%~4 seed=%~5
>> "%HISTORY_FILE%" echo !STEP_TS!,%~1,SKIPPED,%~2,%~3,%~4,%~5,%SPLIT%,
set /a N_SKIP+=1
goto :eof
