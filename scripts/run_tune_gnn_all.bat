@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Tune GNN hyperparameters across datasets x LLMs x seeds x GNN encoders, then
REM verify that GNN training resolves (and reproduces) the tuned hyperparameters.
REM Run this AFTER scripts/run_cache_embedding_all_datasets.bat has cached the
REM LLM embeddings for the same (dataset, llm, init, seed) combinations.
REM
REM Tuned files: gnn_hyperparameters/<dataset>_<llm>_<gnn>[_semi_supervised]_seed<seed>.json
REM Trial logs : results/gnn_tuning/<run>/trials.jsonl   (interrupted runs resume)
REM Summary    : results/gnn_tuning/tuning_summary.csv
REM Verify     : results/gnn_tuning/verify_report.csv
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM 0 = semi-supervised (seed-tagged embedding caches), 1 = supervised
set RE_SPLIT=1

REM Datasets
set DATASETS=cora

REM LLMs (must have cached embeddings under artifacts/cache)
set LLM_NAMES=llama_3.2_1B
REM set LLM_NAMES=llama_3.2_1B smollm2_1.7B gemma2_2B

REM Seeds (must match the seeds the LoRA adapters / embedding caches were made with)
set SEEDS=42

REM GNN encoders to tune: SAGE GCN GAT GIN TransformerConv
set GNN_MODELS=SAGE

REM Embedding cache to tune on
set INIT_APPROACH=eva

REM Search settings (see python -m gnns.tune_gnn_hyperparameter --help)
REM Efficient full grid (512 configs) via successive halving:
REM   rung 0 : all 512 configs, 1 seed, RUNG0_EPOCHS epochs (early_stop = epochs/5),
REM            runs pruned once they fall below PRUNE_RATIO * best so far
REM   final  : TOP_K best configs x N_EVAL_SEEDS seeds at the full budget (200 epochs)
REM   => ~512 short runs + 48 full runs instead of 512 x 3 full runs
set SEARCH=random
set N_TRIALS=512
set RUNG0_EPOCHS=50
set TOP_K=16
set N_EVAL_SEEDS=3
set PRUNE_RATIO=0.9
REM Exhaustive (slow) grid instead: RUNG0_EPOCHS=0, TOP_K=512, PRUNE_RATIO=0

REM Set to 1 to re-tune combinations that already have a tuned file
set RETUNE=1
set SKIP_FLAG=--skip_existing
if "%RETUNE%"=="1" set SKIP_FLAG=

set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set HISTORY_FILE=%STATUS_DIR%\gnn_tuning_history.csv
set FAILED_FILE=%STATUS_DIR%\GNN_TUNING_FAILED_RUNS.txt
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,gnn_model,seed,re_split,init,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE GNN hyperparameter tuning sweep
echo History CSV: %HISTORY_FILE%
echo ==========================================

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%s in (%SEEDS%) do (
            for %%g in (%GNN_MODELS%) do (
                echo.
                echo ==========================================
                echo START: dataset=%%d ^| llm=%%m ^| seed=%%s ^| gnn=%%g ^| init=%INIT_APPROACH%
                echo ==========================================
                python -m gnns.tune_gnn_hyperparameter --dataset_name %%d --llm_name %%m --seed %%s --gnn_model_name %%g --re_split %RE_SPLIT% --init_weight_approach %INIT_APPROACH% --search %SEARCH% --n_trials %N_TRIALS% --rung0_epochs %RUNG0_EPOCHS% --top_k %TOP_K% --n_eval_seeds %N_EVAL_SEEDS% --prune_ratio %PRUNE_RATIO% %SKIP_FLAG%
                set EXIT_CODE=!ERRORLEVEL!
                set "TS_END=!DATE! !TIME!"
                if !EXIT_CODE! NEQ 0 (
                    echo ERROR: dataset=%%d llm=%%m seed=%%s gnn=%%g exit_code=!EXIT_CODE!
                    echo !TS_END!,FAILED,%%d,%%m,%%g,%%s,%RE_SPLIT%,%INIT_APPROACH%,!EXIT_CODE!>> "%HISTORY_FILE%"
                    echo !TS_END! FAILED dataset=%%d llm=%%m seed=%%s gnn=%%g exit=!EXIT_CODE!>> "%FAILED_FILE%"
                ) else (
                    echo !TS_END!,OK,%%d,%%m,%%g,%%s,%RE_SPLIT%,%INIT_APPROACH%,0>> "%HISTORY_FILE%"
                )
            )
        )
    )
)

echo.
echo ==========================================
echo Verifying GNN training uses the tuned hyperparameters
echo ==========================================
python -m gnns.verify_gnn_hyperparameters --dataset_names %DATASETS% --llm_names %LLM_NAMES% --seeds %SEEDS% --gnn_model_names %GNN_MODELS% --re_split %RE_SPLIT% --retrain --check_training_csv
if !ERRORLEVEL! NEQ 0 (
    echo Verification found combinations NOT using / reproducing the tuned hyperparameters - see above.
) else (
    echo Verification OK: all combinations use and reproduce their tuned hyperparameters.
)

pause
endlocal
