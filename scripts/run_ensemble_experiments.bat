@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Adaptive-ensemble analysis experiments (reviewer R2-C5, also R1-C6 / R4-C3):
REM "how do the learned weights behave and why do they beat simple averaging?"
REM
REM   exp1_disagreement_stratified  where the gain comes from (+ uniform->oracle gap)
REM   exp2_weak_view_injection      corrupt / duplicate one view on purpose
REM   exp3_fusion_baselines         non-learned fusion rules vs learned heads (+ #params)
REM   exp4_stability_significance   weight stability across seeds + McNemar / t / Wilcoxon
REM   exp5_node_level_routing       does the node-MLP trust a view where it is correct?
REM   exp6_per_class_weights        view specialisation per class
REM   exp7_logit_scale_calibration  how much of the gain is just recalibration?
REM   exp8_label_efficiency         labels needed by the fusion head / overfitting
REM   exp9_view_contribution        leave-one-view-out + Shapley vs learned weights
REM
REM Fusion protocol (PROTOCOL below, see experiments/ensemble_experiments/ensemble_common.py):
REM   paper      fit the fusion head on train logits, pick epoch (+ hyperparameters with
REM              TUNE=1) on the TEST nodes - same as ensemble/ensemble_gnns_learning.py and
REM              tune_ensemble_hyperparameter. Test accuracy is optimistically biased.
REM   val_split  fit on half of the val nodes, select on the other half, test scored once
REM   train_fit  fit on train logits, select on val, test scored once
REM exp3 also runs COMPARE_PROTOCOLS and reports how much selecting on test inflates
REM each rule (paper - val_split, and paper - train_fit).
REM Output files carry the protocol in their name, so runs never overwrite each other.
REM
REM Requires the LoRA embedding caches of every init (see
REM scripts/run_cache_embedding_all_datasets.bat). The first experiment of a
REM (dataset, llm) trains one GNN per view per GNN seed and caches the logits in
REM %OUT_DIR%\gnn_views\; the others reuse them.
REM
REM Outputs: %OUT_DIR%\<experiment>\  (CSVs, figures, <tag>_report.txt)
REM Progress: results/run_status/ENSEMBLE_EXP_CURRENT_RUN.txt
REM           results/run_status/ensemble_exp_run_history.csv
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
REM SPLIT=1 = supervised embeddings/split (paper setting), 0 = semi-supervised
set SPLIT=1
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
REM LoRA seeds of the adapters / embedding caches (space-separated; one run per seed).
REM Must match the seed-tagged caches (..._pool-mean_seed<seed>_semi_supervised.pt);
REM without it the dataset config seed is used (e.g. cora = 40), which has no cache.
set LORA_SEEDS=42
REM GNN seeds (one GNN per view per seed) and fusion seeds (val fit/select split + head init)
set GNN_SEEDS=42,43,44,45
set FUSE_SEEDS=0,1,2
REM Fusion protocol: paper (select on TEST, as in the paper code), val_split or train_fit
set PROTOCOL=paper
set FIT_FRAC=0.5
REM 1 = grid-search the heads like tune_ensemble_hyperparameter (layers {2,3} x hidden {64,128,256}
REM     x dropout {0.3,0.5} x lr {0.001,0.01,0.0001}; 36 configs for the MLP, 3 lrs for scalar/per-class)
set TUNE=1
REM Fusion-head epochs (tune_ensemble_hyperparameter uses 200)
set EPOCHS=200
REM Protocols exp3 runs next to PROTOCOL for the side-by-side comparison
set COMPARE_PROTOCOLS=val_split,train_fit
set OUT_DIR=results\ensemble_experiments
REM Which node sets the fused predictions are scored on (1 = on, 0 = off).
REM   RUN_TEST  normal results (test nodes)
REM   RUN_VAL   same fusion heads, scored on the validation nodes; output files get a
REM             "_validation" suffix (labels inside the figures/reports still say "test")
set RUN_TEST=1
set RUN_VAL=1
REM Extra flags for every experiment, e.g. "--gnn_model_name GCN" or "--retrain_gnn" or "--lora_seed 42"
set EXTRA_ARGS=

REM Per-experiment extra flags (appended after EXTRA_ARGS)
set ARGS_exp1_disagreement_stratified=--methods scalar,per_class,mlp
set ARGS_exp2_weak_view_injection=--target best --noise_levels 0,0.5,1,2,4,8 --corruptions shuffle,random,base --dup_counts 0,1,2,3,4
set ARGS_exp3_fusion_baselines=--methods all --compare_protocols %COMPARE_PROTOCOLS%
REM more seeds for the significance tests
set ARGS_exp4_stability_significance=--methods scalar,per_class,mlp --gnn_seeds 42,43,44,45,46 --fuse_seeds 0,1,2,3,4
set ARGS_exp5_node_level_routing=
set ARGS_exp6_per_class_weights=
set ARGS_exp7_logit_scale_calibration=--methods scalar,per_class,mlp
set ARGS_exp8_label_efficiency=--methods scalar,per_class,mlp,stacking_lr --fractions 0.05,0.1,0.25,0.5,1.0 --repeats 3
set ARGS_exp9_view_contribution=--methods scalar,per_class

REM Which experiments to run (1 = on, 0 = off)
set RUN_EXP1=1
set RUN_EXP2=1
set RUN_EXP3=1
set RUN_EXP4=0
set RUN_EXP5=1
set RUN_EXP6=1
set RUN_EXP7=1
set RUN_EXP8=0
set RUN_EXP9=1

set PYTHONIOENCODING=utf-8

REM Datasets
set DATASETS=cora citeseer

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=llama_3.2_1B smollm2_1.7B gemma2_2B

set EVAL_SETS=
if "%RUN_TEST%"=="1" set EVAL_SETS=!EVAL_SETS! test
if "%RUN_VAL%"=="1" set EVAL_SETS=!EVAL_SETS! val
if "!EVAL_SETS!"=="" (
    echo ERROR: RUN_TEST and RUN_VAL are both 0 - nothing to run.
    pause
    exit /b 1
)

set EXPERIMENTS=
if "%RUN_EXP1%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp1_disagreement_stratified
if "%RUN_EXP2%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp2_weak_view_injection
if "%RUN_EXP3%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp3_fusion_baselines
if "%RUN_EXP4%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp4_stability_significance
if "%RUN_EXP5%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp5_node_level_routing
if "%RUN_EXP6%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp6_per_class_weights
if "%RUN_EXP7%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp7_logit_scale_calibration
if "%RUN_EXP8%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp8_label_efficiency
if "%RUN_EXP9%"=="1" set EXPERIMENTS=!EXPERIMENTS! exp9_view_contribution

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\ENSEMBLE_EXP_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\ensemble_exp_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\ENSEMBLE_EXP_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\ENSEMBLE_EXP_FAILED_RUNS.txt

REM A history file from before the lora_seed column was added: keep it aside, start a new one
if exist "%HISTORY_FILE%" (
    findstr /c:",lora_seed," "%HISTORY_FILE%" >nul || (
        move /y "%HISTORY_FILE%" "%STATUS_DIR%\ensemble_exp_run_history_old.csv" >nul
        echo [info] old run history moved to %STATUS_DIR%\ensemble_exp_run_history_old.csv
    )
)
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,experiment,dataset,llm_name,lora_seed,peft_type,split,gnn_seeds,fuse_seeds,protocol,tune,eval_set,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE adaptive-ensemble experiments
echo Experiments:%EXPERIMENTS%
echo Datasets: %DATASETS%
echo Protocol: %PROTOCOL%   Tune: %TUNE%   Epochs: %EPOCHS%
echo Eval sets:%EVAL_SETS%
echo Status dir: %STATUS_DIR%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
    for %%s in (%LORA_SEEDS%) do (
        for %%e in (%EXPERIMENTS%) do (
        for %%v in (%EVAL_SETS%) do (
            set "EXP_ARGS=!ARGS_%%e!"
            set "RUN_ID=experiment=%%e | dataset=%%d | llm=%%m | lora_seed=%%s | peft=%PEFT_TYPE% | split=%SPLIT% | protocol=%PROTOCOL% | tune=%TUNE% | eval_set=%%v"
            set "TS=!DATE! !TIME!"

            echo ==========================================
            echo START: !RUN_ID!
            echo Time: !TS!
            echo ==========================================

            (
                echo status=RUNNING
                echo started=!TS!
                echo experiment=%%e
                echo dataset=%%d
                echo llm_name=%%m
                echo lora_seed=%%s
                echo eval_set=%%v
                echo command=python -m experiments.ensemble_experiments.%%e --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --lora_seed %%s --approaches %APPROACHES% --split %SPLIT% --gnn_seeds %GNN_SEEDS% --fuse_seeds %FUSE_SEEDS% --protocol %PROTOCOL% --fit_frac %FIT_FRAC% --tune %TUNE% --epochs %EPOCHS% --eval_set %%v --out_dir %OUT_DIR% %EXTRA_ARGS% !EXP_ARGS!
            ) > "%CURRENT_FILE%"

            REM per-experiment args come last so they override the shared seeds (argparse keeps the last value)
            python -m experiments.ensemble_experiments.%%e --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --lora_seed %%s --approaches "%APPROACHES%" --split %SPLIT% --gnn_seeds "%GNN_SEEDS%" --fuse_seeds "%FUSE_SEEDS%" --protocol %PROTOCOL% --fit_frac %FIT_FRAC% --tune %TUNE% --epochs %EPOCHS% --eval_set %%v --out_dir "%OUT_DIR%" %EXTRA_ARGS% !EXP_ARGS!
            set EXIT_CODE=!ERRORLEVEL!
            set "TS_END=!DATE! !TIME!"

            if !EXIT_CODE! NEQ 0 (
                echo ERROR: !RUN_ID!  exit_code=!EXIT_CODE!
                REM redirect first: "...,0>> file" would be parsed as a redirect of handle 0
                >> "%HISTORY_FILE%" echo !TS_END!,FAILED,%%e,%%d,%%m,%%s,%PEFT_TYPE%,%SPLIT%,"%GNN_SEEDS%","%FUSE_SEEDS%",%PROTOCOL%,%TUNE%,%%v,!EXIT_CODE!
                >> "%FAILED_FILE%" echo !TS_END! FAILED !RUN_ID! exit=!EXIT_CODE!
                echo Continuing to next run despite failure...
                echo.
            ) else (
                echo OK: !RUN_ID!
                >> "%HISTORY_FILE%" echo !TS_END!,OK,%%e,%%d,%%m,%%s,%PEFT_TYPE%,%SPLIT%,"%GNN_SEEDS%","%FUSE_SEEDS%",%PROTOCOL%,%TUNE%,%%v,0
                (
                    echo status=COMPLETED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo experiment=%%e
                    echo dataset=%%d
                    echo llm_name=%%m
                echo lora_seed=%%s
                    echo eval_set=%%v
                    echo exit_code=0
                ) > "%LAST_OK_FILE%"
                echo.
            )
        )
        )
    )
    )
)

echo ==========================================
echo Adaptive-ensemble experiments finished.
echo Check:
echo   %OUT_DIR%\exp1_disagreement\exp1_summary.csv
echo   %OUT_DIR%\exp2_weak_view\exp2_corruption_rows.csv
echo   %OUT_DIR%\exp3_fusion_baselines\exp3_summary.csv
echo   %OUT_DIR%\exp4_stability\exp4_summary.csv
echo   %OUT_DIR%\exp5_routing\exp5_run_rows.csv
echo   %OUT_DIR%\exp6_per_class\exp6_summary.csv
echo   %OUT_DIR%\exp7_calibration\exp7_summary.csv
echo   %OUT_DIR%\exp8_label_efficiency\exp8_summary.csv
echo   %OUT_DIR%\exp9_view_contribution\exp9_summary.csv
if "%RUN_VAL%"=="1" echo   validation results: same folders, files named *_validation.csv / *_validation_*.png
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued ensemble experiment runs finished; see ensemble_exp_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
