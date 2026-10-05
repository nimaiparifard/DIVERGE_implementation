@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Run the GNN-embedding CKA analysis (experiments/cka_for_gnn_embeddings.py)
REM across datasets x LLMs x seeds. Each run trains one GNN per LoRA-init
REM approach on its cached LLM embeddings, extracts the GNN node embeddings
REM right before the classification logits, and computes the pairwise
REM linear-CKA matrix between them. Requires that, for every combination, you
REM already ran:
REM   1) scripts/run_all_datasets_lora_init.bat       (trains the LoRA adapters)
REM   2) scripts/run_cache_embedding_all_datasets.bat (caches their embeddings)
REM
REM Outputs:
REM   results/cka_gnn/cka_gnn_pairwise_results.csv      (one row per ordered approach pair)
REM   results/cka_gnn/cka_gnn_per_approach_results.csv  (GNN accuracy + CKA(LLM emb, GNN emb))
REM   results/cka_gnn/cka_gnn_summary_results.csv       (one row per run: mean off-diag CKA, ...)
REM   results/cka_gnn/<dataset>_<llm>_<peft>_<gnn>_seed<seed>_<nodes>_cka_gnn_heatmap.png
REM   results/train_info/<dataset>_cka_gnn_seed<seed>/  (per-run text report)
REM
REM Progress / identity of the active run is also written to:
REM   results/run_status/CKA_GNN_CURRENT_RUN.txt
REM   results/run_status/cka_gnn_run_history.csv
REM   results/run_status/CKA_GNN_LAST_COMPLETED.txt
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set PEFT_TYPE=lora
REM SPLIT=0 (semi-supervised) matches run_cache_embedding_all_datasets.bat, where
REM --seed selects a seed-tagged embedding cache. Under SPLIT=1 the cache is
REM seed-independent (only the GNN seed/split would change across seeds).
set SPLIT=0
set APPROACHES=pissa,orthogonal,gaussian,loftq,eva
REM Nodes whose embeddings enter the CKA: all, train, val or test
set NODE_SUBSET=all
REM Extra flags, e.g. "--gnn_model_name GCN" or "--save_embeddings" (empty = defaults)
set EXTRA_ARGS=
set CKA_PAIRWISE_CSV=results\cka_gnn\cka_gnn_pairwise_results.csv
set CKA_SUMMARY_CSV=results\cka_gnn\cka_gnn_summary_results.csv
set CKA_PER_APPROACH_CSV=results\cka_gnn\cka_gnn_per_approach_results.csv

set PYTHONIOENCODING=utf-8

REM Datasets
set DATASETS=cora

REM LLMs (must match configs/llm_configs/<name>.json and local_models/<name>)
set LLM_NAMES=llama_3.2_1B
REM Seeds (must match the seeds used to train/cache the adapters being loaded)
set SEEDS=42 43 44 45

REM Status / tracking files
set STATUS_DIR=results\run_status
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set CURRENT_FILE=%STATUS_DIR%\CKA_GNN_CURRENT_RUN.txt
set HISTORY_FILE=%STATUS_DIR%\cka_gnn_run_history.csv
set LAST_OK_FILE=%STATUS_DIR%\CKA_GNN_LAST_COMPLETED.txt
set FAILED_FILE=%STATUS_DIR%\CKA_GNN_FAILED_RUNS.txt

REM Create history CSV header if missing
if not exist "%HISTORY_FILE%" (
    echo timestamp,status,dataset,llm_name,peft_type,split,seed,node_subset,exit_code> "%HISTORY_FILE%"
)

echo ==========================================
echo DIVERGE GNN-embedding CKA multi-model sweep
echo Status dir: %STATUS_DIR%
echo Current run file: %CURRENT_FILE%
echo History CSV: %HISTORY_FILE%
echo ==========================================
echo.

for %%d in (%DATASETS%) do (
    for %%m in (%LLM_NAMES%) do (
        for %%s in (%SEEDS%) do (
            set "RUN_ID=dataset=%%d | llm=%%m | peft=%PEFT_TYPE% | split=%SPLIT% | seed=%%s | nodes=%NODE_SUBSET%"
            set "TS=%DATE% %TIME%"

            echo ==========================================
            echo START: !RUN_ID!
            echo Time: !TS!
            echo ==========================================

            REM Write identity of the active run (overwrite)
            (
                echo status=RUNNING
                echo started=!TS!
                echo dataset=%%d
                echo llm_name=%%m
                echo peft_type=%PEFT_TYPE%
                echo split=%SPLIT%
                echo seed=%%s
                echo node_subset=%NODE_SUBSET%
                echo command=python -m experiments.cka_for_gnn_embeddings --dataset_name %%d --llm_name %%m --peft_type %PEFT_TYPE% --seed %%s --split %SPLIT% --approaches %APPROACHES% --node_subset %NODE_SUBSET% --csv_path %CKA_PAIRWISE_CSV% --summary_csv_path %CKA_SUMMARY_CSV% --per_approach_csv_path %CKA_PER_APPROACH_CSV% %EXTRA_ARGS%
            ) > "%CURRENT_FILE%"

            python -m experiments.cka_for_gnn_embeddings --dataset_name "%%d" --llm_name "%%m" --peft_type "%PEFT_TYPE%" --seed %%s --split %SPLIT% --approaches "%APPROACHES%" --node_subset %NODE_SUBSET% --csv_path "%CKA_PAIRWISE_CSV%" --summary_csv_path "%CKA_SUMMARY_CSV%" --per_approach_csv_path "%CKA_PER_APPROACH_CSV%" %EXTRA_ARGS%
            set EXIT_CODE=!ERRORLEVEL!
            set "TS_END=%DATE% %TIME%"

            if !EXIT_CODE! NEQ 0 (
                echo ERROR: !RUN_ID!  exit_code=!EXIT_CODE!
                echo !TS_END!,FAILED,%%d,%%m,%PEFT_TYPE%,%SPLIT%,%%s,%NODE_SUBSET%,!EXIT_CODE!>> "%HISTORY_FILE%"
                echo !TS_END! FAILED !RUN_ID! exit=!EXIT_CODE!>> "%FAILED_FILE%"
                (
                    echo status=FAILED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo peft_type=%PEFT_TYPE%
                    echo split=%SPLIT%
                    echo seed=%%s
                    echo node_subset=%NODE_SUBSET%
                    echo exit_code=!EXIT_CODE!
                ) > "%CURRENT_FILE%"
                echo Continuing to next run despite failure...
                echo.
            ) else (
                echo OK: !RUN_ID!
                echo !TS_END!,OK,%%d,%%m,%PEFT_TYPE%,%SPLIT%,%%s,%NODE_SUBSET%,0>> "%HISTORY_FILE%"
                (
                    echo status=COMPLETED
                    echo started=!TS!
                    echo finished=!TS_END!
                    echo dataset=%%d
                    echo llm_name=%%m
                    echo peft_type=%PEFT_TYPE%
                    echo split=%SPLIT%
                    echo seed=%%s
                    echo node_subset=%NODE_SUBSET%
                    echo exit_code=0
                ) > "%LAST_OK_FILE%"
                copy /Y "%LAST_OK_FILE%" "%CURRENT_FILE%" >nul
                echo.
            )
        )
    )
)

echo ==========================================
echo GNN CKA sweep finished.
echo Check:
echo   %CKA_PAIRWISE_CSV%
echo   %CKA_PER_APPROACH_CSV%
echo   %CKA_SUMMARY_CSV%
echo   results\cka_gnn\*_cka_gnn_heatmap.png
echo   %HISTORY_FILE%
echo   %FAILED_FILE%  ^(if any failures^)
echo ==========================================
(
    echo status=IDLE
    echo finished=%DATE% %TIME%
    echo note=all queued GNN CKA runs finished; see cka_gnn_run_history.csv
) > "%CURRENT_FILE%"

pause
endlocal
