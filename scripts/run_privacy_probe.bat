@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Keyword-inference privacy probe on saved DIVERGE embeddings (Reviewer 1, Comment 8):
REM   experiments/privacy/privacy_probe.py
REM
REM A linear probe predicts the top-100 content words of each node text from the stored embeddings
REM (mean test ROC-AUC, 0.5 = no leakage), against a label-only control, for the base model and the
REM five LoRA inits, plus Gaussian-noise defense (sigma 0.5/1/2) with the GraphSAGE accuracy on e'.
REM
REM LLM      cache/hyperparameter prefix: llama_3.2_1B, smollm2_1.7B, gemma2_2B
REM DATASETS comma-separated
REM SEEDS    comma-separated, used for every dataset; leave empty for the per-dataset defaults
REM          (cora/citeseer 42-45, pubmed 43, wikics/reddit/computer 42)
REM
REM Needs artifacts/cache/<LLM>_<ds>_seqcls_pool-mean_based_model.pt, the per-seed
REM ..._lora_init-<init>_pool-mean_seed<s>_semi_supervised.pt caches and
REM gnn_hyperparameters/<ds>_<LLM>_SAGE_seed<s>.json (falls back to seed42).
REM
REM Resumes: a dataset whose requested seeds are already in the results JSON is skipped.
REM
REM Outputs: results/privacy/privacy_probe_seeds.json (llama_3.2_1B) or
REM          results/privacy/privacy_probe_seeds_<LLM>.json, plus a matching .log
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set LLM=llama_3.2_1B
set DATASETS=cora,citeseer
set SEEDS=42,43,44,45

set SEED_ARG=
if not "%SEEDS%"=="" set SEED_ARG=--seeds %SEEDS%

if "%LLM%"=="llama_3.2_1B" (
    set LOG=results\privacy\privacy_probe_seeds.log
) else (
    set LOG=results\privacy\privacy_probe_seeds_%LLM%.log
)
if not exist results\privacy mkdir results\privacy

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8

echo ==========================================
echo Privacy probe: llm=%LLM% datasets=[%DATASETS%] seeds=[%SEEDS%]
echo Log: !LOG!
echo Time: !DATE! !TIME!
echo ==========================================
python experiments\privacy\privacy_probe.py --llm %LLM% --datasets %DATASETS% %SEED_ARG% >> "!LOG!" 2>&1
set EXIT_CODE=!ERRORLEVEL!
if !EXIT_CODE! NEQ 0 (
    echo ERROR: exit_code=!EXIT_CODE!, see !LOG!
) else (
    echo OK: see results\privacy and !LOG!
)
pause
endlocal
exit /b %EXIT_CODE%
