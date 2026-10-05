@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM Fine-tuned vs base LLM embeddings, McNemar test on all datasets (paper Table tab:mcnemar):
REM   experiments/hypothesis_testing_fintuned_unfinetuned.py
REM
REM Per dataset: GNN on base embeddings vs GNN on the EVA-initialised LoRA view (FINETUNED; best_val picks the
REM view by validation accuracy instead), both on the split the adapter was trained on
REM (cora/citeseer 42-45, wikics/computer/reddit 42, pubmed 43, arxiv 50).
REM One python call, because the Bonferroni / Holm adjustment is over all datasets together.
REM GNNs are cached in results/ensemble_experiments/gnn_views (shared with the significance tests).
REM
REM Needs the base-model embedding cache artifacts/cache/<llm>_<dataset>_seqcls_pool-mean_based_model.pt
REM for every dataset (scripts/run_tape_base.bat builds it as the 'text' view).
REM
REM Outputs: results/finetuning_significance/{mcnemar.csv, mcnemar_all_views.csv, report.txt}
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set DATASETS=cora,citeseer,pubmed,wikics,arxiv,computer,reddit
REM best_val (default) or a fixed init: pissa orthogonal gaussian loftq eva
set FINETUNED=eva

set WANDB_DISABLED=true
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8

echo ==========================================
echo McNemar fine-tuned vs base: datasets=[%DATASETS%] finetuned=%FINETUNED%
echo Time: !DATE! !TIME!
echo ==========================================
python -m experiments.hypothesis_testing_fintuned_unfinetuned --datasets %DATASETS% --finetuned %FINETUNED%
set EXIT_CODE=!ERRORLEVEL!
if !EXIT_CODE! NEQ 0 (
    echo ERROR: exit_code=!EXIT_CODE!
) else (
    echo OK: see results\finetuning_significance\report.txt
)
pause
endlocal
exit /b %EXIT_CODE%
