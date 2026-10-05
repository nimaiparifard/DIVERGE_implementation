@echo off
REM =============================================================================
REM Efficient DIVERGE: LoRA fine-tune + in-process embedding cache for all
REM 5 init approaches, running as many jobs in parallel as the GPU allows.
REM
REM   1) checks whether Flash-Attention 2 is installed (falls back to SDPA)
REM   2) probes the GPU to pick micro-batch size + number of parallel jobs
REM   3) runs the jobs through a pool (logs in logs\train_eff\)
REM
REM Set JOBS / MICRO_BS below to override the automatic plan.
REM Caches go to artifacts_eff\cache by default; add
REM   --cache_root ./artifacts/cache
REM to EXTRA_ARGS to feed the GNN/ensemble code directly (overwrites old caches).
REM =============================================================================
setlocal
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set DATASET=cora
set LLM_NAME=llama_3.2_1B
set PEFT_TYPE=lora
set SPLIT=1
set SEED=42
set INITS=pissa,gaussian,eva,loftq,orthogonal
set JOBS=
set MICRO_BS=
set EXTRA_ARGS=

echo ==========================================
echo [1/3] Flash-Attention 2 check
echo ==========================================
python -c "from transformers.utils import is_flash_attn_2_available as f; import sys; ok=f(); print('flash_attn 2 installed:', ok); print('-> using', 'flash_attention_2' if ok else 'PyTorch SDPA (fallback)'); sys.exit(0)"
echo.

echo ==========================================
echo [2/3] Estimating parallelism for %DATASET% / %LLM_NAME%
echo ==========================================
python -m train_eff_llm.estimate_parallel --dataset_name %DATASET% --llm_name %LLM_NAME% --peft_type %PEFT_TYPE% --out outputs_eff\parallel_plan.json
if errorlevel 1 (
    echo Estimation failed. Set JOBS and MICRO_BS manually in this file.
    if "%JOBS%"=="" set JOBS=1
    if "%MICRO_BS%"=="" set MICRO_BS=2
    goto run
)
for /f "usebackq tokens=1,2" %%a in (`python -c "import json;p=json.load(open(r'outputs_eff\parallel_plan.json'));print(p['jobs'],p['micro_bs'])"`) do (
    if "%JOBS%"=="" set JOBS=%%a
    if "%MICRO_BS%"=="" set MICRO_BS=%%b
)

:run
echo.
echo ==========================================
echo [3/3] Running %INITS%
echo       dataset=%DATASET% llm=%LLM_NAME% seed=%SEED% split=%SPLIT%
echo       parallel jobs=%JOBS%  micro_bs=%MICRO_BS%
echo ==========================================
python -m train_eff_llm.run_parallel --dataset_name %DATASET% --llm_name %LLM_NAME% --peft_type %PEFT_TYPE% --inits %INITS% --seed %SEED% --split %SPLIT% --jobs %JOBS% --micro_bs %MICRO_BS% %EXTRA_ARGS%

echo.
echo Efficiency rows: results\efficiency\lora_finetune_efficiency_eff.csv
echo Watch VRAM with: nvidia-smi -l 2
endlocal
pause
