@echo off
REM =============================================================================
REM Run LoRA-init fine-tuning jobs SIMULTANEOUSLY on the same GPU.
REM Same dataset + same LLM, two init_weight_approach values x the seed list.
REM Each job opens in its own console window so you can watch it live.
REM
REM Only do this if a single run does not use all your VRAM (e.g. ~1/3 each),
REM since both jobs share the same GPU's compute + memory at the same time.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set DATASET=cora
set LLM_NAME=smollm2_1.7B
set PEFT_TYPE=lora
set SPLIT=1

REM Two init approaches to run in parallel (choices: pissa gaussian eva loftq orthogonal)
set INIT_A=pissa
set INIT_B=gaussian
set SEEDS=42 43 44 45

REM Disable HF Trainer logging integrations (wandb prompts block unattended runs)
set WANDB_DISABLED=true
set WANDB_MODE=disabled
set HF_HUB_DISABLE_TELEMETRY=1

echo ==========================================
echo Launching parallel LoRA-init runs
echo   dataset=%DATASET%  llm=%LLM_NAME%  peft=%PEFT_TYPE%  split=%SPLIT%
echo   Job A: init=%INIT_A%
echo   Job B: init=%INIT_B%
echo   seeds=%SEEDS%
echo ==========================================
echo.

for %%s in (%SEEDS%) do (
    start "LoRA-init %INIT_A% seed%%s" cmd /k python -m train_llm.train_other_lora_init_apporaches --dataset_name "%DATASET%" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%INIT_A%" --split %SPLIT% --seed %%s
    start "LoRA-init %INIT_B% seed%%s" cmd /k python -m train_llm.train_other_lora_init_apporaches --dataset_name "%DATASET%" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%INIT_B%" --split %SPLIT% --seed %%s
)

echo Jobs launched in separate windows (2 inits x 4 seeds = 8 windows).
echo Each window stays open after finishing (cmd /k) so you can check for errors.
echo Watch VRAM usage with: nvidia-smi -l 2
echo If VRAM is tight, shrink SEEDS or run only INIT_A or INIT_B.
echo.
pause
