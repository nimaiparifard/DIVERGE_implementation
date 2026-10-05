@echo off

REM Fast homophily / structural metrics experiment (no GNN training)

REM Run from repository root.



set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR%.."



set LLM_NAME=llama_3.2_1B

set PEFT_TYPE=lora

set DATASETS=cora pubmed citeseer wikics



for %%d in (%DATASETS%) do (

    echo ==========================================

    echo Homophily analysis: %%d

    echo ==========================================



    python "%SCRIPT_DIR%experiments\run_homophily_structural_experiment.py" ^

        --dataset_name "%%d" ^

        --llm_name "%LLM_NAME%" ^

        --peft_type "%PEFT_TYPE%" ^

        --retrained_with_gnn_mistakes ^

        --use_all_nodes



    if errorlevel 1 (

        echo Error processing %%d

        exit /b 1

    )



    echo Successfully completed %%d

    echo.

)



echo ==========================================

echo All homophily experiments completed!

echo Results: cosine_similarity_enhancer\experiment_results

echo ==========================================

pause

