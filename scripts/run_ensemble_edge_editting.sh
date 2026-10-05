#!/bin/bash

# Script to run ensemble edge editing analysis for multiple datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Default values
LLM_NAME="llama_3.2_1B"
PEFT_TYPE="lora"

# List of datasets to process
DATASETS=("cora" "citeseer" "pubmed" "wikics")

# Loop through each dataset
for dataset_name in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $dataset_name"
    echo "=========================================="
    
    python -m experiments.edge_editing \
        --dataset_name "$dataset_name" \
        --llm_name "$LLM_NAME" \
        --peft_type "$PEFT_TYPE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed $dataset_name"
    else
        echo "Error processing $dataset_name"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="
