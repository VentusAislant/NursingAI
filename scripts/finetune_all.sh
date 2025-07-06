#!/bin/bash

# Set CUDA device (adjust as needed for your setup)
export CUDA_VISIBLE_DEVICES=1

# Define the variables to iterate over
BASE_MODELS=("DeepSeek_R1_Distill_Qwen_7B" "MMedIns_Llama3_8B")
ROLES=("patient" "teacher" "expert")
LORA_RANKS=(32)

# Loop through all combinations
for lora_rank in "${LORA_RANKS[@]}"; do
    for role in "${ROLES[@]}"; do
        for base_model in "${BASE_MODELS[@]}"; do
            echo "--------------------------------------------------"
            echo "Running training with:"
            echo "BASE_MODEL: $base_model"
            echo "ROLE: $role"
            echo "LORA_RANK: $lora_rank"

            # Build the config path
            CFG_PATH="./configs/${base_model}/${role}_lora${lora_rank}.py"
            
            # Run the training command
            echo "Executing: xtuner train $CFG_PATH --deepspeed deepspeed_zero3"
            xtuner train "$CFG_PATH" --deepspeed deepspeed_zero3 --work-dir "work_dirs/${base_model}/${role}/lora_${lora_rank}"
            
            # Add a separator between runs
            echo "--------------------------------------------------"
        done
    done
done

echo "All training jobs completed!"