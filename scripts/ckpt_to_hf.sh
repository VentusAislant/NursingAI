#!/bin/bash

# Set CUDA device (adjust as needed for your setup)
export CUDA_VISIBLE_DEVICES=0
export DEEPSPEED_PORT=12345

work_dir="work_dirs"
model_save_dir="checkpoints"

# Define the variables to iterate over
#BASE_MODELS=("DeepSeek_R1_Distill_Qwen_7B_ft" "MMedIns_Llama3_8B_ft" "DeepSeek_R1_Distill_Qwen_7B_ft_pt" "MMedIns_Llama3_8B_ft_pt")
BASE_MODELS=("MMedIns_Llama3_8B_ft" "MMedIns_Llama3_8B_ft_pt")
ROLES=("patient" "teacher" "expert")
LORA_RANKS=(32 64 128)

# Loop through all combinations
for lora_rank in "${LORA_RANKS[@]}"; do
    for role in "${ROLES[@]}"; do
        for base_model in "${BASE_MODELS[@]}"; do
            echo "--------------------------------------------------"
            echo "Processing:"
            echo "BASE_MODEL: $base_model"
            echo "ROLE: $role"
            echo "LORA_RANK: $lora_rank"

            # Build the config path
            CFG_PATH="./configs/${base_model}/${role}_lora${lora_rank}.py"
            CUR_PATH="${work_dir}/${base_model}/${role}/lora_${lora_rank}"

            # 检查路径是否存在
            if [ ! -d "$CUR_PATH" ]; then
                echo "Directory not found: $CUR_PATH"
                continue
            fi

            # Find all *.pth files under current path
            for CKPT_FILE in "$CUR_PATH"/*.pth; do
                # 提取 checkpoint 文件名（不带路径和后缀）
                CKPT_NAME=$(basename "$CKPT_FILE" .pth)
                TARGET_PATH="${model_save_dir}/${base_model}/${role}/lora_${lora_rank}/${CKPT_NAME}"

                # Run the xtuner conversion
                echo "running: xtuner convert pth_to_hf $CFG_PATH $CKPT_FILE $TARGET_PATH"
                xtuner convert pth_to_hf $CFG_PATH $CKPT_FILE $TARGET_PATH
                echo "--------------------------------------------------"
            done
        done
    done
done

echo "✅ All conversion jobs completed!"
