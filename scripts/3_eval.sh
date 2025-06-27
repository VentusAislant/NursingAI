#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_BASE="./pretrained_models/internlm2-chat-7b"

# ["plaintiff", "defendant"]
choice="defendant"
MODEL_ADAPTERS="./checkpoints/${choice}_model"
TEST_FILE_PATH="./data/train_data/test_${choice}_model.json"

python ./eval_src/eval.py \
    --model_name_or_path $MODEL_BASE \
    --checkpoint_dir $MODEL_ADAPTERS \
    --test_file_path $TEST_FILE_PATH \
    --template internlm2_chat \
    --system_prompt 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n' \
    --result_save_path ./results/${choice}_model_test_result.txt