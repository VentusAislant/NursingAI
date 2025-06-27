#!/bin/bash

MODEL_BASE="./pretrained_models/internlm2-chat-7b"
MODEL_ADAPTERS="./checkpoints/model_yuangao"

CUDA_VISIBLE_DEVICES=0 python ./eval_src/eval.py \
    --model_name_or_path $MODEL_BASE \
    --checkpoint_dir $MODEL_ADAPTERS \
    --test_file_path ./data/eval_data2.json \
    --template internlm2_chat \
    --system_prompt 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n'