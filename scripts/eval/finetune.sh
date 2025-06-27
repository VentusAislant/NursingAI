#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python src/train_my.py \
    --stage 'sft' \
    --model_name_or_path /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-base \
    --do_train \
    --dataset_dir ./data \
    --template ziya \
    --dataset acp_knowledge \
    --finetuning_type lora \
    --lora_rank 32 \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --resume_lora_training False \
    --checkpoint_dir /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_13,/repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_26 \
    --output_dir ./checkpoints/acp_knowledge_zhongjing \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 7e-4 \
    --num_train_epochs 10.0 \
    --plot_loss \
    --bf16