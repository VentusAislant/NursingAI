#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train_rm.py \
    --model_name_or_path /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-base \
    --do_train \
    --dataset zhongjing_rlhf \
    --finetuning_type lora \
    --lora_rank 32 \
    --resume_lora_training False \
    --checkpoint_dir /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_13,/repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_26 \
    --output_dir ./checkpoints/acp_zhongjing \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate 7e-4 \
    --num_train_epochs 10.0 \
    --plot_loss \
    --fp16 \
    --dev_ratio 0.1