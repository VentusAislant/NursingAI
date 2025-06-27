#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

#--checkpoint_dir /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_13,/repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_26 \


python ./src/cli_demo.py \
    --model_name_or_path /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-base \
    --checkpoint_dir /repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_13,/repository/users/wind/models/zhongjing/Zhongjing-LLaMA-lora/zhongjing_7_26,./checkpoints/acp_knowledge_zhongjing \
    --template ziya \
    --repetition_penalty 1.2
