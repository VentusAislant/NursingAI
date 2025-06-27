#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# ["lawgpt", "lawyer_llama", "internlm2_chat", "hanfei", "chatlaw"]
model_type="hanfei"
# ["plaintiff", "defendant"]
#choice="plaintiff"

function eval() {
  choice=$1
  MODEL_BASE="./pretrained_models/internlm2-chat-7b"
  MODEL_ADAPTERS="none"

  TEST_FILE_PATH="./data/train_data/test_${choice}_model.json"
  result_save_path="./results/${choice}"

  if [[ "$model_type" == "lawgpt" ]]; then
    template="lawgpt"
    result_save_path="${result_save_path}_${model_type}_result.txt"
    MODEL_ADAPTERS="./pretrained_models/law_llms/lawgpt/lawgpt-lora-7b"
    MODEL_BASE="./pretrained_models/law_llms/chinese_llama/chinese-llama-7b-merged"
  elif [[ "$model_type" == "lawyer_llama" ]]; then
    template="lawyer_llama"
    result_save_path="${result_save_path}_${model_type}_result.txt"
    MODEL_BASE="./pretrained_models/law_llms/lawyer-llama/lawyer-llama-13b-v2"
  elif [[ "$model_type" == "internlm2_chat" ]]; then
    template="internlm2_chat"
    result_save_path="${result_save_path}_${model_type}_result.txt"
  elif [[ "$model_type" == "hanfei" ]]; then
    template="default"
    result_save_path="${result_save_path}_${model_type}_result.txt"
    MODEL_BASE="./pretrained_models/law_llms/hanfei/hanfei/model/hanfei-1.0"
  elif [[ "$model_type" == "chatlaw" ]]; then
    template="internlm2_chat"
    result_save_path="${result_save_path}_${model_type}_result.txt"
    MODEL_BASE="./pretrained_models/law_llms/chatlaw/ChatLaw2_plain_7B"
  fi

  # Ensure the test file exists
  if [ ! -f "$TEST_FILE_PATH" ]; then
    echo "Test file $TEST_FILE_PATH not found!"
    exit 1
  fi

  if [[ "$MODEL_ADAPTERS" == "none" ]]; then
    python ./eval_src/eval.py \
        --model_name_or_path "$MODEL_BASE" \
        --test_file_path "$TEST_FILE_PATH" \
        --template "$template" \
        --result_save_path "$result_save_path"
  else
    python ./eval_src/eval.py \
        --model_name_or_path "$MODEL_BASE" \
        --checkpoint_dir "$MODEL_ADAPTERS" \
        --test_file_path "$TEST_FILE_PATH" \
        --template "$template" \
        --result_save_path "$result_save_path"
  fi
}

eval "plaintiff"
eval "defendant"
