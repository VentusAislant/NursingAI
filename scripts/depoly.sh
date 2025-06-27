# pip install lmdeploy
export CUDA_VISIBLE_DEVICES=0

# ["plaintiff", "defendant"]
choice="defendant"

if [ "$choice" = "plaintiff" ]; then
  ADAPTER="./checkpoints/plaintiff_model"
elif [ "$choice" = "defendant" ]; then
  ADAPTER="./checkpoints/defendant_model"
fi

LLM="./pretrained_models/internlm2-chat-7b"
PROMPT_TEMPLATE="internlm2_chat"
SYSTEM_TEMPLATE=""
xtuner chat $LLM --adapter $ADAPTER --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
