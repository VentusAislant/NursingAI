# pip install lmdeploy
# pip install lmdeploy
export CUDA_VISIBLE_DEVICES=0

# ["plaintiff", "defendant"]
choice="defendant"

if [ "$choice" = "plaintiff" ]; then
  NAME_OR_PATH_TO_LLM="./checkpoints/plaintiff_model"
elif [ "$choice" = "defendant" ]; then
  NAME_OR_PATH_TO_LLM="./checkpoints/defendant_model"
fi

PROMPT_TEMPLATE="internlm2_chat"

#python -m lmdeploy.pytorch.chat ${NAME_OR_PATH_TO_LLM} \
#    --prompt-template internlm2_chat \
#    --max_new_tokens 256 \
#    --temperture 0.8 \
#    --top_p 0.95 \
#    --seed 0

xtuner chat ${NAME_OR_PATH_TO_LLM} --prompt-template $PROMPT_TEMPLATE