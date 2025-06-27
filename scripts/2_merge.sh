# pip install lmdeploy
export CUDA_VISIBLE_DEVICES=0

# ["plaintiff", "defendant"]
choice="plaintiff"

if [ "$choice" = "plaintiff" ]; then
  adapter="./checkpoints/plaintiff_model"
  target_dir="./checkpoints/plaintiff_full_model"
elif [ "$choice" = "defendant" ]; then
  adapter="./checkpoints/defendant_model"
  target_dir="./checkpoints/defendant_full_model"
fi

xtuner convert merge \
    ./pretrained_models/internlm2-chat-7b\
    $adapter \
    $target_dir \
    --max-shard-size 2GB