export CUDA_VISIBLE_DEVICES=1

# ["plaintiff", "defendant"]
choice="defendant"
#CKPT_PATH="/home/zhanghj/projects/law_llm/InternLM/work_dirs/defendant_model/iter_2300.pth"
#CKPT_PATH="/home/zhanghj/projects/law_llm/InternLM/work_dirs/plaintiff_model/iter_2300.pth"
CKPT_PATH="/home/zhanghj/projects/law_llm/InternLM/work_dirs/plaintiff_model/iter_3180.pth"
CKPT_PATH="/home/zhanghj/projects/law_llm/InternLM/work_dirs/defendant_model/iter_3170.pth"

if [ "$choice" = "plaintiff" ]; then
  CFG_PATH="./cfgs/plaintiff_model.py"
  TARGET_PATH="./checkpoints/plaintiff_model"
elif [ "$choice" = "defendant" ]; then
  CFG_PATH="./cfgs/defendant_model.py"
  TARGET_PATH="./checkpoints/defendant_model"
fi

xtuner convert pth_to_hf $CFG_PATH $CKPT_PATH $TARGET_PATH