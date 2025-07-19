export CUDA_VISIBLE_DEVICES=7
export DEEPSPEED_PORT=23456

# ("DeepSeek_R1_Distill_Qwen_7B_ft_pt" "DeepSeek_R1_Distill_Qwen_7B_ft" "MMedIns_Llama3_8B_ft" "MMedIns_Llama3_8B_ft_pt")
BASE_MODEL="DeepSeek_R1_Distill_Qwen_7B_ft_pt"

# ["patient", "expert", "teacher"]
ROLE="patient"

# [32, 64]
LORA_RANK=32

CFG_PATH="./configs/${BASE_MODEL}/${ROLE}_lora${LORA_RANK}.py"
WORK_DIR="work_dirs/${BASE_MODEL}/${ROLE}/lora_${LORA_RANK}"

xtuner train $CFG_PATH --deepspeed deepspeed_zero3 --work-dir $WORK_DIR
