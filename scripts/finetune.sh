export CUDA_VISIBLE_DEVICES=7
export DEEPSPEED_PORT=23456

# ["DeepSeek_R1_Distill_Qwen_7B", "MMedIns_Llama3_8B"]
BASE_MODEL="DeepSeek_R1_Distill_Qwen_7B"
#BASE_MODEL="MMedIns_Llama3_8B"

# ["patient", "expert", "teacher"]
ROLE="expert"

# [32, 64]
LORA_RANK=64

CFG_PATH="./configs/${BASE_MODEL}/${ROLE}_lora${LORA_RANK}.py"
WORK_DIR="work_dirs/${BASE_MODEL}/${ROLE}/lora_${LORA_RANK}"

xtuner train $CFG_PATH --deepspeed deepspeed_zero3 --work-dir $WORK_DIR

# ["DeepSeek_R1_Distill_Qwen_7B", "MMedIns_Llama3_8B"]
BASE_MODEL="MMedIns_Llama3_8B"
#BASE_MODEL="MMedIns_Llama3_8B"

# ["patient", "expert", "teacher"]
ROLE="expert"

# [32, 64]
LORA_RANK=64

CFG_PATH="./configs/${BASE_MODEL}/${ROLE}_lora${LORA_RANK}.py"
WORK_DIR="work_dirs/${BASE_MODEL}/${ROLE}/lora_${LORA_RANK}"

xtuner train $CFG_PATH --deepspeed deepspeed_zero3 --work-dir $WORK_DIR
