export CUDA_VISIBLE_DEVICES=1

# ["DeepSeek_R1_Distill_Qwen_7B", "MMedIns_Llama3_8B"]
BASE_MODEL="DeepSeek_R1_Distill_Qwen_7B"

# ["patient", "expert", "teacher"]
ROLE="patient"

CFG_PATH="./configs/${BASE_MODEL}/${ROLE}.py"

xtuner train $CFG_PATH --deepspeed deepspeed_zero3
