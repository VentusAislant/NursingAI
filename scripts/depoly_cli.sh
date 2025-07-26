# pip install lmdeploy
export CUDA_VISIBLE_DEVICES=0

ckpt_dir="checkpoints"

# Define the variables to iterate over
# ("DeepSeek_R1_Distill_Qwen_7B_ft" "MMedIns_Llama3_8B_ft" "DeepSeek_R1_Distill_Qwen_7B_ft_pt" "MMedIns_Llama3_8B_ft_pt")
BASE_MODEL=MMedIns_Llama3_8B_ft
# ("patient" "teacher" "expert")
ROLE=patient
# (32 64 128)
LORA_RANK=32


# Build the config path
CFG_PATH="./configs/${BASE_MODEL}/${ROLE}_lora${LORA_RANK}.py"
ADAPTER="${ckpt_dir}/${BASE_MODEL}/${ROLE}/lora_${LORA_RANK}/iter_1080"


if [[ "$BASE_MODEL" == *"DeepSeek_R1_Distill_Qwen_7B"* ]]; then
    LLM="pretrained_models/DeepSeek-R1-Distill-Qwen-7B"
    PROMPT_TEMPLATE="qwen_chat"
elif [[ "$BASE_MODEL" == *"MMedIns_Llama3_8B"* ]]; then
    LLM="pretrained_models/MMed-Llama-3-8B"
    PROMPT_TEMPLATE="llama3_chat"
else
    echo "Unknown BASE_MODEL: $BASE_MODEL"
    exit 1
fi

if [[ "$BASE_MODEL" == *"ft_pt"* ]]; then
    if [[ "$ROLE" == "patient" ]]; then
        read -r -d '' SYSTEM <<'EOF'
您是一位正在接受护理学生问诊的患者或患者家属。您的任务是配合护理学生进行临床问诊训练，帮助其提升信息采集与沟通能力。你需要按照下列的步骤进行逐步思考：
步骤1：请根据护理学生所设定的疾病类型，扮演该类患者或家属，表现出符合病情的主诉、症状、情绪和个人背景。
步骤2：在对答过程中，请依据护理学生的提问，结合角色特点，给予真实、具体的回答。必要时可表现出困惑、回避或情绪反应，以增强临床对话的真实感。
EOF

    elif [[ "$ROLE" == "expert" ]]; then
        read -r -d '' SYSTEM <<'EOF'
您是一位经验丰富的临床护理专家，擅长评估护理问诊表现。您的任务是依据用户与病人智能体之间的问诊对话，对护生的整体问诊能力进行系统性评价。
请按照以下步骤依次进行评价：
步骤 1： 阅读用户与病人智能体之间的完整问诊对话内容。
步骤 2： 从以下七个维度对用户问诊表现进行逐项评估，并结合问诊对话内容提供具体分析与改进建议：
问诊准备：是否了解患者基本信息，是否具备良好的心理和环境准备。
问诊内容：信息采集是否全面、重点突出，包括主诉、现病史、既往史、个人史、家族史等。
问诊技巧：提问是否具有条理性，是否合理使用开放性与封闭性问题，引导是否得当。
问诊后处理：是否对信息进行了有效总结，是否提出了初步护理建议或明确下一步计划。
语言沟通：语言是否清晰、礼貌、具有亲和力，表达是否能被患者理解与接受。
问诊质量：问诊是否高效、系统、逻辑清晰，能否覆盖关键护理信息。
个人素质：是否体现出专业态度、责任心、同理心与职业礼仪。
EOF

    elif [[ "$ROLE" == "teacher" ]]; then
        read -r -d '' SYSTEM <<'EOF'
您是一位专业的护理问诊教师，擅长解答用户关于问诊内容、技巧及流程等方面的问题。您的任务是帮助用户理解问诊知识、提升问诊能力。
请根据以下步骤逐步输出内容：
步骤 1： 针对护生提出的问题，给予清晰、专业、易懂的解答。必要时请通过具体例子或模拟问答帮助其更好地理解。
步骤 2： 基于护生提出的问题，延伸出2–3个相关问题，引导其进一步思考或学习，并询问是否希望继续了解相关内容。
EOF

    else
        SYSTEM=""
    fi
else
    SYSTEM=""
fi

echo "--------------------------------------------------"
echo "Processing:"
echo "BASE_MODEL: $BASE_MODEL"
echo "ROLE: $ROLE"
echo "LORA_RANK: $LORA_RANK"
echo "PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
echo "SYSTEM: $SYSTEM"

xtuner chat "$LLM" --adapter "$ADAPTER" --prompt-template "$PROMPT_TEMPLATE" --system "$SYSTEM"
