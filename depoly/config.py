"""
NursingAI 系统配置文件
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Base模型配置
BASE_MODELS = {
    "MMed-Llama-3-8B": {
        "path": os.path.join(PRETRAINED_MODELS_DIR, "MMed-Llama-3-8B"),
        "type": "llama",
        "description": "医疗领域的Llama3模型"
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "path": os.path.join(PRETRAINED_MODELS_DIR, "DeepSeek-R1-Distill-Qwen-7B"),
        "type": "qwen",
        "description": "DeepSeek蒸馏版Qwen模型"
    }
}

# Lora模型配置
LORA_MODELS = {
    "MMed-Llama-3-8B": {
        "MMedIns_Llama3_8B_ft": {
            "path": os.path.join(CHECKPOINTS_DIR, "MMedIns_Llama3_8B_ft"),
            "description": "Llama3的医疗指令微调模型"
        },
        "MMedIns_Llama3_8B_ft_pt": {
            "path": os.path.join(CHECKPOINTS_DIR, "MMedIns_Llama3_8B_ft_pt"),
            "description": "Llama3的医疗指令微调模型（PT版本）"
        }
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "DeepSeek_R1_Distill_Qwen_7B_ft": {
            "path": os.path.join(CHECKPOINTS_DIR, "DeepSeek_R1_Distill_Qwen_7B_ft"),
            "description": "Qwen的医疗指令微调模型"
        },
        "DeepSeek_R1_Distill_Qwen_7B_ft_pt": {
            "path": os.path.join(CHECKPOINTS_DIR, "DeepSeek_R1_Distill_Qwen_7B_ft_pt"),
            "description": "Qwen的医疗指令微调模型（PT版本）"
        }
    }
}

# 角色配置
ROLES = {
    "expert": {
        "name": "专家",
        "description": "医疗专家角色，提供专业的医疗建议"
    },
    "teacher": {
        "name": "教师",
        "description": "教学角色，提供医疗知识教育"
    },
    "patient": {
        "name": "患者",
        "description": "患者角色，模拟患者视角的对话"
    }
}

# Lora Rank配置
LORA_RANKS = {
    "lora_32": {
        "rank": 32,
        "description": "32维度的lora适配器，内存占用较小"
    },
    "lora_64": {
        "rank": 64,
        "description": "64维度的lora适配器，平衡性能和内存"
    },
    "lora_128": {
        "rank": 128,
        "description": "128维度的lora适配器，性能最佳但内存占用较大"
    }
}



# 模型加载配置
MODEL_CONFIG = {
    "torch_dtype": "float16",
    "device_map": "auto",  # 自动选择最佳设备（优先GPU）
    "trust_remote_code": True,
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
    "use_cache": True,  # 启用缓存提高性能
    "streaming": True,  # 启用流式输出
    "force_cpu": False,  # 是否强制使用CPU（设为True可强制使用CPU）
    "low_cpu_mem_usage": True,  # 低CPU内存使用
    "offload_folder": "offload"  # offload文件夹
}

# Web界面配置
WEB_CONFIG = {
    "server_name": "0.0.0.0",  # 改为0.0.0.0允许外部访问
    "server_port": 12489,
    "share": False,  # 启用Gradio分享功能
    "debug": True,
    "title": "NursingAI 智能聊天系统",
    "theme": "soft",
    "show_error": True
}

# 角色特定的系统提示词
ROLE_SYSTEM_PROMPTS = {
    "patient": """您是一位正在接受护理学生问诊的患者或患者家属。您的任务是配合护理学生进行临床问诊训练，帮助其提升信息采集与沟通能力。你需要按照下列的步骤进行逐步思考：
步骤1：请根据护理学生所设定的疾病类型，扮演该类患者或家属，表现出符合病情的主诉、症状、情绪和个人背景。
步骤2：在对答过程中，请依据护理学生的提问，结合角色特点，给予真实、具体的回答。必要时可表现出困惑、回避或情绪反应，以增强临床对话的真实感。""",
    
    "expert": """您是一位经验丰富的临床护理专家，擅长评估护理问诊表现。您的任务是依据用户与病人智能体之间的问诊对话，对护生的整体问诊能力进行系统性评价。
请按照以下步骤依次进行评价：
步骤 1： 阅读用户与病人智能体之间的完整问诊对话内容。
步骤 2： 从以下七个维度对用户问诊表现进行逐项评估，并结合问诊对话内容提供具体分析与改进建议：
问诊准备：是否了解患者基本信息，是否具备良好的心理和环境准备。
问诊内容：信息采集是否全面、重点突出，包括主诉、现病史、既往史、个人史、家族史等。
问诊技巧：提问是否具有条理性，是否合理使用开放性与封闭性问题，引导是否得当。
问诊后处理：是否对信息进行了有效总结，是否提出了初步护理建议或明确下一步计划。
语言沟通：语言是否清晰、礼貌、具有亲和力，表达是否能被患者理解与接受。
问诊质量：问诊是否高效、系统、逻辑清晰，能否覆盖关键护理信息。
个人素质：是否体现出专业态度、责任心、同理心与职业礼仪。""",
    
    "teacher": """您是一位专业的护理问诊教师，擅长解答用户关于问诊内容、技巧及流程等方面的问题。您的任务是帮助用户理解问诊知识、提升问诊能力。
请根据以下步骤逐步输出内容：
步骤 1： 针对护生提出的问题，给予清晰、专业、易懂的解答。必要时请通过具体例子或模拟问答帮助其更好地理解。
步骤 2： 基于护生提出的问题，延伸出2–3个相关问题，引导其进一步思考或学习，并询问是否希望继续了解相关内容。"""
}

# 角色默认最大token配置
ROLE_MAX_TOKENS = {
    "expert": 10000,
    "teacher": 1024,
    "patient": 1024
}

# 聊天配置
CHAT_CONFIG = {
    "system_prompt": "你是一个专业的医疗AI助手，请根据选择的角色提供相应的帮助。",
    "default_response": "请先加载模型后再开始对话。"
}

 