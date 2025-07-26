# NursingAI 智能聊天系统

## 项目概述

NursingAI 是一个基于 Gradio 的多模型医疗对话系统，由中南大学计算机学院阚课题组和中南大学湘雅护理学院丁课题组合作开发。该系统支持多种预训练模型和微调模型，提供医疗专家、教师、患者三种角色的智能对话服务。

## 项目结构

```
NursingAI/
├── src/                    # 源代码目录
│   ├── app.py             # 主应用文件 (25KB)
│   ├── config.py          # 配置文件 (5.8KB)
│   ├── xtuner_chat.py     # XTuner聊天模块 (11KB)
│   └── __init__.py
├── pretrained_models/      # 预训练模型目录
├── checkpoints/           # 训练检查点目录
├── pyproject.toml         # 项目依赖配置
├── run.sh                 # 启动脚本
└── README.md              # 项目文档
```

## 核心文件说明

### 1. `src/app.py` - 主应用文件

**功能**：Gradio Web界面的主应用，负责模型管理、界面交互和对话处理。

**主要组件**：

#### ModelManager 类
- **模型管理**：管理base模型和lora模型的加载、切换
- **动态配置**：根据选择的模型、角色、rank动态获取可用的iter选项
- **对话处理**：处理用户输入，调用XTuner进行对话生成
- **系统提示词管理**：根据角色和模型类型选择合适的系统提示词

#### 主要方法：
- `load_model()`: 加载指定的模型组合
- `chat()`: 非流式对话生成
- `chat_stream()`: 流式对话生成
- `get_iters()`: 动态获取iter选项
- `_get_system_prompt()`: 获取系统提示词

#### Gradio界面组件：
- **模型选择区**：Base模型、Lora模型、角色、Rank、Iter选择
- **聊天对话区**：聊天界面、消息输入、系统提示显示
- **事件绑定**：下拉框联动、模型加载、对话发送

### 2. `src/config.py` - 配置文件

**功能**：集中管理项目的所有配置参数。

**配置项**：

#### 模型配置
```python
BASE_MODELS = {
    "MMed-Llama-3-8B": {
        "path": "预训练模型路径",
        "type": "llama",
        "description": "医疗领域的Llama3模型"
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "path": "预训练模型路径", 
        "type": "qwen",
        "description": "DeepSeek蒸馏版Qwen模型"
    }
}
```

#### Lora模型配置
```python
LORA_MODELS = {
    "MMed-Llama-3-8B": {
        "MMedIns_Llama3_8B_ft": {
            "path": "微调模型路径",
            "description": "Llama3的医疗指令微调模型"
        },
        "MMedIns_Llama3_8B_ft_pt": {
            "path": "微调模型路径",
            "description": "Llama3的医疗指令微调模型（PT版本）"
        }
    }
}
```

#### 角色配置
```python
ROLES = {
    "expert": {"name": "专家", "description": "医疗专家角色"},
    "teacher": {"name": "教师", "description": "教学角色"},
    "patient": {"name": "患者", "description": "患者角色"}
}
```

#### 系统提示词
```python
ROLE_SYSTEM_PROMPTS = {
    "patient": "患者角色的系统提示词...",
    "expert": "专家角色的系统提示词...", 
    "teacher": "教师角色的系统提示词..."
}
```

#### 其他配置
- `MODEL_CONFIG`: 模型加载参数（dtype、device、生成参数等）
- `WEB_CONFIG`: Web界面配置（端口、主题等）
- `CHAT_CONFIG`: 聊天配置（默认响应、系统提示词等）

### 3. `src/xtuner_chat.py` - XTuner聊天模块

**功能**：使用XTuner风格进行对话生成，支持流式和非流式输出。

**主要组件**：

#### XTunerChat 类
- **对话模板**：定义llama和qwen模型的对话格式
- **对话构建**：按照XTuner方式构建完整的对话文本
- **生成控制**：处理停止词、流式输出等

#### 核心方法：

##### `build_conversation()`
构建完整的对话文本，包括：
- 系统提示词（第一轮对话）
- 历史对话记录
- 当前用户输入
- 模型特定的特殊标记

##### `generate_response()`
非流式对话生成：
- 编码输入文本
- 设置生成参数和停止条件
- 生成回复并清理输出

##### `generate_response_stream()`
流式对话生成：
- 使用TextIteratorStreamer实现流式输出
- 后台线程处理生成
- 实时返回生成内容

#### 支持的模型类型：
- **llama**: 使用Llama3的特殊标记格式
- **qwen**: 使用Qwen的特殊标记格式

## 技术特点

### 1. 多模型支持
- 支持多种预训练模型（Llama3、Qwen等）
- 支持多种微调模型（ft、ft_pt版本）
- 动态模型加载和切换

### 2. 多角色对话
- **专家角色**：提供专业的医疗建议和评估
- **教师角色**：提供医疗知识教育和指导
- **患者角色**：模拟患者视角的对话训练

### 3. 智能配置管理
- 动态获取可用的模型组合
- 根据文件系统自动发现iter选项
- 角色特定的系统提示词

### 4. 流式对话
- 支持实时流式输出
- 优化的生成参数配置
- 智能停止词处理

### 5. 用户友好界面
- 直观的模型选择界面
- 实时系统提示显示
- 响应式布局设计

## 安装和运行

### 环境要求
- Python >= 3.8
- CUDA支持（推荐）
- 16GB+ 内存

### 安装依赖
```bash
pip install -e .
```

### 运行应用
```bash
# 使用启动脚本
./run.sh

# 或直接运行
cd src && python app.py
```

### 访问界面
打开浏览器访问：`http://127.0.0.1:7860`

## 使用流程

1. **选择模型**：选择Base模型和对应的Lora模型
2. **选择角色**：选择专家、教师或患者角色
3. **选择参数**：选择Lora Rank和Iter
4. **加载模型**：点击"加载模型"按钮
5. **开始对话**：在聊天界面输入消息开始对话

## 模型说明

### Base模型
- **MMed-Llama-3-8B**: 医疗领域的Llama3模型
- **DeepSeek-R1-Distill-Qwen-7B**: DeepSeek蒸馏版Qwen模型

### Lora模型
- **ft版本**: 指令微调模型，不使用系统提示词
- **ft_pt版本**: 指令微调模型，使用角色特定的系统提示词

### 角色功能
- **专家**: 评估护理问诊表现，提供专业建议
- **教师**: 解答问诊相关问题，提供教育指导
- **患者**: 模拟患者对话，配合问诊训练

## 开发说明

### 添加新模型
1. 在`config.py`中添加模型配置
2. 确保模型文件存在于对应目录
3. 重启应用即可使用

### 添加新角色
1. 在`config.py`中添加角色配置
2. 添加对应的系统提示词
3. 重启应用即可使用

### 自定义配置
- 修改`config.py`中的配置参数
- 调整`MODEL_CONFIG`中的生成参数
- 自定义`WEB_CONFIG`中的界面设置

## 项目优势

1. **模块化设计**：清晰的代码结构，易于维护和扩展
2. **配置驱动**：通过配置文件管理所有参数，无需修改代码
3. **动态发现**：自动发现可用的模型和参数组合
4. **用户友好**：直观的Web界面，支持实时交互
5. **高性能**：支持GPU加速，流式输出优化

## 技术栈

- **后端框架**: Gradio
- **深度学习**: PyTorch, Transformers, PEFT
- **模型格式**: XTuner风格对话
- **界面**: Gradio Blocks
- **部署**: 本地Web服务

## 许可证

MIT License

## 贡献者

- 中南大学计算机学院阚课题组
- 中南大学湘雅护理学院丁课题组

## 联系方式

- 项目维护者：张浩杰
- 邮箱：ai.ventus.aislant@gmail.com 