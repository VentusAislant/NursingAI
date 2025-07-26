import os
import sys
import warnings

# 忽略Pydantic相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# 设置环境变量避免某些兼容性问题
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"

try:
    import gradio as gr
except ImportError as e:
    print(f"❌ Gradio导入失败: {e}")
    print("请运行: pip install gradio==4.16.0")
    sys.exit(1)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError as e:
    print(f"❌ 模型相关库导入失败: {e}")
    print("请运行: pip install torch transformers peft")
    sys.exit(1)

try:
    from chat import chat
except ImportError as e:
    try:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from chat import chat
    except ImportError as e:
        print(f"❌ chat模块导入失败: {e}")
        print("请确保chat.py文件存在")
        sys.exit(1)

try:
    from config import (
        BASE_MODELS, LORA_MODELS, ROLES, LORA_RANKS,
        MODEL_CONFIG, WEB_CONFIG, CHAT_CONFIG, ROLE_SYSTEM_PROMPTS, ROLE_MAX_TOKENS
    )
except ImportError as e:
    # 尝试从当前目录导入
    try:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from config import (
            BASE_MODELS, LORA_MODELS, ROLES, LORA_RANKS,
            MODEL_CONFIG, WEB_CONFIG, CHAT_CONFIG, ROLE_SYSTEM_PROMPTS, ROLE_MAX_TOKENS
        )
    except ImportError as e2:
        print(f"❌ 配置文件导入失败: {e2}")
        sys.exit(1)

class ModelManager:
    def __init__(self):
        self.base_models = BASE_MODELS
        self.lora_models = LORA_MODELS
        self.roles = list(ROLES.keys())
        self.ranks = list(LORA_RANKS.keys())
        
        self.current_role = None
        self.current_lora_model = None
        self.current_base_model = None
        
    def get_available_models(self):
        """获取可用的模型列表"""
        return list(self.base_models.keys())
    
    def get_lora_models_for_base(self, base_model):
        """获取指定base模型对应的lora模型，支持None"""
        if base_model in self.lora_models:
            # 返回显示名和实际值
            return [("无LoRA", None)] + [(k, k) for k in self.lora_models[base_model].keys()]
        return [("无LoRA", None)]

    def get_roles(self):
        """获取可用的角色列表"""
        return self.roles
    
    def get_ranks(self):
        """获取可用的rank列表"""
        return self.ranks
    
    def get_iters(self, base_model, lora_model, role, rank):
        """动态获取指定路径下的iter列表"""
        try:
            # 参数验证
            if not all([base_model, role, rank]):
                print(f"参数不完整: base_model={base_model}, role={role}, rank={rank}")
                return []
            
            # 检查base_model是否存在
            if base_model not in self.lora_models:
                print(f"base_model不存在: {base_model}")
                return []
            
            # 检查lora_model是否存在
            if lora_model not in self.lora_models[base_model]:
                print(f"lora_model不存在: {lora_model} in {base_model}")
                return []
            
            # 构建路径
            lora_path = os.path.join(
                self.lora_models[base_model][lora_model]["path"],
                role,
                rank
            )
            
            if not os.path.exists(lora_path):
                print(f"路径不存在: {lora_path}")
                return []
            
            # 获取所有iter_开头的目录
            iter_dirs = []
            for item in os.listdir(lora_path):
                if item.startswith("iter_") and os.path.isdir(os.path.join(lora_path, item)):
                    iter_dirs.append(item)
            
            # 按迭代次数排序
            iter_dirs.sort(key=lambda x: int(x.split("_")[1]))
            return iter_dirs
            
        except Exception as e:
            print(f"获取iter列表失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_model_info(self, base_model, lora_model, role, rank, iter_name):
        """获取模型信息"""
        # 处理lora_model为None的情况
        lora_description = ""
        if lora_model is not None and base_model in self.lora_models and lora_model in self.lora_models[base_model]:
            lora_description = self.lora_models[base_model][lora_model]["description"]
        elif lora_model is None:
            lora_description = "无LoRA"
        
        # 处理rank为"无"的情况
        rank_description = ""
        if rank == "无":
            rank_description = "无"
        elif rank in LORA_RANKS:
            rank_description = LORA_RANKS[rank]["description"]
        
        info = {
            "base_model": self.base_models[base_model]["description"] if base_model in self.base_models else "",
            "lora_model": lora_description,
            "role": ROLES[role]["description"] if role in ROLES else "",
            "rank": rank_description,
            "iter": f"迭代次数: {iter_name.split('_')[1]}" if iter_name and iter_name != "无" and iter_name.startswith("iter_") else (iter_name if iter_name else "")
        }
        return info
    
    def load_model(self, base_model, lora_model, role, rank, iter_name):
        """加载指定的模型 - 使用XTuner进行模型加载"""
        try:
            # 保存当前选择的角色和模型信息
            self.current_role = role
            self.current_lora_model = lora_model
            self.current_base_model = base_model
            
            # 构建lora模型路径
            if lora_model is None:
                adapter_path = None
            else:
                adapter_path = os.path.join(
                    self.lora_models[base_model][lora_model]["path"],
                    role,
                    rank,
                    iter_name
                )
                if not os.path.exists(adapter_path):
                    return f"错误：模型路径不存在 - {adapter_path}"
            
            # 加载base模型
            base_path = self.base_models[base_model]["path"]
            print(f"正在加载base模型: {base_path}")
            
            # 根据模型类型选择不同的加载方式
            model_type = self.base_models[base_model]["type"]
            
            # 使用Chat进行模型加载
            success = chat.load_model(
                model_name_or_path=base_path,
                adapter_path=adapter_path,
                torch_dtype=MODEL_CONFIG["torch_dtype"],
                bits=None  # 可以根据需要设置量化
            )
            
            if not success:
                return f"❌ 模型加载失败"
            
            # 验证模型是否正确加载
            if chat.model is None or chat.tokenizer is None:
                return f"❌ 模型加载验证失败"
            
            # 清空历史记录，准备新的对话
            chat.clear_history()
            
            # 打印模型详细信息
            print(f"\n🔍 模型详细信息:")
            print(f"   Base模型: {base_model}")
            print(f"   LoRA模型: {lora_model if lora_model else '无LoRA'}")
            print(f"   角色: {role}")
            print(f"   Rank: {rank}")
            print(f"   Iter: {iter_name}")
            print(f"   模型类型: {model_type}")
            
            # 显示设备信息
            if torch.cuda.is_available():
                print(f"   🚀 GPU设备: {torch.cuda.get_device_name()}")
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated() / 1024**3
                reserved_memory = torch.cuda.memory_reserved() / 1024**3
                free_memory = total_memory - reserved_memory
                
                print(f"   📊 GPU总内存: {total_memory:.1f}GB")
                print(f"   💾 已用内存: {allocated_memory:.1f}GB")
                print(f"   🔄 缓存内存: {reserved_memory:.1f}GB")
                print(f"   🆓 可用内存: {free_memory:.1f}GB")
                
                # 内存使用百分比
                memory_usage = (reserved_memory / total_memory) * 100
                print(f"   📈 内存使用率: {memory_usage:.1f}%")
                
                # 检查模型设备
                if hasattr(chat.model, 'device'):
                    print(f"   🎯 模型设备: {chat.model.device}")
                else:
                    model_device = next(chat.model.parameters()).device
                    print(f"   🎯 模型设备: {model_device}")
            else:
                print(f"   💻 使用CPU模式")
            
            # 获取模型信息
            model_info = self.get_model_info(base_model, lora_model, role, rank, iter_name)
            
            # 获取当前系统提示
            current_system_prompt = self._get_system_prompt()
            
            return f"✅ 模型加载成功！\n\n📋 模型信息：\n• Base模型: {base_model}\n  {model_info['base_model']}\n• Lora模型: {lora_model if lora_model else '无LoRA'}\n  {model_info['lora_model']}\n• 角色: {ROLES[role]['name']}\n  {model_info['role']}\n• Rank: {rank}\n  {model_info['rank']}\n• Iter: {iter_name}\n  {model_info['iter']}", current_system_prompt
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"❌ 模型加载失败: {str(e)}"
    
    # def chat(self, message, history, max_new_tokens=None, temperature=None, stop_words=None):
    #     """进行聊天对话 - 使用Chat进行对话生成"""
    #     if chat.model is None or chat.tokenizer is None:
    #         return CHAT_CONFIG["default_response"]
    #
    #     try:
    #         # 获取system prompt
    #         system_prompt = self._get_system_prompt()
    #
    #         # 使用默认值或传入的参数
    #         max_new_tokens = max_new_tokens or MODEL_CONFIG["max_new_tokens"]
    #         temperature = temperature or MODEL_CONFIG["temperature"]
    #
    #         # 设置默认的stop words
    #         default_stop_words = ["</s>", "<eot_id>", "END", "Human:", "Assistant:", "User:", "Bot:"]
    #         if stop_words and stop_words.strip():
    #             stop_words_list = [word.strip() for word in stop_words.split(',') if word.strip()]
    #             stop_words_list.extend(default_stop_words)
    #         else:
    #             stop_words_list = default_stop_words
    #
    #         # 如果chat的历史为空且需要设置system prompt，直接添加到历史中
    #         if not chat.history and system_prompt:
    #             chat.history.append({"role": "system", "content": system_prompt})
    #
    #         # 使用Chat进行对话生成
    #         response = chat.generate_response(
    #             message=[{"role": "user", "content": message}],
    #             max_new_tokens=max_new_tokens,
    #             temperature=temperature,
    #             stop_words=stop_words_list
    #         )
    #
    #         return response
    #
    #     except Exception as e:
    #         print(f"❌ 对话生成失败: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         return f"❌ 对话生成失败: {str(e)}"

    def chat_stream(self, message,  max_new_tokens, temperature):
        """进行流式聊天对话 - 使用Chat进行流式对话生成"""
        if chat.model is None or chat.tokenizer is None:
            yield CHAT_CONFIG["default_response"]
            return
        
        try:
            # 获取system prompt
            system_prompt = self._get_system_prompt()
            
            # 使用默认值或传入的参数
            max_new_tokens = max_new_tokens or MODEL_CONFIG["max_new_tokens"]
            temperature = temperature or MODEL_CONFIG["temperature"]
            

            # 如果chat的历史为空且需要设置system prompt，直接添加到历史中
            if not chat.history and system_prompt:
                chat.history.append({"role": "system", "content": system_prompt})

            chat.history.append({"role": "user", "content": message})
            
            # 使用Chat进行流式对话生成
            for response in chat.generate_response_stream(
                max_new_tokens=max_new_tokens,
                temperature=temperature
            ):
                yield response
                
        except Exception as e:
            print(f"❌ 流式对话生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"❌ 流式对话生成失败: {str(e)}"
    
    def _get_model_type(self):
        """获取当前模型类型"""
        if self.current_base_model is None:
            return "llama"
        
        model_type = self.base_models[self.current_base_model]["type"]
        return model_type
    
    def _get_system_prompt(self):
        """根据角色和模型类型获取system prompt"""
        if self.current_role is None:
            return CHAT_CONFIG["system_prompt"]
        # LoRA为None时，直接返回角色的系统消息
        if self.current_lora_model is None:
            return ROLE_SYSTEM_PROMPTS.get(self.current_role, CHAT_CONFIG["system_prompt"])
        # ft_pt版本
        if self.current_lora_model.endswith("_ft_pt"):
            return ROLE_SYSTEM_PROMPTS.get(self.current_role, CHAT_CONFIG["system_prompt"])
        # 其他情况返回空字符串
        return ""
    


# 创建模型管理器实例
model_manager = ModelManager()

def update_lora_models(base_model):
    """更新lora模型选项"""
    if not base_model:
        return gr.Dropdown(choices=[], value=None)
    
    choices = model_manager.get_lora_models_for_base(base_model)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

def update_iters(base_model, lora_model, role, rank):
    """更新iter选项"""
    if not all([base_model, role, rank]):
        return gr.Dropdown(choices=[], value=None)
    if lora_model is None:
        return gr.Dropdown(choices=["无"], value="无")
    choices = model_manager.get_iters(base_model, lora_model, role, rank)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

# 新增update_ranks函数
def update_ranks(lora_model):
    if lora_model is None:
        return gr.Dropdown(choices=["无"], value="无", interactive=False)
    choices = model_manager.get_ranks()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None, interactive=True)

def load_model_wrapper(base_model, lora_model, role, rank, iter_name):
    """加载模型的包装函数"""
    result = model_manager.load_model(base_model, lora_model, role, rank, iter_name)
    if isinstance(result, tuple):
        return result
    else:
        return result, ""

def clear_model_wrapper():
    """清理模型的包装函数"""
    try:
        chat.clear_model()
        return "✅ 模型已清理，GPU内存已释放", ""
    except Exception as e:
        return f"❌ 模型清理失败: {str(e)}", ""

# def chat_wrapper(message, history):
#     """聊天的包装函数（兼容Gradio格式）"""
#     response = model_manager.chat(message, history)
#     # Gradio Chatbot期望元组格式 (user_message, assistant_message)
#     return history + [[message, response]]

def chat_stream_wrapper(message, history, max_new_tokens, temperature):
    """流式聊天的包装函数"""
    # 确保参数类型正确
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    
    print(f"🔧 调试信息 - max_new_tokens: {max_new_tokens}, temperature: {temperature}")
    
    # 流式生成回复
    for response in model_manager.chat_stream(message, max_new_tokens, temperature):
        # Gradio Chatbot期望格式: [[user_msg, bot_msg], [user_msg2, bot_msg2], ...]
        # 将用户消息和当前回复组合成新的对话轮次
        current_conversation = [message, response]
        # 将历史对话和当前对话组合
        full_history = history + [current_conversation]
        yield full_history



# 创建Gradio界面
with gr.Blocks(title=WEB_CONFIG["title"], theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 NursingAI 智能聊天系统")
    gr.Markdown("请选择base模型和对应的lora模型，然后开始聊天对话")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 模型选择")
            
            # Base模型选择
            base_model_dropdown = gr.Dropdown(
                choices=model_manager.get_available_models(),
                label="选择Base模型",
                value=model_manager.get_available_models()[0] if model_manager.get_available_models() else None
            )
            
            # Lora模型选择
            initial_base_model = model_manager.get_available_models()[0] if model_manager.get_available_models() else None
            initial_lora_models = model_manager.get_lora_models_for_base(initial_base_model) if initial_base_model else []
            initial_lora_model = initial_lora_models[0][1] if initial_lora_models else None
            
            lora_model_dropdown = gr.Dropdown(
                choices=initial_lora_models,
                label="选择Lora模型",
                interactive=True,
                value=initial_lora_model
            )
            
            # 角色选择
            role_dropdown = gr.Dropdown(
                choices=model_manager.get_roles(),
                label="选择角色",
                value=model_manager.get_roles()[0] if model_manager.get_roles() else None
            )
            
            # Rank选择
            initial_rank_choices = ["无"] if initial_lora_model is None else model_manager.get_ranks()
            initial_rank_value = "无" if initial_lora_model is None else (initial_rank_choices[0] if initial_rank_choices else None)
            rank_dropdown = gr.Dropdown(
                choices=initial_rank_choices,
                label="选择Lora Rank",
                value=initial_rank_value,
                interactive=(initial_lora_model is not None)
            )
            
            # Iter选择
            initial_roles = model_manager.get_roles()
            initial_ranks = model_manager.get_ranks()
            initial_role = initial_roles[0] if initial_roles else None
            initial_rank = initial_ranks[0] if initial_ranks else None
            
            initial_iters = model_manager.get_iters(initial_base_model, initial_lora_model, initial_role, initial_rank) if all([initial_base_model, initial_lora_model, initial_role, initial_rank]) else []
            initial_iter = initial_iters[0] if initial_iters else None
            
            iter_dropdown = gr.Dropdown(
                choices=initial_iters,
                label="选择Iter",
                interactive=True,
                value=initial_iter
            )
            
            # 加载模型按钮
            load_btn = gr.Button("🚀 加载模型", variant="primary")
            
            # 清理模型按钮
            clear_model_btn = gr.Button("🗑️ 清理模型", variant="secondary")
            
            # 加载状态显示
            load_status = gr.Textbox(
                label="加载状态",
                interactive=False,
                lines=4
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 聊天对话")
            
            # 当前系统提示显示
            system_prompt_display = gr.Textbox(
                label="当前系统提示",
                interactive=False,
                lines=3,
                max_lines=5,
                placeholder="加载模型后显示当前使用的系统提示..."
            )
            
            # 聊天界面
            chatbot = gr.Chatbot(
                label="聊天记录",
                height=500,
                show_label=True
            )
            
            # 高级设置面板
            initial_role = model_manager.get_roles()[0] if model_manager.get_roles() else None
            initial_max_tokens = ROLE_MAX_TOKENS.get(initial_role, 1024) if initial_role else 1024
            
            with gr.Accordion("🔧 高级设置", open=False):
                max_new_tokens_slider = gr.Slider(
                    minimum=32, maximum=10000, value=initial_max_tokens, step=1, 
                    label="最大生成Token数", info="专家角色默认10000，其他角色默认1024"
                )
                temperature_slider = gr.Slider(
                    minimum=0.01, maximum=2.0, value=0.1, step=0.01, 
                    label="温度系数", info="控制回复的随机性，值越高越随机"
                )

            
            # 输入框
            msg = gr.Textbox(
                label="输入消息",
                placeholder="请输入您的问题...",
                lines=2
            )
            
            # 发送按钮
            send_btn = gr.Button("💬 发送", variant="primary")
            
            # 按钮行
            with gr.Row():
                # 清除按钮
                clear_btn = gr.Button("🗑️ 清除对话")
                # 复制聊天记录按钮
                copy_btn = gr.Button("📋 复制聊天记录")
    
    # 绑定事件
    base_model_dropdown.change(
        fn=update_lora_models,
        inputs=[base_model_dropdown],
        outputs=[lora_model_dropdown]
    ).then(
        fn=update_iters,
        inputs=[base_model_dropdown, lora_model_dropdown, role_dropdown, rank_dropdown],
        outputs=[iter_dropdown]
    )
    
    lora_model_dropdown.change(
        fn=update_ranks,
        inputs=[lora_model_dropdown],
        outputs=[rank_dropdown]
    ).then(
        fn=update_iters,
        inputs=[base_model_dropdown, lora_model_dropdown, role_dropdown, rank_dropdown],
        outputs=[iter_dropdown]
    )
    
    role_dropdown.change(
        fn=update_iters,
        inputs=[base_model_dropdown, lora_model_dropdown, role_dropdown, rank_dropdown],
        outputs=[iter_dropdown]
    )
    
    rank_dropdown.change(
        fn=update_iters,
        inputs=[base_model_dropdown, lora_model_dropdown, role_dropdown, rank_dropdown],
        outputs=[iter_dropdown]
    )
    
    # 角色切换时更新最大token数
    def update_max_tokens(role):
        return ROLE_MAX_TOKENS.get(role, 1024)
    
    role_dropdown.change(
        fn=update_max_tokens,
        inputs=[role_dropdown],
        outputs=[max_new_tokens_slider]
    )
    
    load_btn.click(
        fn=load_model_wrapper,
        inputs=[base_model_dropdown, lora_model_dropdown, role_dropdown, rank_dropdown, iter_dropdown],
        outputs=[load_status, system_prompt_display]
    )
    
    clear_model_btn.click(
        fn=clear_model_wrapper,
        inputs=None,
        outputs=[load_status, system_prompt_display]
    )

    send_btn.click(
        fn=chat_stream_wrapper,
        inputs=[msg, chatbot, max_new_tokens_slider, temperature_slider],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        [msg]
    )
    
    msg.submit(
        fn=chat_stream_wrapper,
        inputs=[msg, chatbot, max_new_tokens_slider, temperature_slider],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        [msg]
    )
    
    def clear_chat():
        """清空聊天历史和界面"""
        chat.clear_history()
        return []
    
    def copy_chat_history(chatbot_history):
        """复制聊天记录到剪贴板"""
        try:
            import pyperclip
        except ImportError:
            return "❌ 请先安装pyperclip: pip install pyperclip"
        
        if not chatbot_history:
            return "❌ 没有聊天记录可复制"
        
        try:
            # 根据当前模型角色设置对话角色名称
            current_role = model_manager.current_role
            if current_role == "patient":
                role_names = ['护士', '患者']
            elif current_role == "expert":
                role_names = ['护生', '专家']
            elif current_role == "teacher":
                role_names = ['护生', '教师']
            else:
                role_names = ['用户', '助手']
            
            # 处理聊天记录格式
            formatted_history = []
            
            # 处理每轮对话
            for i, (user_msg, assistant_msg) in enumerate(chatbot_history, 1):
                formatted_history.append(f"{role_names[0]}: {user_msg}")
                formatted_history.append(f"{role_names[1]}: {assistant_msg}")
            
            # 合并为字符串
            chat_text = "\n".join(formatted_history)
            
            # 复制到剪贴板
            pyperclip.copy(chat_text)
            return f"✅ 聊天记录已复制到剪贴板！\n共{len(chatbot_history)}轮对话，{len(chat_text)}个字符"
            
        except Exception as e:
            return f"❌ 复制失败: {str(e)}"
    
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot]
    )
    
    copy_btn.click(
        fn=copy_chat_history,
        inputs=[chatbot],
        outputs=[load_status]  # 使用load_status显示复制结果
    )
    


if __name__ == "__main__":
    demo.launch(
        server_name=WEB_CONFIG["server_name"],
        server_port=WEB_CONFIG["server_port"],
        share=WEB_CONFIG["share"],
        debug=WEB_CONFIG["debug"],
        show_error=WEB_CONFIG["show_error"]
    ) 