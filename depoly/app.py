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
    from xtuner_chat import xtuner_chat
except ImportError as e:
    try:
        import sys

        sys.path.append(os.path.dirname(__file__))
        from xtuner_chat import xtuner_chat
    except ImportError as e:
        print(f"❌ XTuner聊天模块导入失败: {e}")
        print("请确保xtuner_chat.py文件存在")
        sys.exit(1)

try:
    from config import (
        BASE_MODELS, LORA_MODELS, ROLES, LORA_RANKS,
        MODEL_CONFIG, WEB_CONFIG, CHAT_CONFIG, ROLE_SYSTEM_PROMPTS
    )
except ImportError as e:
    # 尝试从当前目录导入
    try:
        import sys

        sys.path.append(os.path.dirname(__file__))
        from config import (
            BASE_MODELS, LORA_MODELS, ROLES, LORA_RANKS,
            MODEL_CONFIG, WEB_CONFIG, CHAT_CONFIG, ROLE_SYSTEM_PROMPTS
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

        self.current_model = None
        self.current_tokenizer = None
        self.current_role = None
        self.current_lora_model = None
        self.current_base_model = None

    def get_available_models(self):
        """获取可用的模型列表"""
        return list(self.base_models.keys())

    def get_lora_models_for_base(self, base_model):
        """获取指定base模型对应的lora模型"""
        if base_model in self.lora_models:
            return list(self.lora_models[base_model].keys())
        return []

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
            if not all([base_model, lora_model, role, rank]):
                print(f"参数不完整: base_model={base_model}, lora_model={lora_model}, role={role}, rank={rank}")
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
        info = {
            "base_model": self.base_models[base_model]["description"] if base_model in self.base_models else "",
            "lora_model": self.lora_models[base_model][lora_model][
                "description"] if base_model in self.lora_models and lora_model in self.lora_models[base_model] else "",
            "role": ROLES[role]["description"] if role in ROLES else "",
            "rank": LORA_RANKS[rank]["description"] if rank in LORA_RANKS else "",
            "iter": f"迭代次数: {iter_name.split('_')[1]}" if iter_name else ""
        }
        return info

    def load_model(self, base_model, lora_model, role, rank, iter_name):
        """加载指定的模型"""
        try:
            # 保存当前选择的角色和模型信息
            self.current_role = role
            self.current_lora_model = lora_model
            self.current_base_model = base_model

            # 构建lora模型路径
            lora_path = os.path.join(
                self.lora_models[base_model][lora_model]["path"],
                role,
                rank,
                iter_name
            )

            if not os.path.exists(lora_path):
                return f"错误：模型路径不存在 - {lora_path}"

            # 加载base模型
            base_path = self.base_models[base_model]["path"]
            print(f"正在加载base模型: {base_path}")

            # 根据模型类型选择不同的加载方式
            model_type = self.base_models[base_model]["type"]

            # 检查GPU可用性
            if torch.cuda.is_available() and not MODEL_CONFIG["force_cpu"]:
                print(f"🚀 使用GPU进行推理")
                device_map = "auto"  # 自动选择最佳设备
            else:
                print(f"💻 使用CPU进行推理")
                device_map = "cpu"

            if model_type == "llama":
                self.current_tokenizer = AutoTokenizer.from_pretrained(
                    base_path,
                    trust_remote_code=MODEL_CONFIG["trust_remote_code"]
                )
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
                    device_map=device_map,
                    trust_remote_code=MODEL_CONFIG["trust_remote_code"],
                    low_cpu_mem_usage=MODEL_CONFIG["low_cpu_mem_usage"]
                )
            else:
                self.current_tokenizer = AutoTokenizer.from_pretrained(
                    base_path,
                    trust_remote_code=MODEL_CONFIG["trust_remote_code"]
                )
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
                    device_map=device_map,
                    trust_remote_code=MODEL_CONFIG["trust_remote_code"],
                    low_cpu_mem_usage=MODEL_CONFIG["low_cpu_mem_usage"]
                )

            # 检查模型设备分配
            if torch.cuda.is_available() and not MODEL_CONFIG["force_cpu"]:
                # 检查模型是否在GPU上
                model_devices = set()
                for param in self.current_model.parameters():
                    model_devices.add(param.device)

                print(f"📊 模型设备分布: {list(model_devices)}")

                # 检查是否使用了Accelerate offload
                has_meta_device = any('meta' in str(device) for device in model_devices)
                has_cpu_device = any('cpu' in str(device) for device in model_devices)

                if has_meta_device:
                    print(f"ℹ️  模型使用了Accelerate offload，保持当前设备分布")
                elif has_cpu_device and len(model_devices) > 1:
                    print(f"⚠️  检测到模型部分在CPU，尝试移动到GPU...")
                    try:
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        self.current_model = self.current_model.to('cuda:0')
                        # 验证移动结果
                        new_devices = set()
                        for param in self.current_model.parameters():
                            new_devices.add(param.device)
                        print(f"✅ 模型已移动到GPU: {list(new_devices)}")
                    except Exception as e:
                        print(f"❌ 移动模型到GPU失败: {e}")
                        print(f"ℹ️  保持当前设备分布")
                else:
                    print(f"✅ 模型设备分布正常")
            else:
                print(f"✅ 模型已加载到CPU")

            # 加载lora模型
            print(f"正在加载lora模型: {lora_path}")
            self.current_model = PeftModel.from_pretrained(
                self.current_model,
                lora_path,
                torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
                is_trainable=False  # 推理模式，节省内存
            )

            # 验证LoRA模型是否正确加载
            print(f"✅ LoRA模型加载完成")
            print(f"📊 模型类型: {type(self.current_model)}")
            if hasattr(self.current_model, 'peft_config'):
                print(f"📋 LoRA配置: {list(self.current_model.peft_config.keys())}")

            # 检查LoRA模型设备分配
            if torch.cuda.is_available() and not MODEL_CONFIG["force_cpu"]:
                lora_devices = set()
                for param in self.current_model.parameters():
                    lora_devices.add(param.device)

                print(f"📊 LoRA模型设备分布: {list(lora_devices)}")

                # 检查是否使用了Accelerate offload
                has_meta_device = any('meta' in str(device) for device in lora_devices)
                has_cpu_device = any('cpu' in str(device) for device in lora_devices)

                if has_meta_device:
                    print(f"ℹ️  LoRA模型使用了Accelerate offload，保持当前设备分布")
                elif has_cpu_device:
                    print(f"⚠️  检测到LoRA模型在CPU，尝试移动到GPU...")
                    try:
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        self.current_model = self.current_model.to('cuda:0')
                        # 验证移动结果
                        new_devices = set()
                        for param in self.current_model.parameters():
                            new_devices.add(param.device)
                        print(f"✅ LoRA模型已移动到GPU: {list(new_devices)}")
                    except Exception as e:
                        print(f"❌ 移动LoRA模型到GPU失败: {e}")
                        print(f"ℹ️  保持当前设备分布，使用CPU推理")
                else:
                    print(f"✅ LoRA模型设备分布正常")
            else:
                print(f"✅ LoRA模型已加载到CPU")

            # 打印模型详细信息
            print(f"\n🔍 模型详细信息:")
            print(f"   Base模型: {base_model}")
            print(f"   LoRA模型: {lora_model}")
            print(f"   角色: {role}")
            print(f"   Rank: {rank}")
            print(f"   Iter: {iter_name}")
            print(f"   模型类型: {model_type}")

            # 显示设备信息
            if torch.cuda.is_available():
                print(f"   🚀 GPU设备: {torch.cuda.get_device_name()}")
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
                reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3
                free_memory = total_memory - reserved_memory

                print(f"   📊 GPU总内存: {total_memory:.1f}GB")
                print(f"   💾 已用内存: {allocated_memory:.1f}GB")
                print(f"   🔄 缓存内存: {reserved_memory:.1f}GB")
                print(f"   🆓 可用内存: {free_memory:.1f}GB")

                # 内存使用百分比
                memory_usage = (reserved_memory / total_memory) * 100
                print(f"   📈 内存使用率: {memory_usage:.1f}%")
            else:
                print(f"   💻 使用CPU模式")

            # 检查模型实际设备分布
            model_devices = set()
            for param in self.current_model.parameters():
                model_devices.add(str(param.device))

            print(f"   模型设备分布: {list(model_devices)}")
            print(f"   参数数量: {sum(p.numel() for p in self.current_model.parameters()):,}")

            # 获取模型信息
            model_info = self.get_model_info(base_model, lora_model, role, rank, iter_name)

            # 获取当前系统提示
            current_system_prompt = self._get_system_prompt()

            return f"✅ 模型加载成功！\n\n📋 模型信息：\n• Base模型: {base_model}\n  {model_info['base_model']}\n• Lora模型: {lora_model}\n  {model_info['lora_model']}\n• 角色: {ROLES[role]['name']}\n  {model_info['role']}\n• Rank: {rank}\n  {model_info['rank']}\n• Iter: {iter_name}\n  {model_info['iter']}", current_system_prompt

        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"

    def chat(self, message, history):
        """进行聊天对话 - 使用XTuner进行对话生成"""
        if self.current_model is None or self.current_tokenizer is None:
            return CHAT_CONFIG["default_response"]

        try:
            # 获取当前模型类型
            model_type = self._get_model_type()

            # 根据角色和模型类型选择system prompt
            system_prompt = self._get_system_prompt()

            # 使用XTuner进行对话生成
            response = xtuner_chat.generate_response(
                model=self.current_model,
                tokenizer=self.current_tokenizer,
                message=message,
                history=history,
                system_prompt=system_prompt,
                model_type=model_type,
                max_new_tokens=MODEL_CONFIG["max_new_tokens"],
                temperature=MODEL_CONFIG["temperature"],
                do_sample=MODEL_CONFIG["do_sample"]
            )

            return response

        except Exception as e:
            print(f"❌ 对话生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"❌ 对话生成失败: {str(e)}"

    def chat_stream(self, message, history):
        """进行流式聊天对话 - 使用XTuner进行流式对话生成"""
        if self.current_model is None or self.current_tokenizer is None:
            yield CHAT_CONFIG["default_response"]
            return

        try:
            # 获取当前模型类型
            model_type = self._get_model_type()

            # 根据角色和模型类型选择system prompt
            system_prompt = self._get_system_prompt()

            # 使用XTuner进行流式对话生成
            for response in xtuner_chat.generate_response_stream(
                    model=self.current_model,
                    tokenizer=self.current_tokenizer,
                    message=message,
                    history=history,
                    system_prompt=system_prompt,
                    model_type=model_type,
                    max_new_tokens=MODEL_CONFIG["max_new_tokens"],
                    temperature=MODEL_CONFIG["temperature"],
                    do_sample=MODEL_CONFIG["do_sample"]
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
        if self.current_role is None or self.current_lora_model is None:
            return CHAT_CONFIG["system_prompt"]

        # 检查是否是ft_pt版本（需要系统提示词）
        if self.current_lora_model.endswith("_ft_pt"):
            # 根据角色返回对应的system prompt
            if self.current_role in ROLE_SYSTEM_PROMPTS:
                return ROLE_SYSTEM_PROMPTS[self.current_role]
            else:
                return CHAT_CONFIG["system_prompt"]
        else:
            # ft版本直接返回空字符串
            return ""


# 创建模型管理器实例
model_manager = ModelManager()


def update_lora_models(base_model):
    """更新lora模型选项"""
    if not base_model:
        return gr.Dropdown(choices=[], value=None)

    choices = model_manager.get_lora_models_for_base(base_model)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def update_iters(base_model, lora_model, role, rank):
    """更新iter选项"""
    if not all([base_model, lora_model, role, rank]):
        return gr.Dropdown(choices=[], value=None)

    choices = model_manager.get_iters(base_model, lora_model, role, rank)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def load_model_wrapper(base_model, lora_model, role, rank, iter_name):
    """加载模型的包装函数"""
    result = model_manager.load_model(base_model, lora_model, role, rank, iter_name)
    if isinstance(result, tuple):
        return result
    else:
        return result, ""


def chat_wrapper(message, history):
    """聊天的包装函数（兼容旧格式）"""
    response = model_manager.chat(message, history)
    # Gradio Chatbot期望元组格式 (user_message, assistant_message)
    return history + [[message, response]]


def chat_stream_wrapper(message, history):
    """流式聊天的包装函数"""
    # 先添加用户消息到历史
    history.append([message, ""])

    # 流式生成回复
    for response in model_manager.chat_stream(message, history[:-1]):  # 传入除最后一条外的历史
        # 只更新助手的回复部分，不重复用户输入
        history[-1][1] = response
        yield history


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
            initial_base_model = model_manager.get_available_models()[
                0] if model_manager.get_available_models() else None
            initial_lora_models = model_manager.get_lora_models_for_base(
                initial_base_model) if initial_base_model else []
            initial_lora_model = initial_lora_models[0] if initial_lora_models else None

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
            rank_dropdown = gr.Dropdown(
                choices=model_manager.get_ranks(),
                label="选择Lora Rank",
                value=model_manager.get_ranks()[0] if model_manager.get_ranks() else None
            )

            # Iter选择
            initial_roles = model_manager.get_roles()
            initial_ranks = model_manager.get_ranks()
            initial_role = initial_roles[0] if initial_roles else None
            initial_rank = initial_ranks[0] if initial_ranks else None

            initial_iters = model_manager.get_iters(initial_base_model, initial_lora_model, initial_role,
                                                    initial_rank) if all(
                [initial_base_model, initial_lora_model, initial_role, initial_rank]) else []
            initial_iter = initial_iters[0] if initial_iters else None

            iter_dropdown = gr.Dropdown(
                choices=initial_iters,
                label="选择Iter",
                interactive=True,
                value=initial_iter
            )

            # 加载模型按钮
            load_btn = gr.Button("🚀 加载模型", variant="primary")

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
                show_label=True,
                type="messages"  # 使用新的消息格式
            )

            # 输入框
            msg = gr.Textbox(
                label="输入消息",
                placeholder="请输入您的问题...",
                lines=2
            )

            # 发送按钮
            send_btn = gr.Button("💬 发送", variant="primary")

            # 清除按钮
            clear_btn = gr.Button("🗑️ 清除对话")

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

    load_btn.click(
        fn=load_model_wrapper,
        inputs=[base_model_dropdown, lora_model_dropdown, role_dropdown, rank_dropdown, iter_dropdown],
        outputs=[load_status, system_prompt_display]
    )

    send_btn.click(
        fn=chat_stream_wrapper,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        [msg]
    )

    msg.submit(
        fn=chat_stream_wrapper,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        [msg]
    )

    clear_btn.click(
        lambda: [],
        None,
        [chatbot]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=WEB_CONFIG["server_name"],
        server_port=WEB_CONFIG["server_port"],
        share=WEB_CONFIG["share"],
        debug=WEB_CONFIG["debug"],
        show_error=WEB_CONFIG["show_error"]
    )
