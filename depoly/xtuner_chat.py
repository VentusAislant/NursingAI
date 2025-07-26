#!/usr/bin/env python3
"""
使用XTuner进行对话生成的模块
"""

import torch
from mmengine.config import ConfigDict

class XTunerChat:
    """使用XTuner进行对话生成的类"""
    
    def __init__(self):
        # 定义XTuner风格的对话模板
        self.prompt_templates = {
            "llama": ConfigDict(
                SYSTEM="<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
                INSTRUCTION="<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                SUFFIX="<|eot_id|>",
                SUFFIX_AS_EOS=True,
                SEP="\n",
                STOP_WORDS=["<|eot_id|>"],
            ),
            "qwen": ConfigDict(
                SYSTEM="{system}\n\n",
                INSTRUCTION="REDACTED_SPECIAL_TOKEN{input}REDACTED_SPECIAL_TOKEN",
                SUFFIX="REDACTED_SPECIAL_TOKEN",
                SUFFIX_AS_EOS=True,
                STOP_WORDS=["REDACTED_SPECIAL_TOKEN"],
                SEP="\n",
            )
        }
    
    def build_prompt_text(self, text, system_prompt, template, n_turn=0, bot_name="BOT"):
        """按照XTuner的方式构建prompt文本"""
        prompt_text = ""
        
        # 处理系统提示（只在第一轮添加）
        if "SYSTEM" in template and n_turn == 0 and system_prompt:
            prompt_text += template.SYSTEM.format(
                system=system_prompt, round=n_turn + 1, bot_name=bot_name
            )
        
        # 添加指令部分
        prompt_text += template.INSTRUCTION.format(
            input=text, round=n_turn + 1, bot_name=bot_name
        )
        
        return prompt_text
    
    def build_conversation(self, message, history, system_prompt, model_type="llama"):
        """按照XTuner方式构建完整对话"""
        template = self.prompt_templates.get(model_type, self.prompt_templates["llama"])
        
        # 构建完整的对话文本
        inputs = ""
        
        # 处理历史对话
        for i, (human, assistant) in enumerate(history):
            # 构建用户输入部分
            prompt_text = self.build_prompt_text(human, "", template, i, "BOT")
            inputs += prompt_text
            
            # 添加助手回复
            inputs += assistant
            
            # 添加分隔符
            if template.get("SEP"):
                inputs += template.SEP
        
        # 添加当前用户输入
        current_prompt = self.build_prompt_text(message, system_prompt, template, len(history), "BOT")
        inputs += current_prompt
        
        return inputs, template
    
    def generate_response(self, model, tokenizer, message, history, system_prompt, model_type="llama", **kwargs):
        """使用XTuner风格生成回复"""
        # 构建对话
        inputs, template = self.build_conversation(message, history, system_prompt, model_type)
        
        # 详细打印调试信息
        print("\n" + "="*80)
        print("🔍 XTuner详细调试信息")
        print("="*80)
        print(f"📋 模型类型: {model_type}")
        print(f"📝 系统提示: {system_prompt}")
        print(f"💬 用户输入: {message}")
        print(f"📚 对话历史: {history}")
        print(f"🔧 使用的模板: {template}")
        print("\n" + "-"*80)
        print("📄 完整对话模板:")
        print("-"*80)
        print(inputs)
        print("-"*80)
        
        # 生成回复
        print(f"\n🤖 开始推理，使用XTuner风格")
        
        # 编码输入
        if len(history) == 0:
            # 第一轮对话
            ids = tokenizer.encode(inputs, return_tensors="pt")
        else:
            # 后续对话，不添加特殊token
            ids = tokenizer.encode(inputs, return_tensors="pt", add_special_tokens=False)
        
        # 确保输入数据在正确的设备上
        if torch.cuda.is_available():
            ids = ids.cuda()
        else:
            ids = ids.cpu()
        print(f"📊 输入tokens长度: {ids.shape[1]}")
        
        # 设置停止条件
        stop_words = template.STOP_WORDS
        print(f"🛑 停止词: {stop_words}")
        
        # 创建停止条件
        def create_stopping_criteria(tokenizer, stop_words):
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopWordsCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_words):
                    self.tokenizer = tokenizer
                    self.stop_words = stop_words
                    self.stop_word_ids = []
                    for stop_word in stop_words:
                        stop_word_ids = tokenizer.encode(stop_word, add_special_tokens=False)
                        self.stop_word_ids.append(stop_word_ids)
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_word_ids in self.stop_word_ids:
                        if len(input_ids[0]) >= len(stop_word_ids):
                            if input_ids[0][-len(stop_word_ids):].tolist() == stop_word_ids:
                                return True
                    return False
            
            return StoppingCriteriaList([StopWordsCriteria(tokenizer, stop_words)])
        
        stopping_criteria = create_stopping_criteria(tokenizer, stop_words)
        
        # 设置生成参数
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        
        with torch.no_grad():
            outputs = model.generate(
                inputs=ids,
                **generation_kwargs
            )
        
        # 解码输出（只取新生成的部分）
        raw_response = tokenizer.decode(outputs[0][ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\n📤 原始模型输出:")
        print("-"*80)
        print(raw_response)
        print("-"*80)
        
        # 清理响应中的停止词（以防万一）
        response = raw_response
        for stop_word in template.STOP_WORDS:
            if stop_word in response:
                response = response.replace(stop_word, "")
                print(f"🧹 清理停止词 '{stop_word}'")
        
        final_response = response.strip()
        print(f"\n✅ 最终响应: {final_response}")
        print("="*80 + "\n")
        
        return final_response

    def generate_response_stream(self, model, tokenizer, message, history, system_prompt, model_type="llama", **kwargs):
        """使用XTuner风格生成流式回复"""
        # 构建对话
        inputs, template = self.build_conversation(message, history, system_prompt, model_type)
        
        # 简化调试信息，提高性能
        print(f"\n🤖 开始流式推理 - 模型类型: {model_type}")
        
        # 编码输入
        if len(history) == 0:
            # 第一轮对话
            ids = tokenizer.encode(inputs, return_tensors="pt")
        else:
            # 后续对话，不添加特殊token
            ids = tokenizer.encode(inputs, return_tensors="pt", add_special_tokens=False)
        
        # 确保输入数据在正确的设备上
        if torch.cuda.is_available():
            ids = ids.cuda()
        else:
            ids = ids.cpu()
        
        # 设置停止条件
        stop_words = template.STOP_WORDS
        
        # 创建停止条件
        def create_stopping_criteria(tokenizer, stop_words):
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopWordsCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_words):
                    self.tokenizer = tokenizer
                    self.stop_words = stop_words
                    self.stop_word_ids = []
                    for stop_word in stop_words:
                        stop_word_ids = tokenizer.encode(stop_word, add_special_tokens=False)
                        self.stop_word_ids.append(stop_word_ids)
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_word_ids in self.stop_word_ids:
                        if len(input_ids[0]) >= len(stop_word_ids):
                            if input_ids[0][-len(stop_word_ids):].tolist() == stop_word_ids:
                                return True
                    return False
            
            return StoppingCriteriaList([StopWordsCriteria(tokenizer, stop_words)])
        
        stopping_criteria = create_stopping_criteria(tokenizer, stop_words)
        
        # 优化生成参数，提高流畅度
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
            "streamer": None,  # 将在下面设置
            "use_cache": True,  # 启用缓存提高性能
        }
        
        # 创建流式生成器
        from transformers import TextIteratorStreamer
        import threading
        
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=10)
        generation_kwargs["streamer"] = streamer
        
        # 在后台线程中运行生成
        generation_thread = threading.Thread(
            target=model.generate,
            kwargs={"inputs": ids, **generation_kwargs}
        )
        generation_thread.start()
        
        # 流式输出
        accumulated_text = ""
        for new_text in streamer:
            if new_text:
                # 检查是否包含停止词
                should_stop = False
                cleaned_text = new_text
                
                for stop_word in template.STOP_WORDS:
                    if stop_word in new_text:
                        cleaned_text = new_text.replace(stop_word, "")
                        should_stop = True
                        break
                
                accumulated_text += cleaned_text
                # 只返回新生成的内容，不包含用户输入
                yield accumulated_text.strip()
                
                # 如果检测到停止词，提前结束
                if should_stop:
                    break
        
        # 等待生成线程完成
        generation_thread.join()
        
        final_response = accumulated_text.strip()
        print(f"✅ 流式生成完成")
        
        return final_response

# 创建全局实例
xtuner_chat = XTunerChat() 