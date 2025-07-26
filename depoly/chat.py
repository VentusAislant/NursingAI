import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from typing import List, Dict, Optional, Generator
import threading
import gc

class Chat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history: List[Dict[str, str]] = []  # [{"role":..., "content":...}]
        self.model_type = None
        self.stopping_criteria = None

    def clear_model(self):
        """清理当前模型，释放GPU内存"""
        if self.model is not None:
            # 将模型移动到CPU
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            # 删除模型
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        print("✅ 模型已清理，GPU内存已释放")

    def load_model(self, model_name_or_path: str, adapter_path: Optional[str] = None, torch_dtype: str = "fp16", bits: Optional[int] = None):
        # 先清理之前的模型
        self.clear_model()
        
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(torch_dtype, torch.float16)
        
        # 确保模型加载到正确的设备
        if torch.cuda.is_available():
            # 使用device_map="auto"让模型自动分配到GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=dtype, 
                device_map="auto"
            )
        else:
            # CPU模式
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=dtype,
                device_map="cpu"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 设置pad_token以避免attention_mask警告
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 检测模型类型
        self._detect_model_type(model_name_or_path)
        
        # 设置停止词标准
        self._setup_stopping_criteria()
            
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
        
        # 打印模型设备信息
        if hasattr(self.model, 'device'):
            print(f"✅ 模型已加载到设备: {self.model.device}")
        else:
            print(f"✅ 模型已加载，当前设备: {self.device}")
        
        return True
    
    def _detect_model_type(self, model_name_or_path: str):
        """检测模型类型"""
        model_name_lower = model_name_or_path.lower()
        if "llama" in model_name_lower or "deepseek" in model_name_lower:
            self.model_type = "llama"
        elif "qwen" in model_name_lower:
            self.model_type = "qwen"
        else:
            # 默认使用llama配置
            self.model_type = "llama"
        print(f"检测到模型类型: {self.model_type}")
    
    def _setup_stopping_criteria(self):
        """根据模型类型设置停止词标准"""
        # 根据模型类型获取stop words
        if self.model_type == "llama":
            stop_words = ["<|eot_id|>", '<|end_of_text|>']
        elif self.model_type == "qwen":
            stop_words = ["REDACTED_SPECIAL_TOKEN", '<｜end▁of▁sentence｜>']
        else:
            # 默认stop words
            stop_words = ["</s>", "<eot_id>", "END", "Human:", "Assistant:", "User:", "Bot:"]
        
        # 创建停止词标准
        class StopWordsCriteria(StoppingCriteria):
            def __init__(self, stop_words_ids):
                self.stop_words_ids = stop_words_ids
            
            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_words_ids:
                    if len(input_ids[0]) >= len(stop_ids):
                        if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                            return True
                return False
        
        # 将stop words转换为token ids
        stop_words_ids = []
        for word in stop_words:
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if word_ids:
                stop_words_ids.append(word_ids)
        
        if stop_words_ids:
            self.stopping_criteria = StoppingCriteriaList([StopWordsCriteria(stop_words_ids)])
            print(f"设置停止词: {stop_words}")
        else:
            self.stopping_criteria = None

    def build_prompt(self) -> str:
        # 合并历史和当前输入
        messages = self.history
        try:
            # 尝试使用tokenizer的chat模板
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except AttributeError:
            print('apply_chat_template ERROR')
            import sys
            sys.exit(1)

    # def generate_response(self, message: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.1, **kwargs) -> str:
    #     if self.model is None or self.tokenizer is None:
    #         raise RuntimeError("模型未加载")
    #     prompt = self.build_prompt(message)
    #     input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
    #
    #     gen_config = GenerationConfig(
    #         max_new_tokens=max_new_tokens,
    #         temperature=temperature,
    #         do_sample=temperature > 0,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         pad_token_id=self.tokenizer.eos_token_id,
    #     )
    #
    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             inputs=input_ids,
    #             generation_config=gen_config,
    #             stopping_criteria=self.stopping_criteria
    #         )
    #     response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    #     # 更新历史
    #     self.history.extend(message)
    #     return response

    def generate_response_stream(self, max_new_tokens: int = 512, temperature: float = 0.1, **kwargs) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载")
        
        print('===========')
        print(f'Current History')
        print(self.history)
        prompt = self.build_prompt()
        print('===========')
        print('Current Prompt:')
        print(prompt)
        
        # 获取模型当前设备
        model_device = next(self.model.parameters()).device
        print(f"模型设备: {model_device}")
        
        # 确保输入数据在正确的设备上
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(model_device)
        print(f"输入数据设备: {input_ids.device}")
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs={
            "inputs": input_ids,
            "generation_config": gen_config,
            "streamer": streamer,
            "stopping_criteria": self.stopping_criteria,
        })
        thread.start()
        accumulated_response = ""
        for new_text in streamer:
            accumulated_response += new_text
            yield accumulated_response

        thread.join()
        del input_ids
        torch.cuda.empty_cache()
        print('*******************************')
        print(accumulated_response)
        # 更新历史
        self.history.append({"role": "assistant", "content": accumulated_response})

    def clear_history(self):
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()

# 全局实例
chat = Chat()
