## Install
1. Create Env

```bash
conda create --name nursing_ai python=3.10 -y
conda activate nursing_ai
```

2. Install XTuner

```bash
pip installt torch torchvision

git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'

# if error `RuntimeError: Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback): No module named triton.ops`
pip install bitsandbytes --upgrade
```

## Fine-tune

Step 0, prepare the config. XTuner provides many ready-to-use configs and we can view all configs by

```bash
xtuner list-cfg
```
Or, if the provided configs cannot meet the requirements, please copy the provided config to the specified directory and make specific modifications by

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
vi ${SAVE_PATH}/${CONFIG_NAME}_copy.py
```

Step 1, start fine-tuning.

```bash
xtuner train ${CONFIG_NAME_OR_PATH}
```
For example, we can start the QLoRA fine-tuning of InternLM2.5-Chat-7B with oasst1 dataset by

### On a single GPU

```bash
xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
```

### On multiple GPUs

```bash
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
(SLURM) srun ${SRUN_ARGS} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
--deepspeed means using DeepSpeed 🚀 to optimize the training. XTuner comes with several integrated strategies including ZeRO-1, ZeRO-2, and ZeRO-3. If you wish to disable this feature, simply remove this argument.
```

Step 2, convert the saved PTH model (if using DeepSpeed, it will be a directory) to Hugging Face model, by

```bash
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
```

## Chat
XTuner provides tools to chat with pretrained / fine-tuned LLMs.

```bash
xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter {NAME_OR_PATH_TO_ADAPTER} [optional arguments]
```
For example, we can start the chat with InternLM2.5-Chat-7B :

```bash
xtuner chat internlm/internlm2_5-chat-7b --prompt-template internlm2_chat
```
For more examples, please see chat.md.

## Deployment

Step 0, merge the Hugging Face adapter to pretrained LLM, by

```bash
xtuner convert merge \
    ${NAME_OR_PATH_TO_LLM} \
    ${NAME_OR_PATH_TO_ADAPTER} \
    ${SAVE_PATH} \
    --max-shard-size 2GB
```
Step 1, deploy fine-tuned LLM with any other framework, such as LMDeploy 🚀.

```bash
pip install lmdeploy
python -m lmdeploy.pytorch.chat ${NAME_OR_PATH_TO_LLM} \
    --max_new_tokens 256 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```

## LOG

- 2025.07.19

  - 发现 `DeepSeek_R1_Distill_Qwen_7B` 版本的所有模型的聊天模板是 deepseek 重写的:

    ```shell
     "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
    ```

  - 但是之前全部默认使用的 `PROMPT_TEMPLATE.qwen_chat`

    ```python
    qwen_chat=dict(
            SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
            INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
            SUFFIX="<|im_end|>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
        ),
    ```

  - 明显模板不符，导致训练出来的模型无法自动停止，所以全部需要重新训练

  - 新写了一个提示模板

    ```python
    deepseek_qwen=dict(
            SYSTEM="{system}\n\n",
            INSTRUCTION="<｜User｜>{input}<｜Assistant｜>",
            SUFFIX="<｜end▁of▁sentence｜>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<｜end▁of▁sentence｜>"],
        ),
    ```

  - 将所有的 `DeepSeek_R1_Distill_Qwen_7B` 模型加入 `archive` 重新训练新的

