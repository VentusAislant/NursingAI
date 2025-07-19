# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (
    DatasetInfoHook,
    EvaluateChatHook,
    VarlenAttnArgsToMessageHubHook,
)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE


def nursing_ai_map_fn(example):
    r"""Example before preprocessing:
    Example after preprocessing:
        example['conversation'] = [
            {
                'input': 'Can you explain xxx',
                'output': 'Sure! xxx'
            }
        ]
    """
    idx = example['id']
    num_turns = example['num_turns']
    input = example['input']
    output = example['output']

    to_return_conversation = [
        {"input": input, "output": output},
    ]

    return dict(
        conversation=to_return_conversation,
        idx=idx,
        num_turns=num_turns
    )


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = "pretrained_models/DeepSeek-R1-Distill-Qwen-7B"
use_varlen_attn = False

# Data
data_path = "./data/expert/data.json"
prompt_template = PROMPT_TEMPLATE.deepseek_qwen
max_length = 10000
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 4
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 30
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 980
save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 5000
SYSTEM = """您是一位经验丰富的临床护理专家，擅长评估护理问诊表现。您的任务是依据用户与病人智能体之间的问诊对话，对护生的整体问诊能力进行系统性评价。
请按照以下步骤依次进行评价：
步骤 1： 阅读用户与病人智能体之间的完整问诊对话内容。
步骤 2： 从以下七个维度对用户问诊表现进行逐项评估，并结合问诊对话内容提供具体分析与改进建议：
问诊准备：是否了解患者基本信息，是否具备良好的心理和环境准备。
问诊内容：信息采集是否全面、重点突出，包括主诉、现病史、既往史、个人史、家族史等。
问诊技巧：提问是否具有条理性，是否合理使用开放性与封闭性问题，引导是否得当。
问诊后处理：是否对信息进行了有效总结，是否提出了初步护理建议或明确下一步计划。
语言沟通：语言是否清晰、礼貌、具有亲和力，表达是否能被患者理解与接受。
问诊质量：问诊是否高效、系统、逻辑清晰，能否覆盖关键护理信息。
个人素质：是否体现出专业态度、责任心、同理心与职业礼仪。"""

evaluation_inputs = [
    "护士：请模拟一位患有克罗恩病的患者，与我进行护理问诊。\n患者：好的，我来扮演这位患有克罗恩病的患者，我们开始对话吧。\n护士：您好！我叫王小美，是负责您这段时间护理的护士。今天我来了解一下您的基本情况，这样我们能更好地为您提供帮助。您这次住院，主要是哪里不舒服呢？\n患者：哦，最近肚子不太舒服，几乎每次上厕所都有脓血便，时间有点长了，差不多四年吧。症状最近加重了，所以才过来看看的。\n护士：您说的这些症状已经有四年多了啊。\n患者：是的。\n护士：能告诉我您的姓名、年龄，还有您来自哪里吗？\n患者：我叫患者，26岁，陕西咸阳的，做快递员的。\n护士：您这次是怎么来的，是自己过来看病的吗？\n患者：对。\n护士：那在住院之前，您有没有去过其他医院治疗呢？\n患者：是的，之前去过门诊，医生说是克罗恩病，给我开了些药，但效果一般，症状没见好转，反而有点加重。\n护士：开的什么药，还知道吗？\n患者：没太记住\n护士：您的体重最近有变化吗？或者其他不舒服的地方呢？比如精神、食欲什么的。\n患者：嗯，体重变化不大，就是有时候没什么食欲，感觉精神也差，睡眠倒是还行。\n护士：您以前身体一直健康吗？有没有得过什么疾病或者做过手术？\n患者：以前身体挺好的，没得过什么大病，也没有做过手术。\n护士：那您的家里人有没有类似的病史呢？比如父母或兄弟姐妹有没有什么遗传性疾病？\n患者：没有，我爸妈身体都很好，家里没有类似病史。\n护士：您的生活习惯，工作环境怎么样？快递员的工作是不是比较累，压力大不大？\n患者：压力大，工作很辛苦，天天都在外面跑单，常常很累。天气热的时候特别容易出汗，体力消耗很大。\n护士：听起来工作确实挺辛苦的。那您平时有做什么锻炼吗？或者是有什么嗜好？\n患者：平时没有时间锻炼，工作就已经很累了，休息的时候就休息。偶尔去外面走走，放松一下。\n护士：那饮食方面有什么特别的习惯吗？喜欢吃辣或者油腻的食物吗？\n患者：我挺喜欢吃辣的，尤其是火锅，油腻的东西也经常吃，可能这也是病情加重的原因吧。\n护士：是的，辛辣油腻的食物对克罗恩病的患者不太好，可能会加重病情。为了我们快速得让疾病痊愈，减少身体的不适，我们这段时间还是要注意饮食清淡，不能再吃太油太辣的东西了。那最近排便的情况怎么样？除了脓血便，还有腹泻或者便秘之类的症状吗？\n患者：嗯，基本上还是腹泻，有时候粘液和脓血都一起，感觉肚子不太舒服。\n护士：那睡眠方面怎么样？有没有觉得特别不好入睡，或者需要药物帮助？\n患者：睡眠倒是还行，就是有时候感觉心情不好，睡得不是特别踏实，不过不算很严重。\n护士：好的，那对自己的病情，您有什么特别担心的地方吗？比如未来的治疗，或者是生活质量这些？\n患者：其实有点担心这个病会越来越严重，毕竟这种病好像是慢性的，听说可能得长期治疗。\n护士：您放心，我们会根据您的病情制定适合的治疗方案，帮助您控制症状，改善生活质量。慢性病需要长期管理，但通过合理的治疗和饮食调整，症状是可以得到控制的。\n患者：嗯，听您这么说心里稍微放心一点。\n护士：我们一起面对。最后，您现在有没有什么特别需要帮助的地方？\n患者：目前没什么，主要就是控制病情，希望能够尽快恢复。\n护士：好的，患者，您也不要太担心，我们会全力以赴帮您管理病情，确保您尽快康复。如果有任何问题，随时找我。感谢您的配合，祝您早日康复！\n患者：谢谢护士！",
 ]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="right",
    eos_token="<|im_end|>",
)

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    ),
    lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset,
        path='json',
        data_files=dict(train=data_path)
    ),
    tokenizer=tokenizer,
    max_length=max_length,
    # dataset_map_fn=oasst1_map_fn,
    dataset_map_fn=nursing_ai_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn,
)

sampler = SequenceParallelSampler if sequence_parallel_size > 1 else DefaultSampler

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
