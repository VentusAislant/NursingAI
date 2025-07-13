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
pretrained_model_name_or_path = "pretrained_models/MMed-Llama-3-8B"
use_varlen_attn = False

# Data
data_path = "./data/expert/data.json"
prompt_template = PROMPT_TEMPLATE.llama3_chat
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
SYSTEM = ""

evaluation_inputs = [
    "护士：请模拟一位患有克罗恩病的患者，与我进行护理问诊。\n患者：好的，我来扮演这位患有克罗恩病的患者，我们开始对话吧。\n护士：您好！我叫王小美，是负责您这段时间护理的护士。今天我来了解一下您的基本情况，这样我们能更好地为您提供帮助。您这次住院，主要是哪里不舒服呢？\n患者：哦，最近肚子不太舒服，几乎每次上厕所都有脓血便，时间有点长了，差不多四年吧。症状最近加重了，所以才过来看看的。\n护士：您说的这些症状已经有四年多了啊。\n患者：是的。\n护士：能告诉我您的姓名、年龄，还有您来自哪里吗？\n患者：我叫患者，26岁，陕西咸阳的，做快递员的。\n护士：您这次是怎么来的，是自己过来看病的吗？\n患者：对。\n护士：那在住院之前，您有没有去过其他医院治疗呢？\n患者：是的，之前去过门诊，医生说是克罗恩病，给我开了些药，但效果一般，症状没见好转，反而有点加重。\n护士：开的什么药，还知道吗？\n患者：没太记住\n护士：您的体重最近有变化吗？或者其他不舒服的地方呢？比如精神、食欲什么的。\n患者：嗯，体重变化不大，就是有时候没什么食欲，感觉精神也差，睡眠倒是还行。\n护士：您以前身体一直健康吗？有没有得过什么疾病或者做过手术？\n患者：以前身体挺好的，没得过什么大病，也没有做过手术。\n护士：那您的家里人有没有类似的病史呢？比如父母或兄弟姐妹有没有什么遗传性疾病？\n患者：没有，我爸妈身体都很好，家里没有类似病史。\n护士：您的生活习惯，工作环境怎么样？快递员的工作是不是比较累，压力大不大？\n患者：压力大，工作很辛苦，天天都在外面跑单，常常很累。天气热的时候特别容易出汗，体力消耗很大。\n护士：听起来工作确实挺辛苦的。那您平时有做什么锻炼吗？或者是有什么嗜好？\n患者：平时没有时间锻炼，工作就已经很累了，休息的时候就休息。偶尔去外面走走，放松一下。\n护士：那饮食方面有什么特别的习惯吗？喜欢吃辣或者油腻的食物吗？\n患者：我挺喜欢吃辣的，尤其是火锅，油腻的东西也经常吃，可能这也是病情加重的原因吧。\n护士：是的，辛辣油腻的食物对克罗恩病的患者不太好，可能会加重病情。为了我们快速得让疾病痊愈，减少身体的不适，我们这段时间还是要注意饮食清淡，不能再吃太油太辣的东西了。那最近排便的情况怎么样？除了脓血便，还有腹泻或者便秘之类的症状吗？\n患者：嗯，基本上还是腹泻，有时候粘液和脓血都一起，感觉肚子不太舒服。\n护士：那睡眠方面怎么样？有没有觉得特别不好入睡，或者需要药物帮助？\n患者：睡眠倒是还行，就是有时候感觉心情不好，睡得不是特别踏实，不过不算很严重。\n护士：好的，那对自己的病情，您有什么特别担心的地方吗？比如未来的治疗，或者是生活质量这些？\n患者：其实有点担心这个病会越来越严重，毕竟这种病好像是慢性的，听说可能得长期治疗。\n护士：您放心，我们会根据您的病情制定适合的治疗方案，帮助您控制症状，改善生活质量。慢性病需要长期管理，但通过合理的治疗和饮食调整，症状是可以得到控制的。\n患者：嗯，听您这么说心里稍微放心一点。\n护士：我们一起面对。最后，您现在有没有什么特别需要帮助的地方？\n患者：目前没什么，主要就是控制病情，希望能够尽快恢复。\n护士：好的，患者，您也不要太担心，我们会全力以赴帮您管理病情，确保您尽快康复。如果有任何问题，随时找我。感谢您的配合，祝您早日康复！\n患者：谢谢护士！",
    "疾病诊断：克罗恩病\n| 三级指标 | 评价内容 | 评分 | 改进建议 |\n| -------- | -------- | ---- | -------- |\n| A1-1护生态度和蔼有礼 | 全程使用“您好”“您”等礼貌用语，态度关切（“不要担心，我们会帮助您”）。 | 3 | 无 |\n| A1-2对患者使用恰当的称呼 | 以“您”称呼患者，但未结合患者姓名或更具体称谓（如“李先生”）。 | 1 | 建议补充：“李先生”以增强尊重感。 |\n| A1-3主动介绍自己，解释说明来意、目的和所需时间，征求患者和家属的同意 | 主动介绍身份和目的（“了解基本情况”），但未说明问诊时间或征求同意。 | 2 | 建议补充：“大约需要15分钟，您方便吗？”。 |\n| A2-1询问患者一般情况、入院原因、症状及治疗经过 | 详细询问入院原因（脓血便、腹泻）、症状（食欲差、精神差）、治疗经过（克罗恩病用药史）。 | 3 | 无 |\n| A2-2问诊患者营养、代谢与排泄相关内容：如患者饮食、每日饮水量、影响进食的因素、排便、排尿习惯等 | 询问饮食（“喜辣、油腻”）及排便（脓血便、腹泻），但未量化饮水量。 | 2 | 建议补充：“您每天大约喝多少水？”。 |\n| A2-3问诊患者活动与运动相关内容：如患者的活动形式、能力、耐力、疾病及用药情况 | 询问活动习惯（“偶尔散步”），但未评估运动耐力或活动受限情况。 | 1 | 建议补充：“您工作后是否感到极度疲劳？”。 |\n| A2-4问诊患者睡眠与休息相关内容：如患者的日常睡眠形态、睡前习惯、有无失眠、有无嗜睡、疾病及药物情况 | 直接询问睡眠质量（“睡得不太踏实”）。 | 3 | 无 |\n| A2-5问诊患者性与生殖相关内容：如患者的家族史、生育史等 | 未涉及生育史或家族遗传病史（仅问及家族健康）。 | 0 | 建议补充：“您是否有生育计划？家族有消化道疾病史吗？”。 |\n| A2-6问诊患者与疾病有关的认知与感知相关内容：如疾病过程中视、听、味、嗅觉等感知功能状态；思维、语言、定向力及意识状态及个体对认知和感知功能改变的反应 | 未评估感知功能（如腹痛是否伴随恶心、呕吐）。 | 0 | 建议补充：“您是否有其他消化道症状？如恶心或呕吐？”。 |\n| A2-7问诊患者压力与应对相关内容：如患者对疾病的了解程度、恐惧心理、渴望救治或放弃等及对压力应对方式等 | 关注患者心理压力（“担心病情加重”），但未深入应对方式。 | 1 | 建议追问：“您目前如何缓解焦虑情绪？”。 |\n| A2-8问诊患者健康感知及健康状态相关内容：如患者对自我健康状况的认知情况、有无影响健康的危险因素、维护健康采取的措施及多长时间进行一次健康体检等 | 询问饮食习惯，但未涉及健康体检频率或疾病管理措施（如肠镜检查）。 | 1 | 建议补充：“您过去做过肠镜检查吗？”。 |\n| A2-9问诊患者自我概念相关内容：如患者的社会及自我认同及自尊的情况等 | 未涉及疾病对自我认同的影响（如工作能力受限）。 | 0 | 建议补充：“疾病是否影响您的工作自信？”。 |\n| A2-10问诊患者角色和关系形态相关内容：如患者的角色、社会关系、家庭角色和家庭关系等 | 询问工作压力（“快递员工作累”），但未分析家庭角色分工。 | 2 | 可补充：“您在家中的责任是否因疾病受到影响？”。 |\n| A2-11问诊患者的价值与信念相关内容：如患者的价值观与信念及精神困扰等 | 未涉及价值观或精神困扰。 | 0 | 建议补充：“您是否有支撑自己的信念或应对策略？”。 |\n| A2-12掌握疾病相关知识，能根据疾病知识重点询问与疾病有关的内容，包括潜在并发症的病情信息 | 未询问克罗恩病潜在并发症（如肠梗阻、肛周病变）。 | 0 | 建议补充：“您是否有过腹部剧痛或肛周脓肿？”。 |\n| A3-1使用开放性问题询问患者情况，不诱导患者 | 使用开放性问题（“您这次住院主要是哪里不舒服？”）。 | 3 | 无 |\n| A3-2合理使用过渡语言，使问诊过程连贯 | 过渡语言较生硬（“接下来我想了解”）。 | 1 | 建议补充自然过渡句（如“关于您的饮食习惯…”）。 |\n| A3-3问诊语言通俗易懂，必要时重复提问，以总结或阐明问题 | 语言通俗易懂（“脓血便”“肚子不舒服”）。 | 3 | 无 |\n| A3-4能控制问诊的内容，自然转变与问诊无关内容，不随意打断患者，鼓励患者提问，灵活回答患者问题 | 未主动鼓励患者提问（如“您对饮食调整有疑问吗？”）。 | 2 | 建议补充：“您对治疗方案有什么想了解的？”。 |\n| A3-5检查患者对谈话内容的理解程度 | 未检查患者对谈话内容的理解程度。 | 0 | 建议补充：“我刚才说的内容您都清楚了吗？”。 |\n| A3-6用复述、意译、澄清、总结等方式引证核实患者提供的信息 | 未通过复述或总结核实信息（如“您说腹泻四年对吗？”）。 | 0 | 建议复述关键信息以确认准确性。 |\n| A4-1安慰患者，对影响其疾病的相关因素适时指导，并对患者的支持和配合表示感谢，尽可能满足患者的就医需求或期望 | 安慰患者并承诺支持（“我们会制定治疗方案”）。 | 3 | 无 |\n| B1-1问诊语言简明扼要，重点突出 | 语言简明（“您这次住院主要是哪里不舒服？”）。 | 3 | 无 |\n| B1-2所说话语要有一定的信任度，取得患者信赖 | 通过共情（“不要担心，我们会帮助您”）建立信任。 | 3 | 无 |\n| C1-1问诊方式合理，有条理，与患者身份相符，内容真实全面 | 内容覆盖症状、饮食习惯，但缺并发症及心理影响评估。 | 2 | 需补充肠梗阻风险及心理状态评估。 |\n| C1-2问诊重点突出，技巧娴熟 | 围绕克罗恩病核心症状展开，但未深入疾病管理细节。 | 1 | 建议补充对营养支持的关注（如维生素缺乏）。 |\n| C2-1问诊过程中体现人文关怀意识，关心病人，尊重患者 | 体现人文关怀（“希望您早日康复”）。 | 3 | 无 |\n| C2-2有较高的专业素养，问诊过程思维清晰 | 逻辑清晰，从症状到治疗逐步推进。 | 3 | 无 |\n\n### 总结部分\n- 总分：46/84\n- 关键优点：\n1. 核心症状覆盖全面：详细询问消化道症状、饮食及工作压力（A2-1询问患者一般情况、入院原因、症状及治疗经过、A2-10问诊患者角色和关系形态相关内容：如患者的角色、社会关系、家庭角色和家庭关系等）。\n2. 沟通亲和力强：语言简洁，共情表达自然（B1-2所说话语要有一定的信任度，取得患者信赖、A3-3问诊语言通俗易懂，必要时重复提问，以总结或阐明问题）。\n3. 人文关怀明确：关注患者心理压力并承诺支持（C2-1问诊过程中体现人文关怀意识，关心病人，尊重患者、A4-1安慰患者，对影响其疾病的相关因素适时指导，并对患者的支持和配合表示感谢，尽可能满足患者的就医需求或期望）。\n- 改进建议：\n1. 并发症管理不足：未评估肠梗阻或肛周病变风险（A2-12掌握疾病相关知识，能根据疾病知识重点询问与疾病有关的内容，包括潜在并发症的病情信息）。\n2. 营养评估缺失：未询问体重变化或维生素缺乏（A2-2问诊患者营养、代谢与排泄相关内容：如患者饮食、每日饮水量、影响进食的因素、排便、排尿习惯等）。\n3. 心理支持薄弱：需补充对患者心理状态及应对方式的关注（A2-9问诊患者自我概念相关内容：如患者的社会及自我认同及自尊的情况等、A2-7问诊患者压力与应对相关内容：如患者对疾病的了解程度、恐惧心理、渴望救治或放弃等及对压力应对方式等）。"
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="right",
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
        r=64,
        lora_alpha=64,
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
