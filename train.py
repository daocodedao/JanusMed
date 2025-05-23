"""
Reproduce JanusPro training
"""
import argparse

# import datetime
import logging
import math
import os
import sys
import time

import yaml

import mindspore as ms
from mindspore import nn
from mindspore._c_expression import reset_op_id
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig
from janus.train.lr_schedule import WarmupCosineDecayLR
from janus.train.t2i_dataset import create_dataloader_t2i
from janus.train.text_dataset import create_dataloader_text
from janus.train.unified_dataset import create_dataloader_unified
from janus.train.unified_dataset_weightedrandsamp import create_unified_dataloader_weightrandsamp
from janus.train.vqa_dataset import create_dataloader_vqa
from janus.utils.io import set_model_param_dtype

from mindone.trainers.checkpoint import CheckpointManager

# from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.transformers.mindspore_adapter.clip_grad import clip_grad_norm

# from mindone.trainers.zero import prepare_train_network
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(mode, seed, distribute=False):
    # 设置随机种子，保证实验可复现
    set_random_seed(seed)
    # ms.set_context(max_device_memory=max_device_memory) # 设置最大设备内存（可选）
    # 设置MindSpore的执行模式（GRAPH_MODE或PYNATIVE_MODE）
    ms.set_context(mode=mode)
    # 设置JIT编译级别，O0表示不优化，有助于调试
    ms.set_context(jit_config={"jit_level": "O0"})

    if distribute:
        # 分布式训练环境初始化
        ms.set_context(mode=mode)
        init()  # 初始化通信环境
        device_num = get_group_size()  # 获取设备总数
        rank_id = get_rank()  # 获取当前设备的ID
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()  # 重置自动并行上下文

        # 设置自动并行配置
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,  # 数据并行模式
            gradients_mean=True,  # 梯度聚合时取平均值
            device_num=device_num,
        )
    else:
        # 单设备训练
        device_num = 1
        rank_id = 0
        ms.set_context(mode=mode)

    return rank_id, device_num


def main(args):
    # 0. 环境初始化
    # time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") # 时间戳，用于创建唯一的输出路径（可选）
    # args.output_path = os.path.join(args.output_path, time_str) # 将时间戳加入输出路径

    # 初始化MindSpore执行环境和分布式设置
    rank_id, device_num = init_env(
        args.ms_mode,
        args.seed,
        distribute=args.use_parallel,
    )

    # 设置日志记录器
    set_logger(name="", output_dir=args.output_path, rank=rank_id)

    # 1. Janus模型初始化
    # 从预训练路径加载VLChatProcessor，用于数据预处理
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)

    # 从预训练路径加载模型配置
    config = MultiModalityConfig.from_pretrained(args.model_path)
    config.torch_dtype = args.dtype  # 设置模型数据类型
    config.language_config.torch_dtype = args.dtype # 设置语言模型数据类型
    config.language_config._attn_implementation = "flash_attention_2"  # 默认使用FlashAttention 2
    if args.load_weight:
        # 从预训练路径加载模型权重
        vl_gpt = MultiModalityCausalLM.from_pretrained(args.model_path, config=config)
    else:
        # 不加载预训练权重，仅初始化模型结构
        # with no_init_parameters(): # （可选）上下文管理器，用于不初始化参数
        vl_gpt = MultiModalityCausalLM(config=config)

    if args.ckpt_path is not None:
        # 如果指定了ckpt路径，则加载指定的checkpoint
        parameter_dict = ms.load_checkpoint(args.ckpt_path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(vl_gpt, parameter_dict, strict_load=True)
        logger.info("网络中未加载的参数: {}".format(param_not_load))
        logger.info("checkpoint中未加载的参数: {}".format(ckpt_not_load))

    # 1.1 混合精度设置
    dtype_map = {"float16": ms.float16, "bfloat16": ms.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.dtype != "float32":
        # 如果不是float32，则将模型参数转换为指定的数据类型（float16或bfloat16）
        vl_gpt = set_model_param_dtype(vl_gpt, dtype)

    # 1.2 设置可训练参数 (参考Janus论文)
    # TODO: 使用config.yaml来设置训练策略
    num_frozen_params = 0  # 冻结参数数量
    num_train_params = 0   # 可训练参数数量
    # 定义模型的所有模块
    all_modules = set(
        [
            vl_gpt.vision_model,
            vl_gpt.gen_vision_model,
            vl_gpt.language_model,
            vl_gpt.aligner,
            vl_gpt.gen_aligner,
            vl_gpt.gen_head,
            vl_gpt.gen_embed,
        ]
    )
    if args.training_stage == 1:
        # 阶段一：训练适配器和图像头
        # 冻结 sigLIP, VQ16, LLM；训练适配器和图像头
        frozen_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model, vl_gpt.language_model])
    elif args.training_stage == 2:
        # 阶段二：统一预训练
        # 进一步解冻LLM
        frozen_modules = set([vl_gpt.vision_model, vl_gpt.gen_vision_model])
    elif args.training_stage == 3:
        # 阶段三：SFT (Supervised Fine-Tuning)
        # 仅冻结生成式视觉自编码器(VQ)；训练其他所有模块：生成适配器，理解编码器(sigLIP) + 理解适配器, LLM, 文本头, 图像头
        # TODO: gen_embed (nn.Embedding) 在阶段三是否应该可训练
        frozen_modules = set([vl_gpt.gen_vision_model])
    else:
        raise NotImplementedError("未知的训练阶段")

    # 计算可训练模块
    trainable_modules = all_modules - frozen_modules

    # 冻结指定模块的参数
    for module in frozen_modules:
        module.set_train(False)  # 设置模块为非训练模式
        for param in module.get_parameters():
            param.requires_grad = False  # 参数不需要梯度
            num_frozen_params += 1

    # 设置可训练模块的参数
    for module in trainable_modules:
        module.set_train(True)  # 设置模块为训练模式
        for param in module.get_parameters():
            param.requires_grad = True  # 参数需要梯度
            num_train_params += 1

    # VQ编码器不需要梯度
    vl_gpt.gen_vision_model.set_grad(requires_grad=False)

    # 当在图模式下进行混合数据SFT时，设置动态shape
    if args.task == "mixed" and args.ms_mode == 0: # ms_mode 0 表示 GRAPH_MODE
        print("设置动态shape")
        input_ids = ms.Tensor(shape=[None, args.max_length], dtype=ms.int32)
        inputs_embeds = ms.Tensor(
            shape=[None, args.max_length, config.language_config.hidden_size],
            dtype=ms.bfloat16, # 注意：这里使用了bfloat16，应与模型dtype一致
        )
        attention_mask = ms.Tensor(shape=[None, args.max_length], dtype=ms.bool_)
        # 为语言模型设置动态输入
        vl_gpt.language_model.model.set_inputs(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
    else:
        print("未设置动态shape")

    # 调试选项：冻结文本和图像的token embedding table
    freeze_embed_tables = args.freeze_embedding
    if freeze_embed_tables:
        for module in (vl_gpt.gen_embed, vl_gpt.language_model.model.embed_tokens):
            module.set_train(False)
            for param in module.get_parameters():
                param.requires_grad = False
    tot_params = len(list(vl_gpt.get_parameters())) # 模型总参数量
    print(f"总参数量: {tot_params}, 可训练参数量: {num_train_params}, 冻结参数量: {num_frozen_params}")
    assert num_frozen_params + num_train_params == tot_params, "所有参数都应设置为可训练或冻结状态。"
    # 1.3 保存模型配置
    config.save_pretrained(args.output_path)

    # 2. 准备数据集和数据加载器
    task = args.task
    if task == "vqa":
        # 创建VQA任务的数据加载器
        dataloader = create_dataloader_vqa(
            dataset_name=args.dataset_name,
            data_dir=args.vqa_data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_samples=args.num_samples,
        )
    elif task == "text":
        # FIXME: 允许设置路径
        # 创建纯文本任务的数据加载器
        dataloader = create_dataloader_text(
            dataset_name=args.dataset_name,
            data_dir=args.text_qa_data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_samples=args.num_samples,
        )
    elif task == "t2i":
        # 创建文本到图像生成任务的数据加载器
        dataloader = create_dataloader_t2i(
            vl_chat_processor=vl_chat_processor,
            csv_path=args.t2i_csv_path,
            data_dir=args.t2i_data_dir,
            parquet_dir=args.t2i_parquet_dir,
            max_token_length=args.max_length,
            image_size=args.image_size,
            null_prompt_prob=args.null_prompt_prob,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_samples=args.num_samples,
        )
    elif task == "mixed":
        # 创建混合任务的数据加载器
        if args.mixed_task_rand_samp:
            # 使用带权重随机采样的方式创建混合数据加载器
            dataloader = create_unified_dataloader_weightrandsamp(
                vl_chat_processor,
                t2i_csv_path=args.t2i_csv_path,
                t2i_data_path=args.t2i_data_path, # 注意：这里参数名可能与create_dataloader_t2i中的data_dir不同
                t2i_parquet_dir=args.t2i_parquet_dir,
                text_data_dir=args.text_qa_data_dir,
                vqa_data_dir=args.vqa_data_dir,
                max_token_length=args.max_length,
                image_size=args.image_size,
                null_prompt_prob=args.null_prompt_prob,
                batch_size=args.batch_size,
                num_samples=100, # 示例样本数，可能需要调整
                shuffle=args.shuffle,
            )
        else:
            # 使用预先切分数据集的方式创建混合数据加载器
            dataloader = create_dataloader_unified(
                vl_chat_processor,
                t2i_csv_path=args.t2i_csv_path,
                t2i_data_path=args.t2i_data_path, # 注意：这里参数名可能与create_dataloader_t2i中的data_dir不同
                vqa_data_dir=args.vqa_data_dir,
                text_qa_data_dir=args.text_qa_data_dir,
                num_samples_vqa=100, # VQA样本数
                num_samples_puretext=20, # 纯文本样本数
                num_samples_t2i=80, # T2I样本数
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                max_token_length=args.max_length,
                image_size=args.image_size,
                null_prompt_prob=args.null_prompt_prob,
            )
    else:
        raise NotImplementedError("未知的任务类型")
    # task_map = {"vqa": 0, "text": 1, "t2i": 2} # 任务映射（可选）

    # 3. 设置训练器和配置超参数
    # loss_scaler = nn.FixedLossScaleUpdateCell(1024)  # 固定损失缩放器（可选，用于混合精度）
    # 定义优化器
    optimizer = ms.mint.optim.AdamW(
        vl_gpt.trainable_params(),  # 仅优化可训练参数
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-6,
    )
    assert args.warmup_steps < args.train_steps, "预热步数必须小于总训练步数"
    # 定义学习率调度器：预热+余弦衰减
    scheduler = WarmupCosineDecayLR(
        optimizer,
        lr_max=args.learning_rate,
        lr_min=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.train_steps - args.warmup_steps, # 总训练步数减去预热步数
    )

    use_value_and_grad = args.use_value_and_grad # 是否使用MindSpore的value_and_grad自定义训练步骤
    if use_value_and_grad:
        # 自定义前向传播函数
        def forward_fn(*data):
            loss = vl_gpt(*data)
            return loss

        # 获取梯度函数
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
        if args.use_parallel:
            # 分布式训练时，需要梯度聚合
            grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

        # 自定义训练步骤函数
        def train_step(data):
            loss, grads = grad_fn(*data) # 计算损失和梯度
            if args.use_parallel:
                grads = grad_reducer(grads) # 聚合梯度

            # FIXME: 添加此分支后，每步时间成本增加约150ms
            if args.clip_grad:
                # 梯度裁剪
                grads = clip_grad_norm(grads, args.max_grad_norm)

            optimizer(grads) # 更新参数

            return loss
    else:
        # 使用MindOne封装的TrainOneStepWrapper
        train_step = TrainOneStepWrapper(
            vl_gpt,
            optimizer=optimizer,
            scale_sense=ms.Tensor(1.0),  # 损失缩放系数（可调）
            clip_grad=True,  # 是否进行梯度裁剪（可调）
            clip_norm=5.0,  # 梯度裁剪范数（可调）
            # ema=ema, # 指数移动平均（可选）
            # zero_stage=args.zero_stage, # ZeRO优化阶段（可选）
        )

    # TODO: 对于序列并行，需要为其他rank保存ckpt
    ckpt_dir = os.path.join(args.output_path, "ckpt") # checkpoint保存目录
    # TODO: 支持断点续训
    start_epoch = 0 # 起始epoch
    start_global_step = 0 # 起始全局步数
    if rank_id == 0: # 仅在rank 0设备上执行保存操作
        # Checkpoint管理器
        ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        # 性能记录器列名
        perf_columns = ["step", "loss", "train_time(s)"]
        output_dir = ckpt_dir.replace("/ckpt", "") # 主输出目录
        if start_epoch == 0:
            # 从头开始训练，创建新的性能记录器
            record = PerfRecorder(output_dir, metric_names=perf_columns)
        else:
            # 断点续训，恢复性能记录器
            record = PerfRecorder(output_dir, resume=True)

        # 保存参数配置到yaml文件
        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

        logger.info("开始训练...")

    # 4. 训练循环
    start_time_s = time.time() # 记录开始时间

    # ds_iter = dataloader.create_tuple_iterator(num_epochs=num_epochs - start_epoch) # 创建数据迭代器，指定epoch数
    ds_iter = dataloader.create_tuple_iterator(num_epochs=-1) # 创建数据迭代器，无限循环（由外部控制停止）
    num_batches = dataloader.get_dataset_size() # 每个epoch的批次数
    num_epochs = math.ceil(args.train_steps / num_batches) # 根据总训练步数和批次数计算总epoch数
    global_step = start_global_step # 当前全局步数

    for epoch in range(start_epoch + 1, num_epochs + 1):
        # for step in range(args.train_steps): # 按总步数迭代（另一种方式）
        for step, data in enumerate(ds_iter, 1): # 遍历数据加载器
            """
            数据格式示例:
            data = (ms.Tensor(input_ids, dtype=ms.int32),      # 输入token ID
                ms.Tensor(labels, dtype=ms.int32),          # 标签token ID
                ms.Tensor(attention_masks, dtype=ms.bool_), # 注意力掩码
                ms.Tensor(image_seq_masks, dtype=ms.bool_), # 图像序列掩码
                ms.Tensor(image, dtype=dtype),              # 图像像素值
                )
            """
            # 将图像数据的类型转换为模型所需的类型 (例如 bfloat16)
            data[-1] = data[-1].to(dtype)

            if use_value_and_grad:
                loss = train_step(data) # 执行自定义训练步骤
            else:
                loss, overflow, scaling_sens = train_step(*data) # 执行TrainOneStepWrapper训练步骤

            step_time = time.time() - start_time_s # 计算当前step耗时
            global_step += 1 # 全局步数加1
            loss_val = float(loss.asnumpy()) # 获取loss值

            scheduler.step() # 更新学习率
            cur_lr = scheduler.get_last_lr()[0].asnumpy() # 获取当前学习率
            # print("lr", [lr for lr in optimizer.lrs]) # 打印优化器中的学习率（调试用）

            logger.info(
                f"epoch {epoch}, step {step}, loss {loss_val:.8f}, lr {cur_lr:.7f}, step time {step_time*1000:.2f}ms"
            )

            if rank_id == 0: # rank 0记录性能数据
                step_pref_value = [global_step, loss_val, step_time]
                record.add(*step_pref_value)

            # 保存checkpoint
            if (global_step > 0) and (global_step % args.ckpt_save_steps == 0):
                ckpt_name = f"model-s{global_step}.ckpt"
                ckpt_manager.save(vl_gpt, None, ckpt_name=ckpt_name, append_dict=None)
            start_time_s = time.time() # 重置step开始时间

            # 允许在最后一个step + 1停止以便测量，因为通常在图模式下会dump第一个step
            if global_step == args.train_steps + 1: # 注意这里是 global_step
                break
        if global_step >= args.train_steps: # 如果全局步数达到目标，则跳出epoch循环
            break


    logger.info(f"训练完成。请在 {args.output_path} 查看结果")
    reset_op_id() # 重置操作ID（MindSpore内部机制）
    logger.info("结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=1, help="MindSpore运行模式, 0: GRAPH_MODE, 1: PYNATIVE_MODE")
    # TODO: 支持模型名称 "deepseek-ai/Janus-Pro-1B" 以简化操作
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/Janus-Pro-1B",
        help="Janus模型所在的路径",
    )
    parser.add_argument(
        "--training_stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="模型训练阶段, 可选 1, 2, 或 3",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="模型checkpoint文件路径(.ckpt格式), 如果为None, 则使用model_path中的预训练权重",
    )
    parser.add_argument(
        "--load_weight",
        type=str2bool,
        default=True,
        help="如果为True, 则加载model_path中的预训练权重; 如果为False, 则不加载。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="模型数据类型",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="是否使用并行训练")
    parser.add_argument(
        "--use_value_and_grad",
        default=True,
        type=str2bool,
        help="如果为False, 使用MindOne封装的trainer。如果为True, 使用基于`value_and_grad` API的自定义训练步骤",
    )
    parser.add_argument(
        "--freeze_embedding",
        default=False,
        type=str2bool,
        help="如果为True, 冻结LLM的embedding table和gen_embed的embedding table (nn.Embedding)",
    )
    parser.add_argument(
        "--output_path",
        default="outputs/janus-sft",
        type=str,
        help="保存训练结果的输出目录",
    )

    # 训练超参数
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="学习率")
    parser.add_argument(
        "--end_learning_rate",
        default=1e-5,
        type=float,
        help="余弦衰减的最终学习率",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="批处理大小")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="权重衰减")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="是否进行梯度裁剪")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="最大梯度L2范数")
    parser.add_argument(
        "--null_prompt_prob",
        default=0.0,
        type=float,
        help="在t2i任务中, 将文本标题替换为空字符串的概率, 用于无条件引导训练",
    )
    parser.add_argument("--train_steps", default=5000, type=int, help="训练步数")
    parser.add_argument("--warmup_steps", default=50, type=int, help="学习率预热步数")
    parser.add_argument("--ckpt_save_steps", default=500, type=int, help="每隔多少步保存一次checkpoint")
    parser.add_argument(
        "--ckpt_max_keep",
        default=3,
        type=int,
        help="训练期间保留的checkpoint数量上限",
    )
    parser.add_argument(
        "--max_length",
        default=1024,
        type=int,
        help="序列最大长度, 输入序列将被填充(左填充)和截断到此最大长度",
    )

    # 训练数据配置
    parser.add_argument("--task", default="t2i", type=str, help="任务类型: text, t2i, vqa, 或 mixed")
    parser.add_argument(
        "--dataset_name",
        default="",
        type=str,
        help="数据集名称, 用于选择正确的vqa和text数据集加载器",
    )
    parser.add_argument(
        "--mixed_task_rand_samp",
        action="store_true", # 表示此参数为开关，出现即为True
        help="若为True, 则从任意数量的数据集条目组合中按照内部比例进行加权随机采样; "
        + "若为False, 则预先切分数据集",
    )
    parser.add_argument(
        "--t2i_csv_path",
        default=None,
        type=str,
        help="csv标注文件路径, 包含`image_path` (图像路径) 和 `text_en` (英文标题) 列",
    )
    parser.add_argument(
        "--t2i_data_dir",
        default=None,
        type=str,
        help="数据集目录, 包含csv_path中`image_path`指定的图像",
    )
    parser.add_argument(
        "--t2i_parquet_dir",
        default=None,
        type=str,
        help="Parquet格式的数据集目录, 包含csv_path中`image_path`指定的图像 (可选, 另一种数据源)",
    )
    parser.add_argument(
        "--text_qa_data_dir",
        default=None,
        type=str,
        help="文本问答数据集目录",
    )
    parser.add_argument(
        "--vqa_data_dir",
        default=None,
        type=str,
        help="视觉问答(VQA)数据集目录",
    )
    parser.add_argument(
        "--num_samples",
        default=-1,
        type=int,
        help="如果为-1, 则在整个数据集上训练; 否则, 将选择指定数量的样本进行训练。",
    )
    parser.add_argument(
        "--image_size",
        default=384,
        type=int,
        help="图像调整和裁剪的目标尺寸。请谨慎更改, 因为Janus使用固定的384图像尺寸进行训练",
    )
    parser.add_argument("--shuffle", default=True, type=str2bool, help="是否打乱数据集")

    args = parser.parse_args()
    main(args)
