"""
Whisper 上海方言微调训练脚本 (集成 WandB)
支持两种模式：
  - cascade: 级联模式，上海话语音 -> 上海话文本（需配合翻译模型）
  - e2e: 端到端模式，上海话语音 -> 普通话文本（直接输出）
自动检测硬件配置，智能调整并行度和批次大小
"""
import os
import multiprocessing

# ==================== 配置镜像（必须在导入 transformers 之前）====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'

import torch
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from datasets import load_from_disk, Audio
import evaluate
import numpy as np

# ==================== 配置日志格式 ====================
class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 获取颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化时间戳
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建日志消息
        log_message = f"{color}[{timestamp}] [{record.levelname}] [{record.filename}:{record.lineno}]{reset} {record.getMessage()}"
        
        return log_message

class PlainFormatter(logging.Formatter):
    """纯文本日志格式化器（用于文件输出，不带颜色）"""
    
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{record.levelname}] [{record.filename}:{record.lineno}] {record.getMessage()}"
        return log_message

def setup_logger(log_file: str = None, is_main_process: bool = True):
    """设置日志配置
    
    Args:
        log_file: 日志文件路径，如果提供则同时输出到文件
        is_main_process: 是否为主进程（DDP 训练时只有主进程输出详细日志）
    """
    logger = logging.getLogger()
    
    # 非主进程只输出 WARNING 及以上级别，避免重复日志
    log_level = logging.INFO if is_main_process else logging.WARNING
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建控制台处理器（带颜色）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件处理器（只有主进程写日志文件）
    if log_file and is_main_process:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(PlainFormatter())
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")
    
    return logger

# 初始化日志
logger = setup_logger()

# WandB 集成
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB 未安装。运行 'pip install wandb' 来启用训练监控。")


def get_hardware_config() -> dict:
    """
    自动检测硬件配置
    """
    config = {
        "cpu_count": multiprocessing.cpu_count(),
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory_gb": [],
        "total_gpu_memory_gb": 0,
        "system_memory_gb": 0,
    }
    
    if torch.cuda.is_available():
        config["gpu_count"] = torch.cuda.device_count()
        for i in range(config["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            config["gpu_names"].append(props.name)
            memory_gb = props.total_memory / (1024 ** 3)
            config["gpu_memory_gb"].append(round(memory_gb, 1))
            config["total_gpu_memory_gb"] += memory_gb
        config["total_gpu_memory_gb"] = round(config["total_gpu_memory_gb"], 1)
    
    try:
        import psutil
        config["system_memory_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    
    return config

def print_trainable_parameters(model):
    """打印模型的可训练参数数量"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
    logger.info(f"可训练参数: {trainable_params:,} / {all_params:,} ({trainable_percent:.2f}%)")


def setup_lora(model, lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):
    """
    配置 LoRA 微调
    
    Args:
        model: Whisper 模型
        lora_r: LoRA 秩（rank），决定低秩矩阵的维度
        lora_alpha: 缩放因子，控制 LoRA 更新的强度
        lora_dropout: Dropout 比例，防止过拟合
    
    Returns:
        配置了 LoRA 的模型
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("请安装 peft 库: pip install peft")
    
    # 配置 LoRA
    # 注意：对于 Whisper 模型，不要指定 task_type，让 PEFT 自动推断
    # 指定 TaskType.SEQ_2_SEQ_LM 会导致 forward() 接收到错误的参数 input_ids
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    logger.info(f"✓ LoRA 配置完成 (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
    
    return model


def setup_adalora(model, init_r: int = 12, target_r: int = 8, lora_alpha: int = 32, 
                  lora_dropout: float = 0.1, beta1: float = 0.85, beta2: float = 0.85,
                  total_step: int = 10000):
    """
    配置 AdaLoRA 微调（自适应低秩适配）
    
    AdaLoRA 会在训练过程中自动调整不同层的秩（rank），
    为重要的层分配更高的秩，为不重要的层分配更低的秩。
    
    Args:
        model: Whisper 模型
        init_r: 初始秩，训练开始时的秩
        target_r: 目标秩，训练结束时的平均秩
        lora_alpha: 缩放因子
        lora_dropout: Dropout 比例
        beta1: 重要性分数的移动平均系数
        beta2: 不确定性的移动平均系数
        total_step: 总训练步数（AdaLoRA 必需参数）
    
    Returns:
        配置了 AdaLoRA 的模型
    """
    try:
        from peft import AdaLoraConfig, get_peft_model
    except ImportError:
        raise ImportError("请安装 peft 库: pip install peft")
    
    # 配置 AdaLoRA
    adalora_config = AdaLoraConfig(
        init_r=init_r,
        target_r=target_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        beta1=beta1,
        beta2=beta2,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        # AdaLoRA 特有参数
        total_step=total_step,  # 总训练步数（必需）
        tinit=200,  # 开始裁剪秩的步数
        tfinal=min(1000, total_step // 2),  # 停止裁剪秩的步数
        deltaT=10,  # 每隔多少步更新一次秩
    )
    
    # 应用 AdaLoRA
    model = get_peft_model(model, adalora_config)
    logger.info(f"✓ AdaLoRA 配置完成 (init_r={init_r}, target_r={target_r}, alpha={lora_alpha}, total_step={total_step})")
    
    return model


def setup_partial_finetune(model, freeze_encoder_layers: int = 0, freeze_decoder: bool = False) -> int:
    """
    配置部分参数微调（冻结部分层）
    
    Args:
        model: Whisper 模型
        freeze_encoder_layers: 冻结 Encoder 的前 N 层
        freeze_decoder: 是否冻结整个 Decoder
    
    Returns:
        Encoder 总层数
    """
    # 获取 Encoder 层数
    total_encoder_layers = len(model.model.encoder.layers)
    
    # 冻结 Encoder 的前 N 层
    if freeze_encoder_layers > 0:
        for i, layer in enumerate(model.model.encoder.layers):
            if i < freeze_encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info(f"✓ 冻结 Encoder 前 {freeze_encoder_layers} 层")
    
    # 冻结 Decoder
    if freeze_decoder:
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        logger.info("✓ 冻结 Decoder")
    
    return total_encoder_layers


def create_cosine_with_plateau_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int,
                                          plateau_ratio: float = 0.1, min_lr_ratio: float = 0.0):
    """
    创建带平顶的余弦学习率调度器
    
    学习率变化曲线：
    1. Warmup 阶段：线性从 0 上升到 max_lr
    2. Plateau 阶段：保持在 max_lr（平顶）
    3. Cosine 衰减阶段：从 max_lr 余弦衰减到 min_lr
    
    Args:
        optimizer: 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        plateau_ratio: 平顶阶段占（总步数 - warmup步数）的比例
        min_lr_ratio: 最小学习率与最大学习率的比例
    
    Returns:
        LambdaLR 调度器
    """
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    # 计算各阶段的步数
    remaining_steps = num_training_steps - num_warmup_steps
    plateau_steps = int(remaining_steps * plateau_ratio)
    cosine_steps = remaining_steps - plateau_steps
    
    def lr_lambda(current_step: int) -> float:
        # 阶段 1: Warmup（线性上升）
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 阶段 2: Plateau（保持最大值）
        step_after_warmup = current_step - num_warmup_steps
        if step_after_warmup < plateau_steps:
            return 1.0
        
        # 阶段 3: Cosine 衰减
        step_in_cosine = step_after_warmup - plateau_steps
        progress = float(step_in_cosine) / float(max(1, cosine_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_optimal_settings(hardware_config: dict, model_size: str) -> Tuple[int, int, int, int]:
    """
    根据硬件配置和模型大小，自动计算最优的训练参数
    
    对于 medium 和 large 模型，使用保守的 batch_size 以避免 OOM
    """
    cpu_count = hardware_config["cpu_count"]
    gpu_count = hardware_config["gpu_count"]
    
    # 固定为单线程预处理，更稳定
    num_proc = 1
    dataloader_num_workers = max(2, min(cpu_count // 4, 8))
    
    # 根据模型大小设置默认的 batch_size（经过实测验证的保守值）
    default_batch_sizes = {
        "tiny": 16,
        "base": 8,
        "small": 4,
        "medium": 2,  # medium 模型使用 batch_size=2 以避免 OOM
        "large": 1,
    }
    batch_size = default_batch_sizes.get(model_size, 2)
    
    # 计算梯度累积步数，保持有效批次大小为 16
    target_effective_batch = 16
    gradient_accumulation_steps = max(1, target_effective_batch // (batch_size * max(1, gpu_count)))
    
    return num_proc, batch_size, gradient_accumulation_steps, dataloader_num_workers


def print_hardware_info(hardware_config: dict, num_proc: int, batch_size: int, 
                        gradient_accumulation_steps: int, dataloader_num_workers: int):
    """打印硬件配置和优化后的训练参数"""
    logger.info("=" * 60)
    logger.info("硬件配置检测")
    logger.info("=" * 60)
    logger.info(f"  CPU 核心数: {hardware_config['cpu_count']}")
    logger.info(f"  系统内存: {hardware_config['system_memory_gb']} GB")
    
    if hardware_config['gpu_count'] > 0:
        logger.info(f"  GPU 数量: {hardware_config['gpu_count']}")
        for i, (name, mem) in enumerate(zip(hardware_config['gpu_names'], hardware_config['gpu_memory_gb'])):
            logger.info(f"    GPU {i}: {name} ({mem} GB)")
        logger.info(f"  GPU 总显存: {hardware_config['total_gpu_memory_gb']} GB")
    else:
        logger.info("  GPU: 未检测到（将使用 CPU 训练）")
    
    logger.info("自动优化的训练参数:")
    logger.info(f"  数据预处理进程数: {num_proc}")
    logger.info(f"  DataLoader workers: {dataloader_num_workers}")
    logger.info(f"  每卡批次大小: {batch_size}")
    logger.info(f"  梯度累积步数: {gradient_accumulation_steps}")
    effective_batch = batch_size * gradient_accumulation_steps * max(1, hardware_config['gpu_count'])
    logger.info(f"  有效批次大小: {effective_batch}")
    logger.info("=" * 60)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    数据整理器：将音频特征和文本标签进行批处理和填充
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 处理音频输入
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 处理标签
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 将填充标记替换为 -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 移除 bos token
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor, keep_text_cn=False, use_text_cn_as_target=False):
    """
    准备数据集：提取音频特征并对文本进行分词
    
    Args:
        batch: 数据批次
        processor: Whisper 处理器
        keep_text_cn: 是否保留普通话文本（用于级联模式评估）
        use_text_cn_as_target: 是否使用普通话文本作为目标（e2e 模式）
    
    Note:
        音频特征提取部分与 asr_tact.data_utils.prepare_audio_features 逻辑一致，
        可通过以下方式复用（当前保持原有实现以确保稳定性）：
        
        from asr_tact.data_utils import prepare_audio_features
        batch = prepare_audio_features(batch, processor)
    """
    audio = batch["audio"]
    
    # ========== 可复用部分：音频特征提取 ==========
    # 此部分与 asr_tact.data_utils.prepare_audio_features 逻辑一致
    # 计算 log-Mel 频谱图特征（WhisperFeatureExtractor 默认 padding 到 30 秒）
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # ========== 可复用部分结束 ==========

    # 根据模式选择目标文本
    if use_text_cn_as_target and "text_cn" in batch:
        # e2e 模式：使用普通话文本作为目标
        target_text = batch["text_cn"]
    else:
        # cascade 模式：使用上海话文本作为目标
        target_text = batch["text"]
    
    batch["labels"] = processor.tokenizer(target_text).input_ids
    
    # 保留普通话文本用于评估（如果存在）
    if keep_text_cn and "text_cn" in batch:
        batch["text_cn_ref"] = batch["text_cn"]
    
    return batch

def compute_metrics(pred, processor, metric):
    """
    计算评估指标（CER - 字符错误率）
    中文 ASR 标准做法：使用 CER 而非 WER，因为中文分词有歧义
    
    CER 基于编辑距离（Levenshtein Distance）计算：
    CER = (Substitutions + Insertions + Deletions) / Reference_Length
    
    同时记录 20 个样本的预测结果和真实标签对比到 WandB
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 复制 label_ids 避免修改原始数据
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    # 解码预测结果
    # 注意：pred_ids 可能包含 -100 或超出词表范围的 token，需要先处理
    # 将无效的 token id 替换为 pad_token_id
    pred_ids_cleaned = np.where(
        (pred_ids >= 0) & (pred_ids < processor.tokenizer.vocab_size),
        pred_ids,
        processor.tokenizer.pad_token_id
    )
    
    pred_str = processor.tokenizer.batch_decode(pred_ids_cleaned, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 使用 CER（字符错误率）：将每个字符用空格分开，让 jiwer 按字符计算
    # 这是中文 ASR 的标准做法，避免分词歧义
    pred_str_chars = [" ".join(list(s.replace(" ", ""))) for s in pred_str]
    label_str_chars = [" ".join(list(s.replace(" ", ""))) for s in label_str]
    cer = 100 * metric.compute(predictions=pred_str_chars, references=label_str_chars)

    # 记录 20 个固定随机样本的预测结果和真实标签对比到 WandB
    # 使用固定随机种子，确保每次实验、每个 epoch 选择的样本 ID 一致，便于对比
    if WANDB_AVAILABLE and len(pred_str) >= 1:
        import random
        # 创建对比表格
        comparison_table = wandb.Table(columns=["样本编号", "真实标签", "预测结果", "是否完全匹配"])
        num_samples_to_log = min(20, len(pred_str))
        # 使用固定种子随机选择样本索引，确保每次实验选择相同的样本
        rng = random.Random(42)
        sample_indices = rng.sample(range(len(pred_str)), num_samples_to_log)
        sample_indices.sort()  # 排序以便查看
        
        for idx, i in enumerate(sample_indices):
            # 处理可能的空字符串情况
            pred_text = pred_str[i].strip() if pred_str[i] else "(空)"
            label_text = label_str[i].strip() if label_str[i] else "(空)"
            is_match = "Y" if pred_text == label_text and pred_text != "(空)" else "N"
            comparison_table.add_data(
                f"样本 {i+1}",
                label_text,
                pred_text,
                is_match
            )
        wandb.log({"asr_comparison": comparison_table})
        # 同时存到 summary，方便在网页上直接查看最新一轮的对比结果
        wandb.run.summary["asr_comparison"] = comparison_table

    # 返回 cer，但字段名保持 wer 以兼容 Trainer 的 metric_for_best_model 配置
    return {"wer": cer}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Whisper 上海方言微调训练 (WandB 集成)")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="cascade",
                        choices=["cascade", "e2e"],
                        help="训练模式: cascade(级联，输出上海话文本), e2e(端到端，输出普通话文本) (默认: cascade)")
    
    # 模型参数
    parser.add_argument("--model_size", type=str, default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="模型大小 (默认: medium)")
    
    # 微调方法
    parser.add_argument("--finetune_method", type=str, default="full",
                        choices=["full", "lora", "adalora", "partial"],
                        help="微调方法: full(全参数), lora(LoRA), adalora(AdaLoRA自适应), partial(部分层) (默认: full)")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (默认: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (默认: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (默认: 0.05)")
    
    # AdaLoRA 参数
    parser.add_argument("--adalora_init_r", type=int, default=12,
                        help="AdaLoRA 初始 rank (默认: 12)")
    parser.add_argument("--adalora_target_r", type=int, default=8,
                        help="AdaLoRA 目标 rank (默认: 8)")
    
    # Partial 微调参数
    parser.add_argument("--freeze_encoder_layers", type=int, default=12,
                        help="冻结 Encoder 前 N 层 (默认: 12)")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="是否冻结 Decoder (默认: False)")
    
    # 训练参数（可选，默认自动计算）
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批次大小 (默认: 自动根据 GPU 显存计算)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="梯度累积步数 (默认: 自动计算)")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="数据预处理进程数 (默认: 自动根据 CPU 核心数计算)")
    parser.add_argument("--dataloader_num_workers", type=int, default=None,
                        help="DataLoader 工作进程数 (默认: 自动计算)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="学习率 (默认: LoRA 使用 1e-4, 其他使用 1e-5)")
    parser.add_argument("--num_train_epochs", type=int, default=30,
                        help="训练轮数 (默认: 30)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="预热步数 (默认: 500)")
    parser.add_argument("--eval_steps", type=int, default=250,
                        help="评估间隔步数 (默认: 250)")
    parser.add_argument("--save_steps", type=int, default=250,
                        help="保存间隔步数 (默认: 250)")
    
    # 学习率调度器参数
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_plateau",
                        choices=["linear", "cosine", "cosine_with_restarts", 
                                 "polynomial", "constant", "constant_with_warmup",
                                 "cosine_with_plateau"],
                        help="学习率调度器类型 (默认: cosine_with_plateau，带平顶的cosine)")
    parser.add_argument("--plateau_ratio", type=float, default=0.1,
                        help="平顶阶段占总训练步数的比例 (默认: 0.1，即10%%)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="最小学习率与最大学习率的比例 (默认: 0.0)")
    
    # GPU 配置
    parser.add_argument("--single_gpu", action="store_true",
                        help="强制使用单卡训练（避免 DataParallel 开销）")
    
    args = parser.parse_args()
    
    # 如果指定单卡模式，设置环境变量只使用第一张 GPU
    if args.single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info("✓ 已启用单卡模式 (CUDA_VISIBLE_DEVICES=0)")
    
    # ==================== 自动检测硬件配置 ====================
    hardware_config = get_hardware_config()
    auto_num_proc, auto_batch_size, auto_grad_accum, auto_dataloader_workers = get_optimal_settings(
        hardware_config, args.model_size
    )
    
    # 使用自动计算的值或用户指定的值
    num_proc = args.num_proc if args.num_proc is not None else auto_num_proc
    batch_size = args.batch_size if args.batch_size is not None else auto_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else auto_grad_accum
    dataloader_num_workers = args.dataloader_num_workers if args.dataloader_num_workers is not None else auto_dataloader_workers
    
    # 学习率：LoRA/AdaLoRA 使用更高的学习率（1e-4），全参微调使用较低的学习率（1e-5）
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    else:
        learning_rate = 1e-4 if args.finetune_method in ["lora", "adalora"] else 1e-5
        lr_desc = "LoRA/AdaLoRA 推荐值" if args.finetune_method in ["lora", "adalora"] else "全参微调推荐值"
        logger.info(f"✓ 自动设置学习率: {learning_rate} ({lr_desc})")
    
    # 打印硬件信息
    print_hardware_info(hardware_config, num_proc, batch_size, gradient_accumulation_steps, dataloader_num_workers)
    
    # ==================== 根据模式配置参数 ====================
    is_e2e = args.mode == "e2e"
    mode_name = "端到端" if is_e2e else "级联"
    task_desc = "上海话语音 -> 普通话文本" if is_e2e else "上海话语音 -> 上海话文本"
    
    # 微调方法名称映射
    method_names = {
        "full": "全参数微调",
        "lora": "LoRA 微调",
        "adalora": "AdaLoRA 自适应微调",
        "partial": "部分层微调"
    }
    method_name = method_names[args.finetune_method]
    is_lora = args.finetune_method in ["lora", "adalora"]  # LoRA 和 AdaLoRA 都使用 PEFT 保存方式
    
    # 数据集路径（端到端和级联使用不同的数据集，但同一个 WandB 项目）
    if is_e2e:
        dataset_path = "dataset/shanghai/shanghai_unified_dataset"
    else:
        dataset_path = "dataset/shanghai/shanghai_unified_dataset"
    
    # 统一的 WandB 项目，用 mode 分组区分
    wandb_project = "whisper-shanghai-finetuning"
    
    logger.info("=" * 70)
    logger.info(f"Whisper 上海方言微调训练 - {mode_name}模式 - {method_name}")
    logger.info(f"任务：{task_desc}")
    logger.info("=" * 70)
    logger.info("✓ 已配置 HuggingFace 镜像: https://hf-mirror.com")
    logger.info("✓ 已配置 WandB 镜像: https://api.bandw.top")
    
    # 模型配置
    model_name = f"openai/whisper-{args.model_size}"
    language = "Chinese"
    task = "transcribe"
    
    # 生成带时间戳的实验目录
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    exp_base_dir = "./exp"
    mode_suffix = "e2e" if is_e2e else "cascade"
    
    # 根据微调方法生成输出目录名
    if args.finetune_method == "lora":
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-{mode_suffix}-lora-r{args.lora_r}-{timestamp}")
    elif args.finetune_method == "adalora":
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-{mode_suffix}-adalora-r{args.adalora_init_r}to{args.adalora_target_r}-{timestamp}")
    elif args.finetune_method == "partial":
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-{mode_suffix}-partial-freeze{args.freeze_encoder_layers}-{timestamp}")
    else:
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-{mode_suffix}-full-{args.model_size}-{timestamp}")
    
    # 判断是否为主进程（DDP 训练时只有 rank 0 输出详细日志）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    # 创建输出目录并设置日志文件（只有主进程创建目录和写日志文件）
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    setup_logger(log_file, is_main_process=is_main_process)
    
    fp16 = torch.cuda.is_available()
    # 对于 medium 及以上模型，启用梯度检查点以节省显存
    # gradient_checkpointing = args.model_size in ["medium", "large"]
    gradient_checkpointing = False
    
    # WandB 配置
    wandb_run_name = f"whisper-{args.model_size}-{mode_suffix}-{args.finetune_method}-lr{learning_rate}-{timestamp}"
    
    logger.info("配置信息:")
    logger.info(f"  模式: {mode_name} ({args.mode})")
    logger.info(f"  微调方法: {method_name} ({args.finetune_method})")
    logger.info(f"  任务: {task_desc}")
    logger.info(f"  模型: {model_name}")
    if args.finetune_method == "lora":
        logger.info(f"  LoRA rank: {args.lora_r}")
        logger.info(f"  LoRA alpha: {args.lora_alpha}")
        logger.info(f"  LoRA dropout: {args.lora_dropout}")
    elif args.finetune_method == "adalora":
        logger.info(f"  AdaLoRA init_r: {args.adalora_init_r}")
        logger.info(f"  AdaLoRA target_r: {args.adalora_target_r}")
        logger.info(f"  LoRA alpha: {args.lora_alpha}")
        logger.info(f"  LoRA dropout: {args.lora_dropout}")
    elif args.finetune_method == "partial":
        logger.info(f"  冻结 Encoder 层数: {args.freeze_encoder_layers}")
        logger.info(f"  冻结 Decoder: {args.freeze_decoder}")
    logger.info(f"  数据集: {dataset_path}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  训练轮数: {args.num_train_epochs}")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  梯度累积步数: {gradient_accumulation_steps}")
    logger.info(f"  有效批次大小: {batch_size * gradient_accumulation_steps * max(1, hardware_config['gpu_count'])}")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  混合精度: {fp16}")
    logger.info(f"  梯度检查点: {gradient_checkpointing}")
    logger.info(f"  WandB 项目: {wandb_project}")
    
    # ==================== 初始化 WandB ====================
    wandb_config = {
        "mode": args.mode,
        "finetune_method": args.finetune_method,
        "task": task_desc,
        "model_name": model_name,
        "model_size": args.model_size,
        "language": language,
        "num_train_epochs": args.num_train_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps * max(1, hardware_config['gpu_count']),
        "learning_rate": learning_rate,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "plateau_ratio": args.plateau_ratio if args.lr_scheduler_type == "cosine_with_plateau" else None,
        "min_lr_ratio": args.min_lr_ratio if args.lr_scheduler_type == "cosine_with_plateau" else None,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "hardware": {
            "gpu_count": hardware_config['gpu_count'],
            "gpu_names": hardware_config['gpu_names'],
            "total_gpu_memory_gb": hardware_config['total_gpu_memory_gb'],
            "cpu_count": hardware_config['cpu_count'],
        },
    }
    
    if args.finetune_method == "lora":
        wandb_config.update({
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        })
    elif args.finetune_method == "adalora":
        wandb_config.update({
            "adalora_init_r": args.adalora_init_r,
            "adalora_target_r": args.adalora_target_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        })
    elif args.finetune_method == "partial":
        wandb_config.update({
            "freeze_encoder_layers": args.freeze_encoder_layers,
            "freeze_decoder": args.freeze_decoder,
        })
    
    # WandB 初始化（只有主进程初始化，local_rank 和 is_main_process 已在前面定义）
    if WANDB_AVAILABLE and is_main_process:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=args.mode,  # 按模式分组：cascade 或 e2e
            tags=[args.mode, args.finetune_method, args.model_size],  # 添加标签便于筛选
            config=wandb_config,
        )
        logger.info("✓ WandB 初始化成功")
        logger.info(f"  查看训练进度: {wandb.run.get_url()}")
    elif WANDB_AVAILABLE and not is_main_process:
        # 非主进程禁用 WandB
        os.environ["WANDB_DISABLED"] = "true"
        logger.info(f"[Rank {local_rank}] 非主进程，跳过 WandB 初始化")
    else:
        logger.warning("⚠️ WandB 未启用，使用 TensorBoard 记录日志")
    
    # ==================== 加载数据集 ====================
    logger.info(f"加载{mode_name}数据集...")
    if not os.path.exists(dataset_path):
        logger.error(f"❌ 数据集未找到: {dataset_path}")
        if is_e2e:
            logger.info("请先运行端到端数据准备脚本:")
            logger.info("  python find_tune/make_data_shanghai_e2e.py")
            logger.info("  python find_tune/load_data_shanghai_e2e.py")
        else:
            logger.info("请先运行级联数据准备脚本:")
            logger.info("  python find_tune/make_data_shanghai.py")
            logger.info("  python find_tune/load_data_shanghai.py")
        return
    
    dataset = load_from_disk(dataset_path)
    logger.info(f"✓ 训练样本: {len(dataset['train'])}")
    logger.info(f"✓ 测试样本: {len(dataset['test'])}")
    
    # 打印样本示例
    sample = dataset['train'][0]
    logger.info(f"样本示例:")
    logger.info(f"  目标文本: {sample['text']}")
    if 'source_text' in sample and is_e2e:
        logger.info(f"  源文本（上海话）: {sample['source_text']}")
    
    # ==================== 加载处理器（用于数据预处理）====================
    logger.info(f"加载处理器: {model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    logger.info("✓ 处理器加载完成")
    
    # ==================== 准备数据集（在加载模型之前，避免多进程与 CUDA/LoRA 冲突）====================
    # 预处理缓存路径：e2e 和 cascade 使用不同的目标文本，需要不同的缓存
    # - cascade: 目标是上海话文本 (text)，缓存后缀 _cascade 或 _joint（兼容旧版本）
    # - e2e: 目标是普通话文本 (text_cn)，缓存后缀 _e2e
    cache_suffix = "_e2e" if is_e2e else "_cascade"
    cache_dir = os.path.join(dataset_path, f"preprocessed{cache_suffix}")
    
    # 兼容旧版本：cascade 模式也检查 preprocessed_joint 缓存
    if not is_e2e and not os.path.exists(cache_dir):
        joint_cache_dir = os.path.join(dataset_path, "preprocessed_joint")
        if os.path.exists(joint_cache_dir):
            cache_dir = joint_cache_dir
            logger.info(f"使用兼容缓存目录: {cache_dir}")
    
    # 检查数据集是否包含普通话文本（用于级联模式最终评估）
    has_text_cn = "text_cn" in dataset["train"].column_names if "train" in dataset else False
    if has_text_cn and not is_e2e:
        logger.info("✓ 检测到普通话文本 (text_cn)，将用于最终评估")
    
    if os.path.exists(cache_dir):
        # 使用缓存的预处理结果
        logger.info(f"发现预处理缓存，直接加载: {cache_dir}")
        dataset = load_from_disk(cache_dir)
        logger.info("✓ 从缓存加载预处理数据完成")
        # 检查缓存的数据集是否包含 text_cn_ref
        has_text_cn = "text_cn_ref" in dataset["test"].column_names if "test" in dataset else False
    else:
        # 首次预处理并保存缓存
        logger.info("预处理数据集...")
        logger.info(f"  使用 {num_proc} 个进程进行数据预处理...")
        
        # 确定需要保留的列（级联模式保留 text_cn 用于最终评估）
        keep_text_cn = has_text_cn and not is_e2e
        original_columns = dataset["train"].column_names
        
        # 需要移除的列（保留 text_cn 如果存在且是级联模式）
        columns_to_remove = [col for col in original_columns if col != "text_cn"] if keep_text_cn else original_columns
        
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(
            lambda batch: prepare_dataset(batch, processor, keep_text_cn=keep_text_cn, use_text_cn_as_target=is_e2e),
            remove_columns=columns_to_remove,
            num_proc=num_proc
        )
        
        # 保存预处理结果到缓存
        dataset.save_to_disk(cache_dir)
        logger.info(f"✓ 数据预处理完成，已缓存到: {cache_dir}")
        
        # 更新 has_text_cn 标志
        has_text_cn = "text_cn_ref" in dataset["test"].column_names if "test" in dataset else False
    
    # ==================== 加载模型 ====================
    logger.info(f"加载模型: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # 配置模型
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    
    # 启用梯度检查点以节省显存
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✓ 梯度检查点已启用")
    
    logger.info("✓ 基座模型加载完成")
    
    # ==================== 配置微调方法 ====================
    if args.finetune_method == "lora":
        logger.info(f"配置 LoRA...")
        model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        print_trainable_parameters(model)
    
    elif args.finetune_method == "adalora":
        logger.info(f"配置 AdaLoRA（自适应低秩适配）...")
        # 计算总训练步数（AdaLoRA 必需参数）
        num_train_samples = len(dataset["train"])
        steps_per_epoch = num_train_samples // (args.batch_size * args.gradient_accumulation_steps)
        total_step = steps_per_epoch * args.num_train_epochs
        logger.info(f"  总训练步数: {total_step}")
        model = setup_adalora(
            model, 
            init_r=args.adalora_init_r, 
            target_r=args.adalora_target_r,
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            total_step=total_step
        )
        print_trainable_parameters(model)
        
    elif args.finetune_method == "partial":
        logger.info(f"配置 Partial Fine-tuning...")
        total_encoder_layers = setup_partial_finetune(
            model, args.freeze_encoder_layers, args.freeze_decoder
        )
        logger.info(f"  Encoder 总层数: {total_encoder_layers}")
        logger.info(f"  冻结前 {args.freeze_encoder_layers} 层")
        logger.info(f"  训练后 {max(0, total_encoder_layers - args.freeze_encoder_layers)} 层")
        logger.info(f"  Decoder: {'冻结' if args.freeze_decoder else '训练'}")
        print_trainable_parameters(model)
        
    else:
        logger.info(f"使用全参数微调...")
        print_trainable_parameters(model)
    
    # ==================== 初始化数据整理器 ====================
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # ==================== 加载评估指标 ====================
    metric = evaluate.load("wer")
    
    # ==================== 配置训练参数 ====================
    # DDP 训练时，只有主进程报告到 wandb，避免非主进程初始化 wandb 失败
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    if WANDB_AVAILABLE and is_main_process:
        report_to = ["wandb", "tensorboard"]
    else:
        report_to = ["tensorboard"]
    
    # 学习率调度器配置
    # 如果用户指定了调度器类型，使用用户指定的；否则根据微调方法自动选择
    use_custom_plateau_scheduler = args.lr_scheduler_type == "cosine_with_plateau"
    
    if use_custom_plateau_scheduler:
        # 使用自定义的带平顶的 cosine 调度器，先用 constant 占位
        lr_scheduler_type = "constant"
        logger.info(f"✓ 使用带平顶的 Cosine 调度器 (plateau_ratio={args.plateau_ratio}, min_lr_ratio={args.min_lr_ratio})")
    else:
        lr_scheduler_type = args.lr_scheduler_type
        logger.info(f"✓ 使用 {lr_scheduler_type} 学习率调度器")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=args.warmup_steps if not use_custom_plateau_scheduler else 0,  # 自定义调度器自己处理 warmup
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=25,
        report_to=report_to,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        # DDP 优化配置（多卡训练时生效）
        ddp_find_unused_parameters=False,  # LoRA 模型不需要查找未使用参数，提升性能
        ddp_backend="nccl",  # 使用 NCCL 后端，GPU 通信更高效
    )
    
    # ==================== 初始化训练器 ====================
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        tokenizer=processor.feature_extractor,
    )
    
    # 如果使用自定义的带平顶的 cosine 调度器，需要在训练开始前替换调度器
    if use_custom_plateau_scheduler:
        # 计算总训练步数
        num_update_steps_per_epoch = len(dataset["train"]) // (batch_size * gradient_accumulation_steps)
        num_training_steps = num_update_steps_per_epoch * args.num_train_epochs
        plateau_steps = int((num_training_steps - args.warmup_steps) * args.plateau_ratio)
        
        logger.info(f"  总训练步数: {num_training_steps}")
        logger.info(f"  Warmup 步数: {args.warmup_steps}")
        logger.info(f"  平顶步数: {plateau_steps}")
        logger.info(f"  Cosine 衰减步数: {num_training_steps - args.warmup_steps - plateau_steps}")
        
        # 创建优化器（Trainer 会延迟创建，我们需要手动触发）
        trainer.create_optimizer()
        
        # 创建自定义调度器并替换
        custom_scheduler = create_cosine_with_plateau_scheduler(
            optimizer=trainer.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
            plateau_ratio=args.plateau_ratio,
            min_lr_ratio=args.min_lr_ratio,
        )
        trainer.lr_scheduler = custom_scheduler
        logger.info("✓ 已替换为自定义的带平顶 Cosine 调度器")
    
    # ==================== 开始训练 ====================
    logger.info("=" * 70)
    logger.info(f"开始{mode_name} {method_name}训练...")
    logger.info(f"任务：{task_desc}")
    logger.info("=" * 70)
    
    trainer.train()
    
    # ==================== 保存模型 ====================
    logger.info("保存模型...")
    
    if is_lora:
        # 保存 LoRA 适配器
        lora_path = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(lora_path)
        processor.save_pretrained(lora_path)
        logger.info(f"✓ LoRA 适配器保存位置: {lora_path}")
        
        # 合并并保存完整模型
        logger.info("合并 LoRA 权重到基座模型...")
        merged_model = model.merge_and_unload()
        merged_path = os.path.join(output_dir, "merged_model")
        merged_model.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        logger.info(f"✓ 合并模型保存位置: {merged_path}")
    else:
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        logger.info(f"✓ 模型保存位置: {output_dir}")
    
    # ==================== 最终评估 ====================
    logger.info("运行最终评估...")
    metrics = trainer.evaluate()
    
    # 级联模式：额外计算到普通话的 CER（需要翻译模型）
    final_mandarin_cer = None
    if not is_e2e and has_text_cn:
        logger.info("=" * 70)
        logger.info("级联模式最终评估：计算到普通话的 CER")
        logger.info("=" * 70)
        
        # 检查翻译模型是否存在
        translation_model_path = "/mnt/workspace/workgroup/qq/ts/whisper/exp/translation-mt5-small-260320-231422"
        if os.path.exists(translation_model_path):
            try:
                from transformers import MT5ForConditionalGeneration, MT5Tokenizer
                
                logger.info(f"加载翻译模型: {translation_model_path}")
                trans_tokenizer = MT5Tokenizer.from_pretrained(translation_model_path)
                trans_model = MT5ForConditionalGeneration.from_pretrained(translation_model_path)
                trans_device = "cuda" if torch.cuda.is_available() else "cpu"
                trans_model.to(trans_device)
                trans_model.eval()
                logger.info(f"✓ 翻译模型加载完成 (设备: {trans_device})")
                
                # 对测试集进行完整的级联评估
                logger.info("开始级联评估（ASR + 翻译）...")
                
                # 获取模型预测
                predictions = trainer.predict(dataset["test"])
                pred_ids = predictions.predictions
                
                # 解码 ASR 预测结果（上海话）
                pred_ids_cleaned = np.where(
                    (pred_ids >= 0) & (pred_ids < processor.tokenizer.vocab_size),
                    pred_ids,
                    processor.tokenizer.pad_token_id
                )
                shanghai_preds = processor.tokenizer.batch_decode(pred_ids_cleaned, skip_special_tokens=True)
                
                # 获取普通话参考文本
                if "text_cn_ref" in dataset["test"].column_names:
                    mandarin_refs = dataset["test"]["text_cn_ref"]
                elif "text_cn" in dataset["test"].column_names:
                    mandarin_refs = dataset["test"]["text_cn"]
                else:
                    mandarin_refs = None
                
                if mandarin_refs is not None:
                    # 翻译上海话预测结果为普通话
                    logger.info(f"翻译 {len(shanghai_preds)} 条预测结果...")
                    mandarin_preds = []
                    batch_size_trans = 16
                    
                    for i in range(0, len(shanghai_preds), batch_size_trans):
                        batch_texts = shanghai_preds[i:i+batch_size_trans]
                        input_texts = [f"翻译上海话到普通话: {text}" for text in batch_texts]
                        
                        inputs = trans_tokenizer(
                            input_texts,
                            return_tensors="pt",
                            max_length=128,
                            truncation=True,
                            padding=True,
                        ).to(trans_device)
                        
                        with torch.no_grad():
                            outputs = trans_model.generate(
                                **inputs,
                                max_length=128,
                                num_beams=4,
                                early_stopping=True,
                            )
                        
                        translated = trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        mandarin_preds.extend(translated)
                        
                        if (i + batch_size_trans) % 100 == 0:
                            logger.info(f"  已翻译 {min(i + batch_size_trans, len(shanghai_preds))}/{len(shanghai_preds)}")
                    
                    # 计算到普通话的 CER
                    pred_chars = [" ".join(list(s.replace(" ", ""))) for s in mandarin_preds]
                    ref_chars = [" ".join(list(s.replace(" ", ""))) for s in mandarin_refs]
                    final_mandarin_cer = 100 * metric.compute(predictions=pred_chars, references=ref_chars)
                    
                    logger.info(f"✓ 级联最终 CER (到普通话): {final_mandarin_cer:.2f}%")
                    
                    # 记录样本对比到 WandB
                    if WANDB_AVAILABLE:
                        cascade_table = wandb.Table(columns=[
                            "样本编号", "上海话预测", "普通话翻译", "普通话参考", "是否匹配"
                        ])
                        num_samples = min(20, len(mandarin_preds))
                        for i in range(num_samples):
                            sh_pred = shanghai_preds[i].strip() if shanghai_preds[i] else "(空)"
                            mn_pred = mandarin_preds[i].strip() if mandarin_preds[i] else "(空)"
                            mn_ref = mandarin_refs[i].strip() if mandarin_refs[i] else "(空)"
                            is_match = "✓" if mn_pred == mn_ref and mn_pred != "(空)" else "✗"
                            cascade_table.add_data(f"样本 {i+1}", sh_pred, mn_pred, mn_ref, is_match)
                        wandb.log({"cascade_final_comparison": cascade_table})
                else:
                    logger.warning("⚠️ 测试集中未找到普通话参考文本 (text_cn_ref)")
                    
            except Exception as e:
                logger.warning(f"⚠️ 级联评估失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info(f"⚠️ 翻译模型不存在: {translation_model_path}")
            logger.info("  跳过级联最终评估。如需评估，请先训练翻译模型：")
            logger.info("  python find_tune/train_translation.py")
    
    # 记录最终指标到 WandB
    if WANDB_AVAILABLE:
        final_metrics = {
            "final/eval_loss": metrics.get("eval_loss", 0),
            "final/eval_cer_shanghai": metrics["eval_wer"],  # 到上海话的 CER
        }
        if final_mandarin_cer is not None:
            final_metrics["final/eval_cer_mandarin"] = final_mandarin_cer  # 到普通话的 CER
        wandb.log(final_metrics)
        
    logger.info("=" * 70)
    logger.info(f"{mode_name} {method_name}训练完成!")
    logger.info(f"任务：{task_desc}")
    logger.info("=" * 70)
    logger.info(f"最终 CER (到上海话): {metrics['eval_wer']:.2f}%")
    if final_mandarin_cer is not None:
        logger.info(f"最终 CER (到普通话): {final_mandarin_cer:.2f}%")
    logger.info(f"最终 Loss: {metrics.get('eval_loss', 0):.4f}")
    logger.info(f"模型保存位置: {output_dir}")
    
    if WANDB_AVAILABLE:
        logger.info(f"查看完整训练报告: {wandb.run.get_url()}")
        wandb.finish()
    
    # 打印使用说明
    if is_lora:
        logger.info("使用 LoRA 适配器:")
        logger.info(f"  from peft import PeftModel")
        logger.info(f"  from transformers import WhisperForConditionalGeneration")
        logger.info(f"  base_model = WhisperForConditionalGeneration.from_pretrained('{model_name}')")
        logger.info(f"  model = PeftModel.from_pretrained(base_model, '{lora_path}')")
        logger.info("使用合并后的模型:")
        logger.info(f"  from transformers import pipeline")
        logger.info(f"  pipe = pipeline('automatic-speech-recognition', model='{merged_path}')")
    else:
        logger.info("使用微调后的模型:")
        logger.info(f"  from transformers import pipeline")
        logger.info(f"  pipe = pipeline('automatic-speech-recognition', model='{output_dir}')")
    
    logger.info("  result = pipe('audio.wav')")
    if is_e2e:
        logger.info("  print(result['text'])  # 输出普通话文本")
    else:
        logger.info("  print(result['text'])  # 输出上海话文本，需配合翻译模型")
if __name__ == "__main__":
    main()
