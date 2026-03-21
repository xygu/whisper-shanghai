"""
级联联合训练脚本：Whisper ASR + mT5 翻译模型联合训练
任务：上海话语音 -> 上海话文本 -> 普通话文本
两个 loss 加权平均作为最终 loss，一起训练

架构：
1. Whisper: 语音 -> 上海话文本 (ASR Loss)
2. mT5: 上海话文本 -> 普通话文本 (Translation Loss)
3. 最终 Loss = α * ASR_Loss + (1-α) * Translation_Loss
"""
import os
from datetime import datetime

# 配置镜像（必须在导入 transformers 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'

# DDP 训练时，非主进程禁用 WandB（必须在 import wandb 之前设置）
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
if _local_rank != 0:
    os.environ["WANDB_DISABLED"] = "true"

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB 未安装。运行 'pip install wandb' 来启用训练监控。")

import sys
import json
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.modeling_outputs import ModelOutput
from datasets import load_from_disk, Audio
from peft import LoraConfig, get_peft_model, TaskType
import evaluate


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger("cascade_joint_training")
    logger.setLevel(logging.DEBUG)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"training_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path


class CascadeJointModel(nn.Module):
    """
    级联联合模型：Whisper ASR + mT5 翻译
    
    训练时：
    1. Whisper 接收音频，输出上海话文本的 loss
    2. mT5 接收上海话文本（ground truth），输出普通话文本的 loss
    3. 最终 loss = α * asr_loss + (1-α) * translation_loss
    """
    
    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        translation_model: MT5ForConditionalGeneration,
        translation_tokenizer: MT5Tokenizer,
        asr_loss_weight: float = 0.5,
        max_translation_length: int = 128,
    ):
        super().__init__()
        self.whisper = whisper_model
        self.translation = translation_model
        self.translation_tokenizer = translation_tokenizer
        self.asr_loss_weight = asr_loss_weight
        self.max_translation_length = max_translation_length
        
        # 用于 generate 的配置
        self.config = whisper_model.config
        self.generation_config = whisper_model.generation_config
        
    def forward(
        self,
        input_features: torch.Tensor = None,
        labels: torch.Tensor = None,
        translation_input_ids: Optional[torch.Tensor] = None,
        translation_attention_mask: Optional[torch.Tensor] = None,
        translation_labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        前向传播，计算联合 loss
        
        Args:
            input_features: Whisper 的音频特征 [batch, mel_dim, time]
            labels: ASR 标签（上海话文本的 token ids）
            translation_input_ids: 翻译模型输入（上海话文本）
            translation_attention_mask: 翻译模型注意力掩码
            translation_labels: 翻译标签（普通话文本的 token ids）
            input_ids: 忽略（Trainer 可能传递）
            attention_mask: 忽略（Trainer 可能传递）
            decoder_input_ids: 忽略（Trainer 可能传递）
        
        Returns:
            dict: 返回字典而非 dataclass，以兼容 DataParallel 的 gather 操作
        """
        # 忽略 Trainer 可能传递的额外参数（不传递给子模型）
        _ = input_ids, attention_mask, decoder_input_ids, kwargs
        
        # 1. 计算 ASR loss
        # 注意：PEFT 的 PeftModelForSeq2SeqLM.forward() 会传递 input_ids 给底层模型
        # 但 Whisper 不接受 input_ids 参数，所以需要特殊处理
        if hasattr(self.whisper, 'base_model') and hasattr(self.whisper.base_model, 'model'):
            # PEFT 模型：直接调用底层的 Whisper 模型（包含 LoRA 层）
            # self.whisper.base_model 是 LoraModel，self.whisper.base_model.model 是带 LoRA 的 Whisper
            asr_outputs = self.whisper.base_model.model(
                input_features=input_features,
                labels=labels,
            )
        else:
            # 非 PEFT 模型：直接调用
            asr_outputs = self.whisper(
                input_features=input_features,
                labels=labels,
            )
        asr_loss = asr_outputs.loss
                # 2. 计算翻译 loss（如果提供了翻译数据）
        if translation_input_ids is not None and translation_labels is not None:
            # 调试：检查输入数据
            # 检查是否有负数 token（除了 -100）
            invalid_input_mask = translation_input_ids < 0
            invalid_label_mask = (translation_labels < 0) & (translation_labels != -100)
            
            if invalid_input_mask.any():
                print(f"⚠️ translation_input_ids 包含负数 token!")
                print(f"  负数位置: {invalid_input_mask.nonzero()}")
                print(f"  负数值: {translation_input_ids[invalid_input_mask]}")
            
            if invalid_label_mask.any():
                print(f"⚠️ translation_labels 包含非 -100 的负数 token!")
                print(f"  负数位置: {invalid_label_mask.nonzero()}")
                print(f"  负数值: {translation_labels[invalid_label_mask]}")
            
            # 检查 token 是否超出词汇表范围
            vocab_size = self.translation.config.vocab_size
            oov_input_mask = translation_input_ids >= vocab_size
            oov_label_mask = (translation_labels >= vocab_size) & (translation_labels != -100)
            
            if oov_input_mask.any():
                print(f"⚠️ translation_input_ids 包含超出词汇表的 token!")
                print(f"  vocab_size: {vocab_size}")
                print(f"  超出范围的值: {translation_input_ids[oov_input_mask]}")
            
            if oov_label_mask.any():
                print(f"⚠️ translation_labels 包含超出词汇表的 token!")
                print(f"  vocab_size: {vocab_size}")
                print(f"  超出范围的值: {translation_labels[oov_label_mask]}")
            
            # 翻译模型使用 FP32 计算以避免 FP16 溢出导致 NaN
            # mT5 在 FP16 下容易出现数值不稳定问题
            with torch.cuda.amp.autocast(enabled=False):
                # 将翻译模型转为 FP32 并前向传播
                translation_outputs = self.translation(
                    input_ids=translation_input_ids,
                    attention_mask=translation_attention_mask,
                    labels=translation_labels,
                )
            translation_loss = translation_outputs.loss
            
            # 调试：如果 translation_loss 是 NaN，打印更多信息
            if torch.isnan(translation_loss).any():
                print(f"\n🔍 翻译模型调试信息:")
                print(f"  translation_input_ids: {translation_input_ids}")
                print(f"  translation_labels: {translation_labels}")
                print(f"  翻译模型 dtype: {next(self.translation.parameters()).dtype}")
                # 检查 logits
                logits = translation_outputs.logits
                print(f"  logits shape: {logits.shape}")
                print(f"  logits 有 NaN: {torch.isnan(logits).any()}")
                print(f"  logits 有 Inf: {torch.isinf(logits).any()}")
                if torch.isnan(logits).any():
                    nan_count = torch.isnan(logits).sum().item()
                    print(f"  logits NaN 数量: {nan_count}/{logits.numel()}")
            
            # 3. 加权平均
            total_loss = (
                self.asr_loss_weight * asr_loss + 
                (1 - self.asr_loss_weight) * translation_loss
            )
            
            # 返回字典，DataParallel 可以正确 gather 字典类型
            return {
                "loss": total_loss,
                "asr_loss": asr_loss,
                "translation_loss": translation_loss,
                "logits": asr_outputs.logits,
            }
        else:
            # 只有 ASR loss
            return {
                "loss": asr_loss,
                "asr_loss": asr_loss,
                "translation_loss": None,
                "logits": asr_outputs.logits,
            }
    
    def generate(self, input_features, **kwargs):
        """生成函数，用于评估时的 ASR 推理"""
        # 过滤掉翻译相关的参数，这些参数 Whisper 模型不认识
        translation_keys = [
            'translation_input_ids', 
            'translation_attention_mask', 
            'translation_labels'
        ]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in translation_keys}
        return self.whisper.generate(input_features=input_features, **filtered_kwargs)
    
    def gradient_checkpointing_enable(self, **kwargs):
        """启用梯度检查点"""
        self.whisper.gradient_checkpointing_enable(**kwargs)
        self.translation.gradient_checkpointing_enable(**kwargs)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """保存模型"""
        # 保存 Whisper
        whisper_path = os.path.join(save_directory, "whisper")
        os.makedirs(whisper_path, exist_ok=True)
        self.whisper.save_pretrained(whisper_path, **kwargs)
        
        # 保存翻译模型
        translation_path = os.path.join(save_directory, "translation")
        os.makedirs(translation_path, exist_ok=True)
        self.translation.save_pretrained(translation_path, **kwargs)
        self.translation_tokenizer.save_pretrained(translation_path)
        
        # 保存配置
        config = {
            "asr_loss_weight": self.asr_loss_weight,
            "max_translation_length": self.max_translation_length,
        }
        with open(os.path.join(save_directory, "cascade_config.json"), "w") as f:
            json.dump(config, f, indent=2)


@dataclass
class CascadeOutput(ModelOutput):
    """
    级联模型输出，继承自 ModelOutput 以支持多 GPU 训练时的 gather 操作
    """
    loss: Optional[torch.Tensor] = None
    asr_loss: Optional[torch.Tensor] = None
    translation_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


@dataclass
class DataCollatorCascadeJoint:
    """
    级联联合训练的数据整理器
    同时处理 ASR 和翻译任务的数据
    """
    whisper_processor: Any
    translation_tokenizer: Any
    decoder_start_token_id: int
    max_translation_length: int = 128
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 处理 Whisper 音频输入
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.whisper_processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # 2. 处理 ASR 标签（上海话文本）
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.whisper_processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        
        # 3. 处理翻译输入（上海话文本 -> 普通话文本）
        if "translation_input_ids" in features[0]:
            # 翻译输入
            trans_input_features = [
                {"input_ids": f["translation_input_ids"]} for f in features
            ]
            trans_input_batch = self.translation_tokenizer.pad(
                trans_input_features, 
                return_tensors="pt",
                padding=True,
            )
            batch["translation_input_ids"] = trans_input_batch["input_ids"]
            batch["translation_attention_mask"] = trans_input_batch["attention_mask"]
            
            # 翻译标签
            trans_label_features = [
                {"input_ids": f["translation_labels"]} for f in features
            ]
            trans_labels_batch = self.translation_tokenizer.pad(
                trans_label_features,
                return_tensors="pt",
                padding=True,
            )
            trans_labels = trans_labels_batch["input_ids"].masked_fill(
                trans_labels_batch.attention_mask.ne(1), -100
            )
            batch["translation_labels"] = trans_labels
        
        return batch


class CascadeJointTrainer(Seq2SeqTrainer):
    """
    级联联合训练器
    支持记录双 loss 到 WandB，并检测 NaN
    """
    
    def __init__(self, *args, log_level: str = "info", **kwargs):
        super().__init__(*args, **kwargs)
        self.asr_loss_history = []
        self.translation_loss_history = []
        self.step_count = 0
        self.log_level = log_level.lower()
        self.debug_mode = self.log_level == "debug"
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 loss，并记录各部分 loss，检测 NaN"""
        self.step_count += 1
        
        # ========== 调试：检查输入数据（仅 debug 级别）==========
        if self.debug_mode and self.step_count == 1:
            print(f"\n{'='*70}")
            print(f"🔍 Step 1 调试信息 - 检查输入数据")
            print(f"{'='*70}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    has_nan = torch.isnan(value).any().item()
                    has_inf = torch.isinf(value).any().item()
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                          f"nan={has_nan}, inf={has_inf}, "
                          f"min={value[value != -100].min().item() if (value != -100).any() else 'N/A':.4f}, "
                          f"max={value[value != -100].max().item() if (value != -100).any() else 'N/A':.4f}")
                    # 检查 labels 是否全是 -100
                    if 'labels' in key:
                        valid_count = (value != -100).sum().item()
                        total_count = value.numel()
                        print(f"    有效标签数: {valid_count}/{total_count}")
                        if valid_count == 0:
                            print(f"    ⚠️ 警告: {key} 全部是 -100，这会导致 loss 为 NaN!")
            print(f"{'='*70}\n")
        
        outputs = model(**inputs)
        
        # outputs 现在是字典类型
        loss = outputs["loss"]
        asr_loss = outputs.get("asr_loss")
        translation_loss = outputs.get("translation_loss")
        
        # ========== 调试：打印各部分 loss（仅 debug 级别）==========
        if self.debug_mode and self.step_count == 1:
            print(f"\n{'='*70}")
            print(f"🔍 Step 1 调试信息 - Loss 值")
            print(f"{'='*70}")
            print(f"  asr_loss = {asr_loss}")
            print(f"  translation_loss = {translation_loss}")
            print(f"  total_loss = {loss}")
            print(f"{'='*70}\n")
        
        # ========== NaN 检测：Loss ==========
        def check_loss_nan(loss_tensor, loss_name):
            if loss_tensor is None:
                return
            if isinstance(loss_tensor, torch.Tensor):
                if torch.isnan(loss_tensor).any():
                    print(f"\n{'='*70}")
                    print(f"❌ NaN 检测到！Step {self.step_count}, {loss_name} 出现 NaN!")
                    print(f"   {loss_name} = {loss_tensor}")
                    # 额外打印其他 loss 帮助定位
                    print(f"   asr_loss = {asr_loss}")
                    print(f"   translation_loss = {translation_loss}")
                    print(f"{'='*70}\n")
                    raise ValueError(f"Step {self.step_count}: {loss_name} 出现 NaN，终止训练！")
                if torch.isinf(loss_tensor).any():
                    print(f"\n{'='*70}")
                    print(f"❌ Inf 检测到！Step {self.step_count}, {loss_name} 出现 Inf!")
                    print(f"   {loss_name} = {loss_tensor}")
                    print(f"{'='*70}\n")
                    raise ValueError(f"Step {self.step_count}: {loss_name} 出现 Inf，终止训练！")
        
        # 先检查子 loss，再检查 total loss，这样能更好定位问题
        check_loss_nan(asr_loss, "asr_loss")
        check_loss_nan(translation_loss, "translation_loss")
        check_loss_nan(loss, "total_loss")
        
        # 记录各部分 loss（处理多 GPU 情况，loss 可能是多元素 tensor）
        if asr_loss is not None:
            if isinstance(asr_loss, torch.Tensor):
                asr_loss_val = asr_loss.mean().item() if asr_loss.numel() > 1 else asr_loss.item()
            else:
                asr_loss_val = asr_loss
            self.asr_loss_history.append(asr_loss_val)
        if translation_loss is not None:
            if isinstance(translation_loss, torch.Tensor):
                trans_loss_val = translation_loss.mean().item() if translation_loss.numel() > 1 else translation_loss.item()
            else:
                trans_loss_val = translation_loss
            self.translation_loss_history.append(trans_loss_val)
        
        # 将字典转换为简单对象以便 return_outputs 时使用
        if return_outputs:
            class SimpleOutput:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            return (loss, SimpleOutput(outputs))
        return loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写 training_step 以检测梯度中的 NaN/Inf（仅警告，不终止）"""
        # 调用父类的 training_step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # ========== NaN/Inf 检测：梯度（仅警告，FP16 的 GradScaler 会自动跳过 Inf 梯度）==========
        # 注意：FP16 混合精度训练时，梯度 Inf 是正常现象，GradScaler 会自动处理
        # 只有 NaN 才是真正的问题
        nan_params = []
        inf_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)
        
        if nan_params:
            print(f"\n{'='*70}")
            print(f"❌ 梯度 NaN 检测到！Step {self.step_count}")
            print(f"   以下参数的梯度包含 NaN:")
            for name in nan_params[:10]:
                print(f"     - {name}")
            if len(nan_params) > 10:
                print(f"     ... 还有 {len(nan_params) - 10} 个参数")
            print(f"{'='*70}\n")
            # NaN 梯度是严重问题，终止训练
            raise ValueError(f"Step {self.step_count}: 梯度出现 NaN，终止训练！涉及 {len(nan_params)} 个参数")
        
        if inf_params and self.step_count <= 5 and self.debug_mode:
            # Inf 梯度在 FP16 训练中是正常的，GradScaler 会自动跳过
            # 只在 debug 模式下打印警告
            print(f"\n{'='*70}")
            print(f"⚠️ 梯度 Inf 检测到（FP16 正常现象，GradScaler 会自动处理）Step {self.step_count}")
            print(f"   涉及 {len(inf_params)} 个参数，例如: {inf_params[0]}")
            print(f"{'='*70}\n")
        
        return loss
    
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """扩展日志，添加各部分 loss
        
        Args:
            logs: 日志字典
            start_time: 开始时间（新版本 transformers 需要此参数）
        """
        if self.asr_loss_history:
            logs["asr_loss"] = np.mean(self.asr_loss_history[-100:])
        if self.translation_loss_history:
            logs["translation_loss"] = np.mean(self.translation_loss_history[-100:])
        # 兼容新旧版本 transformers
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


class LossLoggingCallback(TrainerCallback):
    """记录各部分 loss 的回调"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            asr_loss = logs.get("asr_loss", "N/A")
            trans_loss = logs.get("translation_loss", "N/A")
            total_loss = logs.get("loss", "N/A")
            
            if isinstance(asr_loss, float) and isinstance(trans_loss, float):
                self.logger.info(
                    f"Step {state.global_step}: "
                    f"Total={total_loss:.4f}, ASR={asr_loss:.4f}, Trans={trans_loss:.4f}"
                )


def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """为 Whisper 配置 LoRA"""
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    return get_peft_model(model, lora_config)


def print_trainable_parameters(model, logger):
    """打印可训练参数数量"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"可训练参数: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")


def prepare_dataset(batch, whisper_processor, translation_tokenizer, max_length=128):
    """
    准备数据集：同时处理 ASR 和翻译任务
    
    Args:
        batch: 数据批次，包含 audio, text (上海话), text_cn (普通话)
        whisper_processor: Whisper 处理器
        translation_tokenizer: mT5 分词器
        max_length: 翻译文本最大长度
    """
    audio = batch["audio"]
    
    # 1. 处理音频特征（用于 Whisper）
    batch["input_features"] = whisper_processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # 2. 处理 ASR 标签（上海话文本）
    batch["labels"] = whisper_processor.tokenizer(batch["text"]).input_ids
    
    # 3. 处理翻译数据（如果有普通话文本）
    if "text_cn" in batch and batch["text_cn"]:
        # 翻译输入：添加任务前缀
        source_text = f"翻译上海话到普通话: {batch['text']}"
        trans_input = translation_tokenizer(
            source_text,
            max_length=max_length,
            truncation=True,
        )
        batch["translation_input_ids"] = trans_input.input_ids
        
        # 翻译标签：普通话文本
        trans_label = translation_tokenizer(
            batch["text_cn"],
            max_length=max_length,
            truncation=True,
        )
        batch["translation_labels"] = trans_label.input_ids
    
    return batch


def compute_metrics(pred, processor, metric):
    """计算 ASR 评估指标（CER）"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 替换 -100
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # 解码
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # 计算 CER（字符级别）
    pred_chars = [" ".join(list(s.replace(" ", ""))) for s in pred_str]
    label_chars = [" ".join(list(s.replace(" ", ""))) for s in label_str]
    
    cer = 100 * metric.compute(predictions=pred_chars, references=label_chars)
    
    return {"wer": cer}  # 使用 wer 字段存储 CER，便于 Trainer 识别


def main():
    parser = argparse.ArgumentParser(description="级联联合训练：Whisper ASR + mT5 翻译")
    
    # 模型配置
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small",
                        help="Whisper 模型名称")
    parser.add_argument("--translation_model", type=str, 
                        default="/mnt/workspace/workgroup/qq/ts/whisper/exp/translation-mt5-small-260320-231422/final_model",
                        help="翻译模型路径（预训练好的 mT5）")
    parser.add_argument("--dataset_path", type=str,
                        default="dataset/shanghai/shanghai_dataset",
                        help="数据集路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    
    # 训练配置
    parser.add_argument("--asr_loss_weight", type=float, default=0.5,
                        help="ASR loss 权重 (0-1)，翻译 loss 权重为 1-α")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小（级联模型显存占用大，建议使用 1）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数（配合小 batch size 使用）")
    parser.add_argument("--warmup_steps", type=int, default=250,
                        help="预热步数")
    parser.add_argument("--eval_steps", type=int, default=250,
                        help="评估间隔")
    parser.add_argument("--save_steps", type=int, default=250,
                        help="保存间隔")
    
    # LoRA 配置
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用 LoRA 微调 Whisper")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # 翻译模型配置
    parser.add_argument("--freeze_translation", action="store_true",
                        help="是否冻结翻译模型（只训练 Whisper）")
    parser.add_argument("--translation_lr_scale", type=float, default=0.1,
                        help="翻译模型学习率缩放因子")
    
    # 其他配置
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="使用 FP16 训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="使用梯度检查点（节省显存，默认启用）")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="数据预处理进程数")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="日志级别：debug 显示详细调试信息，info 为默认级别")
    
    args = parser.parse_args()
    
    # 生成输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        lora_suffix = "-lora" if args.use_lora else "-full"
        args.output_dir = f"./exp/cascade-joint{lora_suffix}-{timestamp}"
    
    # 判断是否为主进程（DDP 训练时只有 rank 0 输出详细日志）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    # 设置日志（只有主进程创建日志文件）
    if is_main_process:
        logger, log_path = setup_logging(args.output_dir)
    else:
        # 非主进程使用简单的 logger，只输出 WARNING 及以上级别
        logger = logging.getLogger("cascade_joint_training")
        logger.setLevel(logging.WARNING)
        log_path = None
    
    if is_main_process:
        logger.info("=" * 70)
        logger.info("级联联合训练：Whisper ASR + mT5 翻译")
        logger.info("=" * 70)
        logger.info(f"任务：上海话语音 -> 上海话文本 -> 普通话文本")
        logger.info(f"Loss = {args.asr_loss_weight:.2f} * ASR + {1-args.asr_loss_weight:.2f} * Translation")
        logger.info(f"日志文件: {log_path}")
        
        # 检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"设备: {device}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU 数量: {gpu_count}")
            for i in range(gpu_count):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    
    # WandB 初始化（只有主进程初始化）
    wandb_project = "whisper-shanghai-finetuning"
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    wandb_run_name = f"joint-α{args.asr_loss_weight}-lr{args.learning_rate}-{timestamp}"
    
    if WANDB_AVAILABLE and is_main_process:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(args),
        )
        logger.info(f"✓ WandB 初始化成功: {wandb.run.url}")
    elif WANDB_AVAILABLE and not is_main_process:
        # 非主进程不初始化 WandB（通过 report_to 参数控制）
        pass
    
    # ==================== 加载数据集 ====================
    if is_main_process:
        logger.info(f"\n加载数据集: {args.dataset_path}")
    
    if not os.path.exists(args.dataset_path):
        logger.error(f"❌ 数据集未找到: {args.dataset_path}")
        logger.info("请先运行数据准备脚本:")
        logger.info("  python find_tune/make_data_shanghai.py")
        logger.info("  python find_tune/load_data_shanghai.py")
        return
    
    dataset = load_from_disk(args.dataset_path)
    if is_main_process:
        logger.info(f"✓ 训练样本: {len(dataset['train'])}")
        logger.info(f"✓ 测试样本: {len(dataset['test'])}")
    
    # 检查数据集是否包含普通话文本
    sample = dataset['train'][0]
    has_text_cn = 'text_cn' in sample and sample['text_cn']
    if not has_text_cn:
        if is_main_process:
            logger.error("❌ 数据集缺少普通话文本 (text_cn)，无法进行联合训练")
            logger.info("请确保数据集包含 text_cn 字段")
        return
    
    if is_main_process:
        logger.info(f"样本示例:")
        logger.info(f"  上海话: {sample['text']}")
        logger.info(f"  普通话: {sample['text_cn']}")
    
    # ==================== 加载处理器和分词器 ====================
    if is_main_process:
        logger.info(f"\n加载 Whisper 处理器: {args.whisper_model}")
    whisper_processor = WhisperProcessor.from_pretrained(
        args.whisper_model, 
        language="Chinese", 
        task="transcribe"
    )
    
    if is_main_process:
        logger.info(f"加载翻译分词器: {args.translation_model}")
    translation_tokenizer = MT5Tokenizer.from_pretrained(args.translation_model)
    
    # ==================== 预处理数据集 ====================
    cache_dir = os.path.join(args.dataset_path, "preprocessed_joint")
    
    if os.path.exists(cache_dir):
        if is_main_process:
            logger.info(f"发现预处理缓存，直接加载: {cache_dir}")
        dataset = load_from_disk(cache_dir)
    else:
        if is_main_process:
            logger.info("预处理数据集...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        original_columns = dataset["train"].column_names
        dataset = dataset.map(
            lambda batch: prepare_dataset(
                batch, whisper_processor, translation_tokenizer
            ),
            remove_columns=original_columns,
            num_proc=args.num_proc,
        )
        
        dataset.save_to_disk(cache_dir)
        if is_main_process:
            logger.info(f"✓ 预处理完成，已缓存到: {cache_dir}")
    
    # ==================== 加载模型 ====================
    if is_main_process:
        logger.info(f"\n加载 Whisper 模型: {args.whisper_model}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    whisper_model.config.forced_decoder_ids = None
    whisper_model.config.suppress_tokens = []
    whisper_model.config.use_cache = False
    
    # 配置 LoRA（如果启用）
    if args.use_lora:
        if is_main_process:
            logger.info(f"配置 LoRA (r={args.lora_r}, α={args.lora_alpha})")
        whisper_model = setup_lora(
            whisper_model, 
            args.lora_r, 
            args.lora_alpha, 
            args.lora_dropout
        )
    
    if is_main_process:
        logger.info(f"加载翻译模型: {args.translation_model}")
    translation_model = MT5ForConditionalGeneration.from_pretrained(args.translation_model)
    
    # 冻结翻译模型（如果指定）
    if args.freeze_translation:
        if is_main_process:
            logger.info("冻结翻译模型参数")
        for param in translation_model.parameters():
            param.requires_grad = False
    
    # 创建级联联合模型
    if is_main_process:
        logger.info("\n创建级联联合模型...")
    model = CascadeJointModel(
        whisper_model=whisper_model,
        translation_model=translation_model,
        translation_tokenizer=translation_tokenizer,
        asr_loss_weight=args.asr_loss_weight,
    )
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("✓ 梯度检查点已启用")
    
    # 打印参数统计（只有主进程打印）
    if is_main_process:
        logger.info("\nWhisper 模型参数:")
        print_trainable_parameters(whisper_model, logger)
        logger.info("翻译模型参数:")
        print_trainable_parameters(translation_model, logger)
    
    # ==================== 初始化数据整理器 ====================
    data_collator = DataCollatorCascadeJoint(
        whisper_processor=whisper_processor,
        translation_tokenizer=translation_tokenizer,
        decoder_start_token_id=whisper_model.config.decoder_start_token_id,
    )
    
    # ==================== 加载评估指标 ====================
    metric = evaluate.load("wer")
    
    # ==================== 配置训练参数 ====================
    # 只有主进程汇报到 wandb，避免多卡训练时的冲突
    if WANDB_AVAILABLE and is_main_process:
        report_to = ["wandb", "tensorboard"]
    else:
        report_to = ["tensorboard"]
    
    # 生成 run_name（避免与 output_dir 相同的警告）
    run_name = f"joint-α{args.asr_loss_weight}-lr{args.learning_rate}-{timestamp}"
    
    # 检测是否支持 BF16（比 FP16 更稳定，不容易溢出）
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_supported and args.fp16  # 如果支持 BF16 且用户启用了混合精度，则使用 BF16
    use_fp16 = args.fp16 and not use_bf16  # 否则使用 FP16
    
    if is_main_process:
        if use_bf16:
            logger.info("✓ 使用 BF16 混合精度训练（更稳定，不易溢出）")
        elif use_fp16:
            logger.info("⚠️ 使用 FP16 混合精度训练（可能出现梯度溢出）")
        else:
            logger.info("使用 FP32 全精度训练")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,  # 设置独立的 run_name，避免与 output_dir 相同
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
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
        # 混合精度配置：优先使用 BF16（更稳定），否则使用 FP16
        fp16=use_fp16,
        bf16=use_bf16,
        # 梯度裁剪：防止梯度爆炸导致 NaN/Inf
        max_grad_norm=1.0,
        gradient_checkpointing=args.gradient_checkpointing,  # 启用梯度检查点节省显存
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # DDP 优化配置（多卡训练时生效，避免 DataParallel 的兼容性问题）
        ddp_find_unused_parameters=False,  # 自定义模型不需要查找未使用参数
        ddp_backend="nccl",  # 使用 NCCL 后端，GPU 通信更高效
    )
    
    # ==================== 初始化训练器 ====================
    trainer = CascadeJointTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, whisper_processor, metric),
        tokenizer=whisper_processor.feature_extractor,
        callbacks=[LossLoggingCallback(logger)],
        log_level=args.log_level,
    )
    
    # ==================== 开始训练 ====================
    if is_main_process:
        logger.info("\n" + "=" * 70)
        logger.info("开始级联联合训练...")
        logger.info(f"Loss = {args.asr_loss_weight:.2f} * ASR + {1-args.asr_loss_weight:.2f} * Translation")
        logger.info("=" * 70)
    
    trainer.train()
    
    # ==================== 保存模型 ====================
    if is_main_process:
        logger.info("\n保存模型...")
        model.save_pretrained(args.output_dir)
        whisper_processor.save_pretrained(os.path.join(args.output_dir, "whisper"))
        
        logger.info(f"✓ 模型保存位置: {args.output_dir}")
        logger.info(f"  - Whisper: {os.path.join(args.output_dir, 'whisper')}")
        logger.info(f"  - 翻译模型: {os.path.join(args.output_dir, 'translation')}")
    
    # ==================== 最终评估 ====================
    if is_main_process:
        logger.info("\n运行最终评估...")
    metrics = trainer.evaluate()
    
    # 级联评估：ASR + 翻译（只有主进程执行）
    if is_main_process:
        logger.info("\n级联评估（ASR -> 翻译）...")
    
    predictions = trainer.predict(dataset["test"])
    pred_ids = predictions.predictions
    
    # 解码 ASR 预测
    pred_ids_cleaned = np.where(
        (pred_ids >= 0) & (pred_ids < whisper_processor.tokenizer.vocab_size),
        pred_ids,
        whisper_processor.tokenizer.pad_token_id
    )
    shanghai_preds = whisper_processor.tokenizer.batch_decode(pred_ids_cleaned, skip_special_tokens=True)
    
    # 翻译预测结果（只有主进程执行）
    if not is_main_process:
        # 非主进程等待主进程完成评估后退出
        if WANDB_AVAILABLE:
            wandb.finish()
        return
    
    logger.info(f"翻译 {len(shanghai_preds)} 条 ASR 预测结果...")
    translation_model.eval()
    trans_device = next(translation_model.parameters()).device
    
    mandarin_preds = []
    batch_size_trans = 16
    
    for i in range(0, len(shanghai_preds), batch_size_trans):
        batch_texts = shanghai_preds[i:i+batch_size_trans]
        input_texts = [f"翻译上海话到普通话: {text}" for text in batch_texts]
        
        inputs = translation_tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True,
        ).to(trans_device)
        
        with torch.no_grad():
            outputs = translation_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        
        translated = translation_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        mandarin_preds.extend(translated)
    
    # 获取普通话参考
    # 需要重新加载原始数据集获取 text_cn
    original_dataset = load_from_disk(args.dataset_path)
    mandarin_refs = original_dataset["test"]["text_cn"]
    
    # 计算到普通话的 CER
    pred_chars = [" ".join(list(s.replace(" ", ""))) for s in mandarin_preds]
    ref_chars = [" ".join(list(s.replace(" ", ""))) for s in mandarin_refs]
    final_cer = 100 * metric.compute(predictions=pred_chars, references=ref_chars)
    
    # ==================== 输出总结 ====================
    logger.info("\n" + "=" * 70)
    logger.info("级联联合训练完成!")
    logger.info("=" * 70)
    logger.info(f"ASR CER (到上海话): {metrics['eval_wer']:.2f}%")
    logger.info(f"级联 CER (到普通话): {final_cer:.2f}%")
    logger.info(f"最终 Loss: {metrics.get('eval_loss', 0):.4f}")
    logger.info(f"模型保存位置: {args.output_dir}")
    
    # 记录到 WandB
    if WANDB_AVAILABLE:
        wandb.log({
            "final/asr_cer": metrics['eval_wer'],
            "final/cascade_cer": final_cer,
            "final/eval_loss": metrics.get('eval_loss', 0),
        })
        
        # 记录样本对比
        comparison_table = wandb.Table(columns=[
            "样本编号", "ASR预测(上海话)", "翻译结果(普通话)", "参考(普通话)"
        ])
        for i in range(min(20, len(mandarin_preds))):
            comparison_table.add_data(
                f"样本 {i+1}",
                shanghai_preds[i],
                mandarin_preds[i],
                mandarin_refs[i],
            )
        wandb.log({"cascade_comparison": comparison_table})
        
        logger.info(f"查看完整训练报告: {wandb.run.get_url()}")
        wandb.finish()
    
    # 打印使用说明
    logger.info("\n使用级联模型:")
    logger.info("  # 加载 Whisper")
    logger.info(f"  from transformers import WhisperForConditionalGeneration, WhisperProcessor")
    logger.info(f"  whisper = WhisperForConditionalGeneration.from_pretrained('{args.output_dir}/whisper')")
    logger.info(f"  processor = WhisperProcessor.from_pretrained('{args.output_dir}/whisper')")
    logger.info("  # 加载翻译模型")
    logger.info(f"  from transformers import MT5ForConditionalGeneration, MT5Tokenizer")
    logger.info(f"  translator = MT5ForConditionalGeneration.from_pretrained('{args.output_dir}/translation')")
    logger.info(f"  trans_tokenizer = MT5Tokenizer.from_pretrained('{args.output_dir}/translation')")


if __name__ == "__main__":
    main()
