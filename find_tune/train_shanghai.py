"""
Whisper 上海方言微调训练脚本
支持三种微调方法：full（全参数）、lora（LoRA）、partial（部分层）
"""
import os
from datetime import datetime

# ==================== 配置镜像（必须在导入 transformers 之前）====================
# HuggingFace 镜像配置（国内访问）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk, Audio
import evaluate
import numpy as np


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    数据整理器：将音频特征和文本标签进行批处理和填充
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """
    准备数据集：提取音频特征并对文本进行分词
    """
    audio = batch["audio"]
    
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch


def compute_metrics(pred, processor, metric):
    """
    计算评估指标（WER - 词错误率）
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def print_trainable_parameters(model):
    """
    打印可训练参数数量
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"可训练参数: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)")


def setup_lora(model, lora_r, lora_alpha, lora_dropout):
    """
    配置 LoRA 微调
    
    Args:
        model: Whisper 模型
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    
    Returns:
        配置了 LoRA 的模型
    """
    from peft import LoraConfig, get_peft_model, TaskType
    
    target_modules = [
        "q_proj", "v_proj",
        "k_proj", "out_proj",
        "fc1", "fc2",
    ]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,gdd
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    model = get_peft_model(model, lora_config)
    return model


def setup_partial_finetune(model, freeze_encoder_layers, freeze_decoder=False):
    """
    配置 Partial Fine-tuning（冻结部分层）
    
    Args:
        model: Whisper 模型
        freeze_encoder_layers: 要冻结的 Encoder 层数（从底层开始）
        freeze_decoder: 是否冻结 Decoder
    
    Returns:
        冻结后的 Encoder 总层数
    """
    # 冻结卷积层和位置编码
    for param in model.model.encoder.conv1.parameters():
        param.requires_grad = False
    for param in model.model.encoder.conv2.parameters():
        param.requires_grad = False
    for param in model.model.encoder.embed_positions.parameters():
        param.requires_grad = False
    
    # 冻结前 N 层 Encoder
    total_encoder_layers = len(model.model.encoder.layers)
    for i in range(min(freeze_encoder_layers, total_encoder_layers)):
        for param in model.model.encoder.layers[i].parameters():
            param.requires_grad = False
    
    # 可选：冻结 Decoder
    if freeze_decoder:
        for param in model.model.decoder.parameters():
            param.requires_grad = False
    
    return total_encoder_layers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Whisper 上海方言微调训练")
    
    # 基础参数
    parser.add_argument("--model_size", type=str, default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="模型大小 (默认: medium)")
    
    # 微调方法选择
    parser.add_argument("--finetune_method", type=str, default="full",
                        choices=["full", "lora", "partial"],
                        help="微调方法: full(全参数), lora(LoRA), partial(部分层) (默认: full)")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (默认: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (默认: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout (默认: 0.1)")
    
    # Partial Fine-tuning 参数
    parser.add_argument("--freeze_encoder_layers", type=int, default=12,
                        help="冻结 Encoder 前 N 层 (默认: 12)")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="是否冻结 Decoder（默认不冻结）")
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="学习率 (默认: full=1e-5, lora=1e-4, partial=1e-5)")
    parser.add_argument("--num_train_epochs", type=int, default=30,
                        help="训练轮数 (默认: 30)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小 (默认: 8)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数 (默认: 2)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="预热步数 (默认: 500)")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估间隔步数 (默认: 500)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存间隔步数 (默认: 500)")
    
    # 学习率调度器参数
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_plateau",
                        choices=["linear", "cosine", "cosine_with_restarts", 
                                 "polynomial", "constant", "constant_with_warmup",
                                 "cosine_with_plateau"],
                        help="学习率调度器类型 (默认: cosine_with_plateau，即带平顶的cosine)")
    parser.add_argument("--plateau_ratio", type=float, default=0.1,
                        help="平顶阶段占总训练步数的比例 (默认: 0.1，即10%%)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="最小学习率与最大学习率的比例 (默认: 0.0)")
    
    args = parser.parse_args()
    
    # 根据微调方法设置默认学习率
    if args.learning_rate is None:
        if args.finetune_method == "lora":
            args.learning_rate = 1e-4
        else:
            args.learning_rate = 1e-5

    # 微调方法名称映射
    method_names = {
        "full": "全参数微调",
        "lora": "LoRA 微调",
        "partial": "部分层微调"
    }
    
    print("=" * 60)
    print(f"Whisper 上海方言微调训练 - {method_names[args.finetune_method]}")
    print("=" * 60)
    print("\n✓ 已配置 HuggingFace 镜像: https://hf-mirror.com")
    
    # ==================== 配置参数 ====================
    dataset_path = "dataset/shanghai/shanghai_dataset"
    model_name = f"openai/whisper-{args.model_size}"
    language = "Chinese"
    task = "transcribe"
    
    # 生成带时间戳和方法名的实验目录
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    exp_base_dir = "./exp"
    
    if args.finetune_method == "lora":
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-lora-r{args.lora_r}-{timestamp}")
    elif args.finetune_method == "partial":
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-partial-freeze{args.freeze_encoder_layers}-{timestamp}")
    else:
        output_dir = os.path.join(exp_base_dir, f"whisper-shanghai-full-{timestamp}")
    
    fp16 = torch.cuda.is_available()
    
    print(f"\n配置信息:")
    print(f"  模型: {model_name}")
    print(f"  微调方法: {method_names[args.finetune_method]}")
    if args.finetune_method == "lora":
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  LoRA dropout: {args.lora_dropout}")
    elif args.finetune_method == "partial":
        print(f"  冻结 Encoder 层数: {args.freeze_encoder_layers}")
        print(f"  冻结 Decoder: {args.freeze_decoder}")
    print(f"  数据集: {dataset_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  训练轮数: {args.num_train_epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  混合精度: {fp16}")
    
    # ==================== 加载数据集 ====================
    print(f"\n加载数据集...")
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集未找到: {dataset_path}")
        print("\n请先运行数据准备脚本:")
        print("  python find_tune/make_data_shanghai.py")
        print("  python find_tune/load_data_shanghai.py")
        return
    
    dataset = load_from_disk(dataset_path)
    print(f"✓ 训练样本: {len(dataset['train'])}")
    print(f"✓ 测试样本: {len(dataset['test'])}")
    
    # ==================== 加载模型和处理器 ====================
    print(f"\n加载模型: {model_name}")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    
    print("✓ 基座模型加载完成")
    
    # ==================== 配置微调方法 ====================
    is_lora = False
    
    if args.finetune_method == "lora":
        print(f"\n配置 LoRA...")
        model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        is_lora = True
        print_trainable_parameters(model)
        
    elif args.finetune_method == "partial":
        print(f"\n配置 Partial Fine-tuning...")
        total_encoder_layers = setup_partial_finetune(
            model, args.freeze_encoder_layers, args.freeze_decoder
        )
        print(f"  Encoder 总层数: {total_encoder_layers}")
        print(f"  冻结前 {args.freeze_encoder_layers} 层")
        print(f"  训练后 {max(0, total_encoder_layers - args.freeze_encoder_layers)} 层")
        print(f"  Decoder: {'冻结' if args.freeze_decoder else '训练'}")
        print_trainable_parameters(model)
        
    else:
        print(f"\n使用全参数微调...")
        print_trainable_parameters(model)
    
    # ==================== 准备数据集 ====================
    print("\n预处理数据集...")
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    
    print("✓ 数据预处理完成")
    
    # ==================== 初始化数据整理器 ====================
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # ==================== 加载评估指标 ====================
    metric = evaluate.load("wer")
    
    # ==================== 配置训练参数 ====================
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
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
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        fp16=fp16,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
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
    
    # ==================== 开始训练 ====================
    print("\n" + "=" * 60)
    print(f"开始 {method_names[args.finetune_method]} 训练...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # ==================== 保存模型 ====================
    print("\n保存模型...")
    
    if is_lora:
        # 保存 LoRA 适配器
        lora_path = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(lora_path)
        processor.save_pretrained(lora_path)
        print(f"✓ LoRA 适配器保存位置: {lora_path}")
        
        # 合并并保存完整模型
        print("合并 LoRA 权重到基座模型...")
        merged_model = model.merge_and_unload()
        merged_path = os.path.join(output_dir, "merged_model")
        merged_model.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"✓ 合并模型保存位置: {merged_path}")
    else:
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
    
    # ==================== 最终评估 ====================
    print("\n运行最终评估...")
    metrics = trainer.evaluate()
    
    print(f"\n{'=' * 60}")
    print(f"{method_names[args.finetune_method]} 训练完成!")
    print(f"{'=' * 60}")
    print(f"最终 WER: {metrics['eval_wer']:.2f}%")
    print(f"模型保存位置: {output_dir}")
    
    if is_lora:
        print(f"\n使用 LoRA 适配器:")
        print(f"  from peft import PeftModel")
        print(f"  from transformers import WhisperForConditionalGeneration")
        print(f"  base_model = WhisperForConditionalGeneration.from_pretrained('{model_name}')")
        print(f"  model = PeftModel.from_pretrained(base_model, '{lora_path}')")
        print(f"\n使用合并后的模型:")
        print(f"  from transformers import pipeline")
        print(f"  pipe = pipeline('automatic-speech-recognition', model='{merged_path}')")
    else:
        print(f"\n使用微调后的模型:")
        print(f"  from transformers import pipeline")
        print(f"  pipe = pipeline('automatic-speech-recognition', model='{output_dir}')")
        print(f"  result = pipe('audio.wav')")
        print(f"  print(result['text'])")

if __name__ == "__main__":
    main()
