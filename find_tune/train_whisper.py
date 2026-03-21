yi'ff"""
Whisper 微调训练脚本
支持使用 Hugging Face Transformers 进行 Whisper 模型微调
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from datasets import load_from_disk, DatasetDict, Audio
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
        # 分离输入和标签，因为它们的长度不同，需要不同的填充方法
        # 首先处理音频输入
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 获取标签特征并填充
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 将填充标记替换为 -100，这样在计算损失时会被忽略
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果所有序列都以 bos token 开头，则移除它，因为我们稍后会添加它
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset(batch, processor):
    """
    准备数据集：提取音频特征并对文本进行分词
    """
    # 加载并重采样音频数据到 16kHz
    audio = batch["audio"]
    
    # 计算输入的 log-Mel 频谱图特征
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # 对目标文本进行编码
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch


def compute_metrics(pred, processor, metric):
    """
    计算评估指标（WER - 词错误率）
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 将 -100 替换为 pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 解码预测和标签
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 计算 WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    # ==================== 配置参数 ====================
    # 数据集配置
    dataset_path = "dataset/shanghai/shanghai_dataset"  # 预处理后的数据集路径
    
    # 模型配置
    model_name = "openai/whisper-small"  # 可选: tiny, base, small, medium, large
    language = "Chinese"  # 目标语言
    task = "transcribe"  # 任务类型: transcribe 或 translate
    
    # 训练配置
    output_dir = "./whisper-finetuned-shanghai"
    num_train_epochs = 10
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 1e-5
    warmup_steps = 500
    logging_steps = 25
    eval_steps = 500
    save_steps = 500
    save_total_limit = 2
    fp16 = True  # 使用混合精度训练
    
    # ==================== 加载数据集 ====================
    print("Loading dataset...")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run the data preparation script first:")
        print("  python find_tune/make_data.py")
        print("  python find_tune/load_data.py")
        return
    
    dataset = load_from_disk(dataset_path)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # ==================== 加载模型和处理器 ====================
    print(f"Loading model: {model_name}")
    
    # 加载特征提取器
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    # 加载分词器
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, 
        language=language, 
        task=task
    )
    
    # 组合成处理器
    processor = WhisperProcessor.from_pretrained(
        model_name, 
        language=language, 
        task=task
    )
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # 配置模型
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # 训练时禁用缓存
    
    # ==================== 准备数据集 ====================
    print("Preparing dataset...")
    
    # 确保音频列的格式正确
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 应用预处理函数
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    
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
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
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
    print("Starting training...")
    trainer.train()
    
    # ==================== 保存最终模型 ====================
    print("Saving final model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # ==================== 最终评估 ====================
    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print(f"Final WER: {metrics['eval_wer']:.2f}%")
    
    print(f"\nTraining completed! Model saved to: {output_dir}")
    print("\nTo use the fine-tuned model:")
    print(f"  from transformers import pipeline")
    print(f"  pipe = pipeline('automatic-speech-recognition', model='{output_dir}')")
    print(f"  result = pipe('audio.wav')")


if __name__ == "__main__":
    main()
