#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带调试信息的Whisper训练脚本
用于诊断dataset.map()后卡住的问题
"""

from datasets import load_dataset, Audio, Dataset
import json
import os
import logging
import traceback
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback

class CustomSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # 每次保存检查点时同时保存processor
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        kwargs['tokenizer'].save_pretrained(checkpoint_dir)
        kwargs['processor'].save_pretrained(checkpoint_dir)
# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq

class WhisperDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, feature_extractor, tokenizer, model, padding=True):
        super().__init__(tokenizer, model=model, padding=padding)
        self.feature_extractor = feature_extractor  # 单独保存feature_extractor

    def __call__(self, features):
        # 1. 处理音频特征
        input_features = [{"input_features": np.array(f["input_features"], dtype=np.float32)} 
                         for f in features]
        
        # 使用feature_extractor进行填充
        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        
        # 2. 处理文本标签
        labels = [{"input_ids": np.array(f["labels"], dtype=np.int32)} 
                 for f in features]
        
        label_batch = self.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt",
        )
        
        # 3. 将填充部分设为-100（忽略损失计算）
        labels = label_batch["input_ids"].masked_fill(
            label_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        
        return batch


def check_data_file(file_path):
    """检查数据文件是否存在且可读"""
    if not os.path.exists(file_path):
        logger.error(f"数据文件不存在: {file_path}")
        return False
    
    with open(file_path, "r", encoding="utf8") as f:
        first_line = f.readline().strip()
        if not first_line:
            logger.error("数据文件为空")
            return False
        
        # 尝试解析第一行JSON
        json.loads(first_line)
        logger.info(f"数据文件格式正确: {file_path}")
        return True
    

def load_and_validate_data(file_path):
    """加载并验证数据"""
    logger.info(f"开始加载数据: {file_path}")
    
    if not check_data_file(file_path):
        return None
    

    with open(file_path, "r", encoding="utf8") as f:
        lines = [json.loads(line) for line in f]
    
    logger.info(f"成功加载 {len(lines)} 条数据")
    
    # 验证数据格式
    for i, line in enumerate(lines[:5]):  # 只检查前5条
        if "audio" not in line or "text" not in line:
            logger.error(f"第 {i} 行缺少必需字段: {line}")
            return None
    
    logger.info("数据格式验证通过")
    return lines

def safe_preprocess(example, processor, sr):

    audio = example["audio"]
    if audio is None or "array" not in audio:
        return None

    # 返回 numpy 数组
    input_features = processor(
        audio["array"],
        sampling_rate=sr,
        return_tensors="np",  # 关键修改
        padding=True,
    ).input_features[0]

    labels = processor.tokenizer(
        example["text"],
        return_tensors="np",  # 关键修改
        padding=True,
    ).input_ids[0]

    return {
        "input_features": input_features,
        "labels": labels,
    }
    

"""主函数"""
sr = 16000
root_dir = 'dataset/shanghai'
data_file = os.path.join(root_dir, "whisper_finetune_data_short.jsonl")

# 1. 检查数据文件
logger.info("=== 步骤1: 检查数据文件 ===")
if not os.path.exists(root_dir):
    logger.error(f"目录不存在: {root_dir}")
    logger.info("请确保数据文件路径正确")


# 2. 加载数据
logger.info("=== 步骤2: 加载数据 ===")
lines = load_and_validate_data(data_file)


# 3. 创建数据集
logger.info("=== 步骤3: 创建HuggingFace数据集 ===")

dataset = Dataset.from_list(lines)
logger.info(f"数据集创建成功，包含 {len(dataset)} 条记录")
logger.info(f"数据集列名: {dataset.column_names}")


# 4. 转换音频列
logger.info("=== 步骤4: 转换音频列 ===")
dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))
logger.info("音频列转换成功")

logger.info("=== 步骤5: 加载模型和处理器 ===")
model_name = "pretrained_models/transformer_small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"

logger.info("加载处理器...")
processor = WhisperProcessor.from_pretrained(model_name)
logger.info("处理器加载成功")
    
logger.info("加载模型...")
model = WhisperForConditionalGeneration.from_pretrained(model_name)
# 修改这部分配置
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
model.config.suppress_tokens = []
logger.info("模型加载成功")

# 6. 数据预处理
logger.info("=== 步骤6: 数据预处理 ===")
logger.info("开始预处理数据...")

logger.info("测试第一条数据...")
test_example = dataset[0]
test_result = safe_preprocess(test_example, processor, sr)
if test_result is None:
        logger.error("第一条数据预处理失败，停止处理")
else:
    logger.info("第一条数据预处理成功")
    
    # 处理所有数据
logger.info("开始处理所有数据...")
processed_dataset = dataset.map(
    lambda x: safe_preprocess(x, processor, sr),
    remove_columns=dataset.column_names,
    desc="预处理数据"
)

# 过滤掉None值
processed_dataset = processed_dataset.filter(
    lambda x: x is not None,
    desc="过滤无效数据"
)
logger.info("示例数据:", processed_dataset[0])
logger.info(f"预处理完成，有效数据: {len(processed_dataset)} 条")


# 7. 设置训练参数
logger.info("=== 步骤7: 设置训练参数 ===")
training_args = TrainingArguments(
    output_dir="./whisper-shanghai",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    fp16=True,
    report_to="none",
    save_total_limit=2,  # 限制保存的检查点数量
    save_safetensors=True,  # 使用更安全的模型保存格式
    include_inputs_for_metrics=True  # 确保保存完整信息
)

# 8. 创建训练器
logger.info("=== 步骤8: 创建训练器 ===")

data_collator = WhisperDataCollator(
    feature_extractor=processor.feature_extractor,  # 传入feature_extractor
    tokenizer=processor.tokenizer,  # 传入tokenizer
    model=model,
    padding=True
)
class ProcessorSavingCallback(TrainerCallback):
    def __init__(self, processor, model):  # 添加model参数
        self.processor = processor
        self.model = model
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存processor相关文件
        self.processor.save_pretrained(checkpoint_dir)
        self.model.config.save_pretrained(checkpoint_dir)  # 保存config
        
        if not os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json")):
            with open(os.path.join(checkpoint_dir, "preprocessor_config.json"), "w") as f:
                json.dump(self.processor.feature_extractor.to_dict(), f)


# 在创建Trainer时修改为：
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    callbacks=[ProcessorSavingCallback(processor, model)]  # 传入model
)
# 9. 开始训练
logger.info("=== 步骤9: 开始训练 ===")
trainer.train()
# 9. 开始训练最后的保存部分
final_dir = "./whisper-shanghai-final"
os.makedirs(final_dir, exist_ok=True)

# 确保配置正确
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

# 保存完整模型和处理器
trainer.save_model(final_dir)
processor.save_pretrained(final_dir)
model.config.save_pretrained(final_dir)

# 确保生成所有必要文件
with open(f"{final_dir}/preprocessor_config.json", "w") as f:
    json.dump(processor.feature_extractor.to_dict(), f)

