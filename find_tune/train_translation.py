"""
上海话到普通话翻译模型训练脚本（集成 WandB）
使用 mT5 模型进行 Seq2Seq 翻译
"""
import os
from datetime import datetime

# ==================== 配置镜像（必须在导入 transformers 之前）====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'

# WandB 集成
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB 未安装。运行 'pip install wandb' 来启用训练监控。")
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import load_from_disk
import evaluate


class EpochLoggingCallback(TrainerCallback):
    """
    自定义回调：在每个 epoch 结束时记录日志
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.current_epoch = 0
    
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """epoch 开始时"""
        self.current_epoch = int(state.epoch) if state.epoch else 0
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"开始 Epoch {self.current_epoch + 1}/{args.num_train_epochs}")
        self.logger.info(f"{'='*60}")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """epoch 结束时"""
        epoch = int(state.epoch) if state.epoch else self.current_epoch + 1
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch}/{args.num_train_epochs} 完成")
        self.logger.info(f"当前步数: {state.global_step}")
        if state.log_history:
            # 获取最近的训练损失
            recent_logs = [log for log in state.log_history if 'loss' in log]
            if recent_logs:
                latest_loss = recent_logs[-1].get('loss', 'N/A')
                self.logger.info(f"最近训练损失: {latest_loss}")
        self.logger.info(f"{'='*60}\n")
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """评估时"""
        epoch = int(state.epoch) if state.epoch else 0
        step = state.global_step
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"评估结果 (Epoch {epoch}, Step {step}):")
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        self.logger.info(f"{'='*60}\n")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """记录训练日志"""
        if logs:
            epoch = state.epoch if state.epoch else 0
            step = state.global_step
            # 只记录包含 loss 的日志
            if 'loss' in logs:
                self.logger.debug(f"Step {step} (Epoch {epoch:.2f}): loss={logs.get('loss', 'N/A'):.4f}, lr={logs.get('learning_rate', 'N/A'):.2e}")


def setup_logging(output_dir: str, log_filename: str = "training.log"):
    """
    设置日志系统，同时输出到控制台和文件
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    
    # 创建 logger
    logger = logging.getLogger("translation_trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # 清除已有的 handlers
    
    # 文件 handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path


def preprocess_function(examples, tokenizer, max_source_length=128, max_target_length=128):
    """
    预处理函数：对源文本和目标文本进行分词
    
    Args:
        examples: 数据样本
        tokenizer: 分词器
        max_source_length: 源文本最大长度
        max_target_length: 目标文本最大长度
    """
    # 添加任务前缀（可选，帮助模型理解任务）
    prefix = "翻译上海话到普通话: "
    inputs = [prefix + text for text in examples["source"]]
    targets = examples["target"]
    
    # 对源文本进行编码
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    
    # 对目标文本进行编码
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer, metric_bleu, metric_rouge, logger=None):
    """
    计算评估指标：BLEU 和 ROUGE
    """
    preds, labels = eval_preds
    
    # 获取 pad_token_id，确保是有效值
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = len(tokenizer)
    
    if logger:
        logger.debug(f"compute_metrics: preds shape={preds.shape}, labels shape={labels.shape}")
        logger.debug(f"compute_metrics: preds 范围=[{preds.min()}, {preds.max()}], vocab_size={vocab_size}")
    
    # 将 -100 替换为 pad_token_id
    labels = np.where(labels != -100, labels, pad_token_id)
    
    # 处理预测结果中的无效 token ID
    preds = np.where(preds < 0, pad_token_id, preds)
    preds = np.clip(preds, 0, vocab_size - 1)
    
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    
    # 解码预测和标签
    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        if logger:
            logger.error(f"解码错误: {e}")
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    # 去除首尾空格
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # 打印一些样本用于调试
    if logger:
        logger.debug("=" * 50)
        logger.debug("评估样本预览 (前5个):")
        for i in range(min(5, len(decoded_preds))):
            logger.debug(f"  [{i}] 预测: '{decoded_preds[i]}'")
            logger.debug(f"      标签: '{decoded_labels[i]}'")
        logger.debug("=" * 50)
        
        # 统计空预测
        empty_preds = sum(1 for p in decoded_preds if not p)
        logger.debug(f"空预测数量: {empty_preds}/{len(decoded_preds)}")
    
    # 过滤空字符串
    valid_pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p and l]
    
    if not valid_pairs:
        if logger:
            logger.warning("没有有效的预测-标签对!")
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    decoded_preds, decoded_labels = zip(*valid_pairs)
    decoded_preds = list(decoded_preds)
    decoded_labels = list(decoded_labels)
    
    # 计算 BLEU
    try:
        bleu_result = metric_bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        bleu_score = bleu_result["bleu"] * 100
    except Exception as e:
        if logger:
            logger.error(f"BLEU 计算错误: {e}")
        bleu_score = 0.0
    
    # 计算 ROUGE
    try:
        rouge_result = metric_rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        rouge1 = rouge_result["rouge1"] * 100
        rouge2 = rouge_result["rouge2"] * 100
        rougeL = rouge_result["rougeL"] * 100
    except Exception as e:
        if logger:
            logger.error(f"ROUGE 计算错误: {e}")
        rouge1 = rouge2 = rougeL = 0.0
    
    if logger:
        logger.info(f"评估结果: BLEU={bleu_score:.2f}, ROUGE-L={rougeL:.2f}")
    
    # 记录样本对比到 WandB
    if WANDB_AVAILABLE and len(decoded_preds) >= 1:
        comparison_table = wandb.Table(columns=["样本编号", "上海话(输入)", "普通话(标签)", "预测结果", "是否匹配"])
        num_samples = min(20, len(decoded_preds))
        for i in range(num_samples):
            pred_text = decoded_preds[i].strip() if decoded_preds[i] else "(空)"
            label_text = decoded_labels[i].strip() if decoded_labels[i] else "(空)"
            is_match = "✓" if pred_text == label_text and pred_text != "(空)" else "✗"
            comparison_table.add_data(f"样本 {i+1}", "-", label_text, pred_text, is_match)
        wandb.log({"translation_comparison": comparison_table})
    
    return {
        "bleu": bleu_score,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="上海话到普通话翻译模型训练")
    parser.add_argument("--model_name", type=str, default="google/mt5-small",
                        help="预训练模型名称 (默认: google/mt5-small)")
    parser.add_argument("--dataset_path", type=str, 
                        default="dataset/shanghai/translation_dataset",
                        help="翻译数据集路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认自动生成）")
    parser.add_argument("--num_train_epochs", type=int, default=30,
                        help="训练轮数 (默认: 30，小数据集需要更多轮次)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="学习率 (默认: 3e-4，mT5 需要较高学习率)")
    parser.add_argument("--max_source_length", type=int, default=64,
                        help="源文本最大长度")
    parser.add_argument("--max_target_length", type=int, default=64,
                        help="目标文本最大长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="评估间隔步数")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="保存间隔步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="早停耐心值")
    
    args = parser.parse_args()
    
    # ==================== 配置参数 ====================
    model_name = args.model_name
    dataset_path = args.dataset_path
    
    # 生成带时间戳的输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        model_short_name = model_name.split("/")[-1]
        output_dir = f"./exp/translation-{model_short_name}-{timestamp}"
    else:
        output_dir = args.output_dir
    
    # 创建输出目录并设置日志
    os.makedirs(output_dir, exist_ok=True)
    logger, log_path = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("上海话到普通话翻译模型训练")
    logger.info("=" * 60)
    logger.info(f"✓ 已配置 HuggingFace 镜像: https://hf-mirror.com")
    logger.info(f"✓ 已配置 WandB 镜像: https://api.bandw.top")
    logger.info(f"✓ 日志文件: {log_path}")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = torch.cuda.is_available()
    
    # 记录所有配置参数
    config = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "warmup_ratio": args.warmup_ratio,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "device": device,
        "fp16": fp16,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 保存配置到文件
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # WandB 配置 - 使用独立项目（与端到端模型分开）
    wandb_project = "whisper-shanghai-cascade"  # 级联方案的翻译模型
    model_short_name = model_name.split("/")[-1]
    wandb_run_name = f"translation-{model_short_name}-lr{args.learning_rate}-{timestamp}"
    
    # 初始化 WandB
    if WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group="translation",  # 分组：翻译模型
            config=config,
        )
        logger.info(f"✓ WandB 初始化成功")
        logger.info(f"  项目: {wandb_project}")
        logger.info(f"  查看训练进度: {wandb.run.get_url()}")
    else:
        logger.warning("⚠️ WandB 未启用，使用 TensorBoard 记录日志")
    
    logger.info(f"\n配置信息:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # ==================== 加载数据集 ====================
    logger.info(f"\n加载数据集: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"❌ 数据集未找到: {dataset_path}")
        logger.info("\n请先运行数据准备脚本:")
        logger.info("  python find_tune/load_data_translation.py")
        return
    
    dataset = load_from_disk(dataset_path)
    logger.info(f"✓ 训练样本: {len(dataset['train'])}")
    logger.info(f"✓ 测试样本: {len(dataset['test'])}")
    
    # 打印数据样本
    logger.debug("\n数据样本预览:")
    for i in range(min(3, len(dataset['train']))):
        sample = dataset['train'][i]
        logger.debug(f"  [{i}] source: {sample['source']}")
        logger.debug(f"      target: {sample['target']}")
    
    # ==================== 加载模型和分词器 ====================
    logger.info(f"\n加载模型: {model_name}")
    
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    
    logger.info("✓ 模型加载完成")
    logger.info(f"  模型参数量: {model.num_parameters() / 1e6:.1f}M")
    logger.info(f"  词汇表大小: {len(tokenizer)}")
    logger.info(f"  pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"  eos_token_id: {tokenizer.eos_token_id}")
    
    # ==================== 预处理数据集 ====================
    logger.info("\n预处理数据集...")
    
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(
            examples, 
            tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )
    
    logger.info("✓ 数据预处理完成")
    
    # 验证预处理结果
    logger.debug("\n预处理后的数据样本:")
    sample = tokenized_dataset["train"][0]
    logger.debug(f"  input_ids 长度: {len(sample['input_ids'])}")
    logger.debug(f"  labels 长度: {len(sample['labels'])}")
    logger.debug(f"  input_ids: {sample['input_ids'][:20]}...")
    logger.debug(f"  labels: {sample['labels'][:20]}...")
    
    # 解码验证
    decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    decoded_label = tokenizer.decode([l if l != -100 else tokenizer.pad_token_id for l in sample['labels']], skip_special_tokens=True)
    logger.debug(f"  解码后 input: {decoded_input}")
    logger.debug(f"  解码后 label: {decoded_label}")
    
    # ==================== 初始化数据整理器 ====================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    
    # ==================== 加载评估指标 ====================
    logger.info("\n加载评估指标...")
    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    logger.info("✓ 评估指标加载完成")
    
    # ==================== 配置训练参数 ====================
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=20,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["wandb", "tensorboard"] if WANDB_AVAILABLE else ["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=False,
        fp16=fp16,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )
    
    # ==================== 初始化训练器 ====================
    # 创建自定义回调
    epoch_logging_callback = EpochLoggingCallback(logger)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, metric_bleu, metric_rouge, logger
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            epoch_logging_callback,
        ],
    )
    
    # ==================== 开始训练 ====================
    logger.info("\n" + "=" * 60)
    logger.info("开始训练...")
    logger.info("=" * 60 + "\n")
    
    train_result = trainer.train()
    
    # 记录训练结果
    logger.info("\n训练统计:")
    logger.info(f"  总步数: {train_result.global_step}")
    logger.info(f"  训练损失: {train_result.training_loss:.4f}")
    
    # ==================== 保存最终模型 ====================
    logger.info("\n保存最终模型...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # ==================== 最终评估 ====================
    logger.info("\n运行最终评估...")
    metrics = trainer.evaluate()
    
    # 保存评估结果
    metrics_path = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # ==================== 测试推理 ====================
    logger.info("\n测试推理...")
    model.eval()
    test_samples = [
        "侬好",
        "阿拉两个人来聊聊金融方面呃",
        "吾已经做了已经到八七年了",
    ]
    
    logger.info("推理测试结果:")
    for sample in test_samples:
        input_text = f"翻译上海话到普通话: {sample}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=args.max_source_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_target_length,
                num_beams=4,
                early_stopping=True,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"  输入: {sample}")
        logger.info(f"  输出: {result}")
    
    # ==================== 输出总结 ====================
    # 记录最终指标到 WandB
    if WANDB_AVAILABLE:
        wandb.log({
            "final/eval_bleu": metrics.get('eval_bleu', 0),
            "final/eval_rouge1": metrics.get('eval_rouge1', 0),
            "final/eval_rouge2": metrics.get('eval_rouge2', 0),
            "final/eval_rougeL": metrics.get('eval_rougeL', 0),
            "final/eval_loss": metrics.get('eval_loss', 0),
        })
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"训练完成!")
    logger.info(f"{'=' * 60}")
    logger.info(f"最终 BLEU: {metrics.get('eval_bleu', 0):.2f}")
    logger.info(f"最终 ROUGE-L: {metrics.get('eval_rougeL', 0):.2f}")
    logger.info(f"模型保存位置: {final_model_path}")
    logger.info(f"日志文件: {log_path}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"评估结果: {metrics_path}")
    
    if WANDB_AVAILABLE:
        logger.info(f"查看完整训练报告: {wandb.run.get_url()}")
        wandb.finish()
    
    logger.info(f"\n使用翻译模型:")
    logger.info(f"  from transformers import MT5ForConditionalGeneration, MT5Tokenizer")
    logger.info(f"  tokenizer = MT5Tokenizer.from_pretrained('{final_model_path}')")
    logger.info(f"  model = MT5ForConditionalGeneration.from_pretrained('{final_model_path}')")
    logger.info(f"  inputs = tokenizer('翻译上海话到普通话: 侬好', return_tensors='pt')")
    logger.info(f"  outputs = model.generate(**inputs, max_length=64, num_beams=4)")
    logger.info(f"  print(tokenizer.decode(outputs[0], skip_special_tokens=True))")


class ShanghaiToMandarinTranslator:
    """
    上海话到普通话翻译器
    可以作为独立模块使用，也可以接在 Whisper ASR 后面
    """
    
    def __init__(
        self,
        model_path: str = "./exp/translation-mt5-small/final_model",
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        """
        初始化翻译器
        
        Args:
            model_path: 翻译模型路径
            device: 设备 (cuda/cpu)
            max_length: 最大生成长度
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.max_length = max_length
        
        print(f"加载翻译模型: {model_path}")
        self.tokenizer = MT5Tokenizer.from_pretrained(model_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        print(f"✓ 翻译模型加载完成 (设备: {device})")
    
    def translate(self, text: str) -> str:
        """
        将上海话文本翻译为普通话
        
        Args:
            text: 上海话文本
        
        Returns:
            普通话文本
        """
        # 添加任务前缀
        input_text = f"翻译上海话到普通话: {text}"
        
        # 编码
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )
        
        # 解码
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        批量翻译
        
        Args:
            texts: 上海话文本列表
        
        Returns:
            普通话文本列表
        """
        # 添加任务前缀
        input_texts = [f"翻译上海话到普通话: {text}" for text in texts]
        
        # 编码
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )
        
        # 解码
        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated


if __name__ == "__main__":
    main()
