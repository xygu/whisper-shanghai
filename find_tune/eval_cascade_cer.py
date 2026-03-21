"""
计算级联模式的最终中文 CER
使用已训练的 Whisper 模型 + 翻译模型，计算到普通话的 CER

借鉴 inference_shanghai.py 的推理方式
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# 配置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import (
    pipeline,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)
from datasets import load_from_disk, Audio
import evaluate

def compute_cer(predictions, references):
    """计算 CER（字符错误率）"""
    metric = evaluate.load("wer")
    pred_chars = [" ".join(list(s.replace(" ", ""))) for s in predictions]
    ref_chars = [" ".join(list(s.replace(" ", ""))) for s in references]
    return 100 * metric.compute(predictions=pred_chars, references=ref_chars)

def main():
    parser = argparse.ArgumentParser(description="计算级联模式的最终中文 CER")
    parser.add_argument(
        "--whisper_model",
        type=str,
        required=True,
        help="Whisper ASR 模型路径"
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        default="/mnt/workspace/workgroup/qq/ts/whisper/exp/translation-mt5-small-260320-231422/final_model",
        help="翻译模型路径"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/shanghai/shanghai_unified_dataset",
        help="数据集路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 70)
    print("级联模式 CER 评估")
    print("=" * 70)
    print(f"Whisper 模型: {args.whisper_model}")
    print(f"翻译模型: {args.translation_model}")
    print(f"数据集: {args.dataset_path}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = load_from_disk(args.dataset_path)
    test_dataset = dataset["test"]
    print(f"测试样本数: {len(test_dataset)}")
    
    # 检查是否有普通话文本
    if "text_cn" not in test_dataset.column_names:
        print("❌ 数据集中没有 text_cn 字段，无法计算到普通话的 CER")
        print("请重新运行 make_data_shanghai.py 和 load_data_shanghai.py")
        return
    
    # 使用 pipeline 加载 Whisper 模型（借鉴 inference_shanghai.py）
    print(f"\n加载 Whisper 模型: {args.whisper_model}")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.whisper_model,
        device=device
    )
    
    # 清除模型中可能冲突的 forced_decoder_ids 配置
    if hasattr(asr_pipe.model.config, 'forced_decoder_ids'):
        asr_pipe.model.config.forced_decoder_ids = None
    if hasattr(asr_pipe.model.generation_config, 'forced_decoder_ids'):
        asr_pipe.model.generation_config.forced_decoder_ids = None
    print("✓ Whisper 模型加载完成")
    
    # 加载翻译模型
    # 自动检测模型路径：如果指定路径下没有 tokenizer 文件，尝试添加 /final_model
    translation_model_path = args.translation_model
    if not os.path.exists(os.path.join(translation_model_path, "spiece.model")):
        final_model_path = os.path.join(translation_model_path, "final_model")
        if os.path.exists(os.path.join(final_model_path, "spiece.model")):
            translation_model_path = final_model_path
            print(f"自动检测到模型在 final_model 子目录中")
    
    print(f"\n加载翻译模型: {translation_model_path}")
    trans_tokenizer = MT5Tokenizer.from_pretrained(translation_model_path)
    trans_model = MT5ForConditionalGeneration.from_pretrained(translation_model_path)
    trans_model.to(device)
    trans_model.eval()
    print("✓ 翻译模型加载完成")
    
    # 准备数据
    print("\n开始评估...")
    
    shanghai_preds = []  # ASR 预测的上海话
    shanghai_refs = []   # 上海话参考
    mandarin_preds = []  # 翻译后的普通话
    mandarin_refs = []   # 普通话参考
    
    # 逐样本处理（避免显存溢出）
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    for i in tqdm(range(len(test_dataset)), desc="ASR 推理"):
        sample = test_dataset[i]
        audio = sample["audio"]
        
        # 使用 pipeline 进行 ASR 推理
        audio_input = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
        result = asr_pipe(audio_input, generate_kwargs={"language": "Chinese", "task": "transcribe"})
        
        shanghai_pred = result['text']
        shanghai_preds.append(shanghai_pred)
        shanghai_refs.append(sample["text"])
        mandarin_refs.append(sample["text_cn"])