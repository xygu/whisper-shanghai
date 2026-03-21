"""
Whisper 上海方言端到端推理脚本
任务：上海话语音 -> 普通话文本（跳过上海话文本中间步骤）
"""
import os
import argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa

def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    加载端到端微调后的模型
    
    Args:
        model_path: 模型路径
        device: 设备
    
    Returns:
        processor, model
    """
    print(f"加载端到端模型: {model_path}")
    print(f"设备: {device}")
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    print("✓ 模型加载完成")
    return processor, model

def transcribe_audio(audio_path, processor, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    端到端转录：上海话语音 -> 普通话文本
    
    Args:
        audio_path: 音频文件路径
        processor: Whisper 处理器
        model: Whisper 模型
        device: 设备
    
    Returns:
        普通话文本
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 提取特征
    input_features = processor.feature_extractor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # 生成文本
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # 解码
    transcription = processor.tokenizer.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription

def transcribe_with_pipeline(audio_path, model_path):
    """
    使用 pipeline 进行端到端转录
    
    Args:
        audio_path: 音频文件路径
        model_path: 模型路径
    
    Returns:
        普通话文本
    """
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    result = pipe(audio_path)
    return result["text"]

def main():
    parser = argparse.ArgumentParser(description="Whisper 上海方言端到端推理（上海话语音 -> 普通话文本）")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="微调后的模型路径")
    parser.add_argument("--audio", type=str, required=True,
                        help="输入音频文件路径")
    parser.add_argument("--use_pipeline", action="store_true",
                        help="使用 pipeline 进行推理")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Whisper 上海方言端到端推理")
    print("任务：上海话语音 -> 普通话文本")
    print("=" * 60)
    
    if not os.path.exists(args.audio):
        print(f"❌ 音频文件不存在: {args.audio}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return
    
    print(f"\n输入音频: {args.audio}")
    print(f"模型路径: {args.model_path}")
    
    if args.use_pipeline:
        print("\n使用 Pipeline 推理...")
        result = transcribe_with_pipeline(args.audio, args.model_path)
    else:
        print("\n加载模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, model = load_model(args.model_path, device)
        
        print("\n开始转录...")
        result = transcribe_audio(args.audio, processor, model, device)
    
    print(f"\n{'=' * 60}")
    print(f"转录结果（普通话）:")
    print(f"{'=' * 60}")
    print(result)
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
