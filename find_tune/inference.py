"""
使用微调后的 Whisper 模型进行推理
"""
import torch
from transformers import pipeline
import argparse


def transcribe_audio(model_path, audio_file, language="Chinese"):
    """
    使用微调后的模型转录音频
    
    Args:
        model_path: 微调后的模型路径
        audio_file: 音频文件路径
        language: 语言（默认中文）
    """
    print(f"Loading model from: {model_path}")
    
    # 检查是否有 GPU
    device = 0 if torch.cuda.is_available() else -1
    
    # 创建 pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device
    )
    
    print(f"Transcribing: {audio_file}")
    
    # 进行转录
    result = pipe(
        audio_file,
        generate_kwargs={
            "language": language,
            "task": "transcribe"
        }
    )
    
    print("\n" + "="*50)
    print("Transcription Result:")
    print("="*50)
    print(result["text"])
    print("="*50)
    
    return result["text"]


def main():
    parser = argparse.ArgumentParser(description="Whisper 模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./whisper-finetuned-shanghai",
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="要转录的音频文件路径"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="音频语言（默认: Chinese）"
    )
    
    args = parser.parse_args()
    
    transcribe_audio(args.model_path, args.audio_file, args.language)


if __name__ == "__main__":
    main()
