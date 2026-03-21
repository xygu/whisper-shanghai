"""
使用微调后的 Whisper 模型进行推理
支持多种音频格式：wav, mp3, m4a, flac, ogg, wma, aac 等

级联方案支持：
- Whisper ASR: 上海话语音 -> 上海话文本
- mT5 翻译（可选）: 上海话文本 -> 普通话文本
"""
import argparse
import os
import torch
from transformers import pipeline, MT5ForConditionalGeneration, MT5Tokenizer
import numpy as np
from typing import Optional, Dict, Any

# 支持的音频格式
SUPPORTED_FORMATS = {
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac',
    '.opus', '.webm', '.mp4', '.avi', '.mkv', '.mov'
}


def load_audio(audio_file, target_sr=16000):
    """
    加载音频文件，支持多种格式
    
    Args:
        audio_file: 音频文件路径
        target_sr: 目标采样率（Whisper 需要 16kHz）
    
    Returns:
        numpy array: 音频数据
        int: 采样率
    """
    ext = os.path.splitext(audio_file)[1].lower()
    
    if ext not in SUPPORTED_FORMATS:
        print(f"警告: 未知格式 {ext}，尝试使用 ffmpeg 加载...")
    
    # 优先使用 librosa（支持大多数格式，底层调用 ffmpeg/soundfile）
    try:
        import librosa
        audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"librosa 加载失败: {e}")
    
    # 备选方案：使用 soundfile（对 wav/flac 支持更好）
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # 转为单声道
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr
    except Exception as e:
        print(f"soundfile 加载失败: {e}")
    
    # 最后方案：使用 pydub（需要 ffmpeg）
    try:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(audio_file)
        audio_segment = audio_segment.set_frame_rate(target_sr).set_channels(1)
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** (audio_segment.sample_width * 8 - 1))  # 归一化
        return samples, target_sr
    except Exception as e:
        print(f"pydub 加载失败: {e}")
    
    raise RuntimeError(
        f"无法加载音频文件: {audio_file}\n"
        f"请确保已安装 ffmpeg: sudo apt install ffmpeg 或 brew install ffmpeg"
    )


class ShanghaiToMandarinTranslator:
    """
    上海话到普通话翻译器
    可以作为独立模块使用，也可以接在 Whisper ASR 后面
    """
    
    def __init__(
        self,
        model_path: str,
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
        input_text = f"翻译上海话到普通话: {text}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )
        
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated


def transcribe_audio(
    audio_file,
    model_path="./whisper-shanghai-finetuned",
    language="Chinese",
    device=None,
    enable_translation=False,
    translation_model_path=None,
) -> Dict[str, Any]:
    """
    使用微调后的模型转录音频，可选翻译为普通话
    
    Args:
        audio_file: 音频文件路径（支持 wav, mp3, m4a, flac, ogg 等格式）
        model_path: 微调后的 Whisper 模型路径
        language: 语言
        device: 设备 (cuda/cpu)
        enable_translation: 是否启用翻译模块
        translation_model_path: 翻译模型路径
    
    Returns:
        dict: 包含 'shanghai_text'（上海话文本）和可选的 'mandarin_text'（普通话文本）
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ext = os.path.splitext(audio_file)[1].lower()
    print(f"音频格式: {ext}")
    print(f"加载 ASR 模型: {model_path}")
    print(f"使用设备: {device}")
    print(f"翻译模块: {'启用' if enable_translation else '禁用'}")
    
    # 创建 ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device
    )
    
    # 清除模型中可能冲突的 forced_decoder_ids 配置
    if hasattr(pipe.model.config, 'forced_decoder_ids'):
        pipe.model.config.forced_decoder_ids = None
    if hasattr(pipe.model.generation_config, 'forced_decoder_ids'):
        pipe.model.generation_config.forced_decoder_ids = None
    
    print(f"\n转录音频: {audio_file}")
    
    # 对于非 wav 格式，先手动加载并转换
    if ext != '.wav':
        print(f"转换音频格式 {ext} -> 16kHz mono...")
        audio_data, sr = load_audio(audio_file, target_sr=16000)
        audio_input = {"array": audio_data, "sampling_rate": sr}
        result = pipe(audio_input, generate_kwargs={"language": language, "task": "transcribe"})
    else:
        result = pipe(audio_file, generate_kwargs={"language": language, "task": "transcribe"})
    
    shanghai_text = result['text']
    
    print(f"\n上海话转录结果:")
    print(f"  {shanghai_text}")
    
    output = {
        'shanghai_text': shanghai_text,
        'mandarin_text': None,
    }
    
    # 如果启用翻译模块
    if enable_translation:
        if translation_model_path is None:
            print("\n⚠️ 未指定翻译模型路径，跳过翻译")
        elif not os.path.exists(translation_model_path):
            print(f"\n⚠️ 翻译模型不存在: {translation_model_path}，跳过翻译")
        else:
            print(f"\n翻译为普通话...")
            translator = ShanghaiToMandarinTranslator(
                model_path=translation_model_path,
                device=device,
            )
            mandarin_text = translator.translate(shanghai_text)
            output['mandarin_text'] = mandarin_text
            
            print(f"\n普通话翻译结果:")
            print(f"  {mandarin_text}")
    
    return output


def transcribe_audio_simple(
    audio_file,
    model_path="./whisper-shanghai-finetuned",
    language="Chinese",
    device=None
) -> str:
    """
    简化版转录函数（向后兼容）
    
    Args:
        audio_file: 音频文件路径
        model_path: 模型路径
        language: 语言
        device: 设备
    
    Returns:
        str: 转录文本
    """
    result = transcribe_audio(
        audio_file=audio_file,
        model_path=model_path,
        language=language,
        device=device,
        enable_translation=False,
    )
    return result['shanghai_text']

def main():
    parser = argparse.ArgumentParser(
        description="使用微调后的 Whisper 模型进行音频转录，可选翻译为普通话"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="音频文件路径"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./whisper-shanghai-finetuned",
        help="微调后的 Whisper ASR 模型路径"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="语言"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="是否启用翻译模块（上海话 -> 普通话）"
    )
    parser.add_argument(
        "--translation_model_path",
        type=str,
        default="./exp/translation-mt5-small/final_model",
        help="翻译模型路径"
    )
    
    args = parser.parse_args()
    
    result = transcribe_audio(
        audio_file=args.audio_file,
        model_path=args.model_path,
        language=args.language,
        device=args.device,
        enable_translation=args.translate,
        translation_model_path=args.translation_model_path,
    )
    
    print("\n" + "=" * 50)
    print("最终结果:")
    print("=" * 50)
    print(f"上海话: {result['shanghai_text']}")
    if result['mandarin_text']:
        print(f"普通话: {result['mandarin_text']}")


if __name__ == "__main__":
    main()
