"""
数据处理工具模块

提供 Whisper 音频特征提取的公共函数，供微调和 SAE 训练复用
"""

from typing import Dict, Any, List, Optional
import torch
import numpy as np

from transformers import WhisperProcessor


def prepare_audio_features(
    batch: Dict[str, Any],
    processor: WhisperProcessor,
) -> Dict[str, Any]:
    """
    准备音频特征：提取 log-Mel 频谱图
    
    这是微调和 SAE 训练共用的音频处理函数，确保数据分布一致。
    
    Args:
        batch: 包含 "audio" 字段的数据批次
        processor: Whisper 处理器
        
    Returns:
        添加了 "input_features" 字段的批次
    """
    audio = batch["audio"]
    
    # 计算 log-Mel 频谱图特征
    # WhisperFeatureExtractor 默认会 padding 到 30 秒（3000 mel 帧）
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    return batch


def audio_length_to_encoder_length(audio_samples: int, sample_rate: int = 16000) -> int:
    """
    计算音频样本数对应的 Whisper encoder 输出序列长度
    
    Whisper 的 mel 特征: hop_length=160, 所以 mel_frames = audio_samples // 160
    Encoder 使用两层 stride=2 的卷积，所以 encoder_len = mel_frames // 4
    最终: encoder_len = audio_samples // 160 // 4 = audio_samples // 640
    但 Whisper 固定输出 1500 帧（对应 30 秒音频），短音频会被 padding
    
    Args:
        audio_samples: 音频样本数
        sample_rate: 采样率（默认 16000）
        
    Returns:
        encoder 输出的序列长度
    """
    mel_frames = audio_samples // 160
    encoder_len = (mel_frames + 1) // 2  # 第一层卷积 stride=2
    encoder_len = (encoder_len + 1) // 2  # 第二层卷积 stride=2
    return encoder_len


class SAEDataCollator:
    """
    SAE 训练的数据整理器
    
    复用 prepare_audio_features 提取 mel 特征，并生成 attention mask
    """
    
    # Whisper 固定参数
    WHISPER_SAMPLE_RATE = 16000
    WHISPER_MAX_AUDIO_SAMPLES = 480000  # 30 秒 * 16000 Hz
    WHISPER_ENCODER_SEQ_LEN = 1500  # 固定输出长度
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 记录原始音频长度（用于生成 attention mask）
        audio_lengths = [len(f["audio"]["array"]) for f in features]
        
        # 使用公共函数提取 mel 特征
        batch_input_features = []
        for feature in features:
            processed = prepare_audio_features(feature, self.processor)
            batch_input_features.append(processed["input_features"])
        
        # 计算每个样本在 encoder 输出中的实际长度
        encoder_lengths = [
            min(audio_length_to_encoder_length(length), self.WHISPER_ENCODER_SEQ_LEN)
            for length in audio_lengths
        ]
        
        # 创建 attention mask: 1 表示有效位置，0 表示 padding
        attention_masks = []
        for enc_len in encoder_lengths:
            mask = torch.zeros(self.WHISPER_ENCODER_SEQ_LEN)
            mask[:enc_len] = 1.0
            attention_masks.append(mask)
        
        return {
            "input_features": torch.tensor(np.stack(batch_input_features), dtype=torch.float32),
            "attention_mask": torch.stack(attention_masks),
            "encoder_lengths": torch.tensor(encoder_lengths, dtype=torch.long),
        }
