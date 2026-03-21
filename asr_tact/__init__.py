"""
ASR-TACT: Targeted Activation Concept Tuning for ASR

基于 SAE (Sparse Autoencoder) 的可解释 ASR 微调框架
参考 TaCT 项目实现，适配 Whisper 语音识别模型

核心组件:
- SAE: 稀疏自编码器，将隐层表征投射到高维稀疏空间
- FeatureExtractor: 音频特征提取器（声学/语言/错误模式）
- NeuronAnalyzer: 神经元激活分析和语义标注
- GatedLoRA: 门控 LoRA 微调模块
"""

from .sae import SAE
from .feature_extractor import ASRFeatureExtractor
from .neuron_analyzer import NeuronAnalyzer
from .gated_lora import GatedLoRA

__version__ = "0.1.0"
__all__ = ["SAE", "ASRFeatureExtractor", "NeuronAnalyzer", "GatedLoRA"]
