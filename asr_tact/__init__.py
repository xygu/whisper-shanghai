"""
ASR-TACT: Targeted Activation Concept Tuning for ASR

基于 SAE (Sparse Autoencoder) 的可解释 ASR 微调框架
参考 TaCT 项目实现，适配 Whisper 语音识别模型

核心组件:
- SAE: 稀疏自编码器，将隐层表征投射到高维稀疏空间
- ASRFeatureExtractor: 音频特征提取器（三层特征：声学/语言/错误模式）
- NeuronAnalyzer: 神经元激活分析，输出供 LLM 手动标注的 JSON
- CounterfactualAnalyzer: 反事实分析，通过梯度和消融找关键神经元
- GatedLoRA: 门控 LoRA 微调模块

输出格式:
1. sample_features.json: 每条样本的完整三层特征 + 神经元激活值
2. neuron_top_samples.json: 每个神经元 top 10% 高激活样本的特征汇总
3. counterfactual_results.json: 反事实分析结果（梯度重要性、消融影响）
"""

from .sae import SAE, SAEConfig
from .feature_extractor import (
    ASRFeatureExtractor,
    AcousticFeatures,
    LinguisticFeatures,
    ErrorPatternFeatures,
    SampleFeatures,
)
from .neuron_analyzer import (
    NeuronAnalyzer,
    SampleActivationRecord,
    NeuronTopSamples,
)
from .gated_lora import GatedLoRALinear, GatedLoRAConfig, GatedLoRAModel, select_gate_neurons
from .counterfactual_analysis import CounterfactualAnalyzer, NeuronImportance

__version__ = "0.1.0"
__all__ = [
    "SAE",
    "SAEConfig",
    "ASRFeatureExtractor",
    "AcousticFeatures",
    "LinguisticFeatures",
    "ErrorPatternFeatures",
    "SampleFeatures",
    "NeuronAnalyzer",
    "SampleActivationRecord",
    "NeuronTopSamples",
    "GatedLoRALinear",
    "GatedLoRAConfig",
    "GatedLoRAModel",
    "select_gate_neurons",
    "CounterfactualAnalyzer",
    "NeuronImportance",
]
