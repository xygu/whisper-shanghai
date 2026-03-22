"""
Neuron Analysis Script

分析 SAE 神经元的激活模式，提取特征关联，生成 LLM 标注 prompt
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, List
from tqdm import tqdm

import torch
import numpy as np

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from datasets import load_from_disk, Audio

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_tact.sae import SAE, SAEConfig
from asr_tact.feature_extractor import ASRFeatureExtractor
from asr_tact.neuron_analyzer import NeuronAnalyzer


def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger('neuron_analysis')
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(os.path.join(output_dir, 'analysis.log'))
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def extract_encoder_hidden_states(
    model: WhisperForConditionalGeneration,
    input_features: torch.Tensor,
    layer: int = 12,
) -> torch.Tensor:
    """提取 Whisper encoder 指定层的隐状态"""
    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
    return encoder_outputs.hidden_states[layer]


def analyze_neurons(
    sae_checkpoint: str,
    model_name: str = "openai/whisper-medium",
    dataset_path: str = "dataset/shanghai/shanghai_dataset",
    output_dir: str = "./exp/neuron_analysis",
    encoder_layer: int = 12,
    topk: int = 64,
    max_samples: int = 1000,
    device: str = "cuda",
):
    """
    分析 SAE 神经元
    """
    # 创建输出目录
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, f"analysis-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("ASR-TACT Neuron Analysis")
    logger.info("=" * 70)
    
    # 加载 SAE 模型
    logger.info(f"Loading SAE checkpoint: {sae_checkpoint}")
    checkpoint = torch.load(sae_checkpoint, map_location='cpu')
    sae_config = SAEConfig.from_dict(checkpoint['config'])
    
    sae = SAE(
        input_dim=sae_config.input_dim,
        latent_dim=sae_config.latent_dim,
        norm_type=sae_config.norm_type,
        use_activate=sae_config.use_activate,
        topk_type=sae_config.topk_type,
        share_weight=sae_config.share_weight,
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    logger.info(f"SAE config: {sae_config.to_dict()}")
    
    # 加载 Whisper 模型
    logger.info(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    # 加载数据集
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 创建特征提取器
    feature_extractor = ASRFeatureExtractor(sample_rate=16000)
    
    # 创建神经元分析器
    analyzer = NeuronAnalyzer(
        latent_dim=sae_config.latent_dim,
        top_percent=0.1,  # top 10%
    )
    
    # 处理样本
    logger.info(f"Processing {min(max_samples, len(dataset['train']))} samples...")
    
    samples_to_process = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
    
    for idx, sample in enumerate(tqdm(samples_to_process, desc="Analyzing")):
        sample_id = str(idx)
        
        # 提取音频特征
        audio_array = sample['audio']['array']
        transcript = sample.get('text', '')
        
        # 获取模型预测
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)
        
        with torch.no_grad():
            # 获取预测
            generated_ids = whisper_model.generate(input_features, max_length=225)
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 获取 encoder 隐状态
            hidden_states = extract_encoder_hidden_states(
                whisper_model, input_features, layer=encoder_layer
            )
            
            # 获取 SAE 激活
            _, _, sparse = sae(hidden_states, topk)
            activations = sparse.squeeze(0)  # [seq_len, latent_dim]
        
        # 提取样本的三层特征
        sample_features = feature_extractor.extract_features(
            audio=audio_array,
            transcript=transcript,
            prediction=prediction,
            sample_id=sample_id,
        )
        
        # 记录样本 (使用新接口，分别传入三层特征)
        analyzer.record_sample(
            sample_id=sample_id,
            activations=activations,
            acoustic_features=sample_features.acoustic.to_dict(),
            linguistic_features=sample_features.linguistic.to_dict(),
            error_pattern_features=sample_features.error_pattern.to_dict(),
            metadata={'audio_path': sample.get('path', '')},
            activation_threshold=0.0,
        )
    
    # 计算每个神经元的 top 10% 样本汇总
    logger.info("Computing neuron top 10% samples...")
    analyzer.compute_neuron_top_samples()
    
    # 导出分析结果
    logger.info("Exporting analysis results...")
    analyzer.export_data(output_dir, top_n_neurons=100)
    
    # 保存分析器状态
    analyzer_path = os.path.join(output_dir, 'analyzer_state.json')
    analyzer.save(analyzer_path)
    
    logger.info("=" * 70)
    logger.info("Analysis completed!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    # 打印摘要
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    logger.info(f"Total neurons: {summary['total_neurons']}")
    logger.info(f"Active neurons: {summary['active_neurons']}")
    logger.info(f"Total samples: {summary['total_samples']}")
    
    logger.info("\nTop 10 neurons by activation rate:")
    for i, neuron in enumerate(summary['top_neurons'][:10]):
        logger.info(f"  #{neuron['id']}: {neuron['activation_rate']:.2%} - {neuron['inferred_concept']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE neurons")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to SAE checkpoint")
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium",
                        help="Whisper model name")
    parser.add_argument("--dataset_path", type=str, default="dataset/shanghai/shanghai_dataset",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./exp/neuron_analysis",
                        help="Output directory")
    parser.add_argument("--encoder_layer", type=int, default=12,
                        help="Encoder layer")
    parser.add_argument("--topk", type=int, default=64,
                        help="TopK for sparse activation")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum samples to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    analyze_neurons(
        sae_checkpoint=args.sae_checkpoint,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        encoder_layer=args.encoder_layer,
        topk=args.topk,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
