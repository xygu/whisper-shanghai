"""
SAE 统计指标计算脚本

计算并输出 SAE 模型的关键指标：
- 特征字典规模 (latent_dim)
- 激活稀疏度 (topk / latent_dim)
- 方差解释率 (explained_var)
- 死亡特征率 (dead_neurons / latent_dim)
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from tqdm import tqdm

import torch
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asr_tact.sae import SAE, SAEConfig
from asr_tact.data_utils import audio_length_to_encoder_length


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


def compute_sae_statistics(
    sae_checkpoint: str,
    model_name: str = "openai/whisper-medium",
    dataset_path: str = "dataset/shanghai/shanghai_dataset",
    encoder_layer: int = 12,
    topk: int = 64,
    max_samples: int = 500,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    计算 SAE 模型的统计指标
    
    Returns:
        包含以下指标的字典:
        - latent_dim: 特征字典规模
        - topk: 每 token 激活的特征数
        - sparsity_rate: 激活稀疏度 (topk / latent_dim)
        - avg_explained_var: 平均方差解释率
        - dead_neuron_count: 死亡神经元数量
        - dead_neuron_rate: 死亡神经元比例
        - active_neuron_count: 活跃神经元数量
        - neuron_activation_distribution: 神经元激活分布
    """
    print("=" * 70)
    print("SAE Statistics Computation")
    print("=" * 70)
    
    # 加载 SAE 模型
    print(f"Loading SAE checkpoint: {sae_checkpoint}")
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
    
    # 加载 Whisper 模型
    print(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    # 加载数据集
    print(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 统计变量
    total_tokens = 0
    total_topk_activations = 0  # topk 选中的激活总数
    explained_vars = []
    neuron_activation_counts = torch.zeros(sae_config.latent_dim, device=device)
    
    # 处理样本
    samples = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
    
    print(f"Processing {len(samples)} samples...")
    for sample in tqdm(samples, desc="Computing statistics"):
        audio_array = sample['audio']['array']
        audio_length = len(audio_array)
        
        # 处理音频
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)
        
        with torch.no_grad():
            # 提取隐状态
            hidden_states = extract_encoder_hidden_states(
                whisper_model, input_features, layer=encoder_layer
            )
            
            # 计算 attention_mask（与训练时一致）
            encoder_seq_len = audio_length_to_encoder_length(audio_length)
            actual_seq_len = hidden_states.shape[1]  # Whisper encoder 输出固定为 1500
            attention_mask = torch.zeros(1, actual_seq_len, device=device)
            attention_mask[:, :min(encoder_seq_len, actual_seq_len)] = 1.0
            
            # SAE 前向传播
            sparse_recover, aux_recover, sparse = sae(hidden_states, topk)
            
            # 使用 SAE 的 compute_loss 计算方差解释率（与训练时完全一致）
            _, loss_dict = sae.compute_loss(
                hidden_states, sparse_recover, aux_recover, attention_mask
            )
            explained_vars.append(loss_dict['explained_var'])
            
            # 统计激活（使用 topk 后的稀疏激活，而不是 ReLU 后的）
            batch_size, seq_len, _ = sparse.shape
            total_tokens += batch_size * seq_len
            
            # 获取 topk 后的稀疏激活和 mask
            sparse_topk, _, mask = sae._get_topk(sparse, topk)
            
            # 统计每个神经元的激活次数（基于 topk mask）
            # mask shape: [batch, seq_len, latent_dim]
            neuron_activation_counts += mask.sum(dim=(0, 1))
            
            # 统计 topk 激活总数
            # 对于 batch_topk: 每个 batch 选 topk * seq_len 个激活
            # 对于 topk: 每个位置选 topk 个激活
            total_topk_activations += mask.sum().item()
    
    # 计算统计指标
    avg_explained_var = np.mean(explained_vars)
    
    # 死亡神经元：从未激活的神经元
    dead_neurons = (neuron_activation_counts == 0).sum().item()
    dead_neuron_rate = dead_neurons / sae_config.latent_dim
    
    # 活跃神经元
    active_neurons = sae_config.latent_dim - dead_neurons
    
    # 激活稀疏度计算
    # 理论稀疏度: topk / latent_dim (每个 token 激活的神经元比例)
    theoretical_sparsity_rate = topk / sae_config.latent_dim
    
    # 实际每 token 平均激活数
    # 对于 batch_topk: 应该约等于 topk (因为 topk * seq_len 个激活分布在 seq_len 个 token 上)
    # 对于 topk: 应该正好等于 topk
    avg_activations_per_token = total_topk_activations / total_tokens if total_tokens > 0 else 0
    
    # 实际稀疏度 (基于实际激活数)
    actual_sparsity_rate = avg_activations_per_token / sae_config.latent_dim
    
    # 神经元激活分布
    activation_counts_np = neuron_activation_counts.cpu().numpy()
    activation_distribution = {
        'min': float(activation_counts_np.min()),
        'max': float(activation_counts_np.max()),
        'mean': float(activation_counts_np.mean()),
        'std': float(activation_counts_np.std()),
        'median': float(np.median(activation_counts_np)),
        'percentile_90': float(np.percentile(activation_counts_np, 90)),
        'percentile_99': float(np.percentile(activation_counts_np, 99)),
    }
    
    stats = {
        # 基本配置
        'input_dim': sae_config.input_dim,
        'latent_dim': sae_config.latent_dim,
        'topk': topk,
        'topk_type': sae_config.topk_type,
        'norm_type': sae_config.norm_type,
        
        # 稀疏度指标
        'theoretical_sparsity_rate': theoretical_sparsity_rate,
        'theoretical_sparsity_rate_percent': f"{theoretical_sparsity_rate * 100:.4f}%",
        'actual_sparsity_rate': actual_sparsity_rate,
        'actual_sparsity_rate_percent': f"{actual_sparsity_rate * 100:.4f}%",
        'avg_activations_per_token': avg_activations_per_token,
        
        # 重建质量
        'avg_explained_var': avg_explained_var,
        'avg_explained_var_percent': f"{avg_explained_var * 100:.2f}%",
        
        # 神经元利用率
        'dead_neuron_count': dead_neurons,
        'dead_neuron_rate': dead_neuron_rate,
        'dead_neuron_rate_percent': f"{dead_neuron_rate * 100:.2f}%",
        'active_neuron_count': active_neurons,
        'active_neuron_rate': 1 - dead_neuron_rate,
        
        # 激活分布
        'neuron_activation_distribution': activation_distribution,
        
        # 样本信息
        'total_samples': len(samples),
        'total_tokens': total_tokens,
    }
    
    return stats


def print_stats(stats: Dict[str, Any]):
    """打印统计结果"""
    print("\n" + "=" * 70)
    print("SAE Model Statistics")
    print("=" * 70)
    
    print("\n### 基本配置")
    print(f"  输入维度 (input_dim):     {stats['input_dim']}")
    print(f"  特征字典规模 (latent_dim): {stats['latent_dim']}")
    print(f"  TopK:                      {stats['topk']}")
    print(f"  TopK 类型:                 {stats['topk_type']}")
    print(f"  归一化类型:                {stats['norm_type']}")
    
    print("\n### 稀疏度指标")
    print(f"  理论稀疏度 (topk/latent):  {stats['theoretical_sparsity_rate_percent']}")
    print(f"  实际稀疏度:                {stats['actual_sparsity_rate_percent']}")
    print(f"  每 token 平均激活数:       {stats['avg_activations_per_token']:.2f}")
    print(f"  (期望值 ≈ topk = {stats['topk']})")
    
    print("\n### 重建质量")
    print(f"  方差解释率:                {stats['avg_explained_var_percent']}")
    
    print("\n### 神经元利用率")
    print(f"  死亡神经元数量:            {stats['dead_neuron_count']}")
    print(f"  死亡神经元比例:            {stats['dead_neuron_rate_percent']}")
    print(f"  活跃神经元数量:            {stats['active_neuron_count']}")
    
    print("\n### 激活分布")
    dist = stats['neuron_activation_distribution']
    print(f"  最小激活次数:              {dist['min']:.0f}")
    print(f"  最大激活次数:              {dist['max']:.0f}")
    print(f"  平均激活次数:              {dist['mean']:.2f}")
    print(f"  中位数激活次数:            {dist['median']:.0f}")
    print(f"  90% 分位数:                {dist['percentile_90']:.0f}")
    print(f"  99% 分位数:                {dist['percentile_99']:.0f}")
    
    print("\n### 与论文/TaCT 对比")
    print("  ┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐")
    print("  │ 指标            │ 本项目           │ TaCT 参考        │ Claude 论文      │")
    print("  ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤")
    print(f"  │ 输入维度        │ {stats['input_dim']:>16,} │ 1536             │ -                │")
    print(f"  │ 特征字典规模    │ {stats['latent_dim']:>16,} │ 2048~4096        │ 34,164,353       │")
    print(f"  │ TopK            │ {stats['topk']:>16} │ 48               │ -                │")
    print(f"  │ 激活稀疏度      │ {stats['actual_sparsity_rate_percent']:>16} │ ~2.3%            │ <0.001%          │")
    print(f"  │ 方差解释率      │ {stats['avg_explained_var_percent']:>16} │ >90%             │ 67%              │")
    print(f"  │ 死亡特征率      │ {stats['dead_neuron_rate_percent']:>16} │ <1%              │ <1%              │")
    print("  └─────────────────┴──────────────────┴──────────────────┴──────────────────┘")
    
    print("\n### 推荐超参数调整")
    if stats['avg_explained_var'] < 0.85:
        print("  ⚠️  方差解释率偏低，建议：")
        print("      - 增加训练轮数")
        print("      - 减小 latent_dim（如 2048 或 4096）")
        print("      - 增大 topk（如 48 或 96）")
    if stats['avg_activations_per_token'] > stats['topk'] * 1.5:
        print("  ⚠️  每 token 激活数异常，可能存在统计 bug")
    if stats['dead_neuron_rate'] > 0.1:
        print("  ⚠️  死亡神经元过多，建议检查辅助损失")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compute SAE statistics")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to SAE checkpoint")
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium",
                        help="Whisper model name")
    parser.add_argument("--dataset_path", type=str, default="dataset/shanghai/shanghai_dataset",
                        help="Path to dataset")
    parser.add_argument("--encoder_layer", type=int, default=12,
                        help="Encoder layer")
    parser.add_argument("--topk", type=int, default=64,
                        help="TopK for sparse activation")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples to process")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    
    args = parser.parse_args()
    
    stats = compute_sae_statistics(
        sae_checkpoint=args.sae_checkpoint,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        encoder_layer=args.encoder_layer,
        topk=args.topk,
        max_samples=args.max_samples,
        device=args.device,
    )
    
    print_stats(stats)
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nStatistics saved to: {args.output}")


if __name__ == "__main__":
    main()
