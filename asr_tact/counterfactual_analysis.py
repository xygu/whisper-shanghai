"""
Counterfactual Analysis for ASR Neurons

通过反事实优化找到对 ASR 输出影响最大的关键神经元

核心方法（参考 TaCT）:
1. 对于每个高错误率样本，优化 delta（神经元激活的变化量）来减少预测误差
2. 选择 |delta| 最大的神经元作为关键神经元
3. 这些神经元是"改变其激活值能最大程度改善输出"的神经元

关键公式:
    delta = argmin_delta [ Loss(f(x + delta)) + λ * |delta| ]
    key_neurons = topk(|delta|)
"""

import os
import sys
import json
import copy
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asr_tact.sae import SAE, SAEConfig


@dataclass
class NeuronImportance:
    """神经元重要性记录"""
    neuron_id: int
    delta_importance: float = 0.0     # delta 优化后的重要性（核心指标）
    gradient_importance: float = 0.0  # 梯度重要性（辅助）
    sample_count: int = 0             # 样本数
    mean_delta: float = 0.0           # 平均 delta 值
    
    def to_dict(self) -> dict:
        return asdict(self)

class CounterfactualAnalyzer:
    """
    反事实分析器（参考 TaCT 实现）
    
    核心思想：
    通过优化 delta（神经元激活的变化量）来减少预测误差，
    然后选择 |delta| 最大的神经元作为关键神经元。
    
    这些神经元是"改变其激活值能最大程度改善输出"的神经元，
    应该用于门控 LoRA 的触发条件。
    """
    
    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        sae: SAE,
        encoder_layer: int = 12,
        topk: int = 64,
        device: str = "cuda",
        optimization_steps: int = 50,
        learning_rate: float = 15.0,
        regularization: float = 1e-2,
    ):
        self.whisper_model = whisper_model
        self.processor = processor
        self.sae = sae
        self.encoder_layer = encoder_layer
        self.topk = topk
        self.device = device
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # 神经元重要性统计
        self.neuron_importance: Dict[int, NeuronImportance] = {}
        self.delta_accumulator: Dict[int, List[float]] = defaultdict(list)
        
        # 初始化
        for i in range(sae.latent_dim):
            self.neuron_importance[i] = NeuronImportance(neuron_id=i)
    
    def optimize_delta(
        self,
        audio_array: np.ndarray,
        transcript: str,
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """
        通过优化 delta 找到关键神经元（参考 TaCT 的 error_statistic.py）
        
        核心思想：
        1. 获取当前 SAE 激活 sparse_temp
        2. 初始化 delta = 0
        3. 迭代优化: delta = delta - lr * grad(Loss + λ|delta|)
        4. 返回优化后的 delta，|delta| 大的神经元就是关键神经元
        
        Returns:
            delta: 优化后的 delta [latent_dim]
            pred_loss: 原始预测 loss
            recover_loss: 优化后的 loss
        """
        # 准备输入
        input_features = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self.device)
        
        # 准备标签
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        encoder = self.whisper_model.model.encoder
        
        # 获取 encoder 隐状态
        with torch.no_grad():
            # Embedding
            inputs_embeds = encoder.conv1(input_features)
            inputs_embeds = F.gelu(inputs_embeds)
            inputs_embeds = encoder.conv2(inputs_embeds)
            inputs_embeds = F.gelu(inputs_embeds)
            inputs_embeds = inputs_embeds.permute(0, 2, 1)
            
            embed_pos = encoder.embed_positions.weight[:inputs_embeds.shape[1]]
            hidden_states = inputs_embeds + embed_pos
            hidden_states = encoder.dropout(hidden_states)
            
            # 通过前面的层
            for layer in encoder.layers[:self.encoder_layer]:
                hidden_states = layer(hidden_states)[0]
            
            # 获取 SAE 激活
            sparse = self.sae.encode(hidden_states)
            sparse_topk, _, mask = self.sae._get_topk(sparse, self.topk)
            sparse_temp = sparse_topk.detach()
            
            # 计算原始预测 loss
            recovered = self.sae.decode(sparse_temp)
            h = recovered
            for layer in encoder.layers[self.encoder_layer:]:
                h = layer(h)[0]
            h = encoder.layer_norm(h)
            
            decoder_input_ids = labels[:, :-1]
            decoder_labels = labels[:, 1:]
            decoder_outputs = self.whisper_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=h,
            )
            logits = self.whisper_model.proj_out(decoder_outputs.last_hidden_state)
            pred_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                decoder_labels.view(-1),
                ignore_index=-100,
            ).item()
        
        # 如果原始 loss 已经很低，跳过优化
        if pred_loss < 0.5:
            return None, pred_loss, pred_loss
        
        # 初始化 delta（对整个序列取平均后的激活）
        # 取序列维度的平均，得到 [batch, latent_dim]
        sparse_mean = sparse_temp.mean(dim=1)  # [1, latent_dim]
        delta = torch.zeros_like(sparse_mean)
        delta.requires_grad_(True)
        
        # 获取 mask（哪些神经元被激活）
        mask_mean = (sparse_temp.abs().mean(dim=1) > 0).float()  # [1, latent_dim]
        
        # 迭代优化 delta
        for step in range(self.optimization_steps):
            # 应用 delta 到激活
            sparse_modified = sparse_temp.clone()
            # 将 delta 广播到所有序列位置
            delta_expanded = delta.unsqueeze(1).expand_as(sparse_modified)
            sparse_modified = sparse_modified + delta_expanded
            
            # 解码
            recovered = self.sae.decode(sparse_modified)
            
            # 通过剩余层
            h = recovered
            for layer in encoder.layers[self.encoder_layer:]:
                h = layer(h)[0]
            h = encoder.layer_norm(h)
            
            # Decoder
            decoder_outputs = self.whisper_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=h,
            )
            logits = self.whisper_model.proj_out(decoder_outputs.last_hidden_state)
            
            # 计算 loss + 正则化
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                decoder_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = loss + self.regularization * torch.abs(delta).mean()
            
            # 反向传播
            total_loss.backward()
            
            # 更新 delta
            with torch.no_grad():
                if delta.grad is not None:
                    delta = delta - self.learning_rate * delta.grad
                    delta = delta * mask_mean  # 只更新被激活的神经元
                delta = delta.detach()
                delta.requires_grad_(True)
        
        # 计算优化后的 loss
        with torch.no_grad():
            sparse_modified = sparse_temp.clone()
            delta_expanded = delta.unsqueeze(1).expand_as(sparse_modified)
            sparse_modified = sparse_modified + delta_expanded
            
            recovered = self.sae.decode(sparse_modified)
            h = recovered
            for layer in encoder.layers[self.encoder_layer:]:
                h = layer(h)[0]
            h = encoder.layer_norm(h)
            
            decoder_outputs = self.whisper_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=h,
            )
            logits = self.whisper_model.proj_out(decoder_outputs.last_hidden_state)
            recover_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                decoder_labels.view(-1),
                ignore_index=-100,
            ).item()
        
        return delta.detach().squeeze(0), pred_loss, recover_loss
    
    def analyze_samples(
        self,
        samples: List[Dict],
    ):
        """
        分析多个样本，通过 delta 优化找到关键神经元
        
        Args:
            samples: 样本列表，每个包含 'audio' 和 'text'
        """
        print(f"Analyzing {len(samples)} samples with delta optimization...")
        
        successful_samples = 0
        total_improvement = 0.0
        
        for sample in tqdm(samples, desc="Delta optimization"):
            audio_array = sample['audio']['array']
            transcript = sample.get('text', '')
            
            if not transcript:
                continue
            
            try:
                delta, pred_loss, recover_loss = self.optimize_delta(audio_array, transcript)
                
                if delta is not None:
                    # 记录每个神经元的 |delta| 值
                    delta_abs = delta.abs().cpu().numpy()
                    for neuron_id in range(len(delta_abs)):
                        if delta_abs[neuron_id] > 0:
                            self.delta_accumulator[neuron_id].append(delta_abs[neuron_id])
                    
                    successful_samples += 1
                    total_improvement += (pred_loss - recover_loss)
                    
            except Exception as e:
                print(f"Error in delta optimization: {e}")
                continue
        
        # 汇总 delta 重要性
        print("Aggregating delta importance...")
        for neuron_id, deltas in self.delta_accumulator.items():
            if deltas:
                self.neuron_importance[neuron_id].delta_importance = float(np.mean(deltas))
                self.neuron_importance[neuron_id].mean_delta = float(np.mean(deltas))
                self.neuron_importance[neuron_id].sample_count = len(deltas)
        
        print(f"Successfully analyzed {successful_samples} samples")
        if successful_samples > 0:
            print(f"Average loss improvement: {total_improvement / successful_samples:.4f}")
    
    def get_key_neurons(
        self,
        top_n: int = 64,
    ) -> List[int]:
        """
        获取关键神经元（按 |delta| 排序）
        
        关键神经元是那些"改变其激活值能最大程度改善输出"的神经元
        
        Args:
            top_n: 返回的神经元数量
            
        Returns:
            关键神经元 ID 列表
        """
        sorted_neurons = sorted(
            self.neuron_importance.items(),
            key=lambda x: x[1].delta_importance,
            reverse=True
        )
        
        return [n[0] for n in sorted_neurons[:top_n]]
    
    def export_results(self, output_path: str):
        """导出分析结果"""
        key_neurons = self.get_key_neurons(100)
        
        results = {
            'neuron_importance': {
                str(k): v.to_dict() 
                for k, v in self.neuron_importance.items()
                if v.delta_importance > 0
            },
            'key_neurons': key_neurons,  # 按 |delta| 排序的关键神经元
            'summary': {
                'total_neurons': len(self.neuron_importance),
                'neurons_with_delta': sum(1 for v in self.neuron_importance.values() if v.delta_importance > 0),
                'top_10_neurons': [
                    {
                        'neuron_id': nid,
                        'delta_importance': self.neuron_importance[nid].delta_importance,
                        'sample_count': self.neuron_importance[nid].sample_count,
                    }
                    for nid in key_neurons[:10]
                ],
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results exported to {output_path}")


def run_counterfactual_analysis(
    sae_checkpoint: str,
    model_name: str = "openai/whisper-medium",
    dataset_path: str = "dataset/shanghai/shanghai_dataset",
    output_dir: str = "./exp/counterfactual",
    encoder_layer: int = 12,
    topk: int = 64,
    max_samples: int = 200,
    optimization_steps: int = 50,
    learning_rate: float = 15.0,
    regularization: float = 1e-2,
    compute_ablation: bool = False,  # 保留参数兼容性，但不再使用
    device: str = "cuda",
):
    """
    运行反事实分析（参考 TaCT）
    
    通过优化 delta 找到关键神经元：
    - 对于每个高错误率样本，优化 delta 来减少预测误差
    - 选择 |delta| 最大的神经元作为关键神经元
    - 这些神经元应该用于门控 LoRA 的触发条件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Counterfactual Analysis for ASR Neurons (TaCT-style)")
    print("=" * 70)
    print(f"Optimization steps: {optimization_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Regularization: {regularization}")
    
    # 加载 SAE
    print(f"Loading SAE: {sae_checkpoint}")
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
    
    # 加载 Whisper
    print(f"Loading Whisper: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    # 加载数据集
    print(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    samples = list(dataset['train'].select(range(min(max_samples, len(dataset['train'])))))
    
    # 创建分析器
    analyzer = CounterfactualAnalyzer(
        whisper_model=whisper_model,
        processor=processor,
        sae=sae,
        encoder_layer=encoder_layer,
        topk=topk,
        device=device,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        regularization=regularization,
    )
    
    # 运行分析
    analyzer.analyze_samples(samples=samples)
    
    # 导出结果
    output_path = os.path.join(output_dir, 'counterfactual_results.json')
    analyzer.export_results(output_path)
    
    # 打印 top 神经元
    print("\n" + "=" * 70)
    print("Top 20 Key Neurons (by |delta| importance):")
    print("=" * 70)
    print("These neurons have the largest impact on improving predictions")
    print("when their activations are modified.\n")
    
    top_neurons = analyzer.get_key_neurons(20)
    for i, nid in enumerate(top_neurons):
        imp = analyzer.neuron_importance[nid]
        print(f"  #{i+1}: Neuron {nid} - delta={imp.delta_importance:.6f}, samples={imp.sample_count}")
    
    return analyzer


def main():
    parser = argparse.ArgumentParser(description="Counterfactual analysis for ASR neurons")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium")
    parser.add_argument("--dataset_path", type=str, default="dataset/shanghai/shanghai_dataset")
    parser.add_argument("--output_dir", type=str, default="./exp/counterfactual")
    parser.add_argument("--encoder_layer", type=int, default=12)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--compute_ablation", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    run_counterfactual_analysis(
        sae_checkpoint=args.sae_checkpoint,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        encoder_layer=args.encoder_layer,
        topk=args.topk,
        max_samples=args.max_samples,
        compute_ablation=args.compute_ablation,
        device=args.device,
    )


if __name__ == "__main__":
    main()
