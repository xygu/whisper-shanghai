"""
Neuron Analyzer

分析 SAE 神经元的激活模式，关联特征，生成 LLM 标注 prompt
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from tqdm import tqdm


@dataclass
class NeuronActivationStats:
    """神经元激活统计"""
    neuron_id: int
    
    # 激活统计
    activation_count: int = 0  # 激活次数
    total_samples: int = 0  # 总样本数
    activation_rate: float = 0.0  # 激活率
    
    # 激活值统计
    mean_activation: float = 0.0
    max_activation: float = 0.0
    std_activation: float = 0.0
    
    # 关联的样本 ID
    top_activated_samples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NeuronFeatureCorrelation:
    """神经元与特征的关联"""
    neuron_id: int
    
    # 声学特征关联
    snr_correlation: Dict[str, float] = field(default_factory=dict)  # low/medium/high -> 激活率
    speech_rate_correlation: Dict[str, float] = field(default_factory=dict)
    pitch_contour_correlation: Dict[str, float] = field(default_factory=dict)
    
    # 语言特征关联
    domain_correlation: Dict[str, float] = field(default_factory=dict)
    sentence_type_correlation: Dict[str, float] = field(default_factory=dict)
    contains_numbers_correlation: float = 0.0
    contains_dialect_correlation: float = 0.0
    
    # 错误模式关联
    error_type_correlation: Dict[str, float] = field(default_factory=dict)
    high_cer_correlation: float = 0.0  # CER > 0.1 时的激活率
    
    # 推断的语义概念
    inferred_concept: str = ""
    concept_confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class NeuronAnalyzer:
    """
    神经元分析器
    
    分析 SAE 神经元的激活模式，关联样本特征，
    生成用于 LLM 语义标注的 prompt
    """
    
    def __init__(
        self,
        latent_dim: int = 8192,
        top_k_samples: int = 20,
    ):
        self.latent_dim = latent_dim
        self.top_k_samples = top_k_samples
        
        # 存储每个神经元的激活记录
        self.neuron_activations: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        
        # 存储样本特征
        self.sample_features: Dict[str, Dict] = {}
        
        # 神经元统计
        self.neuron_stats: Dict[int, NeuronActivationStats] = {}
        self.neuron_correlations: Dict[int, NeuronFeatureCorrelation] = {}
    
    def record_activation(
        self,
        sample_id: str,
        activations: torch.Tensor,
        features: Dict,
        threshold: float = 0.0,
    ):
        """
        记录一个样本的神经元激活
        
        Args:
            sample_id: 样本 ID
            activations: 神经元激活值 [seq_len, latent_dim] 或 [latent_dim]
            features: 样本特征字典
            threshold: 激活阈值
        """
        # 存储样本特征
        self.sample_features[sample_id] = features
        
        # 处理激活值
        if activations.dim() == 2:
            # 取序列平均或最大
            activations = activations.mean(dim=0)
        
        activations = activations.detach().cpu().numpy()
        
        # 记录每个神经元的激活
        for neuron_id in range(min(len(activations), self.latent_dim)):
            activation_value = float(activations[neuron_id])
            if activation_value > threshold:
                self.neuron_activations[neuron_id].append((sample_id, activation_value))
    
    def compute_statistics(self):
        """计算所有神经元的统计信息"""
        total_samples = len(self.sample_features)
        
        for neuron_id in tqdm(range(self.latent_dim), desc="Computing neuron stats"):
            activations = self.neuron_activations.get(neuron_id, [])
            
            stats = NeuronActivationStats(neuron_id=neuron_id)
            stats.total_samples = total_samples
            stats.activation_count = len(activations)
            stats.activation_rate = len(activations) / max(total_samples, 1)
            
            if activations:
                values = [v for _, v in activations]
                stats.mean_activation = float(np.mean(values))
                stats.max_activation = float(np.max(values))
                stats.std_activation = float(np.std(values))
                
                # 获取 top-k 激活样本
                sorted_activations = sorted(activations, key=lambda x: x[1], reverse=True)
                stats.top_activated_samples = [
                    sid for sid, _ in sorted_activations[:self.top_k_samples]
                ]
            
            self.neuron_stats[neuron_id] = stats
    
    def compute_correlations(self):
        """计算神经元与特征的关联"""
        for neuron_id in tqdm(range(self.latent_dim), desc="Computing correlations"):
            correlation = NeuronFeatureCorrelation(neuron_id=neuron_id)
            
            activated_samples = set(
                sid for sid, _ in self.neuron_activations.get(neuron_id, [])
            )
            
            if not activated_samples:
                self.neuron_correlations[neuron_id] = correlation
                continue
            
            # 统计各特征的激活率
            feature_counts = defaultdict(lambda: defaultdict(int))
            feature_totals = defaultdict(lambda: defaultdict(int))
            
            for sample_id, features in self.sample_features.items():
                is_activated = sample_id in activated_samples
                
                # 声学特征
                acoustic = features.get('acoustic', {})
                snr_level = acoustic.get('snr_level', 'unknown')
                feature_totals['snr'][snr_level] += 1
                if is_activated:
                    feature_counts['snr'][snr_level] += 1
                
                speech_rate_level = acoustic.get('speech_rate_level', 'unknown')
                feature_totals['speech_rate'][speech_rate_level] += 1
                if is_activated:
                    feature_counts['speech_rate'][speech_rate_level] += 1
                
                pitch_contour = acoustic.get('pitch_contour', 'unknown')
                feature_totals['pitch_contour'][pitch_contour] += 1
                if is_activated:
                    feature_counts['pitch_contour'][pitch_contour] += 1
                
                # 语言特征
                linguistic = features.get('linguistic', {})
                domain = linguistic.get('domain', 'general')
                feature_totals['domain'][domain] += 1
                if is_activated:
                    feature_counts['domain'][domain] += 1
                
                sentence_type = linguistic.get('sentence_type', 'declarative')
                feature_totals['sentence_type'][sentence_type] += 1
                if is_activated:
                    feature_counts['sentence_type'][sentence_type] += 1
                
                # 数字
                has_numbers = linguistic.get('contains_numbers', False)
                feature_totals['numbers']['yes' if has_numbers else 'no'] += 1
                if is_activated:
                    feature_counts['numbers']['yes' if has_numbers else 'no'] += 1
                
                # 方言
                has_dialect = len(linguistic.get('dialect_markers', [])) > 0
                feature_totals['dialect']['yes' if has_dialect else 'no'] += 1
                if is_activated:
                    feature_counts['dialect']['yes' if has_dialect else 'no'] += 1
                
                # 错误模式
                error = features.get('error_pattern', {})
                error_type = error.get('error_type', 'none')
                feature_totals['error_type'][error_type] += 1
                if is_activated:
                    feature_counts['error_type'][error_type] += 1
                
                # 高 CER
                cer = error.get('cer', 0)
                high_cer = cer > 0.1
                feature_totals['high_cer']['yes' if high_cer else 'no'] += 1
                if is_activated:
                    feature_counts['high_cer']['yes' if high_cer else 'no'] += 1
            
            # 计算关联率
            for level in feature_totals['snr']:
                total = feature_totals['snr'][level]
                if total > 0:
                    correlation.snr_correlation[level] = feature_counts['snr'][level] / total
            
            for level in feature_totals['speech_rate']:
                total = feature_totals['speech_rate'][level]
                if total > 0:
                    correlation.speech_rate_correlation[level] = feature_counts['speech_rate'][level] / total
            
            for level in feature_totals['pitch_contour']:
                total = feature_totals['pitch_contour'][level]
                if total > 0:
                    correlation.pitch_contour_correlation[level] = feature_counts['pitch_contour'][level] / total
            
            for domain in feature_totals['domain']:
                total = feature_totals['domain'][domain]
                if total > 0:
                    correlation.domain_correlation[domain] = feature_counts['domain'][domain] / total
            
            for stype in feature_totals['sentence_type']:
                total = feature_totals['sentence_type'][stype]
                if total > 0:
                    correlation.sentence_type_correlation[stype] = feature_counts['sentence_type'][stype] / total
            
            # 数字关联
            if feature_totals['numbers']['yes'] > 0:
                correlation.contains_numbers_correlation = (
                    feature_counts['numbers']['yes'] / feature_totals['numbers']['yes']
                )
            
            # 方言关联
            if feature_totals['dialect']['yes'] > 0:
                correlation.contains_dialect_correlation = (
                    feature_counts['dialect']['yes'] / feature_totals['dialect']['yes']
                )
            
            # 错误类型关联
            for etype in feature_totals['error_type']:
                total = feature_totals['error_type'][etype]
                if total > 0:
                    correlation.error_type_correlation[etype] = feature_counts['error_type'][etype] / total
            
            # 高 CER 关联
            if feature_totals['high_cer']['yes'] > 0:
                correlation.high_cer_correlation = (
                    feature_counts['high_cer']['yes'] / feature_totals['high_cer']['yes']
                )
            
            # 推断语义概念
            correlation.inferred_concept, correlation.concept_confidence = self._infer_concept(correlation)
            
            self.neuron_correlations[neuron_id] = correlation
    
    def _infer_concept(
        self, 
        correlation: NeuronFeatureCorrelation
    ) -> Tuple[str, float]:
        """基于关联推断神经元的语义概念"""
        candidates = []
        
        # 检查 SNR 关联
        if correlation.snr_correlation.get('low', 0) > 0.5:
            candidates.append(("低信噪比/噪声敏感", correlation.snr_correlation['low']))
        if correlation.snr_correlation.get('high', 0) > 0.5:
            candidates.append(("高质量音频", correlation.snr_correlation['high']))
        
        # 检查语速关联
        if correlation.speech_rate_correlation.get('fast', 0) > 0.5:
            candidates.append(("快速语音", correlation.speech_rate_correlation['fast']))
        if correlation.speech_rate_correlation.get('slow', 0) > 0.5:
            candidates.append(("慢速语音", correlation.speech_rate_correlation['slow']))
        
        # 检查音高关联
        if correlation.pitch_contour_correlation.get('rising', 0) > 0.5:
            candidates.append(("升调/疑问", correlation.pitch_contour_correlation['rising']))
        if correlation.pitch_contour_correlation.get('falling', 0) > 0.5:
            candidates.append(("降调/陈述", correlation.pitch_contour_correlation['falling']))
        
        # 检查领域关联
        for domain, rate in correlation.domain_correlation.items():
            if rate > 0.5 and domain != 'general':
                candidates.append((f"{domain}领域", rate))
        
        # 检查数字关联
        if correlation.contains_numbers_correlation > 0.5:
            candidates.append(("数字序列", correlation.contains_numbers_correlation))
        
        # 检查方言关联
        if correlation.contains_dialect_correlation > 0.5:
            candidates.append(("方言特征", correlation.contains_dialect_correlation))
        
        # 检查错误类型关联
        for etype, rate in correlation.error_type_correlation.items():
            if rate > 0.5 and etype != 'none':
                candidates.append((f"{etype}错误", rate))
        
        # 检查高 CER 关联
        if correlation.high_cer_correlation > 0.5:
            candidates.append(("高错误率样本", correlation.high_cer_correlation))
        
        if candidates:
            # 返回置信度最高的概念
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[1]
        
        return "未知", 0.0
    
    def generate_llm_prompt(
        self, 
        neuron_id: int,
        include_samples: bool = True,
    ) -> str:
        """
        生成用于 LLM 语义标注的 prompt
        
        Args:
            neuron_id: 神经元 ID
            include_samples: 是否包含样本详情
            
        Returns:
            LLM prompt 字符串
        """
        stats = self.neuron_stats.get(neuron_id)
        correlation = self.neuron_correlations.get(neuron_id)
        
        if not stats or not correlation:
            return f"神经元 #{neuron_id} 没有足够的统计信息。"
        
        prompt = f"""# ASR 模型神经元语义分析

## 任务说明
你是一个语音识别模型的可解释性分析专家。我将提供一个神经元的激活统计信息和关联特征，请分析这个神经元可能代表的语义概念。

## 神经元 #{neuron_id} 统计信息

### 基本统计
- 激活次数: {stats.activation_count} / {stats.total_samples} 样本
- 激活率: {stats.activation_rate:.2%}
- 平均激活值: {stats.mean_activation:.4f}
- 最大激活值: {stats.max_activation:.4f}
- 激活值标准差: {stats.std_activation:.4f}

### 声学特征关联
- 信噪比关联: {json.dumps(correlation.snr_correlation, ensure_ascii=False)}
- 语速关联: {json.dumps(correlation.speech_rate_correlation, ensure_ascii=False)}
- 音高轮廓关联: {json.dumps(correlation.pitch_contour_correlation, ensure_ascii=False)}

### 语言特征关联
- 领域关联: {json.dumps(correlation.domain_correlation, ensure_ascii=False)}
- 句型关联: {json.dumps(correlation.sentence_type_correlation, ensure_ascii=False)}
- 数字内容关联: {correlation.contains_numbers_correlation:.2%}
- 方言特征关联: {correlation.contains_dialect_correlation:.2%}

### 错误模式关联
- 错误类型关联: {json.dumps(correlation.error_type_correlation, ensure_ascii=False)}
- 高错误率样本关联: {correlation.high_cer_correlation:.2%}
"""
        
        if include_samples and stats.top_activated_samples:
            prompt += "\n### 高激活样本示例\n"
            for i, sample_id in enumerate(stats.top_activated_samples[:10]):
                features = self.sample_features.get(sample_id, {})
                linguistic = features.get('linguistic', {})
                error = features.get('error_pattern', {})
                
                transcript = linguistic.get('transcript', '')[:50]
                prediction = error.get('prediction', '')[:50]
                cer = error.get('cer', 0)
                
                prompt += f"""
**样本 {i+1}** (ID: {sample_id})
- 真实文本: {transcript}
- 预测文本: {prediction}
- CER: {cer:.2%}
- 方言标记: {linguistic.get('dialect_markers', [])}
"""
        
        prompt += f"""
## 初步推断
基于统计分析，该神经元可能与 **{correlation.inferred_concept}** 相关（置信度: {correlation.concept_confidence:.2%}）

## 请回答
1. **概念名称**: 用一个简洁的短语描述这个神经元的功能（如"上海话声调检测器"、"数字序列识别"、"背景噪声敏感"）
2. **置信度**: 高/中/低
3. **解释**: 简要说明为什么这些样本共享这个特征，以及这个神经元在 ASR 任务中的作用
4. **建议**: 如果要针对这个神经元进行微调优化，应该关注什么类型的数据？
"""
        
        return prompt
    
    def export_neuron_data(
        self, 
        output_dir: str,
        top_n: int = 100,
    ):
        """
        导出神经元分析数据
        
        Args:
            output_dir: 输出目录
            top_n: 导出激活率最高的 N 个神经元
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 按激活率排序
        sorted_neurons = sorted(
            self.neuron_stats.items(),
            key=lambda x: x[1].activation_rate,
            reverse=True
        )[:top_n]
        
        # 导出统计信息
        stats_data = {
            nid: stats.to_dict() 
            for nid, stats in sorted_neurons
        }
        with open(os.path.join(output_dir, 'neuron_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        # 导出关联信息
        correlation_data = {
            nid: self.neuron_correlations[nid].to_dict()
            for nid, _ in sorted_neurons
            if nid in self.neuron_correlations
        }
        with open(os.path.join(output_dir, 'neuron_correlations.json'), 'w', encoding='utf-8') as f:
            json.dump(correlation_data, f, ensure_ascii=False, indent=2)
        
        # 导出 LLM prompts
        prompts_dir = os.path.join(output_dir, 'llm_prompts')
        os.makedirs(prompts_dir, exist_ok=True)
        
        for nid, _ in sorted_neurons:
            prompt = self.generate_llm_prompt(nid)
            with open(os.path.join(prompts_dir, f'neuron_{nid}.md'), 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        # 导出摘要
        summary = {
            'total_neurons': self.latent_dim,
            'total_samples': len(self.sample_features),
            'active_neurons': sum(1 for s in self.neuron_stats.values() if s.activation_count > 0),
            'top_neurons': [
                {
                    'id': nid,
                    'activation_rate': stats.activation_rate,
                    'inferred_concept': self.neuron_correlations.get(nid, NeuronFeatureCorrelation(nid)).inferred_concept,
                }
                for nid, stats in sorted_neurons[:20]
            ]
        }
        with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Exported neuron analysis to {output_dir}")
        print(f"  - {len(sorted_neurons)} neurons analyzed")
        print(f"  - {len(self.sample_features)} samples processed")
    
    def save(self, path: str):
        """保存分析器状态"""
        data = {
            'latent_dim': self.latent_dim,
            'top_k_samples': self.top_k_samples,
            'neuron_activations': {
                k: list(v) for k, v in self.neuron_activations.items()
            },
            'sample_features': self.sample_features,
            'neuron_stats': {
                k: v.to_dict() for k, v in self.neuron_stats.items()
            },
            'neuron_correlations': {
                k: v.to_dict() for k, v in self.neuron_correlations.items()
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'NeuronAnalyzer':
        """加载分析器状态"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analyzer = cls(
            latent_dim=data['latent_dim'],
            top_k_samples=data['top_k_samples'],
        )
        analyzer.neuron_activations = {
            int(k): [(sid, val) for sid, val in v]
            for k, v in data['neuron_activations'].items()
        }
        analyzer.sample_features = data['sample_features']
        
        for k, v in data['neuron_stats'].items():
            stats = NeuronActivationStats(neuron_id=int(k))
            for key, val in v.items():
                if hasattr(stats, key):
                    setattr(stats, key, val)
            analyzer.neuron_stats[int(k)] = stats
        
        for k, v in data['neuron_correlations'].items():
            corr = NeuronFeatureCorrelation(neuron_id=int(k))
            for key, val in v.items():
                if hasattr(corr, key):
                    setattr(corr, key, val)
            analyzer.neuron_correlations[int(k)] = corr
        
        return analyzer
