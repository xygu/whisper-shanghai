"""
Neuron Analyzer

分析 SAE 神经元的激活模式，提取样本特征，生成供 LLM 标注的 JSON 数据

输出格式:
1. sample_features.json: 每条样本的完整三层特征 + 神经元激活值
2. neuron_top_samples.json: 每个神经元 top 10% 高激活样本的特征汇总
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
class SampleActivationRecord:
    """单个样本的激活记录"""
    sample_id: str
    
    # 三层特征
    acoustic_features: Dict[str, Any] = field(default_factory=dict)
    linguistic_features: Dict[str, Any] = field(default_factory=dict)
    error_pattern_features: Dict[str, Any] = field(default_factory=dict)
    
    # 神经元激活值 (只保存非零激活)
    neuron_activations: Dict[int, float] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'sample_id': self.sample_id,
            'acoustic_features': self.acoustic_features,
            'linguistic_features': self.linguistic_features,
            'error_pattern_features': self.error_pattern_features,
            'neuron_activations': {str(k): v for k, v in self.neuron_activations.items()},
            'metadata': self.metadata,
        }


@dataclass
class NeuronTopSamples:
    """神经元 top 激活样本汇总"""
    neuron_id: int
    
    # 激活统计
    total_samples: int = 0
    activation_count: int = 0
    activation_rate: float = 0.0
    mean_activation: float = 0.0
    max_activation: float = 0.0
    percentile_90_activation: float = 0.0  # 90 分位数激活值
    
    # top 10% 样本的特征汇总
    top_samples: List[Dict] = field(default_factory=list)
    
    # 特征分布统计 (top 10% 样本中各特征的分布)
    acoustic_distribution: Dict[str, Any] = field(default_factory=dict)
    linguistic_distribution: Dict[str, Any] = field(default_factory=dict)
    error_distribution: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        def convert_to_dict(obj):
            """递归将 defaultdict 转换为普通 dict"""
            if isinstance(obj, defaultdict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        
        return {
            'neuron_id': self.neuron_id,
            'total_samples': self.total_samples,
            'activation_count': self.activation_count,
            'activation_rate': self.activation_rate,
            'mean_activation': self.mean_activation,
            'max_activation': self.max_activation,
            'percentile_90_activation': self.percentile_90_activation,
            'top_samples': convert_to_dict(self.top_samples),
            'acoustic_distribution': convert_to_dict(self.acoustic_distribution),
            'linguistic_distribution': convert_to_dict(self.linguistic_distribution),
            'error_distribution': convert_to_dict(self.error_distribution),
        }


class NeuronAnalyzer:
    """
    神经元分析器
    
    分析 SAE 神经元的激活模式，提取样本特征，
    输出供 LLM 手动标注的 JSON 数据
    
    输出:
    1. sample_features.json: 每条样本的完整三层特征 + 神经元激活值
    2. neuron_top_samples.json: 每个神经元 top 10% 高激活样本的特征汇总
    """
    
    def __init__(
        self,
        latent_dim: int = 8192,
        top_percent: float = 0.1,  # top 10%
    ):
        self.latent_dim = latent_dim
        self.top_percent = top_percent
        
        # 存储每个样本的完整记录
        self.sample_records: Dict[str, SampleActivationRecord] = {}
        
        # 存储每个神经元的激活记录 (sample_id, activation_value)
        self.neuron_activations: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        
        # 神经元 top 样本汇总
        self.neuron_top_samples: Dict[int, NeuronTopSamples] = {}
    
    def record_sample(
        self,
        sample_id: str,
        activations: torch.Tensor,
        acoustic_features: Dict,
        linguistic_features: Dict,
        error_pattern_features: Dict,
        metadata: Optional[Dict] = None,
        activation_threshold: float = 0.0,
    ):
        """
        记录一个样本的完整信息
        
        Args:
            sample_id: 样本 ID
            activations: 神经元激活值 [seq_len, latent_dim] 或 [latent_dim]
            acoustic_features: 声学特征字典
            linguistic_features: 语言特征字典
            error_pattern_features: 错误模式特征字典
            metadata: 额外元数据
            activation_threshold: 激活阈值 (低于此值不记录)
        """
        # 处理激活值
        if activations.dim() == 2:
            # 取序列最大值 (更能体现神经元是否被激活)
            activations = activations.max(dim=0)[0]
        
        activations = activations.detach().cpu().numpy()
        
        # 创建样本记录
        record = SampleActivationRecord(
            sample_id=sample_id,
            acoustic_features=acoustic_features,
            linguistic_features=linguistic_features,
            error_pattern_features=error_pattern_features,
            metadata=metadata or {},
        )
        
        # 记录非零激活的神经元
        for neuron_id in range(min(len(activations), self.latent_dim)):
            activation_value = float(activations[neuron_id])
            if activation_value > activation_threshold:
                record.neuron_activations[neuron_id] = activation_value
                self.neuron_activations[neuron_id].append((sample_id, activation_value))
        
        self.sample_records[sample_id] = record
    
    def record_activation(
        self,
        sample_id: str,
        activations: torch.Tensor,
        features: Dict,
        threshold: float = 0.0,
    ):
        """
        兼容旧接口：记录一个样本的神经元激活
        """
        acoustic = features.get('acoustic', {})
        linguistic = features.get('linguistic', {})
        error_pattern = features.get('error_pattern', {})
        metadata = features.get('metadata', {})
        
        self.record_sample(
            sample_id=sample_id,
            activations=activations,
            acoustic_features=acoustic,
            linguistic_features=linguistic,
            error_pattern_features=error_pattern,
            metadata=metadata,
            activation_threshold=threshold,
        )
    
    def compute_neuron_top_samples(self):
        """
        计算每个神经元的 top 10% 高激活样本汇总
        
        输出每个神经元的:
        - 激活统计 (总样本数、激活数、激活率、均值、最大值、90分位数)
        - top 10% 样本的完整特征列表
        - top 10% 样本的特征分布统计
        """
        total_samples = len(self.sample_records)
        
        for neuron_id in tqdm(range(self.latent_dim), desc="Computing neuron top samples"):
            activations = self.neuron_activations.get(neuron_id, [])
            
            top_samples_data = NeuronTopSamples(neuron_id=neuron_id)
            top_samples_data.total_samples = total_samples
            top_samples_data.activation_count = len(activations)
            top_samples_data.activation_rate = len(activations) / max(total_samples, 1)
            
            if not activations:
                self.neuron_top_samples[neuron_id] = top_samples_data
                continue
            
            # 计算激活值统计
            values = [v for _, v in activations]
            top_samples_data.mean_activation = float(np.mean(values))
            top_samples_data.max_activation = float(np.max(values))
            top_samples_data.percentile_90_activation = float(np.percentile(values, 90))
            
            # 获取 top 10% 样本
            sorted_activations = sorted(activations, key=lambda x: x[1], reverse=True)
            top_count = max(1, int(len(sorted_activations) * self.top_percent))
            top_sample_ids = [sid for sid, _ in sorted_activations[:top_count]]
            
            # 收集 top 样本的完整特征
            top_samples_list = []
            acoustic_stats = defaultdict(lambda: defaultdict(int))
            linguistic_stats = defaultdict(lambda: defaultdict(int))
            error_stats = defaultdict(lambda: defaultdict(int))
            
            for sample_id in top_sample_ids:
                record = self.sample_records.get(sample_id)
                if not record:
                    continue
                
                # 获取该样本在此神经元的激活值
                activation_value = record.neuron_activations.get(neuron_id, 0)
                
                # 添加到 top 样本列表
                sample_info = {
                    'sample_id': sample_id,
                    'activation_value': activation_value,
                    'acoustic': record.acoustic_features,
                    'linguistic': record.linguistic_features,
                    'error_pattern': record.error_pattern_features,
                }
                top_samples_list.append(sample_info)
                
                # 统计声学特征分布
                for key, value in record.acoustic_features.items():
                    if isinstance(value, (str, bool)):
                        acoustic_stats[key][str(value)] += 1
                    elif isinstance(value, (int, float)):
                        acoustic_stats[key]['values'] = acoustic_stats[key].get('values', [])
                        acoustic_stats[key]['values'].append(value)
                
                # 统计语言特征分布
                for key, value in record.linguistic_features.items():
                    if isinstance(value, (str, bool)):
                        linguistic_stats[key][str(value)] += 1
                    elif isinstance(value, list):
                        linguistic_stats[key]['count'] = linguistic_stats[key].get('count', 0) + len(value)
                        for item in value:
                            linguistic_stats[key][str(item)] = linguistic_stats[key].get(str(item), 0) + 1
                    elif isinstance(value, (int, float)):
                        linguistic_stats[key]['values'] = linguistic_stats[key].get('values', [])
                        linguistic_stats[key]['values'].append(value)
                
                # 统计错误模式分布
                for key, value in record.error_pattern_features.items():
                    if isinstance(value, (str, bool)):
                        error_stats[key][str(value)] += 1
                    elif isinstance(value, (int, float)):
                        error_stats[key]['values'] = error_stats[key].get('values', [])
                        error_stats[key]['values'].append(value)
            
            # 处理数值型特征的统计
            for key in acoustic_stats:
                if 'values' in acoustic_stats[key]:
                    vals = acoustic_stats[key]['values']
                    acoustic_stats[key] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                    }
            
            for key in linguistic_stats:
                if 'values' in linguistic_stats[key]:
                    vals = linguistic_stats[key]['values']
                    linguistic_stats[key] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                    }
            
            for key in error_stats:
                if 'values' in error_stats[key]:
                    vals = error_stats[key]['values']
                    error_stats[key] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                    }
            
            top_samples_data.top_samples = top_samples_list
            top_samples_data.acoustic_distribution = dict(acoustic_stats)
            top_samples_data.linguistic_distribution = dict(linguistic_stats)
            top_samples_data.error_distribution = dict(error_stats)
            
            self.neuron_top_samples[neuron_id] = top_samples_data
    
    def compute_statistics(self):
        """兼容旧接口"""
        self.compute_neuron_top_samples()
    
    def export_data(
        self, 
        output_dir: str,
        top_n_neurons: int = 100,
    ):
        """
        导出分析数据，供 LLM 手动标注
        
        输出文件:
        1. sample_features.json: 每条样本的完整三层特征 + 神经元激活值
        2. neuron_top_samples.json: 每个神经元 top 10% 高激活样本的特征汇总
        3. summary.json: 摘要信息
        
        Args:
            output_dir: 输出目录
            top_n_neurons: 导出激活率最高的 N 个神经元
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 导出每条样本的完整特征 + 神经元激活值
        sample_features_data = {
            sample_id: record.to_dict()
            for sample_id, record in self.sample_records.items()
        }
        sample_features_path = os.path.join(output_dir, 'sample_features.json')
        with open(sample_features_path, 'w', encoding='utf-8') as f:
            json.dump(sample_features_data, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(sample_features_data)} samples to {sample_features_path}")
        
        # 2. 导出每个神经元的 top 10% 样本汇总
        # 按激活率排序，取 top N
        sorted_neurons = sorted(
            self.neuron_top_samples.items(),
            key=lambda x: x[1].activation_rate,
            reverse=True
        )[:top_n_neurons]
        
        neuron_top_samples_data = {
            str(nid): data.to_dict()
            for nid, data in sorted_neurons
        }
        neuron_top_samples_path = os.path.join(output_dir, 'neuron_top_samples.json')
        with open(neuron_top_samples_path, 'w', encoding='utf-8') as f:
            json.dump(neuron_top_samples_data, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(neuron_top_samples_data)} neurons to {neuron_top_samples_path}")
        
        # 3. 导出摘要
        summary = {
            'total_samples': len(self.sample_records),
            'total_neurons': self.latent_dim,
            'active_neurons': sum(1 for data in self.neuron_top_samples.values() if data.activation_count > 0),
            'top_percent': self.top_percent,
            'top_neurons_by_activation_rate': [
                {
                    'neuron_id': nid,
                    'activation_rate': data.activation_rate,
                    'activation_count': data.activation_count,
                    'mean_activation': data.mean_activation,
                    'max_activation': data.max_activation,
                    'top_sample_count': len(data.top_samples),
                }
                for nid, data in sorted_neurons[:20]
            ],
        }
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Export Summary ===")
        print(f"Output directory: {output_dir}")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Total neurons: {summary['total_neurons']}")
        print(f"Active neurons: {summary['active_neurons']}")
        print(f"\nFiles:")
        print(f"  1. sample_features.json - 每条样本的完整三层特征 + 神经元激活值")
        print(f"  2. neuron_top_samples.json - 每个神经元 top {self.top_percent*100:.0f}% 高激活样本的特征汇总")
        print(f"  3. summary.json - 摘要信息")
    
    def export_neuron_data(self, output_dir: str, top_n: int = 100):
        """兼容旧接口"""
        self.export_data(output_dir, top_n_neurons=top_n)
    
    def compute_correlations(self):
        """兼容旧接口 - 不再需要单独计算关联，已在 compute_neuron_top_samples 中完成"""
        pass
    
    def save(self, path: str):
        """保存分析器状态"""
        data = {
            'latent_dim': self.latent_dim,
            'top_percent': self.top_percent,
            'sample_records': {
                k: v.to_dict() for k, v in self.sample_records.items()
            },
            'neuron_activations': {
                str(k): list(v) for k, v in self.neuron_activations.items()
            },
            'neuron_top_samples': {
                str(k): v.to_dict() for k, v in self.neuron_top_samples.items()
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Saved analyzer state to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NeuronAnalyzer':
        """加载分析器状态"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analyzer = cls(
            latent_dim=data['latent_dim'],
            top_percent=data.get('top_percent', 0.1),
        )
        
        # 加载样本记录
        for sample_id, record_dict in data.get('sample_records', {}).items():
            record = SampleActivationRecord(sample_id=sample_id)
            record.acoustic_features = record_dict.get('acoustic_features', {})
            record.linguistic_features = record_dict.get('linguistic_features', {})
            record.error_pattern_features = record_dict.get('error_pattern_features', {})
            record.neuron_activations = {
                int(k): v for k, v in record_dict.get('neuron_activations', {}).items()
            }
            record.metadata = record_dict.get('metadata', {})
            analyzer.sample_records[sample_id] = record
        
        # 加载神经元激活
        analyzer.neuron_activations = {
            int(k): [(sid, val) for sid, val in v]
            for k, v in data.get('neuron_activations', {}).items()
        }
        
        # 加载神经元 top 样本
        for k, v in data.get('neuron_top_samples', {}).items():
            top_data = NeuronTopSamples(neuron_id=int(k))
            for key, val in v.items():
                if hasattr(top_data, key):
                    setattr(top_data, key, val)
            analyzer.neuron_top_samples[int(k)] = top_data
        
        return analyzer
