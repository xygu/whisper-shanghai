"""
Gated LoRA Module

门控 LoRA 微调模块：只有当特定神经元高度激活时才启用 LoRA 适配
实现可解释、精准的 ASR 模型微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
import math
import json


class GatedLoRALinear(nn.Module):
    """
    门控 LoRA 线性层
    
    当指定的 SAE 神经元激活时，才应用 LoRA 适配
    
    Args:
        base_layer: 原始线性层
        gate_neurons: 门控神经元 ID 列表
        lora_rank: LoRA 秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: LoRA dropout 率
        gate_threshold: 门控激活阈值
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        gate_neurons: List[int],
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        gate_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.gate_neurons = gate_neurons
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.gate_threshold = gate_threshold
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.scaling = lora_alpha / lora_rank
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 冻结基础层
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        x: torch.Tensor, 
        sae_activations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, in_features]
            sae_activations: SAE 激活值 [batch, seq_len, latent_dim] 或 [batch, latent_dim]
            
        Returns:
            输出张量 [batch, seq_len, out_features]
        """
        # 基础层输出
        base_output = self.base_layer(x)
        
        # 计算 LoRA 输出
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        # 计算门控信号
        if sae_activations is not None:
            gate = self._compute_gate(sae_activations, x.device)
            # 广播门控信号
            if gate.dim() == 2:
                gate = gate.unsqueeze(-1)  # [batch, seq_len, 1]
            return base_output + gate * lora_output
        else:
            # 无门控时，直接应用 LoRA
            return base_output + lora_output
    
    def _compute_gate(
        self, 
        sae_activations: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """计算门控信号"""
        # 获取门控神经元的激活值
        gate_indices = torch.tensor(self.gate_neurons, device=device)
        
        if sae_activations.dim() == 3:
            # [batch, seq_len, latent_dim]
            gate_activations = sae_activations.index_select(-1, gate_indices)
            gate_signal = gate_activations.mean(dim=-1)  # [batch, seq_len]
        else:
            # [batch, latent_dim]
            gate_activations = sae_activations.index_select(-1, gate_indices)
            gate_signal = gate_activations.mean(dim=-1)  # [batch]
        
        # Sigmoid 门控
        gate = torch.sigmoid((gate_signal - self.gate_threshold) * 10)
        
        return gate
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取 LoRA 参数"""
        return [self.lora_A, self.lora_B]


class GatedLoRAConfig:
    """门控 LoRA 配置"""
    
    def __init__(
        self,
        target_modules: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        gate_threshold: float = 0.5,
        gate_neurons: Dict[str, List[int]] = None,
    ):
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "out_proj",
            "fc1", "fc2",
        ]
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gate_threshold = gate_threshold
        self.gate_neurons = gate_neurons or {}
    
    def to_dict(self) -> dict:
        return {
            'target_modules': self.target_modules,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'gate_threshold': self.gate_threshold,
            'gate_neurons': self.gate_neurons,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GatedLoRAConfig':
        return cls(**config_dict)
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GatedLoRAConfig':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


class GatedLoRAModel(nn.Module):
    """
    门控 LoRA 模型包装器
    
    将 Whisper 模型的指定层替换为门控 LoRA 层
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        sae: nn.Module,
        config: GatedLoRAConfig,
        encoder_layer: int = 12,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.sae = sae
        self.config = config
        self.encoder_layer = encoder_layer
        
        # 存储替换的层
        self.gated_lora_layers: Dict[str, GatedLoRALinear] = {}
        
        # 应用门控 LoRA
        self._apply_gated_lora()
        
        # 冻结基础模型
        self._freeze_base_model()
    
    def _apply_gated_lora(self):
        """应用门控 LoRA 到目标模块"""
        for name, module in self.base_model.named_modules():
            # 检查是否是目标模块
            module_name = name.split('.')[-1]
            if module_name in self.config.target_modules and isinstance(module, nn.Linear):
                # 获取该模块的门控神经元
                gate_neurons = self.config.gate_neurons.get(name, [])
                if not gate_neurons:
                    # 使用默认门控神经元
                    gate_neurons = self.config.gate_neurons.get('default', list(range(64)))
                
                # 创建门控 LoRA 层
                gated_lora = GatedLoRALinear(
                    base_layer=module,
                    gate_neurons=gate_neurons,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    gate_threshold=self.config.gate_threshold,
                )
                
                # 替换原始层
                self._replace_module(name, gated_lora)
                self.gated_lora_layers[name] = gated_lora
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """替换模型中的模块"""
        parts = name.split('.')
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def _freeze_base_model(self):
        """冻结基础模型参数"""
        for name, param in self.base_model.named_parameters():
            # 只训练 LoRA 参数
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False
    
    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        前向传播
        
        在 encoder 的指定层提取隐状态，通过 SAE 获取激活，
        然后将激活传递给门控 LoRA 层
        """
        # 获取 encoder 输出和中间层隐状态
        encoder_outputs = self._forward_encoder_with_hidden_states(input_features)
        
        # 获取指定层的隐状态
        hidden_state = encoder_outputs['hidden_states'][self.encoder_layer]
        
        # 通过 SAE 获取稀疏激活
        with torch.no_grad():
            sae_activations = self.sae.encode(hidden_state)
        
        # 将激活传递给所有门控 LoRA 层
        self._set_sae_activations(sae_activations)
        
        # 正常前向传播
        return self.base_model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
    
    def _forward_encoder_with_hidden_states(
        self, 
        input_features: torch.Tensor
    ) -> Dict:
        """获取 encoder 的所有隐状态"""
        encoder = self.base_model.model.encoder
        
        # 嵌入
        inputs_embeds = encoder.conv1(input_features)
        inputs_embeds = F.gelu(inputs_embeds)
        inputs_embeds = encoder.conv2(inputs_embeds)
        inputs_embeds = F.gelu(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        
        embed_pos = encoder.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos
        hidden_states = encoder.dropout(hidden_states)
        
        all_hidden_states = [hidden_states]
        
        for layer in encoder.layers:
            hidden_states = layer(hidden_states)[0]
            all_hidden_states.append(hidden_states)
        
        hidden_states = encoder.layer_norm(hidden_states)
        all_hidden_states.append(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
        }
    
    def _set_sae_activations(self, activations: torch.Tensor):
        """设置所有门控 LoRA 层的 SAE 激活"""
        # 这里需要一个更优雅的方式传递激活
        # 暂时使用模块属性
        for layer in self.gated_lora_layers.values():
            layer._current_sae_activations = activations
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        params = []
        for layer in self.gated_lora_layers.values():
            params.extend(layer.get_lora_parameters())
        return params
    
    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def save_lora_weights(self, path: str):
        """保存 LoRA 权重"""
        state_dict = {}
        for name, layer in self.gated_lora_layers.items():
            state_dict[f"{name}.lora_A"] = layer.lora_A.data
            state_dict[f"{name}.lora_B"] = layer.lora_B.data
        torch.save(state_dict, path)
    
    def load_lora_weights(self, path: str):
        """加载 LoRA 权重"""
        state_dict = torch.load(path, map_location='cpu')
        for name, layer in self.gated_lora_layers.items():
            if f"{name}.lora_A" in state_dict:
                layer.lora_A.data = state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in state_dict:
                layer.lora_B.data = state_dict[f"{name}.lora_B"]


def find_critical_neurons(
    sae: nn.Module,
    model: nn.Module,
    error_samples: List[Dict],
    processor,
    top_k: int = 64,
    device: str = 'cuda',
) -> Dict[int, float]:
    """
    通过反事实优化找到关键神经元
    
    对于高错误率样本，找到改变哪些神经元最能提升性能
    
    Args:
        sae: SAE 模型
        model: Whisper 模型
        error_samples: 高错误率样本列表
        processor: Whisper processor
        top_k: SAE topk 参数
        device: 设备
        
    Returns:
        关键神经元及其重要性分数
    """
    from .feature_extractor import ASRFeatureExtractor
    
    sae = sae.to(device)
    model = model.to(device)
    sae.eval()
    model.eval()
    
    neuron_importance = {}
    
    for sample in error_samples:
        audio = sample.get('audio')
        if isinstance(audio, dict):
            audio = audio['array']
        
        # 准备输入
        input_features = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # 获取 encoder 隐状态
        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features, output_hidden_states=True)
            hidden_state = encoder_outputs.hidden_states[12]  # 中间层
            
            # 获取 SAE 激活
            _, _, sparse = sae(hidden_state, top_k)
            
            # 计算每个神经元的梯度重要性
            sparse.requires_grad_(True)
            sparse.retain_grad()
            
            # 重建并解码
            sparse_topk, _, _ = sae._get_topk(sparse, top_k)
            recover = sae.decode(sparse_topk)
            
            # 使用重建的隐状态继续前向传播
            # 这里简化处理，使用激活值的 L2 范数作为重要性
            importance = sparse.abs().mean(dim=(0, 1))
            
            for i, imp in enumerate(importance.cpu().numpy()):
                if i not in neuron_importance:
                    neuron_importance[i] = 0
                neuron_importance[i] += imp
    
    # 归一化
    n_samples = len(error_samples)
    for k in neuron_importance:
        neuron_importance[k] /= n_samples
    
    # 排序返回
    sorted_neurons = sorted(neuron_importance.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_neurons)


def select_gate_neurons(
    neuron_importance: Dict[int, float] = None,
    neuron_correlations: Dict[int, Dict] = None,
    top_n: int = 64,
    min_importance: float = 0.0,
    counterfactual_results: Optional[Dict] = None,
) -> List[int]:
    """
    选择门控神经元（参考 TaCT）
    
    核心原则：选择那些"改变其激活值能最大程度改善输出"的神经元
    即反事实分析中 |delta| 最大的神经元
    
    Args:
        neuron_importance: 神经元激活率（已弃用，仅作兼容）
        neuron_correlations: 神经元特征关联（已弃用，仅作兼容）
        top_n: 选择的神经元数量
        min_importance: 最小重要性阈值
        counterfactual_results: 反事实分析结果（必需）
            包含 'neuron_importance' 字段，每个神经元有:
            - delta_importance: delta 优化后的重要性（核心指标）
        
    Returns:
        选中的神经元 ID 列表（按 |delta| 排序）
    """
    # 如果有反事实分析结果，直接使用 key_neurons
    if counterfactual_results:
        key_neurons = counterfactual_results.get('key_neurons', [])
        if key_neurons:
            selected = key_neurons[:top_n]
            print(f"Selected {len(selected)} gate neurons from counterfactual analysis:")
            
            cf_neurons = counterfactual_results.get('neuron_importance', {})
            for i, nid in enumerate(selected[:10]):
                nid_str = str(nid)
                if nid_str in cf_neurons:
                    delta = cf_neurons[nid_str].get('delta_importance', 0)
                    samples = cf_neurons[nid_str].get('sample_count', 0)
                    print(f"  #{i+1}: Neuron {nid} - delta={delta:.6f}, samples={samples}")
            
            return selected
    
    # 如果没有反事实结果，从 neuron_importance 中提取 delta_importance
    if counterfactual_results:
        cf_neurons = counterfactual_results.get('neuron_importance', {})
        candidates = []
        for nid_str, data in cf_neurons.items():
            nid = int(nid_str)
            delta = data.get('delta_importance', 0)
            if delta > min_importance:
                candidates.append((nid, delta))
        
        # 按 delta 排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [nid for nid, _ in candidates[:top_n]]
        
        print(f"Selected {len(selected)} gate neurons by delta importance:")
        for i, (nid, delta) in enumerate(candidates[:min(10, top_n)]):
            print(f"  #{i+1}: Neuron {nid} - delta={delta:.6f}")
        
        return selected
    
    # 兜底：如果没有反事实结果，使用激活率（不推荐）
    if neuron_importance:
        print("WARNING: No counterfactual results provided. Using activation rate as fallback.")
        print("This is NOT recommended. Please run counterfactual analysis first.")
        
        candidates = [
            (nid, rate) for nid, rate in neuron_importance.items()
            if rate > min_importance
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [nid for nid, _ in candidates[:top_n]]
        
        return selected
    
    # 如果什么都没有，返回前 top_n 个神经元
    print("WARNING: No neuron importance data provided. Using default neurons 0 to {top_n-1}.")
    return list(range(top_n))
