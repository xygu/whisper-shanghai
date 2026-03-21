"""
Sparse Autoencoder (SAE) for ASR

参考 TaCT 项目的 SAE 实现，适配 Whisper 模型
支持 batch_topk 稀疏化和多种归一化方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Literal


class SAE(nn.Module):
    """
    稀疏自编码器 (Sparse Autoencoder)
    
    将 Whisper encoder 的隐层表征投射到高维稀疏空间，
    每个神经元对应一个可解释的语义概念。
    
    Args:
        input_dim: 输入维度 (Whisper encoder hidden size, e.g., 1024 for medium)
        latent_dim: 稀疏表征维度 (神经元数量, e.g., 8192)
        norm_type: 归一化类型 ('z-norm', 'layer_norm', 'rms_norm')
        use_activate: 是否使用 ReLU 激活
        topk_type: TopK 类型 ('topk', 'batch_topk')
        share_weight: 是否共享编码器和解码器权重
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # Whisper medium encoder dim
        latent_dim: int = 8192,
        norm_type: Literal['z-norm', 'layer_norm', 'rms_norm'] = 'z-norm',
        use_activate: bool = True,
        topk_type: Literal['topk', 'batch_topk'] = 'batch_topk',
        share_weight: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.use_activate = use_activate
        self.topk_type = topk_type
        self.share_weight = share_weight
        
        # 初始化编码器权重 (Kaiming 初始化 + 归一化)
        encoder_weight = torch.randn(latent_dim, input_dim)
        nn.init.kaiming_uniform_(encoder_weight)
        encoder_weight = encoder_weight / torch.sqrt(
            (encoder_weight ** 2).sum(dim=-1, keepdim=True).expand(-1, input_dim)
        )
        encoder_weight = 0.1 * encoder_weight
        
        self.encoder_linear = nn.Parameter(encoder_weight)  # [latent_dim, input_dim]
        self.decoder_linear = nn.Parameter(encoder_weight.clone().detach().T)  # [input_dim, latent_dim]
        self.encoder_bias = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        self.activate = nn.ReLU()
        
        # 记录每个神经元的使用次数 (用于检测死神经元)
        self.register_buffer('use_number', torch.zeros(latent_dim))
        self.scaler = 0.0  # 辅助损失缩放因子
        
        # 归一化统计量 (在 forward 中计算)
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, input_dim]
            topk: 保留的 top-k 激活数量
            
        Returns:
            sparse_recover: 重建的表征 [batch, seq_len, input_dim]
            aux_recover: 辅助重建 (用于激活死神经元)
            sparse: 稀疏激活 [batch, seq_len, latent_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 输入归一化
        x_normed = self._input_norm(x)
        x_centered = x_normed - self.decoder_bias.expand(batch_size, seq_len, -1)
        
        # 编码到稀疏空间
        sparse = x_centered @ self.encoder_linear.T + self.encoder_bias.expand(batch_size, seq_len, -1)
        
        if self.use_activate:
            sparse = self.activate(sparse)
        
        # TopK 稀疏化
        sparse_topk, topk_index, mask = self._get_topk(sparse, topk)
        
        # 解码重建
        if self.share_weight:
            sparse_recover = sparse_topk @ self.encoder_linear + self.decoder_bias.expand(batch_size, seq_len, -1)
        else:
            sparse_recover = sparse_topk @ self.decoder_linear.T + self.decoder_bias.expand(batch_size, seq_len, -1)
        
        # 输出反归一化
        sparse_recover = self._output_norm(sparse_recover, aux=False)
        
        # 辅助重建 (激活死神经元)
        aux_recover = self._aux_recover(sparse, embed_dim // 2, topk_index)
        aux_recover = self._output_norm(aux_recover, aux=True)
        
        return sparse_recover, aux_recover, sparse
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码，返回稀疏激活"""
        batch_size, seq_len, _ = x.shape
        x_normed = self._input_norm(x)
        x_centered = x_normed - self.decoder_bias.expand(batch_size, seq_len, -1)
        sparse = x_centered @ self.encoder_linear.T + self.encoder_bias.expand(batch_size, seq_len, -1)
        if self.use_activate:
            sparse = self.activate(sparse)
        return sparse
    
    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """仅解码，从稀疏激活重建"""
        batch_size, seq_len, _ = sparse.shape
        if self.share_weight:
            recover = sparse @ self.encoder_linear + self.decoder_bias.expand(batch_size, seq_len, -1)
        else:
            recover = sparse @ self.decoder_linear.T + self.decoder_bias.expand(batch_size, seq_len, -1)
        return self._output_norm(recover, aux=False)
    
    def _aux_recover(
        self, 
        x: torch.Tensor, 
        topk: int, 
        topk_index: torch.Tensor
    ) -> torch.Tensor:
        """辅助重建：激活长期未使用的神经元"""
        batch_size, seq_len, _ = x.shape
        
        # 更新使用计数
        self.use_number = self.use_number + (seq_len * batch_size)
        self.use_number[topk_index.reshape(-1) % self.latent_dim] = 0
        
        # 计算死神经元比例
        self.scaler = min((self.use_number > 1e6).sum().item() / topk, 1.0)
        
        # 创建死神经元掩码
        use_mask = self.use_number.reshape(1, 1, -1).expand(batch_size, seq_len, -1).to(x.device)
        use_mask = torch.where(use_mask > 1e6, 1.0, 0.0)
        
        # 仅对死神经元进行辅助重建
        aux_sparse = x * use_mask
        aux_sparse, _, _ = self._get_topk(aux_sparse, topk)
        
        if self.share_weight:
            aux_recover = aux_sparse @ self.encoder_linear + self.decoder_bias.expand(batch_size, seq_len, -1)
        else:
            aux_recover = aux_sparse @ self.decoder_linear.T + self.decoder_bias.expand(batch_size, seq_len, -1)
        
        return aux_recover
    
    def _input_norm(self, x: torch.Tensor) -> torch.Tensor:
        """输入归一化"""
        batch_size, seq_len, embed_dim = x.shape
        
        if self.norm_type == 'layer_norm':
            self.x_mean = x.mean(dim=-1, keepdim=True).expand(-1, -1, embed_dim)
            self.x_std = x.std(dim=-1, keepdim=True).expand(-1, -1, embed_dim)
        elif self.norm_type == 'rms_norm':
            self.x_mean = torch.zeros_like(x)
            self.x_std = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True)).expand(-1, -1, embed_dim)
        elif self.norm_type == 'z-norm':
            self.x_mean = x.mean(dim=-2, keepdim=True).expand(-1, seq_len, -1)
            self.x_std = x.std(dim=-2, keepdim=True).expand(-1, seq_len, -1)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
        
        return (x - self.x_mean) / (self.x_std + 1e-6)
    
    def _output_norm(self, x: torch.Tensor, aux: bool) -> torch.Tensor:
        """输出反归一化"""
        if aux:
            return x * self.x_std
        else:
            return x * self.x_std + self.x_mean
    
    def _get_topk(
        self, 
        x: torch.Tensor, 
        topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取 TopK 稀疏激活"""
        if self.topk_type == 'topk':
            mask, topk_index = self._get_topk_mask(x, topk)
        elif self.topk_type == 'batch_topk':
            mask, topk_index = self._get_batch_topk_mask(x, topk)
        else:
            raise ValueError(f"Unknown topk_type: {self.topk_type}")
        
        return x * mask, topk_index, mask
    
    def _get_topk_mask(
        self, 
        x: torch.Tensor, 
        topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """每个位置独立的 TopK"""
        batch_size, seq_len, embed_dim = x.shape
        topk_index = torch.topk(x, k=topk, dim=-1, largest=True)[1]
        mask = torch.zeros(batch_size, seq_len, embed_dim, device=x.device)
        mask.scatter_(2, topk_index, 1)
        return mask, topk_index
    
    def _get_batch_topk_mask(
        self, 
        x: torch.Tensor, 
        topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """整个序列共享的 TopK (更稀疏)"""
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.reshape(batch_size, -1)
        topk_index = torch.topk(x_flat, k=topk * seq_len, dim=-1, largest=True)[1]
        mask = torch.zeros(batch_size, seq_len * embed_dim, device=x.device)
        mask.scatter_(1, topk_index, 1)
        mask = mask.reshape(batch_size, seq_len, embed_dim)
        return mask, topk_index
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        sparse_recover: torch.Tensor, 
        aux_recover: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 SAE 损失
        
        Args:
            x: 原始输入
            sparse_recover: 稀疏重建
            aux_recover: 辅助重建
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        # 重建损失
        recover_loss = F.mse_loss(sparse_recover, x)
        
        # 辅助损失 (激活死神经元)
        aux_loss = F.mse_loss(x - sparse_recover.detach(), aux_recover)
        
        # 总损失
        total_loss = recover_loss + self.scaler * aux_loss
        
        # 计算解释方差
        explained_var = 1 - ((x - sparse_recover).var(dim=-2) / (x.var(dim=-2) + 1e-6)).mean()
        
        loss_dict = {
            'recover_loss': recover_loss.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item(),
            'explained_var': explained_var.item(),
            'dead_neurons': (self.use_number > 1e6).sum().item(),
        }
        
        return total_loss, loss_dict
    
    def get_neuron_activations(
        self, 
        x: torch.Tensor, 
        topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取神经元激活情况
        
        Returns:
            activations: 稀疏激活值 [batch, seq_len, latent_dim]
            active_mask: 激活掩码 [batch, seq_len, latent_dim]
        """
        _, _, sparse = self.forward(x, topk)
        _, _, mask = self._get_topk(sparse, topk)
        return sparse, mask


class SAEConfig:
    """SAE 配置类"""
    
    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 8192,
        norm_type: str = 'z-norm',
        use_activate: bool = True,
        topk_type: str = 'batch_topk',
        topk: int = 64,
        share_weight: bool = False,
        learning_rate: float = 1e-4,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.use_activate = use_activate
        self.topk_type = topk_type
        self.topk = topk
        self.share_weight = share_weight
        self.learning_rate = learning_rate
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SAEConfig':
        return cls(**config_dict)
    
    @classmethod
    def for_whisper(cls, model_size: str = 'medium') -> 'SAEConfig':
        """根据 Whisper 模型大小返回推荐配置"""
        size_to_dim = {
            'tiny': 384,
            'base': 512,
            'small': 768,
            'medium': 1024,
            'large': 1280,
            'large-v2': 1280,
            'large-v3': 1280,
        }
        input_dim = size_to_dim.get(model_size, 1024)
        latent_dim = input_dim * 8  # 8x 扩展
        
        return cls(
            input_dim=input_dim,
            latent_dim=latent_dim,
            topk=max(32, latent_dim // 128),  # 约 0.8% 稀疏度
        )
