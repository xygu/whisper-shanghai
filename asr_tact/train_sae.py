"""
SAE Training Script for Whisper ASR

训练 SAE 模型，将 Whisper encoder 的隐层表征投射到高维稀疏空间
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)
from datasets import load_from_disk, Audio

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_tact.sae import SAE, SAEConfig
from asr_tact.feature_extractor import ASRFeatureExtractor, SampleFeatures


def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger('sae_training')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
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
    """
    提取 Whisper encoder 指定层的隐状态
    
    Args:
        model: Whisper 模型
        input_features: 输入特征 [batch, mel_bins, time]
        layer: 要提取的层号
        
    Returns:
        hidden_states: [batch, seq_len, hidden_dim]
    """
    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # 获取指定层的隐状态
    hidden_states = encoder_outputs.hidden_states[layer]
    
    return hidden_states


def train_sae(
    model_name: str = "openai/whisper-medium",
    dataset_path: str = "dataset/shanghai/shanghai_dataset",
    output_dir: str = "./exp/sae",
    encoder_layer: int = 12,
    latent_dim: int = 8192,
    topk: int = 64,
    norm_type: str = "z-norm",
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_every: int = 1000,
    log_every: int = 100,
):
    """
    训练 SAE 模型
    """
    # 创建输出目录
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, f"sae-layer{encoder_layer}-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("ASR-TACT SAE Training")
    logger.info("=" * 70)
    
    # 设置 TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # 加载 Whisper 模型
    logger.info(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    # 冻结 Whisper 模型
    for param in whisper_model.parameters():
        param.requires_grad = False
    
    # 获取 encoder hidden dim
    encoder_dim = whisper_model.config.d_model
    logger.info(f"Encoder hidden dim: {encoder_dim}")
    
    # 创建 SAE 模型
    logger.info(f"Creating SAE model: input_dim={encoder_dim}, latent_dim={latent_dim}")
    sae_config = SAEConfig(
        input_dim=encoder_dim,
        latent_dim=latent_dim,
        norm_type=norm_type,
        use_activate=True,
        topk_type="batch_topk",
        topk=topk,
        share_weight=False,
        learning_rate=learning_rate,
    )
    sae = SAE(
        input_dim=sae_config.input_dim,
        latent_dim=sae_config.latent_dim,
        norm_type=sae_config.norm_type,
        use_activate=sae_config.use_activate,
        topk_type=sae_config.topk_type,
        share_weight=sae_config.share_weight,
    )
    sae.to(device)
    
    # 保存配置
    with open(os.path.join(output_dir, 'sae_config.json'), 'w') as f:
        json.dump(sae_config.to_dict(), f, indent=2)
    
    # 加载数据集
    logger.info(f"Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Test samples: {len(dataset['test'])}")
    
    # 数据预处理函数
    def preprocess_batch(batch):
        audio_arrays = [item["array"] for item in batch["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        return {"input_features": inputs.input_features}
    
    # 创建 DataLoader
    train_dataset = dataset["train"].map(
        preprocess_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,
    )
    train_dataset.set_format(type="torch", columns=["input_features"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1,
    )
    
    # 训练循环
    logger.info("Starting training...")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        sae.train()
        epoch_loss = 0.0
        epoch_recover_loss = 0.0
        epoch_aux_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_features = batch["input_features"].to(device)
            
            # 提取 encoder 隐状态
            hidden_states = extract_encoder_hidden_states(
                whisper_model, input_features, layer=encoder_layer
            )
            
            # SAE 前向传播
            sparse_recover, aux_recover, sparse = sae(hidden_states, topk)
            
            # 计算损失
            loss, loss_dict = sae.compute_loss(hidden_states, sparse_recover, aux_recover)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 更新统计
            epoch_loss += loss.item()
            epoch_recover_loss += loss_dict['recover_loss']
            epoch_aux_loss += loss_dict['aux_loss']
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recover': f"{loss_dict['recover_loss']:.4f}",
                'exp_var': f"{loss_dict['explained_var']:.2%}",
            })
            
            # 日志记录
            if global_step % log_every == 0:
                writer.add_scalar('loss/total', loss.item(), global_step)
                writer.add_scalar('loss/recover', loss_dict['recover_loss'], global_step)
                writer.add_scalar('loss/aux', loss_dict['aux_loss'], global_step)
                writer.add_scalar('metric/explained_var', loss_dict['explained_var'], global_step)
                writer.add_scalar('metric/dead_neurons', loss_dict['dead_neurons'], global_step)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            
            # 保存检查点
            if global_step % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': sae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Epoch 统计
        avg_loss = epoch_loss / len(train_loader)
        avg_recover = epoch_recover_loss / len(train_loader)
        avg_aux = epoch_aux_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Recover: {avg_recover:.4f}, Aux: {avg_aux:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'config': sae_config.to_dict(),
                'loss': avg_loss,
            }, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    # 保存最终模型
    final_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'step': global_step,
        'epoch': num_epochs,
        'model_state_dict': sae.state_dict(),
        'config': sae_config.to_dict(),
        'loss': avg_loss,
    }, final_path)
    logger.info(f"Saved final model: {final_path}")
    
    writer.close()
    logger.info("Training completed!")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train SAE for Whisper ASR")
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium",
                        help="Whisper model name")
    parser.add_argument("--dataset_path", type=str, default="dataset/shanghai/shanghai_dataset",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./exp/sae",
                        help="Output directory")
    parser.add_argument("--encoder_layer", type=int, default=12,
                        help="Encoder layer to extract hidden states from")
    parser.add_argument("--latent_dim", type=int, default=8192,
                        help="SAE latent dimension")
    parser.add_argument("--topk", type=int, default=64,
                        help="TopK for sparse activation")
    parser.add_argument("--norm_type", type=str, default="z-norm",
                        choices=["z-norm", "layer_norm", "rms_norm"],
                        help="Normalization type")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N steps")
    
    args = parser.parse_args()
    
    train_sae(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        encoder_layer=args.encoder_layer,
        latent_dim=args.latent_dim,
        topk=args.topk,
        norm_type=args.norm_type,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
