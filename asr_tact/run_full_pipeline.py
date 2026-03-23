#!/usr/bin/env python3
"""
ASR-TACT 完整流程脚本

独立步骤，可以一个一个跑：
1. train_sae: 训练 SAE (latent_dim=9000, topk=100)
2. compute_stats: 统计量汇报 (死亡神经元率、R²、稀疏度)
3. find_worst: 找最差 1/10 样本
4. counterfactual: 反事实优化找 top10 神经元
5. extract_features: 提取神经元激活的 token 特征
6. generate_prompt: 生成 LLM 解释 prompt
7. train_lora: 门控 LoRA 微调

使用方法:
    python asr_tact/run_full_pipeline.py train_sae
    python asr_tact/run_full_pipeline.py compute_stats --sae_checkpoint exp/sae/xxx/best_model.pt
    python asr_tact/run_full_pipeline.py find_worst --sae_checkpoint exp/sae/xxx/best_model.pt
    python asr_tact/run_full_pipeline.py counterfactual --sae_checkpoint exp/sae/xxx/best_model.pt --worst_samples_path exp/worst_samples.json
    python asr_tact/run_full_pipeline.py extract_features --sae_checkpoint exp/sae/xxx/best_model.pt --key_neurons_path exp/counterfactual/key_neurons.json
    python asr_tact/run_full_pipeline.py generate_prompt --features_path exp/neuron_features.json
    python asr_tact/run_full_pipeline.py train_lora --sae_checkpoint exp/sae/xxx/best_model.pt --key_neurons_path exp/counterfactual/key_neurons.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# 设置 HuggingFace 镜像（中国可用）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 配置
# ============================================================
DEFAULT_CONFIG = {
    'model_name': 'openai/whisper-medium',
    'dataset_path': 'dataset/shanghai/shanghai_dataset',
    'encoder_layer': 12,
    'latent_dim': 9000,  # 约 8.8x 扩展
    'topk': 256,         # 稀疏度 256/9000 ≈ 2.84%
    'norm_type': 'z-norm',
    'topk_type': 'batch_topk',  # topk 或 batch_topk
    'batch_size': 4,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'dead_neuron_threshold': 5e5,  # 死神经元阈值
    'device': 'cuda',
}

# ============================================================
# LLM 解释 Prompt 生成函数 (参考气象领域风格)
# ============================================================
def generate_neuron_explanation_prompt(
    neuron_id: int,
    transcripts: List[str],
    acoustic_stats: Dict[str, List],
    linguistic_stats: Dict[str, List],
) -> str:
    """
    生成用于 LLM 解释 SAE 神经元的 prompt
    参考气象领域的结构化 prompt 风格
    """
    
    # 构建声学特征统计字符串
    acoustic_section = ""
    for k, values in acoustic_stats.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            acoustic_section += f"  - {k}: mean={mean_val:.3f}, std={std_val:.3f}, n={len(values)}\n"
    
    # 构建语言特征统计字符串
    linguistic_section = ""
    for k, values in linguistic_stats.items():
        if values:
            if all(isinstance(v, bool) for v in values):
                true_count = sum(values)
                total = len(values)
                linguistic_section += f"  - {k}: {true_count}/{total} ({true_count/total*100:.1f}% True)\n"
            else:
                mean_val = np.mean(values)
                std_val = np.std(values)
                linguistic_section += f"  - {k}: mean={mean_val:.3f}, std={std_val:.3f}\n"
    
    # 构建转录文本样本
    transcript_samples = "\n".join([f"  {i+1}. \"{t}\"" for i, t in enumerate(transcripts[:15])])
    
    prompt = f"""You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Example interpretations:
- "Tone contour pattern" detected → specific tonal words are transcribed
- "Fricative consonant" detected → words with 's', 'sh', 'x' sounds appear
- "Sentence boundary" detected → punctuation or pause-related tokens generated

Your Task:
Using the provided concept data (acoustic features, linguistic features, activation distribution, and sample transcripts), identify **three possible speech/linguistic phenomena** this neuron concept could represent, and rank them in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Acoustic Features Section**:
   Statistics of audio-level features for tokens where this neuron activates highly.
   - duration: segment duration in seconds
   - energy: RMS energy level
   - pitch_mean/std: fundamental frequency statistics
   - spectral_centroid: brightness of sound
   - zero_crossing_rate: indicator of noisiness/fricatives

2. **Linguistic Features Section**:
   Statistics of text-level features for highly-activated tokens.
   - char_count: number of characters in transcript
   - contains_punctuation: whether punctuation is present
   - is_question: whether it's a question sentence
   - dialect_markers: presence of Shanghainese-specific words

3. **Sample Transcripts**:
   Top 15 transcripts where this neuron had highest activation values.
   These are the actual text outputs associated with high neuron activation.

============================================================
CURRENT CONCEPT DATA: Neuron #{neuron_id}
============================================================

### Acoustic Features Statistics
{acoustic_section if acoustic_section else "  (No acoustic features available)"}

### Linguistic Features Statistics
{linguistic_section if linguistic_section else "  (No linguistic features available)"}

### Sample Transcripts (Top 15 by activation)
{transcript_samples if transcript_samples else "  (No transcripts available)"}

============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {{
    "Reasoning": "<Detailed explanation of why this phenomenon matches the data>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  }},
  ...
]

Confidence Scoring:
- 4: All acoustic/linguistic trends strongly match one phenomenon
- 3: Most trends match, minor inconsistencies
- 2: Some trends match, weaker evidence
- 1: Very little matches, highly uncertain

============================================================
CHECKLIST FOR REASONING (You must address each item)
============================================================

1. **Transcript Pattern Analysis**: 
   - What common words, phrases, or structures appear across samples?
   - Are there repeated characters, syllables, or grammatical patterns?

2. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories?
   - Does duration indicate word boundaries, pauses, or specific syllable types?

3. **Dialect-Specific Markers**:
   - Are there Shanghainese-specific vocabulary or grammatical structures?
   - Do patterns suggest tone sandhi or dialect-specific phonological rules?

4. **Phonetic Category Hypothesis**:
   - Could this neuron encode consonant types (stops, fricatives, nasals)?
   - Could it encode vowel qualities or tonal patterns?

5. **Linguistic Level**:
   - Is this a phoneme-level, word-level, or sentence-level feature?
   - Does it relate to syntax, semantics, or pragmatics?

6. **Error Pattern Correlation**:
   - If this neuron relates to ASR errors, what type of confusion might it cause?
   - Homophone confusion? Tone errors? Word boundary errors?
"""
    
    return prompt


# ============================================================
# Step 1: 训练 SAE
# ============================================================
def cmd_train_sae(args):
    """训练 SAE (latent_dim=9000, topk=100)"""
    from asr_tact.train_sae import train_sae
    
    print("=" * 70)
    print("Step 1: Train SAE")
    print("=" * 70)
    print(f"  model_name: {args.model_name}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  topk: {args.topk}")
    print(f"  稀疏度: {args.topk / args.latent_dim * 100:.2f}%")
    print(f"  dead_neuron_threshold: {args.dead_neuron_threshold}")
    print("=" * 70)
    
    train_sae(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        encoder_layer=args.encoder_layer,
        latent_dim=args.latent_dim,
        topk=args.topk,
        norm_type=args.norm_type,
        topk_type=args.topk_type,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        dead_neuron_threshold=args.dead_neuron_threshold,
        device=args.device,
        use_wandb=not args.no_wandb,
    )

# ============================================================
# Step 2: 统计量汇报
# ============================================================
def cmd_compute_stats(args):
    """统计量汇报：死亡神经元率、重建率(R²)、稀疏度"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from datasets import load_from_disk, Audio
    from asr_tact.sae import SAE, SAEConfig
    
    print("=" * 70)
    print("Step 2: Compute SAE Statistics")
    print("=" * 70)
    
    device = args.device
    
    # 加载 SAE
    print(f"Loading SAE: {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location='cpu')
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
    print(f"Loading Whisper: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    # 加载数据集
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 统计变量
    total_tokens = 0
    total_topk_activations = 0
    explained_vars = []
    neuron_activation_counts = torch.zeros(sae_config.latent_dim, device=device)
    
    # 处理样本
    samples = dataset['train'].select(range(min(args.max_samples, len(dataset['train']))))
    
    print(f"Processing {len(samples)} samples...")
    for sample in tqdm(samples, desc="Computing statistics"):
        audio_array = sample['audio']['array']
        
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)
        
        with torch.no_grad():
            # 提取隐状态
            encoder_outputs = whisper_model.model.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = encoder_outputs.hidden_states[args.encoder_layer]
            
            # SAE 前向传播
            sparse_recover, aux_recover, sparse = sae(hidden_states, args.topk)
            
            # 计算 R² (方差解释率)
            x = hidden_states
            ss_res = ((x - sparse_recover) ** 2).sum()
            ss_tot = ((x - x.mean()) ** 2).sum()
            r_squared = 1 - (ss_res / (ss_tot + 1e-6))
            explained_vars.append(r_squared.item())
            
            # 统计激活
            batch_size, seq_len, _ = sparse.shape
            total_tokens += batch_size * seq_len
            
            sparse_topk, _, mask = sae._get_topk(sparse, args.topk)
            neuron_activation_counts += mask.sum(dim=(0, 1))
            total_topk_activations += mask.sum().item()
    
    # 计算统计量
    avg_r_squared = np.mean(explained_vars)
    dead_neurons = (neuron_activation_counts == 0).sum().item()
    dead_neuron_rate = dead_neurons / sae_config.latent_dim
    sparsity = args.topk / sae_config.latent_dim
    avg_activations_per_token = total_topk_activations / total_tokens
    
    # 打印结果
    print("\n" + "=" * 70)
    print("SAE Statistics")
    print("=" * 70)
    print(f"\n### 基本配置")
    print(f"  输入维度 (input_dim):     {sae_config.input_dim}")
    print(f"  特征字典规模 (latent_dim): {sae_config.latent_dim}")
    print(f"  TopK:                      {args.topk}")
    
    print(f"\n### 稀疏度指标")
    print(f"  稀疏度 (topk/latent_dim):  {sparsity * 100:.4f}%")
    print(f"  每 token 平均激活数:       {avg_activations_per_token:.2f}")
    print(f"  (期望值 ≈ topk = {args.topk})")
    
    print(f"\n### 重建质量")
    print(f"  R² (方差解释率):           {avg_r_squared * 100:.2f}%")
    
    print(f"\n### 神经元利用率")
    print(f"  死亡神经元数量:            {dead_neurons}")
    print(f"  死亡神经元率:              {dead_neuron_rate * 100:.2f}%")
    print(f"  活跃神经元数量:            {sae_config.latent_dim - dead_neurons}")
    
    print("\n" + "=" * 70)
    
    # 保存结果
    stats = {
        'input_dim': sae_config.input_dim,
        'latent_dim': sae_config.latent_dim,
        'topk': args.topk,
        'sparsity': sparsity,
        'r_squared': avg_r_squared,
        'dead_neuron_count': dead_neurons,
        'dead_neuron_rate': dead_neuron_rate,
        'avg_activations_per_token': avg_activations_per_token,
        'total_samples': len(samples),
        'total_tokens': total_tokens,
    }
    
    output_path = os.path.join(os.path.dirname(args.sae_checkpoint), 'sae_stats.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {output_path}")
    
    return stats

# ============================================================
# Step 3: 找最差 1/10 样本
# ============================================================
def cmd_find_worst(args):
    """找出最差 1/10 训练样本"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from datasets import load_from_disk, Audio
    from asr_tact.sae import SAE, SAEConfig
    
    print("=" * 70)
    print("Step 3: Find Worst 1/10 Samples")
    print("=" * 70)
    
    device = args.device
    
    # 加载模型
    print(f"Loading SAE: {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location='cpu')
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
    
    print(f"Loading Whisper: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 计算每个样本的预测 loss
    sample_losses = []
    
    print("Computing prediction loss for each sample...")
    for idx, sample in enumerate(tqdm(dataset['train'], desc="Evaluating samples")):
        audio_array = sample['audio']['array']
        transcript = sample.get('text', '')
        
        if not transcript:
            continue
        
        try:
            input_features = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)
            
            labels = processor.tokenizer(
                transcript,
                return_tensors="pt",
            ).input_ids.to(device)
            
            with torch.no_grad():
                outputs = whisper_model(
                    input_features=input_features,
                    labels=labels,
                )
                loss = outputs.loss.item()
            
            sample_losses.append({
                'idx': idx,
                'loss': loss,
                'text': transcript,
            })
        except Exception as e:
            continue
    
    # 排序找最差 1/10
    sample_losses.sort(key=lambda x: x['loss'], reverse=True)
    worst_count = len(sample_losses) // 10
    worst_samples = sample_losses[:worst_count]
    
    print(f"\nFound {len(worst_samples)} worst samples (top 10% by loss)")
    print(f"Loss range: {worst_samples[-1]['loss']:.4f} - {worst_samples[0]['loss']:.4f}")
    
    # 保存结果
    output_path = args.output_path or os.path.join(
        os.path.dirname(args.sae_checkpoint), 'worst_samples.json'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'worst_samples': worst_samples,
            'total_samples': len(sample_losses),
            'worst_count': worst_count,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Worst samples saved to: {output_path}")
    
    return worst_samples

# ============================================================
# Step 4: 反事实优化找 top10 神经元
# ============================================================
def cmd_counterfactual(args):
    """反事实优化找变化最大的 10 个神经元"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from datasets import load_from_disk, Audio
    from asr_tact.sae import SAE, SAEConfig
    
    print("=" * 70)
    print("Step 4: Counterfactual Analysis - Find Top 10 Neurons")
    print("=" * 70)
    
    device = args.device
    
    # 加载模型
    print(f"Loading SAE: {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location='cpu')
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
    
    print(f"Loading Whisper: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    # 加载最差样本
    print(f"Loading worst samples: {args.worst_samples_path}")
    with open(args.worst_samples_path, 'r') as f:
        worst_data = json.load(f)
    worst_samples = worst_data['worst_samples']
    
    # 加载数据集
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 反事实优化参数
    optimization_steps = args.optimization_steps
    learning_rate = args.learning_rate
    regularization = args.regularization
    
    # 累积每个神经元的 delta
    delta_accumulator = defaultdict(list)
    
    print(f"Running counterfactual optimization on {len(worst_samples)} samples...")
    
    for sample_info in tqdm(worst_samples[:args.max_samples], desc="Counterfactual"):
        idx = sample_info['idx']
        sample = dataset['train'][idx]
        audio_array = sample['audio']['array']
        transcript = sample.get('text', '')
        
        if not transcript:
            continue
        
        try:
            # 准备输入
            input_features = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)
            
            labels = processor.tokenizer(
                transcript,
                return_tensors="pt",
            ).input_ids.to(device)
            
            encoder = whisper_model.model.encoder
            
            # 获取 encoder 隐状态
            with torch.no_grad():
                inputs_embeds = encoder.conv1(input_features)
                inputs_embeds = F.gelu(inputs_embeds)
                inputs_embeds = encoder.conv2(inputs_embeds)
                inputs_embeds = F.gelu(inputs_embeds)
                inputs_embeds = inputs_embeds.permute(0, 2, 1)
                
                embed_pos = encoder.embed_positions.weight[:inputs_embeds.shape[1]]
                hidden_states = inputs_embeds + embed_pos
                hidden_states = encoder.dropout(hidden_states)
                
                for layer in encoder.layers[:args.encoder_layer]:
                    hidden_states = layer(hidden_states)[0]
                
                # 获取 SAE 激活
                sparse = sae.encode(hidden_states)
                sparse_topk, _, mask = sae._get_topk(sparse, args.topk)
                sparse_temp = sparse_topk.detach()
            
            # 初始化 delta
            sparse_mean = sparse_temp.mean(dim=1)
            delta = torch.zeros_like(sparse_mean)
            delta.requires_grad_(True)
            
            mask_mean = (sparse_temp.abs().mean(dim=1) > 0).float()
            
            decoder_input_ids = labels[:, :-1]
            decoder_labels = labels[:, 1:]
            
            # 迭代优化 delta
            for step in range(optimization_steps):
                sparse_modified = sparse_temp.clone()
                delta_expanded = delta.unsqueeze(1).expand_as(sparse_modified)
                sparse_modified = sparse_modified + delta_expanded
                
                recovered = sae.decode(sparse_modified)
                
                h = recovered
                for layer in encoder.layers[args.encoder_layer:]:
                    h = layer(h)[0]
                h = encoder.layer_norm(h)
                
                decoder_outputs = whisper_model.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=h,
                )
                logits = whisper_model.proj_out(decoder_outputs.last_hidden_state)
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    decoder_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = loss + regularization * torch.abs(delta).mean()
                
                total_loss.backward()
                
                with torch.no_grad():
                    delta = delta - learning_rate * delta.grad
                    delta = delta * mask_mean
                    delta = delta.detach()
                    delta.requires_grad_(True)
            
            # 记录 delta
            delta_abs = delta.abs().squeeze(0).cpu().numpy()
            for neuron_id in range(len(delta_abs)):
                if delta_abs[neuron_id] > 0:
                    delta_accumulator[neuron_id].append(delta_abs[neuron_id])
                    
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # 计算每个神经元的平均 delta
    neuron_importance = {}
    for neuron_id, deltas in delta_accumulator.items():
        neuron_importance[neuron_id] = {
            'mean_delta': float(np.mean(deltas)),
            'max_delta': float(np.max(deltas)),
            'sample_count': len(deltas),
        }
    
    # 排序找 top 10
    sorted_neurons = sorted(
        neuron_importance.items(),
        key=lambda x: x[1]['mean_delta'],
        reverse=True
    )
    
    top_10_neurons = [n[0] for n in sorted_neurons[:10]]
    
    print("\n" + "=" * 70)
    print("Top 10 Key Neurons (by |delta| importance)")
    print("=" * 70)
    for i, (nid, data) in enumerate(sorted_neurons[:10]):
        print(f"  #{i+1}: Neuron {nid} - mean_delta={data['mean_delta']:.6f}, samples={data['sample_count']}")
    
    # 保存结果
    output_dir = args.output_dir or os.path.dirname(args.sae_checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'key_neurons.json')
    with open(output_path, 'w') as f:
        json.dump({
            'top_10_neurons': top_10_neurons,
            'neuron_importance': {str(k): v for k, v in neuron_importance.items()},
            'total_neurons_analyzed': len(neuron_importance),
        }, f, indent=2)
    
    print(f"\nKey neurons saved to: {output_path}")
    
    return top_10_neurons, neuron_importance

# ============================================================
# Step 5: 提取神经元激活的 token 特征
# ============================================================
def cmd_extract_features(args):
    """找这 10 个神经元激活值最大的前 10% token，提取特征"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from datasets import load_from_disk, Audio
    from asr_tact.sae import SAE, SAEConfig
    from asr_tact.feature_extractor import ASRFeatureExtractor
    
    print("=" * 70)
    print("Step 5: Extract Features for Top 10% Tokens")
    print("=" * 70)
    
    device = args.device
    
    # 加载 key neurons
    print(f"Loading key neurons: {args.key_neurons_path}")
    with open(args.key_neurons_path, 'r') as f:
        key_data = json.load(f)
    top_10_neurons = key_data['top_10_neurons']
    
    print(f"Key neurons: {top_10_neurons}")
    
    # 加载模型
    print(f"Loading SAE: {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location='cpu')
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
    
    print(f"Loading Whisper: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    whisper_model.to(device)
    whisper_model.eval()
    
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 初始化特征提取器
    feature_extractor = ASRFeatureExtractor()
    
    # 收集每个神经元的 token 激活
    neuron_token_activations = {nid: [] for nid in top_10_neurons}
    
    print(f"Collecting token activations for {len(top_10_neurons)} neurons...")
    
    for idx, sample in enumerate(tqdm(dataset['train'].select(range(min(args.max_samples, len(dataset['train'])))), desc="Collecting")):
        audio_array = sample['audio']['array']
        transcript = sample.get('text', '')
        
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)
        
        with torch.no_grad():
            encoder_outputs = whisper_model.model.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = encoder_outputs.hidden_states[args.encoder_layer]
            
            sparse = sae.encode(hidden_states)
            sparse_topk, _, _ = sae._get_topk(sparse, args.topk)
        
        # 提取样本特征
        sample_features = feature_extractor.extract_features(
            audio=audio_array,
            transcript=transcript,
            sample_id=str(idx),
        )
        
        # 记录每个神经元的激活
        for nid in top_10_neurons:
            activations = sparse_topk[0, :, nid].cpu().numpy()
            for token_idx, act_val in enumerate(activations):
                if act_val > 0:
                    neuron_token_activations[nid].append({
                        'sample_idx': idx,
                        'token_idx': token_idx,
                        'activation': float(act_val),
                        'transcript': transcript,
                        'acoustic_features': sample_features.acoustic.to_dict() if sample_features.acoustic else {},
                        'linguistic_features': sample_features.linguistic.to_dict() if sample_features.linguistic else {},
                    })
    
    # 对每个神经元，取激活值最大的前 10% token
    neuron_top_tokens = {}
    
    for nid in top_10_neurons:
        tokens = neuron_token_activations[nid]
        tokens.sort(key=lambda x: x['activation'], reverse=True)
        top_10_pct = max(1, len(tokens) // 10)
        neuron_top_tokens[nid] = tokens[:top_10_pct]
        print(f"Neuron {nid}: {len(tokens)} total tokens, top 10% = {top_10_pct} tokens")
    
    # 保存结果
    output_dir = args.output_dir or os.path.dirname(args.key_neurons_path)
    output_path = os.path.join(output_dir, 'neuron_features.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'top_10_neurons': top_10_neurons,
            'neuron_top_tokens': {str(k): v for k, v in neuron_top_tokens.items()},
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nFeatures saved to: {output_path}")
    
    return neuron_top_tokens

# ============================================================
# Step 6: 生成 LLM 解释 prompt
# ============================================================
def cmd_generate_prompt(args):
    """生成 LLM 解释 prompt"""
    print("=" * 70)
    print("Step 6: Generate LLM Explanation Prompts")
    print("=" * 70)
    
    # 加载特征
    print(f"Loading features: {args.features_path}")
    with open(args.features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    top_10_neurons = data['top_10_neurons']
    neuron_top_tokens = data['neuron_top_tokens']
    
    prompts = {}
    
    for nid in top_10_neurons:
        nid_str = str(nid)
        tokens = neuron_top_tokens.get(nid_str, [])
        
        if not tokens:
            continue
        
        # 收集特征统计
        transcripts = [t['transcript'] for t in tokens[:20]]
        
        acoustic_stats = defaultdict(list)
        linguistic_stats = defaultdict(list)
        
        for t in tokens[:50]:
            acoustic = t.get('acoustic_features', {})
            linguistic = t.get('linguistic_features', {})
            
            for k, v in acoustic.items():
                if isinstance(v, (int, float)):
                    acoustic_stats[k].append(v)
            
            for k, v in linguistic.items():
                if isinstance(v, (int, float, bool)):
                    linguistic_stats[k].append(v)
        
        # 生成 prompt
        prompt = generate_neuron_explanation_prompt(nid, transcripts, acoustic_stats, linguistic_stats)
        
        prompts[nid] = prompt
        
        print(f"\n{'='*50}")
        print(f"Neuron {nid} Prompt:")
        print('='*50)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # 保存 prompts
    output_dir = os.path.dirname(args.features_path)
    output_path = os.path.join(output_dir, 'llm_prompts.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    
    # 也保存为 markdown 文件方便阅读
    md_path = os.path.join(output_dir, 'llm_prompts.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        for nid, prompt in prompts.items():
            f.write(f"\n{'='*70}\n")
            f.write(prompt)
            f.write(f"\n{'='*70}\n\n")
    
    print(f"\nPrompts saved to: {output_path}")
    print(f"Markdown saved to: {md_path}")
    
    return prompts

# ============================================================
# Step 7: 门控 LoRA 微调
# ============================================================
def cmd_train_lora(args):
    """使用 TaCT 比例的神经元作为门控来 LoRA 微调"""
    import torch
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import load_from_disk, Audio
    from asr_tact.sae import SAE, SAEConfig
    from asr_tact.gated_lora import GatedLoRAConfig, GatedLoRAModel
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    
    print("=" * 70)
    print("Step 7: Train Gated LoRA")
    print("=" * 70)
    
    device = args.device
    
    # 加载 key neurons
    print(f"Loading key neurons: {args.key_neurons_path}")
    with open(args.key_neurons_path, 'r') as f:
        key_data = json.load(f)
    
    # TaCT 比例：使用 top 神经元作为门控
    # TaCT 使用 topk=48 作为门控神经元数量
    gate_neurons = key_data['top_10_neurons']
    
    # 扩展到 TaCT 比例 (约 48 个)
    neuron_importance = key_data.get('neuron_importance', {})
    sorted_neurons = sorted(
        neuron_importance.items(),
        key=lambda x: x[1].get('mean_delta', 0),
        reverse=True
    )
    gate_neurons = [int(n[0]) for n in sorted_neurons[:48]]
    
    print(f"Using {len(gate_neurons)} gate neurons (TaCT-style)")
    
    # 加载 SAE
    print(f"Loading SAE: {args.sae_checkpoint}")
    checkpoint = torch.load(args.sae_checkpoint, map_location='cpu')
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
    print(f"Loading Whisper: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # 创建门控 LoRA 配置
    lora_config = GatedLoRAConfig(
        target_modules=['q_proj', 'v_proj'],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        gate_threshold=0.5,
        gate_neurons={'default': gate_neurons},
    )
    
    # 创建门控 LoRA 模型
    gated_model = GatedLoRAModel(
        base_model=whisper_model,
        sae=sae,
        config=lora_config,
        encoder_layer=args.encoder_layer,
    )
    gated_model.to(device)
    gated_model.print_trainable_parameters()
    
    # 加载数据集
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 数据预处理
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch
    
    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)
    
    # 数据整理器
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
            return batch
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=whisper_model.config.decoder_start_token_id,
    )
    
    # 输出目录
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = args.output_dir or f"./exp/gated_lora-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    lora_config.save(os.path.join(output_dir, 'gated_lora_config.json'))
    
    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        num_train_epochs=args.num_epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
    )
    
    # 创建训练器
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=gated_model.base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    gated_model.save_lora_weights(os.path.join(output_dir, 'gated_lora_weights.pt'))
    processor.save_pretrained(output_dir)
    
    print("=" * 70)
    print("Training completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="ASR-TACT Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # train_sae
    p1 = subparsers.add_parser('train_sae', help='Step 1: Train SAE')
    p1.add_argument('--model_name', default=DEFAULT_CONFIG['model_name'])
    p1.add_argument('--dataset_path', default=DEFAULT_CONFIG['dataset_path'])
    p1.add_argument('--output_dir', default='./exp/sae')
    p1.add_argument('--encoder_layer', type=int, default=DEFAULT_CONFIG['encoder_layer'])
    p1.add_argument('--latent_dim', type=int, default=DEFAULT_CONFIG['latent_dim'])
    p1.add_argument('--topk', type=int, default=DEFAULT_CONFIG['topk'])
    p1.add_argument('--norm_type', default=DEFAULT_CONFIG['norm_type'])
    p1.add_argument('--topk_type', default=DEFAULT_CONFIG['topk_type'],
                    choices=['topk', 'batch_topk'],
                    help="TopK type: 'topk' for per-position, 'batch_topk' for sequence-level")
    p1.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    p1.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_epochs'])
    p1.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'])
    p1.add_argument('--dead_neuron_threshold', type=float, default=DEFAULT_CONFIG['dead_neuron_threshold'],
                    help='Dead neuron threshold for auxiliary loss')
    p1.add_argument('--device', default=DEFAULT_CONFIG['device'])
    p1.add_argument('--no_wandb', action='store_true')
    
    # compute_stats
    p2 = subparsers.add_parser('compute_stats', help='Step 2: Compute statistics')
    p2.add_argument('--sae_checkpoint', required=True)
    p2.add_argument('--model_name', default=DEFAULT_CONFIG['model_name'])
    p2.add_argument('--dataset_path', default=DEFAULT_CONFIG['dataset_path'])
    p2.add_argument('--encoder_layer', type=int, default=DEFAULT_CONFIG['encoder_layer'])
    p2.add_argument('--topk', type=int, default=DEFAULT_CONFIG['topk'])
    p2.add_argument('--max_samples', type=int, default=500)
    p2.add_argument('--device', default=DEFAULT_CONFIG['device'])
    
    # find_worst
    p3 = subparsers.add_parser('find_worst', help='Step 3: Find worst 1/10 samples')
    p3.add_argument('--sae_checkpoint', required=True)
    p3.add_argument('--model_name', default=DEFAULT_CONFIG['model_name'])
    p3.add_argument('--dataset_path', default=DEFAULT_CONFIG['dataset_path'])
    p3.add_argument('--output_path', default=None)
    p3.add_argument('--device', default=DEFAULT_CONFIG['device'])
    
    # counterfactual
    p4 = subparsers.add_parser('counterfactual', help='Step 4: Counterfactual analysis')
    p4.add_argument('--sae_checkpoint', required=True)
    p4.add_argument('--worst_samples_path', required=True)
    p4.add_argument('--model_name', default=DEFAULT_CONFIG['model_name'])
    p4.add_argument('--dataset_path', default=DEFAULT_CONFIG['dataset_path'])
    p4.add_argument('--encoder_layer', type=int, default=DEFAULT_CONFIG['encoder_layer'])
    p4.add_argument('--topk', type=int, default=DEFAULT_CONFIG['topk'])
    p4.add_argument('--max_samples', type=int, default=100)
    p4.add_argument('--optimization_steps', type=int, default=50)
    p4.add_argument('--learning_rate', type=float, default=15.0)
    p4.add_argument('--regularization', type=float, default=1e-2)
    p4.add_argument('--output_dir', default=None)
    p4.add_argument('--device', default=DEFAULT_CONFIG['device'])
    
    # extract_features
    p5 = subparsers.add_parser('extract_features', help='Step 5: Extract token features')
    p5.add_argument('--sae_checkpoint', required=True)
    p5.add_argument('--key_neurons_path', required=True)
    p5.add_argument('--model_name', default=DEFAULT_CONFIG['model_name'])
    p5.add_argument('--dataset_path', default=DEFAULT_CONFIG['dataset_path'])
    p5.add_argument('--encoder_layer', type=int, default=DEFAULT_CONFIG['encoder_layer'])
    p5.add_argument('--topk', type=int, default=DEFAULT_CONFIG['topk'])
    p5.add_argument('--max_samples', type=int, default=500)
    p5.add_argument('--output_dir', default=None)
    p5.add_argument('--device', default=DEFAULT_CONFIG['device'])
    
    # generate_prompt
    p6 = subparsers.add_parser('generate_prompt', help='Step 6: Generate LLM prompts')
    p6.add_argument('--features_path', required=True)
    
    # train_lora
    p7 = subparsers.add_parser('train_lora', help='Step 7: Train Gated LoRA')
    p7.add_argument('--sae_checkpoint', required=True)
    p7.add_argument('--key_neurons_path', required=True)
    p7.add_argument('--model_name', default=DEFAULT_CONFIG['model_name'])
    p7.add_argument('--dataset_path', default=DEFAULT_CONFIG['dataset_path'])
    p7.add_argument('--encoder_layer', type=int, default=DEFAULT_CONFIG['encoder_layer'])
    p7.add_argument('--lora_rank', type=int, default=8)
    p7.add_argument('--lora_alpha', type=float, default=16.0)
    p7.add_argument('--batch_size', type=int, default=2)
    p7.add_argument('--num_epochs', type=int, default=10)
    p7.add_argument('--learning_rate', type=float, default=1e-5)
    p7.add_argument('--output_dir', default=None)
    p7.add_argument('--device', default=DEFAULT_CONFIG['device'])
    
    args = parser.parse_args()
    
    if args.command == 'train_sae':
        cmd_train_sae(args)
    elif args.command == 'compute_stats':
        cmd_compute_stats(args)
    elif args.command == 'find_worst':
        cmd_find_worst(args)
    elif args.command == 'counterfactual':
        cmd_counterfactual(args)
    elif args.command == 'extract_features':
        cmd_extract_features(args)
    elif args.command == 'generate_prompt':
        cmd_generate_prompt(args)
    elif args.command == 'train_lora':
        cmd_train_lora(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
