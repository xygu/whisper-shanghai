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
    top_tokens: List[Dict],
    char_distribution: Dict[str, Dict],
    total_token_count: int,
    acoustic_stats: Dict[str, List],
    linguistic_stats: Dict[str, List],
) -> str:
    """
    生成用于 LLM 解释 SAE 神经元的 prompt。

    以字符（aligned_char）为主体，char_distribution 作为主线，
    top_tokens 里的 aligned_context 和 transcript 作为具体证据。
    """

    # ── 字符分布表（主线）──────────────────────────────────────────
    char_dist_lines = []
    for rank, (char, stat) in enumerate(char_distribution.items(), start=1):
        ctx_list = stat['contexts']
        context_examples = "\u3001".join(ctx_list) if ctx_list else "\u2014"
        count = stat['count']
        mean_act = stat['mean_activation']
        char_dist_lines.append(
            f"  {rank:2d}. \u5b57=[{char}]  \u51fa\u73b0={count}\u6b21  "
            f"\u5e73\u5747\u6fc0\u6d3b={mean_act:.3f}  \u4e0a\u4e0b\u6587\u793a\u4f8b: {context_examples}"
        )
    char_dist_section = "\n".join(char_dist_lines) if char_dist_lines else "  (\u65e0\u5b57\u7b26\u5bf9\u9f50\u6570\u636e)"

    # ── Top 30 具体 token 证据（以 char 为标识）────────────────────
    token_evidence_lines = []
    for i, token in enumerate(top_tokens[:30], start=1):
        char = token.get('aligned_char', '')
        context = token.get('aligned_context', '')
        transcript = token.get('transcript', '')
        activation = token.get('activation', 0.0)
        token_evidence_lines.append(
            f"  {i:2d}. 激活={activation:.3f}  字=[{char}]  上下文=[{context}]  整句=[{transcript}]"
        )
    token_evidence_section = "\n".join(token_evidence_lines) if token_evidence_lines else "  (无 token 数据)"

    # ── 声学特征统计 ───────────────────────────────────────────────
    acoustic_section = ""
    for k, values in acoustic_stats.items():
        if values:
            acoustic_section += (
                f"  - {k}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, n={len(values)}\n"
            )

    # ── 语言特征统计 ───────────────────────────────────────────────
    linguistic_section = ""
    for k, values in linguistic_stats.items():
        if values:
            if all(isinstance(v, bool) for v in values):
                true_count = sum(values)
                total = len(values)
                linguistic_section += (
                    f"  - {k}: {true_count}/{total} ({true_count/total*100:.1f}% True)\n"
                )
            else:
                linguistic_section += (
                    f"  - {k}: mean={np.mean(values):.3f}, std={np.std(values):.3f}\n"
                )

    prompt = f"""You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #{neuron_id}
(Total activation events: {total_token_count})
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
{char_dist_section}

### Top 30 Token Evidence (highest activation, per-token detail)
{token_evidence_section}

### Acoustic Features Statistics (from top-30 transcripts)
{acoustic_section if acoustic_section else "  (No acoustic features available)"}

### Linguistic Features Statistics (from top-30 transcripts)
{linguistic_section if linguistic_section else "  (No linguistic features available)"}

============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {{
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  }},
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?
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
    
    # 使用 checkpoint 中的 topk 配置，除非用户显式指定
    topk = args.topk if args.topk != DEFAULT_CONFIG['topk'] else sae_config.topk
    print(f"Using topk={topk} (from {'args' if args.topk != DEFAULT_CONFIG['topk'] else 'checkpoint'})")
    
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
    total_valid_tokens = 0  # 只统计有效（非 padding）的 token
    total_topk_activations = 0
    explained_vars = []
    neuron_activation_counts = torch.zeros(sae_config.latent_dim, device=device)
    
    # 导入 attention mask 计算函数
    from asr_tact.data_utils import audio_length_to_encoder_length
    WHISPER_ENCODER_SEQ_LEN = 1500  # Whisper encoder 固定输出长度
    
    # 处理样本
    samples = dataset['train'].select(range(min(args.max_samples, len(dataset['train']))))
    
    print(f"Processing {len(samples)} samples...")
    for sample in tqdm(samples, desc="Computing statistics"):
        audio_array = sample['audio']['array']
        audio_length = len(audio_array)
        
        # 计算实际有效的 encoder 长度
        encoder_length = min(audio_length_to_encoder_length(audio_length), WHISPER_ENCODER_SEQ_LEN)
        
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
            sparse_recover, aux_recover, sparse = sae(hidden_states, topk)
            
            # 创建 attention mask: 只计算有效位置
            batch_size, seq_len, hidden_dim = hidden_states.shape
            attention_mask = torch.zeros(batch_size, seq_len, device=device)
            attention_mask[:, :encoder_length] = 1.0
            
            # 计算 R² (方差解释率) - 只在有效位置计算
            x = hidden_states
            mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq, 1]
            
            # 只计算有效位置的残差和总方差
            x_masked = x * mask_expanded
            recover_masked = sparse_recover * mask_expanded
            
            num_valid = attention_mask.sum() * hidden_dim
            
            if num_valid > 0:
                # 计算有效位置的均值
                x_mean = (x_masked.sum() / num_valid)
                
                # 残差平方和（只在有效位置）
                ss_res = (((x - sparse_recover) ** 2) * mask_expanded).sum()
                
                # 总平方和（只在有效位置）
                ss_tot = (((x - x_mean) ** 2) * mask_expanded).sum()
                
                r_squared = 1 - (ss_res / (ss_tot + 1e-6))
                explained_vars.append(r_squared.item())
            
            # 统计激活 - 只统计有效位置
            total_tokens += batch_size * seq_len
            total_valid_tokens += encoder_length * batch_size
            
            sparse_topk, _, topk_mask = sae._get_topk(sparse, topk)
            
            # 只统计有效位置的激活
            valid_topk_mask = topk_mask * mask_expanded
            neuron_activation_counts += valid_topk_mask.sum(dim=(0, 1))
            total_topk_activations += valid_topk_mask.sum().item()
    
    # 计算统计量
    avg_r_squared = np.mean(explained_vars) if explained_vars else 0.0
    dead_neurons = (neuron_activation_counts == 0).sum().item()
    dead_neuron_rate = dead_neurons / sae_config.latent_dim
    sparsity = topk / sae_config.latent_dim
    avg_activations_per_token = total_topk_activations / total_valid_tokens if total_valid_tokens > 0 else 0.0
    padding_ratio = 1 - (total_valid_tokens / total_tokens) if total_tokens > 0 else 0.0
    
    # 打印结果
    print("\n" + "=" * 70)
    print("SAE Statistics")
    print("=" * 70)
    print(f"\n### 基本配置")
    print(f"  输入维度 (input_dim):     {sae_config.input_dim}")
    print(f"  特征字典规模 (latent_dim): {sae_config.latent_dim}")
    print(f"  TopK:                      {topk}")
    
    print(f"\n### Token 统计")
    print(f"  总 token 数:               {total_tokens}")
    print(f"  有效 token 数:             {total_valid_tokens}")
    print(f"  Padding 比例:              {padding_ratio * 100:.2f}%")
    
    print(f"\n### 稀疏度指标")
    print(f"  稀疏度 (topk/latent_dim):  {sparsity * 100:.4f}%")
    print(f"  每有效 token 平均激活数:   {avg_activations_per_token:.2f}")
    print(f"  (期望值 ≈ topk = {topk})")
    
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
        'topk': topk,
        'sparsity': sparsity,
        'r_squared': avg_r_squared,
        'dead_neuron_count': dead_neurons,
        'dead_neuron_rate': dead_neuron_rate,
        'avg_activations_per_token': avg_activations_per_token,
        'total_samples': len(samples),
        'total_tokens': total_tokens,
        'total_valid_tokens': total_valid_tokens,
        'padding_ratio': padding_ratio,
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
    
    # 使用 checkpoint 中的 topk 配置，除非用户显式指定
    topk = args.topk if args.topk != DEFAULT_CONFIG['topk'] else sae_config.topk
    print(f"Using topk={topk} (from {'args' if args.topk != DEFAULT_CONFIG['topk'] else 'checkpoint'})")
    
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
            hidden_states = F.dropout(hidden_states, p=encoder.dropout, training=encoder.training)
            
            for layer in encoder.layers[:args.encoder_layer]:
                hidden_states = layer(hidden_states, attention_mask=None, layer_head_mask=None)[0]
            
            # 获取 SAE 激活
            sparse = sae.encode(hidden_states)
            sparse_topk, _, mask = sae._get_topk(sparse, topk)
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
                h = layer(h, attention_mask=None, layer_head_mask=None)[0]
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
        delta_abs = delta.abs().squeeze(0).cpu().detach().numpy()
        for neuron_id in range(len(delta_abs)):
            if delta_abs[neuron_id] > 0:
                delta_accumulator[neuron_id].append(delta_abs[neuron_id])
                

    
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
# Step 5 辅助函数：token 时间位置对齐
# ============================================================

def align_token_by_time_ratio(
    token_idx: int,
    total_encoder_frames: int,
    audio_array,
    transcript: str,
    sample_rate: int = 16000,
) -> dict:
    """
    方法一：基于时间比例估算 token 对应的字符位置。

    Whisper encoder 每帧对应 hop_length=160 个采样点（即 10ms），
    但 encoder 输出做了 2x 下采样，所以每个 encoder token 对应 20ms。

    思路：
    1. 计算 token 在整段音频中的时间比例
    2. 用同样的比例在 transcript 中定位对应字符
    3. 取该位置前后各 1 个字作为上下文窗口

    Args:
        token_idx: encoder 输出的帧索引
        total_encoder_frames: encoder 输出的总帧数
        audio_array: 原始音频波形
        transcript: 对应的文字转录
        sample_rate: 采样率

    Returns:
        dict 包含 aligned_char、aligned_context、time_sec、time_ratio
    """
    # 每个 encoder token 对应 20ms（hop=160, 下采样×2）
    seconds_per_encoder_token = (160 * 2) / sample_rate
    token_time_sec = token_idx * seconds_per_encoder_token
    audio_duration_sec = len(audio_array) / sample_rate

    time_ratio = token_time_sec / max(audio_duration_sec, 1e-6)
    time_ratio = min(time_ratio, 1.0)

    # 按比例定位到 transcript 中的字符位置
    chars = [ch for ch in transcript if ch.strip()]  # 去掉空格
    char_count = len(chars)

    if char_count == 0:
        return {
            'aligned_char': '',
            'aligned_context': '',
            'time_sec': round(token_time_sec, 3),
            'time_ratio': round(time_ratio, 3),
            'align_method': 'time_ratio',
        }

    char_idx = min(int(time_ratio * char_count), char_count - 1)
    window_start = max(0, char_idx - 1)
    window_end = min(char_count, char_idx + 2)

    aligned_char = chars[char_idx]
    aligned_context = ''.join(chars[window_start:window_end])

    return {
        'aligned_char': aligned_char,
        'aligned_context': aligned_context,
        'time_sec': round(token_time_sec, 3),
        'time_ratio': round(time_ratio, 3),
        'align_method': 'time_ratio',
    }


def align_token_by_whisper_timestamp(
    token_idx: int,
    whisper_model,
    processor,
    audio_array,
    transcript: str,
    device: str,
    sample_rate: int = 16000,
) -> dict:
    """
    方法二：使用 Whisper word-level 时间戳对齐 token 到具体字符。

    思路：
    1. 调用 whisper_model.generate() 带 return_timestamps=True
    2. 得到每个 token 的起止时间
    3. 计算当前 encoder token_idx 对应的时间点
    4. 找到时间上最近的 decode token，取其对应文字

    Args:
        token_idx: encoder 输出的帧索引
        whisper_model: WhisperForConditionalGeneration 模型
        processor: WhisperProcessor
        audio_array: 原始音频波形
        transcript: 对应的文字转录（用于 fallback）
        device: 推理设备
        sample_rate: 采样率

    Returns:
        dict 包含 aligned_char、aligned_context、time_sec、align_method
    """
    seconds_per_encoder_token = (160 * 2) / sample_rate
    token_time_sec = token_idx * seconds_per_encoder_token

    try:
        input_features = processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            outputs = whisper_model.generate(
                input_features,
                return_timestamps=True,
                return_token_timestamps=True,
                language="zh",
            )

        # 解码带时间戳的 token 序列
        # outputs.token_timestamps: [batch, seq_len]，每个值是该 token 的起始时间（秒）
        token_timestamps = outputs.token_timestamps[0].cpu().tolist()  # [seq_len]
        token_ids = outputs.sequences[0].cpu().tolist()               # [seq_len]

        # 过滤掉特殊 token（id < 50000 通常是文字 token，具体阈值看 tokenizer）
        text_token_pairs = []
        for tid, ttime in zip(token_ids, token_timestamps):
            decoded = processor.tokenizer.decode([tid])
            # 跳过特殊 token（<|...|> 格式）
            if decoded.startswith('<|') and decoded.endswith('|>'):
                continue
            if decoded.strip():
                text_token_pairs.append((decoded.strip(), ttime))

        if not text_token_pairs:
            raise ValueError("No text tokens found in timestamp output")

        # 找到时间上最接近 token_time_sec 的 decode token
        best_text, best_time = min(
            text_token_pairs,
            key=lambda pair: abs(pair[1] - token_time_sec),
        )

        # 找该字在 text_token_pairs 中的位置，取前后各 1 个字作为上下文
        best_idx = next(
            (i for i, (text, _) in enumerate(text_token_pairs) if text == best_text and abs(_ - best_time) < 1e-6),
            0,
        )
        context_tokens = text_token_pairs[max(0, best_idx - 1): best_idx + 2]
        aligned_context = ''.join(t for t, _ in context_tokens)

        return {
            'aligned_char': best_text,
            'aligned_context': aligned_context,
            'time_sec': round(token_time_sec, 3),
            'closest_timestamp': round(best_time, 3),
            'align_method': 'whisper_timestamp',
        }

    except Exception as alignment_error:
        # 降级到 time_ratio 方法
        fallback = align_token_by_time_ratio(
            token_idx=token_idx,
            total_encoder_frames=1500,  # Whisper 最大帧数
            audio_array=audio_array,
            transcript=transcript,
            sample_rate=sample_rate,
        )
        fallback['align_method'] = f'time_ratio_fallback({alignment_error})'
        return fallback



# ============================================================
# Step 5 辅助函数：token 时间位置对齐
# ============================================================

def align_token_by_time_ratio(
    token_idx: int,
    total_encoder_frames: int,
    audio_array,
    transcript: str,
    sample_rate: int = 16000,
) -> dict:
    """
    方法一：基于时间比例估算 token 对应的字符位置。

    Whisper encoder 每个输出 token 对应 20ms（hop_length=160, 下采样×2）。
    思路：用 token 在总帧数中的比例，映射到 transcript 字符序列中的位置。
    """
    seconds_per_encoder_token = (160 * 2) / sample_rate
    token_time_sec = token_idx * seconds_per_encoder_token
    audio_duration_sec = (total_encoder_frames * 160 * 2) / sample_rate

    time_ratio = token_time_sec / max(audio_duration_sec, 1e-6)
    time_ratio = min(time_ratio, 1.0)

    chars = [ch for ch in transcript if ch.strip()]
    char_count = len(chars)

    if char_count == 0:
        return {
            'aligned_char': '',
            'aligned_context': '',
            'time_sec': round(token_time_sec, 3),
            'time_ratio': round(time_ratio, 3),
            'align_method': 'time_ratio',
        }

    char_idx = min(int(time_ratio * char_count), char_count - 1)
    window_start = max(0, char_idx - 1)
    window_end = min(char_count, char_idx + 2)

    return {
        'aligned_char': chars[char_idx],
        'aligned_context': ''.join(chars[window_start:window_end]),
        'time_sec': round(token_time_sec, 3),
        'time_ratio': round(time_ratio, 3),
        'align_method': 'time_ratio',
    }


def align_token_by_whisper_timestamp(
    token_idx: int,
    whisper_model,
    processor,
    audio_array,
    transcript: str,
    device: str,
    sample_rate: int = 16000,
) -> dict:
    """
    方法二：使用 Whisper word-level 时间戳精确对齐 token 到具体字符。

    思路：
    1. 调用 whisper_model.generate() 带 return_token_timestamps=True
    2. 得到每个 decode token 的起始时间
    3. 计算 encoder token_idx 对应的时间点
    4. 找时间上最近的 decode token，取其对应文字及上下文
    """
    seconds_per_encoder_token = (160 * 2) / sample_rate
    token_time_sec = token_idx * seconds_per_encoder_token

    try:
        input_features = processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            outputs = whisper_model.generate(
                input_features,
                return_timestamps=True,
                return_token_timestamps=True,
                language="zh",
            )

        token_timestamps = outputs.token_timestamps[0].cpu().tolist()
        token_ids = outputs.sequences[0].cpu().tolist()

        # 过滤特殊 token，只保留文字 token
        text_token_pairs = []
        for tid, ttime in zip(token_ids, token_timestamps):
            decoded = processor.tokenizer.decode([tid])
            if decoded.startswith('<|') and decoded.endswith('|>'):
                continue
            if decoded.strip():
                text_token_pairs.append((decoded.strip(), ttime))

        if not text_token_pairs:
            raise ValueError("No text tokens found in timestamp output")

        # 找时间上最近的 decode token
        best_idx, (best_text, best_time) = min(
            enumerate(text_token_pairs),
            key=lambda pair: abs(pair[1][1] - token_time_sec),
        )
        context_tokens = text_token_pairs[max(0, best_idx - 1): best_idx + 2]

        return {
            'aligned_char': best_text,
            'aligned_context': ''.join(t for t, _ in context_tokens),
            'time_sec': round(token_time_sec, 3),
            'closest_timestamp': round(best_time, 3),
            'align_method': 'whisper_timestamp',
        }

    except Exception as alignment_error:
        # 降级到 time_ratio 方法
        fallback = align_token_by_time_ratio(
            token_idx=token_idx,
            total_encoder_frames=1500,
            audio_array=audio_array,
            transcript=transcript,
            sample_rate=sample_rate,
        )
        fallback['align_method'] = f'time_ratio_fallback({alignment_error})'
        return fallback


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
    
    # 使用 checkpoint 中的 topk 配置，除非用户显式指定
    topk = args.topk if args.topk != DEFAULT_CONFIG['topk'] else sae_config.topk
    print(f"Using topk={topk} (from {'args' if args.topk != DEFAULT_CONFIG['topk'] else 'checkpoint'})")
    
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
            sparse_topk, _, _ = sae._get_topk(sparse, topk)
        
        # 提取样本特征
        sample_features = feature_extractor.extract_features(
            audio=audio_array,
            transcript=transcript,
            sample_id=str(idx),
        )
        
        total_encoder_frames = hidden_states.shape[1]

        # whisper_timestamp 模式下，每条样本只做一次 generate，结果缓存复用
        whisper_timestamp_cache = None
        if args.token_align == 'whisper_timestamp':
            try:
                with torch.no_grad():
                    ts_outputs = whisper_model.generate(
                        input_features,
                        return_timestamps=True,
                        return_token_timestamps=True,
                        language="zh",
                    )
                token_timestamps = ts_outputs.token_timestamps[0].cpu().tolist()
                token_ids = ts_outputs.sequences[0].cpu().tolist()
                text_token_pairs = []
                for tid, ttime in zip(token_ids, token_timestamps):
                    decoded = processor.tokenizer.decode([tid])
                    if decoded.startswith('<|') and decoded.endswith('|>'):
                        continue
                    if decoded.strip():
                        text_token_pairs.append((decoded.strip(), ttime))
                whisper_timestamp_cache = text_token_pairs
            except Exception:
                whisper_timestamp_cache = None  # 降级到 time_ratio

        # 记录每个神经元的激活
        for nid in top_10_neurons:
            activations = sparse_topk[0, :, nid].cpu().numpy()
            for token_idx, act_val in enumerate(activations):
                if act_val > 0:
                    # 计算 token 对齐信息
                    if args.token_align == 'none':
                        alignment = {}
                    elif args.token_align == 'whisper_timestamp' and whisper_timestamp_cache is not None:
                        seconds_per_token = (160 * 2) / 16000
                        token_time_sec = token_idx * seconds_per_token
                        best_idx, (best_text, best_time) = min(
                            enumerate(whisper_timestamp_cache),
                            key=lambda pair: abs(pair[1][1] - token_time_sec),
                        )
                        context = whisper_timestamp_cache[max(0, best_idx - 1): best_idx + 2]
                        alignment = {
                            'aligned_char': best_text,
                            'aligned_context': ''.join(t for t, _ in context),
                            'time_sec': round(token_time_sec, 3),
                            'closest_timestamp': round(best_time, 3),
                            'align_method': 'whisper_timestamp',
                        }
                    else:
                        # time_ratio 或 whisper_timestamp 降级
                        alignment = align_token_by_time_ratio(
                            token_idx=token_idx,
                            total_encoder_frames=total_encoder_frames,
                            audio_array=audio_array,
                            transcript=transcript,
                        )

                    neuron_token_activations[nid].append({
                        'sample_idx': idx,
                        'token_idx': token_idx,
                        'activation': float(act_val),
                        'transcript': transcript,
                        'alignment': alignment,
                        'acoustic_features': sample_features.acoustic.to_dict() if sample_features.acoustic else {},
                        'linguistic_features': sample_features.linguistic.to_dict() if sample_features.linguistic else {},
                    })
    
    # 对每个神经元，取激活值最大的前 top_percent token
    neuron_top_tokens = {}
    top_percent = getattr(args, 'top_percent', 1.0)
    
    for nid in top_10_neurons:
        tokens = neuron_token_activations[nid]
        tokens.sort(key=lambda x: x['activation'], reverse=True)
        top_count = max(1, int(len(tokens) * top_percent))
        neuron_top_tokens[nid] = tokens[:top_count]
        print(f"Neuron {nid}: {len(tokens)} total tokens, top {top_percent*100:.0f}% = {top_count} tokens")
    
    # 保存结果
    output_dir = args.output_dir or os.path.dirname(args.key_neurons_path)
    os.makedirs(output_dir, exist_ok=True)
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

        # 收集特征统计：前 30 条整句用于声学/语言特征，前 500 条做 char 频率统计
        acoustic_stats = defaultdict(list)
        linguistic_stats = defaultdict(list)

        for t in tokens[:30]:
            acoustic = t.get('acoustic_features', {})
            linguistic = t.get('linguistic_features', {})

            for k, v in acoustic.items():
                if isinstance(v, (int, float)):
                    acoustic_stats[k].append(v)

            for k, v in linguistic.items():
                if isinstance(v, (int, float, bool)):
                    linguistic_stats[k].append(v)

        # 构建 char 频率统计（前 500 条，取 freq 前 10）
        char_stats = defaultdict(lambda: {'count': 0, 'activation_sum': 0.0, 'contexts': []})
        for t in tokens[:500]:
            alignment = t.get('alignment', {})
            char = alignment.get('aligned_char', '')
            context = alignment.get('aligned_context', '')
            if not char:
                continue
            char_stats[char]['count'] += 1
            char_stats[char]['activation_sum'] += t.get('activation', 0.0)
            if context and context not in char_stats[char]['contexts'] and len(char_stats[char]['contexts']) < 5:
                char_stats[char]['contexts'].append(context)

        sorted_chars = sorted(char_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        char_distribution = {}
        for char, stat in sorted_chars[:10]:
            char_distribution[char] = {
                'count': stat['count'],
                'mean_activation': round(stat['activation_sum'] / stat['count'], 4),
                'contexts': stat['contexts'],
            }

        # 构建 top 30 token 证据列表（以 char 为标识，保留 aligned_context 和整句）
        top_tokens_evidence = []
        for t in tokens[:30]:
            alignment = t.get('alignment', {})
            top_tokens_evidence.append({
                'aligned_char': alignment.get('aligned_char', ''),
                'aligned_context': alignment.get('aligned_context', ''),
                'transcript': t.get('transcript', ''),
                'activation': t.get('activation', 0.0),
            })

        total_token_count = len(tokens)

        # 生成 prompt
        prompt = generate_neuron_explanation_prompt(
            nid,
            top_tokens_evidence,
            char_distribution,
            total_token_count,
            acoustic_stats,
            linguistic_stats,
        )

        prompts[nid] = prompt
        
        print(f"\n{'='*50}")
        print(f"Neuron {nid} Prompt:")
        print('='*50)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # 保存 prompts
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.features_path)
    os.makedirs(output_dir, exist_ok=True)
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

# ============================================================
# Step 7 辅助函数：级联 CER 评估（Whisper + MT5 → 普通话 CER）
# ============================================================

def evaluate_cascade_cer(
    whisper_model,
    processor,
    translation_model_path: str,
    eval_dataset_path: str,
    device: str,
    label: str = "Model",
) -> dict:
    """
    用 Whisper + MT5 翻译模型对 test 集做推理，计算：
    - 整体 CER（所有样本）
    - 后 10% CER（CER 最高的 10% 样本，即最差样本）

    Args:
        whisper_model: 已加载的 WhisperForConditionalGeneration 模型
        processor: WhisperProcessor
        translation_model_path: MT5 翻译模型路径
        eval_dataset_path: 评估数据集路径（需含 text_cn 字段）
        device: 推理设备
        label: 打印时的标签名称

    Returns:
        dict 包含 overall_cer, worst10_cer, per_sample_cer
    """
    from transformers import MT5ForConditionalGeneration, MT5Tokenizer
    from datasets import load_from_disk, Audio
    import evaluate as hf_evaluate

    print(f"\n{'='*60}")
    print(f"CER 评估: {label}")
    print(f"{'='*60}")

    # 加载评估数据集
    eval_ds = load_from_disk(eval_dataset_path)
    test_ds = eval_ds["test"].cast_column("audio", Audio(sampling_rate=16000))

    if "text_cn" not in test_ds.column_names:
        print(f"  ⚠️  数据集缺少 text_cn 字段，跳过 CER 评估")
        return {}

    print(f"  测试样本数: {len(test_ds)}")

    # 加载 MT5 翻译模型
    trans_model_path = translation_model_path
    final_model_subdir = os.path.join(trans_model_path, "final_model")
    if not os.path.exists(os.path.join(trans_model_path, "spiece.model")) and \
       os.path.exists(os.path.join(final_model_subdir, "spiece.model")):
        trans_model_path = final_model_subdir

    print(f"  加载 MT5 翻译模型: {trans_model_path}")
    trans_tokenizer = MT5Tokenizer.from_pretrained(trans_model_path)
    trans_model = MT5ForConditionalGeneration.from_pretrained(trans_model_path)
    trans_model.to(device)
    trans_model.eval()

    wer_metric = hf_evaluate.load("wer")

    def compute_cer_single(prediction: str, reference: str) -> float:
        pred_chars = " ".join(list(prediction.replace(" ", "")))
        ref_chars = " ".join(list(reference.replace(" ", "")))
        if not ref_chars.strip():
            return 0.0
        return 100 * wer_metric.compute(predictions=[pred_chars], references=[ref_chars])

    def translate_to_mandarin(shanghai_text: str) -> str:
        """用 MT5 把上海话转写翻译成普通话"""
        inputs = trans_tokenizer(
            shanghai_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = trans_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

    per_sample_cer = []

    for sample in tqdm(test_ds, desc=f"  推理 ({label})"):
        audio_array = sample["audio"]["array"]
        mandarin_ref = sample["text_cn"]

        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                input_features,
                language="zh",
                task="transcribe",
            )
        shanghai_pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        mandarin_pred = translate_to_mandarin(shanghai_pred)
        cer = compute_cer_single(mandarin_pred, mandarin_ref)
        per_sample_cer.append(cer)

    # 整体 CER（所有样本的平均）
    overall_cer = float(np.mean(per_sample_cer))

    # 后 10% CER（最差样本）
    sorted_cer = sorted(per_sample_cer, reverse=True)
    worst10_count = max(1, int(len(sorted_cer) * 0.1))
    worst10_cer = float(np.mean(sorted_cer[:worst10_count]))

    print(f"  整体 CER:       {overall_cer:.2f}%  (n={len(per_sample_cer)})")
    print(f"  后 10% CER:     {worst10_cer:.2f}%  (n={worst10_count}, 最差样本)")

    # 释放翻译模型显存
    del trans_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "overall_cer": overall_cer,
        "worst10_cer": worst10_cer,
        "per_sample_cer": per_sample_cer,
    }

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
    

    # ── Baseline 评估（微调前）──────────────────────────────────────────────
    baseline_cer_results = {}
    if args.translation_model and os.path.exists(args.eval_dataset_path):
        print("\n[Baseline] 评估微调前的后 10% CER...")
        baseline_cer_results = evaluate_cascade_cer(
            whisper_model=whisper_model,
            processor=processor,
            translation_model_path=args.translation_model,
            eval_dataset_path=args.eval_dataset_path,
            device=device,
            label="Baseline（微调前）",
        )
    else:
        print("\n[Baseline] 跳过 CER 评估（未指定 --translation_model 或 --eval_dataset_path）")

    # 开始训练
    print("Starting training...")
    trainer.train()
    

    # ── 训练后评估 ──────────────────────────────────────────────────────────
    post_cer_results = {}
    if args.translation_model and os.path.exists(args.eval_dataset_path):
        print("\n[训练后] 评估微调后的 CER...")
        post_cer_results = evaluate_cascade_cer(
            whisper_model=whisper_model,
            processor=processor,
            translation_model_path=args.translation_model,
            eval_dataset_path=args.eval_dataset_path,
            device=device,
            label="微调后（Gated LoRA）",
        )

        # 打印对比摘要
        print("\n" + "="*60)
        print("CER 评估对比摘要")
        print("="*60)
        if baseline_cer_results:
            print(f"  Baseline 后 10% CER:  {baseline_cer_results.get('worst10_cer', 'N/A'):.2f}%")
        if post_cer_results:
            print(f"  微调后 整体 CER:      {post_cer_results.get('overall_cer', 'N/A'):.2f}%")
            print(f"  微调后 后 10% CER:    {post_cer_results.get('worst10_cer', 'N/A'):.2f}%")
        if baseline_cer_results and post_cer_results:
            delta = post_cer_results.get('worst10_cer', 0) - baseline_cer_results.get('worst10_cer', 0)
            print(f"  后 10% CER 变化:      {delta:+.2f}%  ({'↑ 变差' if delta > 0 else '↓ 改善'})")
        print("="*60)

        # 保存评估结果到文件
        cer_report_path = os.path.join(output_dir, 'cer_evaluation.json')
        with open(cer_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'baseline': {k: v for k, v in baseline_cer_results.items() if k != 'per_sample_cer'},
                'post_finetune': {k: v for k, v in post_cer_results.items() if k != 'per_sample_cer'},
            }, f, ensure_ascii=False, indent=2)
        print(f"  评估结果已保存到: {cer_report_path}")

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
    p5.add_argument('--top_percent', type=float, default=1.0, help='Top percent of tokens to keep (0.1=10%%, 1.0=100%%)')
    p5.add_argument('--token_align', choices=['none', 'time_ratio', 'whisper_timestamp'], default='time_ratio',
                    help='Method to align encoder token to transcript characters: '
                         'none=no alignment, '
                         'time_ratio=estimate by time position ratio, '
                         'whisper_timestamp=use Whisper word-level timestamps (slower but accurate)')
    p5.add_argument('--output_dir', default=None)
    p5.add_argument('--device', default=DEFAULT_CONFIG['device'])
    
    # generate_prompt
    p6 = subparsers.add_parser('generate_prompt', help='Step 6: Generate LLM prompts')
    p6.add_argument('--features_path', required=True)
    p6.add_argument('--output_dir', default=None, help='Output directory for prompts')
    
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
    p7.add_argument('--translation_model', default='/mnt/workspace/workgroup/qq/ts/whisper/exp/translation-mt5-small-260320-231422/final_model',
                    help='MT5 翻译模型路径（上海话→普通话），用于训练后 CER 评估')
    p7.add_argument('--eval_dataset_path', default='dataset/shanghai/shanghai_unified_dataset',
                    help='用于 CER 评估的数据集路径（需含 text_cn 字段）')
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
