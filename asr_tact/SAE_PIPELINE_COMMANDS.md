# ASR-TACT SAE Pipeline 完整启动命令

基于已有上海话能力的 Whisper 模型，使用 SAE 进行可解释性分析和门控 LoRA 微调。

## 配置参数说明

| 参数 | 配置 B | 配置 C | 配置 D |
|------|--------|--------|--------|
| `encoder_layer` | 12 (中间层) | 8 (浅层) | 18 (深层) |
| `norm_type` | layer_norm | layer_norm | layer_norm |
| `topk_type` | topk | topk | topk |
| `latent_dim` | 15000 | 15000 | 15000 |
| `topk` | 500 | 500 | 500 |
| `num_epochs` | 20 | 20 | 20 |

---

## Step 1: 训练 SAE

### 配置 B（Layer 12，中间层）
```bash
nohup python asr_tact/run_full_pipeline.py train_sae \
    --model_name exp/whisper-shanghai-260318-004259 \
    --latent_dim 15000 \
    --topk 500 \
    --norm_type layer_norm \
    --topk_type topk \
    --encoder_layer 12 \
    --dead_neuron_threshold 1e6 \
    --num_epochs 20 \
    --output_dir ./exp/sae_config_B \
    > nohup_config_B.out 2>&1 &
```

### 配置 C（Layer 8，浅层）
```bash
nohup python asr_tact/run_full_pipeline.py train_sae \
    --model_name exp/whisper-shanghai-260318-004259 \
    --latent_dim 15000 \
    --topk 500 \
    --norm_type layer_norm \
    --topk_type topk \
    --encoder_layer 8 \
    --dead_neuron_threshold 1e6 \
    --num_epochs 20 \
    --output_dir ./exp/sae_config_C \
    > nohup_config_C.out 2>&1 &
```

### 配置 D（Layer 18，深层）
```bash
nohup python asr_tact/run_full_pipeline.py train_sae \
    --model_name exp/whisper-shanghai-260318-004259 \
    --latent_dim 15000 \
    --topk 500 \
    --norm_type layer_norm \
    --topk_type topk \
    --encoder_layer 18 \
    --dead_neuron_threshold 1e6 \
    --num_epochs 20 \
    --output_dir ./exp/sae_config_D \
    > nohup_config_D.out 2>&1 &
```

---

## Step 2: 统计量汇报

计算死亡神经元率、重建率(R²)、稀疏度等统计指标。

```bash
# 配置 B
python asr_tact/run_full_pipeline.py compute_stats \
    --sae_checkpoint ./exp/sae_config_B/sae-layer12-XXXXXX/best_model.pt \
    --model_name exp/whisper-shanghai-260318-004259

# 配置 C
python asr_tact/run_full_pipeline.py compute_stats \
    --sae_checkpoint ./exp/sae_config_C/sae-layer8-XXXXXX/best_model.pt \
    --model_name exp/whisper-shanghai-260318-004259

# 配置 D
python asr_tact/run_full_pipeline.py compute_stats \
    --sae_checkpoint ./exp/sae_config_D/sae-layer18-XXXXXX/best_model.pt \
    --model_name exp/whisper-shanghai-260318-004259
```

---

## Step 3: 找最差 1/10 样本

找出重建误差最大的样本，用于后续反事实分析。

```bash
# 配置 B
python asr_tact/run_full_pipeline.py find_worst \
    --sae_checkpoint ./exp/sae_config_B/sae-layer12-XXXXXX/best_model.pt \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_B

# 配置 C
python asr_tact/run_full_pipeline.py find_worst \
    --sae_checkpoint ./exp/sae_config_C/sae-layer8-XXXXXX/best_model.pt \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_C

# 配置 D
python asr_tact/run_full_pipeline.py find_worst \
    --sae_checkpoint ./exp/sae_config_D/sae-layer18-XXXXXX/best_model.pt \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_D
```

---

## Step 4: 反事实分析

通过梯度优化找到关键神经元（对预测影响最大的神经元）。

```bash
# 配置 B
python asr_tact/run_full_pipeline.py counterfactual \
    --sae_checkpoint ./exp/sae_config_B/sae-layer12-XXXXXX/best_model.pt \
    --worst_samples_path ./exp/sae_config_B/worst_samples.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_B/counterfactual

# 配置 C
python asr_tact/run_full_pipeline.py counterfactual \
    --sae_checkpoint ./exp/sae_config_C/sae-layer8-XXXXXX/best_model.pt \
    --worst_samples_path ./exp/sae_config_C/worst_samples.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_C/counterfactual

# 配置 D
python asr_tact/run_full_pipeline.py counterfactual \
    --sae_checkpoint ./exp/sae_config_D/sae-layer18-XXXXXX/best_model.pt \
    --worst_samples_path ./exp/sae_config_D/worst_samples.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_D/counterfactual
```

---

## Step 5: 提取神经元激活的 Token 特征

提取关键神经元激活时对应的 token 特征，用于 LLM 解释。

```bash
# 配置 B
python asr_tact/run_full_pipeline.py extract_features \
    --sae_checkpoint ./exp/sae_config_B/sae-layer12-XXXXXX/best_model.pt \
    --key_neurons_path ./exp/sae_config_B/counterfactual/key_neurons.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_B/features

# 配置 C
python asr_tact/run_full_pipeline.py extract_features \
    --sae_checkpoint ./exp/sae_config_C/sae-layer8-XXXXXX/best_model.pt \
    --key_neurons_path ./exp/sae_config_C/counterfactual/key_neurons.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_C/features

# 配置 D
python asr_tact/run_full_pipeline.py extract_features \
    --sae_checkpoint ./exp/sae_config_D/sae-layer18-XXXXXX/best_model.pt \
    --key_neurons_path ./exp/sae_config_D/counterfactual/key_neurons.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_D/features
```

---

## Step 6: 生成 LLM 解释 Prompt

生成用于 LLM 解释神经元语义的 prompt。

```bash
# 配置 B
python asr_tact/run_full_pipeline.py generate_prompt \
    --features_path ./exp/sae_config_B/features/neuron_features.json \
    --output_dir ./exp/sae_config_B/llm_prompts

# 配置 C
python asr_tact/run_full_pipeline.py generate_prompt \
    --features_path ./exp/sae_config_C/features/neuron_features.json \
    --output_dir ./exp/sae_config_C/llm_prompts

# 配置 D
python asr_tact/run_full_pipeline.py generate_prompt \
    --features_path ./exp/sae_config_D/features/neuron_features.json \
    --output_dir ./exp/sae_config_D/llm_prompts
```

---

## Step 7: 训练门控 LoRA

使用关键神经元作为门控，训练 LoRA 微调模型。

```bash
# 配置 B
nohup python asr_tact/run_full_pipeline.py train_lora \
    --sae_checkpoint ./exp/sae_config_B/sae-layer12-XXXXXX/best_model.pt \
    --key_neurons_path ./exp/sae_config_B/counterfactual/key_neurons.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_B/gated_lora \
    > nohup_lora_B.out 2>&1 &

# 配置 C
nohup python asr_tact/run_full_pipeline.py train_lora \
    --sae_checkpoint ./exp/sae_config_C/sae-layer8-XXXXXX/best_model.pt \
    --key_neurons_path ./exp/sae_config_C/counterfactual/key_neurons.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_C/gated_lora \
    > nohup_lora_C.out 2>&1 &

# 配置 D
nohup python asr_tact/run_full_pipeline.py train_lora \
    --sae_checkpoint ./exp/sae_config_D/sae-layer18-XXXXXX/best_model.pt \
    --key_neurons_path ./exp/sae_config_D/counterfactual/key_neurons.json \
    --model_name exp/whisper-shanghai-260318-004259 \
    --output_dir ./exp/sae_config_D/gated_lora \
    > nohup_lora_D.out 2>&1 &
```

---

## 快速执行脚本

### 一键执行配置 B 全流程
```bash
#!/bin/bash
CONFIG="B"
SAE_DIR="./exp/sae_config_${CONFIG}"
MODEL="exp/whisper-shanghai-260318-004259"
LAYER=12

# Step 1: Train SAE
python asr_tact/run_full_pipeline.py train_sae \
    --model_name $MODEL \
    --latent_dim 15000 --topk 500 \
    --norm_type layer_norm --topk_type topk \
    --encoder_layer $LAYER \
    --dead_neuron_threshold 1e6 --num_epochs 20 \
    --output_dir $SAE_DIR

# 获取最新的 checkpoint 目录
SAE_CKPT=$(ls -td ${SAE_DIR}/sae-layer*/ | head -1)best_model.pt

# Step 2-6
python asr_tact/run_full_pipeline.py compute_stats --sae_checkpoint $SAE_CKPT --model_name $MODEL
python asr_tact/run_full_pipeline.py find_worst --sae_checkpoint $SAE_CKPT --model_name $MODEL --output_dir $SAE_DIR
python asr_tact/run_full_pipeline.py counterfactual --sae_checkpoint $SAE_CKPT --worst_samples_path ${SAE_DIR}/worst_samples.json --model_name $MODEL --output_dir ${SAE_DIR}/counterfactual
python asr_tact/run_full_pipeline.py extract_features --sae_checkpoint $SAE_CKPT --key_neurons_path ${SAE_DIR}/counterfactual/key_neurons.json --model_name $MODEL --output_dir ${SAE_DIR}/features
python asr_tact/run_full_pipeline.py generate_prompt --features_path ${SAE_DIR}/features/neuron_features.json --output_dir ${SAE_DIR}/llm_prompts

# Step 7: Train LoRA
python asr_tact/run_full_pipeline.py train_lora --sae_checkpoint $SAE_CKPT --key_neurons_path ${SAE_DIR}/counterfactual/key_neurons.json --model_name $MODEL --output_dir ${SAE_DIR}/gated_lora
```

---

## 注意事项

1. **XXXXXX** 需要替换为实际的时间戳目录名（如 `260323-170000`）
2. 每个 Step 依赖前一个 Step 的输出，需按顺序执行
3. Step 1 和 Step 7 是耗时操作，建议使用 `nohup` 后台运行
4. 如需指定 GPU，在命令前加 `CUDA_VISIBLE_DEVICES=0,1`
