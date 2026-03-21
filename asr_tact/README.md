# ASR-TACT: Targeted Activation Concept Tuning for ASR

基于 SAE (Sparse Autoencoder) 的可解释 ASR 微调框架，参考 TaCT 项目实现，适配 Whisper 语音识别模型。

## 核心思想

1. **SAE 稀疏化**: 将 Whisper encoder 的隐层表征投射到高维稀疏空间，每个神经元对应一个可解释的语义概念
2. **神经元语义标注**: 通过分析神经元激活与样本特征的关联，用 LLM 给每个神经元赋予人类可理解的概念
3. **反事实优化**: 对于高错误率样本，找到改变哪些神经元最能提升性能
4. **门控 LoRA**: 将关键神经元作为门控，高度激活时才启用 LoRA，实现精准可解释的微调

## 项目结构

```
asr_tact/
├── __init__.py           # 模块初始化
├── sae.py                # SAE 模型实现
├── feature_extractor.py  # 音频特征提取器
├── neuron_analyzer.py    # 神经元分析器
├── gated_lora.py         # 门控 LoRA 模块
├── train_sae.py          # SAE 训练脚本
├── analyze_neurons.py    # 神经元分析脚本
├── config.yaml           # 配置文件
└── README.md             # 说明文档

run_asr_tact.py           # 主运行脚本
```

## 快速开始

### 1. 训练 SAE

```bash
python run_asr_tact.py train_sae --config asr_tact/config.yaml
```

主要参数:
- `--model_name`: Whisper 模型名称 (默认: openai/whisper-medium)
- `--dataset_path`: 数据集路径
- `--latent_dim`: SAE 稀疏维度 (默认: 8192)
- `--topk`: TopK 稀疏激活数量 (默认: 64)
- `--encoder_layer`: 提取隐状态的 encoder 层 (默认: 12)

### 2. 分析神经元

```bash
python run_asr_tact.py analyze \
    --sae_checkpoint exp/asr_tact/sae/best_model.pt \
    --max_samples 1000
```

分析结果包括:
- `neuron_stats.json`: 神经元激活统计
- `neuron_correlations.json`: 神经元与特征的关联
- `llm_prompts/`: LLM 语义标注 prompt

### 3. 训练门控 LoRA

```bash
python run_asr_tact.py train_gated_lora \
    --sae_checkpoint exp/asr_tact/sae/best_model.pt \
    --analysis_dir exp/asr_tact/neuron_analysis/analysis-xxx
```

### 4. 推理

```bash
python run_asr_tact.py inference \
    --model_path exp/asr_tact/gated_lora-xxx \
    --audio_path test.wav
```

## 特征维度

### 声学特征 (Acoustic)
- 信噪比 (SNR): low/medium/high
- 语速: slow/normal/fast
- 音高轮廓: rising/falling/flat/varied
- 能量分布、静音比例

### 语言特征 (Linguistic)
- 领域: general/medical/legal/tech/finance
- 句型: declarative/interrogative/imperative/exclamatory
- 数字/英文/标点检测
- 方言标记 (上海话特征词)

### 错误模式 (Error Pattern)
- 同音字混淆
- 边界错误 (分词)
- OOV 词汇
- CER/WER 统计

## LLM 神经元标注

分析完成后，会在 `llm_prompts/` 目录生成每个神经元的标注 prompt，格式如下:

```markdown
# ASR 模型神经元语义分析

## 神经元 #1234 统计信息
- 激活率: 15.2%
- 平均激活值: 0.8234

### 声学特征关联
- 信噪比关联: {"low": 0.65, "medium": 0.20, "high": 0.15}
- 语速关联: {"fast": 0.55, "normal": 0.30, "slow": 0.15}

### 高激活样本示例
**样本 1**
- 真实文本: 侬好，阿拉上海人
- 预测文本: 你好，阿拉上海人
- CER: 16.67%
- 方言标记: ['侬', '阿拉']

## 请回答
1. 概念名称: (如"上海话声调检测器")
2. 置信度: 高/中/低
3. 解释: ...
```

## 配置说明

详见 `asr_tact/config.yaml`，主要配置项:

```yaml
# SAE 配置
sae:
  latent_dim: 8192      # 神经元数量
  topk: 64              # 稀疏度
  encoder_layer: 12     # 提取层

# 门控 LoRA 配置
gated_lora:
  lora_rank: 8
  lora_alpha: 16.0
  gate_threshold: 0.5
  num_gate_neurons: 64
```

## 参考

- TaCT 项目: 气象预测可解释性框架
- Whisper: OpenAI 语音识别模型
- SAE: Sparse Autoencoder 稀疏自编码器
- LoRA: Low-Rank Adaptation 低秩适配
