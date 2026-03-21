# Whisper 上海方言微调实验记录

**实验编号**: whisper-small-shanghai-lr1e-05-bs2-260315-210012  
**实验日期**: 2026年3月15日 21:00 - 2026年3月16日 02:32  
**实验人员**: 范钦  
**实验目标**: 使用上海方言数据集微调 Whisper 模型，实现方言到普通话的语音识别转录

---

## 1. 数据集

### 1.1 数据来源
- **数据集名称**: 上海方言语音数据集
- **数据集路径**: `dataset/shanghai/`
- **数据组成**:
  - WAV 音频文件（16kHz 采样率）
  - TXT 文件：上海方言文本标注（带时间戳）
  - TXT_CN 文件：普通话翻译文本（带时间戳）

### 1.2 数据格式
每个样本包含以下信息：
```
[start_time, end_time] speaker_id gender text
示例: [3.080, 7.060] G0003 male 我们两个人来聊聊金融方面的
```

### 1.3 数据预处理流程

#### 步骤1: 数据解析与切片 (`make_data_shanghai.py`)
- 从 WAV、TXT_CN 文件中提取音频片段和对应文本
- 使用**普通话翻译**作为标注（`use_chinese=True`）
- 过滤条件：
  - 移除特殊标记 `[ENS]`、`[NOISE]`
  - 过滤时长 < 0.5秒的片段
- 输出格式：JSONL 文件 (`shanghai_hf_data.jsonl`)

每条记录包含：
```json
{
  "audio_path": "dataset/shanghai/WAV/xxx.wav",
  "start": 3.080,
  "end": 7.060,
  "speaker_id": "G0003",
  "gender": "male",
  "text": "我们两个人来聊聊金融方面的"
}
```

#### 步骤2: 数据集构建 (`load_data_shanghai.py`)
- 使用 HuggingFace Datasets 加载 JSONL
- 音频处理：
  - 使用 librosa 加载音频片段
  - 重采样到 16kHz
  - 根据时间戳切片（offset + duration）
- 数据划分：
  - 训练集：80%
  - 测试集：20%
  - 随机种子：42
- 并行处理：4个进程 (`num_proc=4`)

### 1.4 数据统计
根据日志推断（具体数值未在日志中显示）：
- **总样本数**: 约 3,090 条（根据 1545 步 × 2 batch × 1 GPU 推算）
- **训练集**: 约 2,472 条
- **测试集**: 约 618 条
- **音频时长**: 每个片段 0.5-10 秒不等

---

## 2. 实验算法框架

### 2.1 模型架构
- **基础模型**: OpenAI Whisper-Medium
  - 注意：实验名称为 "small" 但实际使用的是 "medium" 模型
- **模型来源**: `openai/whisper-medium`
- **参数量**: 763,857,920 (约 7.64 亿参数)

#### 模型结构详情：
```
Encoder:
  - Layers: 24
  - Attention Heads: 16
  - FFN Dimension: 4096
  - d_model: 1024

Decoder:
  - Layers: 24
  - Attention Heads: 16
  - FFN Dimension: 4096
  - d_model: 1024

其他配置:
  - Vocabulary Size: 51,865
  - Mel Bins: 80
  - Max Source Positions: 1,500
  - Max Target Positions: 448
```

### 2.2 训练框架
- **框架**: HuggingFace Transformers (v4.52.4)
- **训练器**: `Seq2SeqTrainer`
- **任务类型**: 序列到序列生成（Seq2Seq）
- **任务**: 转录（transcribe）
- **语言**: 中文（Chinese）

### 2.3 评估指标
- **主要指标**: WER (Word Error Rate, 词错误率)
- **计算方式**: 使用 `jiwer` 库
- **优化目标**: 最小化 WER（`metric_for_best_model: wer`, `greater_is_better: false`）

---

## 3. 超参数配置

### 3.1 训练超参数

| 参数类别 | 参数名称 | 参数值 | 说明 |
|---------|---------|--------|------|
| **学习率** | learning_rate | 1e-05 | 初始学习率 |
| | lr_scheduler_type | linear | 线性衰减 |
| | warmup_steps | 500 | 预热步数 |
| **批次大小** | per_device_train_batch_size | 2 | 每设备训练批次 |
| | per_device_eval_batch_size | 2 | 每设备评估批次 |
| | gradient_accumulation_steps | 8 | 梯度累积步数 |
| | **effective_batch_size** | **16** | 有效批次大小 |
| **训练轮数** | num_train_epochs | 15 | 总训练轮数 |
| | max_steps | -1 | 不限制步数 |
| **优化器** | optim | adamw_torch | AdamW 优化器 |
| | adam_beta1 | 0.9 | Adam β1 |
| | adam_beta2 | 0.999 | Adam β2 |
| | adam_epsilon | 1e-08 | Adam ε |
| | weight_decay | 0 | 权重衰减 |
| **梯度** | max_grad_norm | 1.0 | 梯度裁剪 |
| **精度** | fp16 | True | 混合精度训练 |
| | fp16_opt_level | O1 | FP16 优化级别 |

### 3.2 评估与保存策略

| 参数名称 | 参数值 | 说明 |
|---------|--------|------|
| eval_strategy | steps | 按步数评估 |
| eval_steps | 500 | 每 500 步评估一次 |
| save_strategy | steps | 按步数保存 |
| save_steps | 500 | 每 500 步保存一次 |
| save_total_limit | 3 | 最多保存 3 个检查点 |
| load_best_model_at_end | True | 训练结束加载最佳模型 |
| logging_steps | 25 | 每 25 步记录日志 |

### 3.3 数据加载配置

| 参数名称 | 参数值 | 说明 |
|---------|--------|------|
| dataloader_num_workers | 2 | 数据加载线程数 |
| dataloader_pin_memory | True | 固定内存 |
| remove_unused_columns | False | 保留所有列 |

### 3.4 生成配置

| 参数名称 | 参数值 | 说明 |
|---------|--------|------|
| predict_with_generate | True | 使用生成模式预测 |
| generation_max_length | 225 | 最大生成长度 |
| generation_num_beams | None | 不使用束搜索 |
| do_sample | False | 不使用采样 |

### 3.5 硬件配置

| 配置项 | 详情 |
|--------|------|
| **GPU** | 2 × Tesla V100-SXM2-32GB (34GB 显存) |
| **CUDA** | 12.4 |
| **CPU** | 48 核心（22 逻辑核心）|
| **内存** | 188 GB |
| **系统** | Linux 4.19.91 |
| **Python** | 3.9.23 |

---

## 4. 实验结果

### 4.1 训练过程

#### 总体统计
- **总训练步数**: 1,545 步
- **总训练时长**: 5小时17分59秒 (19,079 秒)
- **训练速度**: 2.578 samples/second
- **平均训练 Loss**: 0.3123

#### 评估检查点表现

| 检查点 | Epoch | 步数 | Eval Loss | Eval WER | 评估时长 |
|--------|-------|------|-----------|----------|----------|
| Checkpoint 1 | 4.86 | 500 | 0.5947 | **88.54%** | 593.17s |
| Checkpoint 2 | 9.71 | 1000 | 0.6938 | **85.85%** | 569.38s |
| Checkpoint 3 | 14.57 | 1500 | 0.7868 | **85.73%** | 568.10s |

#### 训练 Loss 变化趋势

| Epoch 范围 | 平均 Loss | 梯度范数范围 |
|-----------|-----------|-------------|
| 5.0 - 6.0 | 0.10 - 0.14 | 165K - 260K |
| 6.0 - 8.0 | 0.03 - 0.09 | 100K - 273K |
| 8.0 - 10.0 | 0.01 - 0.03 | 35K - 238K |
| 10.0 - 12.0 | 0.004 - 0.007 | 15K - 140K |
| 12.0 - 14.0 | 0.001 - 0.004 | 1K - 5K |
| 14.0 - 15.0 | 0.0002 - 0.001 | 674 - 10K |

### 4.2 最终结果

#### 性能指标
```
✓ 训练完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最终 WER:        85.73%
最终 Loss:       0.7868
训练 Loss:       0.0006 (最后一步)
训练轮数:        15 epochs
总步数:          1,545 steps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模型保存位置:    ./whisper-shanghai-finetuned
WandB 报告:      https://app.bandw.top/glegexy-fun/whisper-shanghai-finetuning/runs/qu4y2jjl
```

#### 性能分析图表

**Loss 变化曲线**:
```
训练 Loss: 0.31 → 0.10 → 0.03 → 0.01 → 0.001 → 0.0006 ✓ (持续下降)
评估 Loss: 0.59 → 0.69 → 0.79 ✗ (持续上升)
```

**WER 变化曲线**:
```
Epoch 4.86:  88.54% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 9.71:  85.85% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 14.57: 85.73% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    (改善幅度极小，几乎停滞)
```

### 4.3 问题诊断

#### 🔴 严重问题

1. **WER 过高 (85.73%)**
   - 词错误率 85.73% 意味着识别准确率仅约 14%
   - 正常语音识别模型 WER 应在 10-30% 之间
   - 方言识别任务目标 WER < 20%
   - **结论**: 模型几乎不可用

2. **严重过拟合**
   - 训练 Loss: 0.0006（极低）
   - 评估 Loss: 0.7868（较高）
   - Loss 差距: 1,311 倍
   - 评估 Loss 从 0.59 上升到 0.79（+34%）
   - **结论**: 模型在训练集上过度拟合，泛化能力极差

3. **梯度不稳定**
   - 梯度范数波动范围: 674 - 273,141
   - 波动幅度: 405 倍
   - 训练后期仍有大梯度（10,336）
   - **结论**: 训练过程不稳定

#### ⚠️ 次要问题

4. **WER 改善停滞**
   - Step 500 → 1000: 改善 2.69%
   - Step 1000 → 1500: 改善 0.12%
   - **结论**: 后期训练无效，浪费计算资源

5. **模型选择不当**
   - 实验名称: "whisper-small"
   - 实际使用: "whisper-medium" (7.64亿参数)
   - 数据量: 约 3,000 条样本
   - **结论**: 模型过大，数据量不足以支撑

---

## 5. 实验迭代记录

### 5.1 实验配置变动历程

| 实验编号 | 实验名称 | 关键变动 | WER 表现 | 结论 |
|---------|---------|---------|---------|------|
| 1 | whisper-medium-shanghai-lr1e-05-bs2-260315-210012 | 误用**普通话**语言标签 | ~85-90% | ❌ 标签错误导致效果差 |
| 2 | whisper-medium-shanghai-lr1e-05-bs2-260317-010002 | 改用**上海话**语言标签 | ~83-86% | ✅ 标签修正后明显改善 |
| 3 | whisper-small-shanghai-lr1e-05-bs2-260317-071028 | 换用 **small 模型** | ~88-89% | ❌ 小模型效果反而变差 |

### 5.2 关键发现

1. **语言标签至关重要**：最初误用普通话标签（Mandarin）训练上海话数据，导致模型学习方向错误。改用正确的上海话标签后，WER 从 ~85% 下降到 ~83%，有明显改善。

2. **模型大小并非越小越好**：尝试从 medium 模型切换到 small 模型，期望减少过拟合，但实际 WER 反而上升到 ~88-89%。说明对于方言识别这类复杂任务，模型容量仍然重要。

3. **后续方向**：应继续使用 medium 模型 + 正确的上海话标签，重点优化正则化策略和数据增强。

---

## 6. 启发与改进建议

### 5.1 根本原因分析

#### 🎯 核心问题：数据质量
```
可能的数据问题：
1. 标注准确性：普通话翻译是否准确对应上海方言音频？
2. 数据分布：训练集和测试集是否来自相同分布？
3. 数据量：3,000 条样本对于 7.64 亿参数模型严重不足
4. 音频质量：是否存在噪音、混响等问题？
5. 文本规范：标点符号、数字、英文等是否统一处理？
```

**建议行动**：
- 人工抽查 100 条样本，验证标注质量
- 检查训练集和测试集的说话人、场景分布
- 统计数据集的音频时长、文本长度分布
- 可视化音频波形和频谱，检查质量

### 5.2 立即改进措施

#### 优先级 1: 模型与数据匹配 ⭐⭐⭐⭐⭐

**问题**: 7.64亿参数模型 + 3,000条样本 = 严重过拟合

**解决方案**:
```python
# 方案 A: 使用更小的模型（推荐）
model_name = "openai/whisper-small"  # 244M 参数
# 或
model_name = "openai/whisper-base"   # 74M 参数

# 方案 B: 增加数据量
# 目标: 至少 10,000+ 条样本
```

**预期效果**: WER 降低 20-30%

#### 优先级 2: 增强正则化 ⭐⭐⭐⭐⭐

**问题**: 无正则化导致过拟合

**解决方案**:
```python
# 在 TrainingArguments 中添加
weight_decay = 0.01              # 添加权重衰减
dropout = 0.1                    # 添加 dropout
attention_dropout = 0.1          # 添加注意力 dropout
activation_dropout = 0.1         # 添加激活 dropout
```

**预期效果**: 评估 Loss 下降，WER 降低 10-15%

#### 优先级 3: 优化学习率 ⭐⭐⭐⭐

**问题**: 学习率过小，训练效率低

**解决方案**:
```python
# 当前配置
learning_rate = 1e-5
lr_scheduler_type = "linear"
warmup_steps = 500

# 改进配置
learning_rate = 5e-5             # 提高 5 倍
lr_scheduler_type = "cosine"     # 使用余弦退火
warmup_ratio = 0.1               # 使用比例而非固定步数
```

**预期效果**: 收敛速度提升 2-3 倍

#### 优先级 4: 减少训练轮数 ⭐⭐⭐⭐

**问题**: 15 轮训练过多，后期无效

**解决方案**:
```python
num_train_epochs = 5             # 从 15 减少到 5
# 或使用早停
early_stopping_patience = 3
early_stopping_threshold = 0.01
```

**预期效果**: 节省 60% 训练时间，避免过拟合

#### 优先级 5: 增大批次大小 ⭐⭐⭐

**问题**: 有效批次 16 偏小，训练不稳定

**解决方案**:
```python
# 当前配置
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
# 有效批次 = 2 × 8 × 2 GPU = 32

# 改进配置
per_device_train_batch_size = 4  # 增加到 4
gradient_accumulation_steps = 8  # 保持不变
# 有效批次 = 4 × 8 × 2 GPU = 64
```

**预期效果**: 梯度更稳定，训练更快

#### 优先级 6: 启用数据增强 ⭐⭐⭐

**问题**: 未使用数据增强，泛化能力差

**解决方案**:
```python
# 在模型配置中启用
apply_spec_augment = True        # 启用频谱增强
mask_time_prob = 0.05            # 时间遮蔽概率
mask_feature_prob = 0.05         # 特征遮蔽概率
mask_time_length = 10            # 时间遮蔽长度
mask_feature_length = 10         # 特征遮蔽长度
```

**预期效果**: WER 降低 5-10%

### 5.3 推荐实验配置

#### 配置 A: 快速验证（推荐首选）
```python
# 模型
model_name = "openai/whisper-small"  # 244M 参数

# 训练
num_train_epochs = 5
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 5e-5
lr_scheduler_type = "cosine"
warmup_ratio = 0.1

# 正则化
weight_decay = 0.01
dropout = 0.1

# 早停
early_stopping_patience = 3

# 数据增强
apply_spec_augment = True
```

**预期结果**: WER 30-40%，训练时间 2-3 小时

#### 配置 B: 高性能（数据充足时）
```python
# 模型
model_name = "openai/whisper-base"   # 74M 参数

# 训练
num_train_epochs = 8
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
learning_rate = 3e-5
lr_scheduler_type = "cosine"
warmup_ratio = 0.1

# 正则化
weight_decay = 0.01
dropout = 0.1
attention_dropout = 0.1

# 数据增强
apply_spec_augment = True
mask_time_prob = 0.08
mask_feature_prob = 0.08
```

**预期结果**: WER 20-30%，训练时间 3-4 小时

### 5.4 调试建议

#### 1. 添加详细日志
```python
# 在训练脚本中添加
def compute_metrics(pred):
    # 记录预测样本
    print(f"预测: {pred.predictions[0]}")
    print(f"标签: {pred.label_ids[0]}")
    # 计算 WER
    wer = ...
    return {"wer": wer}
```

#### 2. 可视化训练过程
```python
# 使用 TensorBoard
tensorboard --logdir ./whisper-shanghai-finetuned/runs

# 或使用 WandB（已启用）
# 查看: https://app.bandw.top/glegexy-fun/whisper-shanghai-finetuning
```

#### 3. 分析错误样本
```python
# 在推理脚本中添加
def analyze_errors(model, test_dataset):
    errors = []
    for sample in test_dataset:
        pred = model.transcribe(sample["audio"])
        if wer(pred, sample["text"]) > 0.5:
            errors.append({
                "audio": sample["audio_path"],
                "pred": pred,
                "truth": sample["text"],
                "wer": wer(pred, sample["text"])
            })
    return errors
```

### 5.5 长期改进方向

#### 1. 数据扩充
- 收集更多上海方言数据（目标: 10,000+ 条）
- 使用数据增强技术（速度扰动、音高变换）
- 考虑使用预训练的中文 Whisper 模型

#### 2. 模型优化
- 尝试 LoRA、QLoRA 等参数高效微调方法
- 使用知识蒸馏从大模型迁移到小模型
- 探索多任务学习（同时训练方言识别和普通话翻译）

#### 3. 评估优化
- 添加 CER（字符错误率）作为辅助指标
- 分析不同说话人、场景的性能差异
- 进行人工评估，了解实际可用性

---

## 6. 实验总结

> ⚠️ **2026-03-18 更新说明**：以下 6.1-6.3 节内容基于首次实验（误用普通话标签）得出，部分结论已被后续实验推翻。请参阅 **6.4 节 基于迭代实验的修正** 获取最新结论。

### 6.1 关键发现 ~~（部分已过时）~~

✅ **成功之处**:
1. 完整走通了 Whisper 微调流程
2. 训练过程稳定，无崩溃或错误
3. 模型成功保存，可用于推理
4. 训练 Loss 持续下降，说明模型有学习能力

❌ **失败之处**:
1. **WER 85.73% 过高**，模型几乎不可用
2. **严重过拟合**，训练 Loss 和评估 Loss 严重背离
3. ~~**模型选择不当**，Medium 模型对于 3K 样本过大~~ → **已修正：实际是语言标签错误，medium 模型表现优于 small**
4. **缺乏正则化**，导致泛化能力极差
5. **训练轮数过多**，后期训练无效

### 6.2 经验教训 ~~（部分已过时）~~

#### ~~教训 1: 模型大小要匹配数据量~~ → **已修正**
```
原结论（已推翻）:
数据量 < 5K:     使用 whisper-tiny 或 whisper-base
数据量 5K-20K:   使用 whisper-small
数据量 20K-100K: 使用 whisper-medium
数据量 > 100K:   使用 whisper-large

新结论（基于实验 3）:
对于方言识别任务，模型容量更重要！
whisper-medium (WER ~83%) 优于 whisper-small (WER ~88%)
不要盲目追求小模型来减少过拟合
```

#### 教训 2: 正则化至关重要
```
没有正则化 = 过拟合
必须使用: weight_decay, dropout, 数据增强
```

#### 教训 3: 监控评估指标
```
不要只看训练 Loss！
评估 Loss 上升 = 立即停止训练
```

#### 教训 4: 数据质量第一
```
垃圾数据 + 完美模型 = 垃圾结果
高质量数据 + 简单模型 = 好结果
```

#### 🆕 教训 5: 语言标签必须正确
```
误用普通话标签训练上海话 = 模型学习方向错误
正确的语言标签是微调成功的前提！
```

### 6.3 下一步行动 ~~（部分已过时）~~

#### 立即行动（本周内）:
1. ~~✅ 切换到 whisper-small 模型~~ → **已验证：small 效果更差，应继续使用 medium**
2. ✅ 添加 weight_decay=0.01 和 dropout=0.1
3. ✅ 提高学习率到 5e-5
4. ✅ 减少训练轮数到 5 轮
5. ✅ 增大批次大小到 64

#### 短期行动（本月内）:
1. 🔍 人工检查 100 条样本的标注质量
2. 📊 分析数据分布，确保训练集和测试集一致
3. 🎵 启用数据增强（频谱增强）
4. 📈 添加 CER 指标，全面评估性能

#### 长期行动（下季度）:
1. 📦 收集更多上海方言数据（目标 10,000+ 条）
2. 🔬 尝试 LoRA 等参数高效微调方法
3. 🤝 探索多任务学习（方言识别 + 普通话翻译）
4. 🚀 部署到生产环境，收集真实用户反馈

### 6.4 基于迭代实验的修正（2026-03-18）

根据第 5 节记录的三次实验迭代，对原有结论进行如下修正：

#### 🔄 修正 1: 模型大小选择

| 原结论 | 新结论 |
|--------|--------|
| Medium 模型对于 3K 样本过大，应使用 small/base | **Medium 模型表现优于 small**，方言识别需要足够的模型容量 |
| 小模型可减少过拟合 | 小模型 WER 反而更高（~88% vs ~83%），过拟合问题应通过正则化解决 |

#### 🔄 修正 2: 问题根因

| 原结论 | 新结论 |
|--------|--------|
| WER 高的主因是模型过大 | **WER 高的主因是语言标签错误**（误用普通话标签） |
| 应优先缩小模型 | 应优先确保语言标签正确，然后优化正则化 |

#### ✅ 当前最佳配置

基于三次实验，当前最佳配置为：
- **模型**: `openai/whisper-medium`（而非 small）
- **语言标签**: 上海话（而非普通话）
- **最佳 WER**: ~83-86%（实验 2）

#### 📋 后续优化方向

1. **继续使用 medium 模型**，不再尝试更小的模型
2. **重点优化正则化**：weight_decay、dropout、数据增强
3. **探索更大模型**：如 whisper-large，观察是否有进一步提升
4. **数据扩充**：增加训练数据量以充分发挥大模型优势

---

## 7. 附录

### 7.1 实验环境

```yaml
硬件:
  GPU: 2 × Tesla V100-SXM2-32GB
  CPU: 48 cores
  Memory: 188 GB
  
软件:
  OS: Linux 4.19.91
  Python: 3.9.23
  CUDA: 12.4
  PyTorch: (从 transformers 推断)
  Transformers: 4.52.4
  Datasets: (版本未记录)
  
依赖:
  - transformers >= 4.30.0
  - datasets
  - librosa
  - evaluate
  - jiwer
  - tensorboard
  - wandb
```

### 7.2 文件清单

```
实验相关文件:
├── find_tune/
│   ├── make_data_shanghai.py          # 数据预处理
│   ├── load_data_shanghai.py          # 数据加载
│   ├── train_shanghai_wandb.py        # 训练脚本（本次实验）
│   ├── inference_shanghai.py          # 推理脚本
│   └── README_SHANGHAI.md             # 文档
├── dataset/shanghai/
│   ├── WAV/                           # 音频文件
│   ├── TXT/                           # 方言标注
│   ├── TXT_CN/                        # 普通话标注
│   ├── shanghai_hf_data.jsonl         # 预处理后的数据
│   └── shanghai_dataset/              # HuggingFace 数据集
├── wandb/
│   └── run-20260315_210025-qu4y2jjl/  # 本次实验日志
│       ├── files/
│       │   ├── config.yaml            # 完整配置
│       │   ├── output.log             # 训练日志
│       │   └── wandb-summary.json     # 结果摘要
│       └── logs/
└── whisper-shanghai-finetuned/        # 训练后的模型
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── checkpoint-1500/
```

### 7.3 参考资料

- [Whisper 论文](https://arxiv.org/abs/2212.04356)
- [HuggingFace Whisper 文档](https://huggingface.co/docs/transformers/model_doc/whisper)
- [微调教程](https://huggingface.co/blog/fine-tune-whisper)
- [WandB 实验报告](https://app.bandw.top/glegexy-fun/whisper-shanghai-finetuning/runs/qu4y2jjl)

### 7.4 联系方式

- **实验人员**: 顾心悦
- **邮箱**: glegexy@163.com

---

**文档版本**: v1.0  
**创建日期**: 2026-03-16  
**最后更新**: 2026-03-16
