# 上海话到普通话翻译模块

## 概述

本模块实现了上海话语音到普通话文本的级联翻译系统，采用 **Whisper ASR + mT5 翻译** 的两阶段架构。

```
┌─────────────────────────────────────────────────────────────────┐
│                      级联翻译系统架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   上海话语音 ──► Whisper ASR ──► 上海话文本 ──► mT5 ──► 普通话文本  │
│      .wav         (微调)           "侬好"      (微调)    "你好"   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 方案选择

### 为什么选择级联方案（Cascade）而非端到端（End-to-End）？

| 特性 | 级联方案 | 端到端方案 |
|------|----------|------------|
| **模块独立性** | ✅ ASR 和翻译可独立优化 | ❌ 需要联合训练 |
| **数据需求** | ✅ 可分别使用 ASR 和翻译数据 | ❌ 需要三元组数据 |
| **灵活性** | ✅ 可选择是否启用翻译 | ❌ 固定输出 |
| **可解释性** | ✅ 可查看中间结果 | ❌ 黑盒 |
| **错误传播** | ❌ ASR 错误会传播 | ✅ 可能更鲁棒 |
| **延迟** | ❌ 两阶段推理 | ✅ 单阶段推理 |

**结论**：考虑到现有数据结构（TXT 和 TXT_CN 平行语料）和灵活性需求，级联方案更适合当前场景。

## 模型架构

### 1. ASR 模块：Whisper

- **基座模型**: `openai/whisper-medium`（或其他大小）
- **任务**: 上海话语音 → 上海话文本
- **训练数据**: WAV 音频 + TXT 上海话标注
- **输出**: 上海话文本

### 2. 翻译模块：mT5

- **基座模型**: `google/mt5-small`
- **任务**: 上海话文本 → 普通话文本
- **训练数据**: TXT（上海话）+ TXT_CN（普通话）平行语料
- **输入格式**: `翻译上海话到普通话: {上海话文本}`
- **输出**: 普通话文本

#### 为什么选择 mT5？

1. **多语言支持**: mT5 在 101 种语言上预训练，包括中文
2. **Seq2Seq 架构**: 天然适合翻译任务
3. **模型大小可选**: small (300M) / base (580M) / large (1.2B)
4. **中文方言理解**: 预训练数据包含多种中文变体

## 数据格式

### 平行语料格式

**TXT（上海话）**:
```
[3.080,7.060]   G0003   male    阿拉两个人来聊聊金融方面呃
[22.598,27.970] G0003   male    搿呃，阿姨喃，应该讲，侬已经交关年数辣辣了解了
```

**TXT_CN（普通话）**:
```
[3.080,7.060] G0003 male 我们两个人来聊聊金融方面的
[22.598,27.970] G0003 male 这个的，阿姨呢，应该说，您已经很多年都在了解这方面了
```

### 数据处理流程

```
TXT/ + TXT_CN/
      │
      ▼
load_data_translation.py  ──► translation_data.jsonl
      │
      ▼
translation_dataset/
├── train/
└── test/
```

## 使用方法

### 1. 准备翻译数据

```bash
python find_tune/load_data_translation.py \
    --root_dir dataset/shanghai/ \
    --output_dir dataset/shanghai/translation_dataset \
    --test_size 0.1
```

### 2. 训练翻译模型

```bash
python find_tune/train_translation.py \
    --model_name google/mt5-small \
    --dataset_path dataset/shanghai/translation_dataset \
    --num_train_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5
```

### 3. 推理（仅 ASR）

```bash
python find_tune/inference_shanghai.py \
    --audio_file test.wav \
    --model_path ./whisper-shanghai-finetuned
```

### 4. 推理（ASR + 翻译）

```bash
python find_tune/inference_shanghai.py \
    --audio_file test.wav \
    --model_path ./whisper-shanghai-finetuned \
    --translate \
    --translation_model_path ./exp/translation-mt5-small/final_model
```

## 代码结构

```
find_tune/
├── load_data_translation.py   # 翻译数据加载
├── train_translation.py       # 翻译模型训练
├── inference_shanghai.py      # 推理（支持可选翻译）
└── README_TRANSLATION.md      # 本文档
```

## 训练参数

### 翻译模型默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `google/mt5-small` | 预训练模型 |
| `num_train_epochs` | 10 | 训练轮数 |
| `batch_size` | 8 | 批次大小 |
| `learning_rate` | 5e-5 | 学习率 |
| `max_source_length` | 128 | 源文本最大长度 |
| `max_target_length` | 128 | 目标文本最大长度 |
| `warmup_steps` | 200 | 预热步数 |
| `gradient_accumulation_steps` | 2 | 梯度累积 |

## 评估指标

- **BLEU**: 衡量翻译质量的标准指标
- **ROUGE-1/2/L**: 衡量文本相似度

## API 使用

### Python API

```python
from find_tune.inference_shanghai import transcribe_audio, ShanghaiToMandarinTranslator

# 方式 1: 完整流程（ASR + 翻译）
result = transcribe_audio(
    audio_file="test.wav",
    model_path="./whisper-shanghai-finetuned",
    enable_translation=True,
    translation_model_path="./exp/translation-mt5-small/final_model"
)
print(f"上海话: {result['shanghai_text']}")
print(f"普通话: {result['mandarin_text']}")

# 方式 2: 仅翻译（已有上海话文本）
translator = ShanghaiToMandarinTranslator(
    model_path="./exp/translation-mt5-small/final_model"
)
mandarin = translator.translate("侬好，阿拉是上海人")
print(f"普通话: {mandarin}")
```

## 目录结构

```
dataset/shanghai/
├── WAV/                        # 音频文件
├── TXT/                        # 上海话标注
├── TXT_CN/                     # 普通话翻译
├── shanghai_dataset/           # ASR 数据集
├── translation_dataset/        # 翻译数据集
│   ├── train/
│   └── test/
└── translation_data.jsonl      # 平行语料 JSONL

exp/
├── whisper-shanghai-{timestamp}/       # ASR 模型
└── translation-mt5-small-{timestamp}/  # 翻译模型
    └── final_model/
```

## 其他可选方案

### 方案 A: 端到端 Whisper 翻译

直接微调 Whisper 的 `translate` 任务，输入上海话语音，输出普通话文本。

**优点**: 单模型，延迟低
**缺点**: 需要 (音频, 普通话文本) 数据对，无法查看中间结果

### 方案 B: 使用 LLM 翻译

使用大语言模型（如 Qwen、ChatGLM）进行上海话到普通话翻译。

**优点**: 无需训练，零样本能力强
**缺点**: 推理成本高，可能不理解方言特有表达

### 方案 C: 联合训练

将 Whisper encoder 输出直接接入翻译 decoder，进行联合训练。

**优点**: 可能减少错误传播
**缺点**: 实现复杂，需要更多工程工作

## 注意事项

1. **数据对齐**: TXT 和 TXT_CN 通过时间戳对齐，确保平行语料正确匹配
2. **过滤规则**: 自动过滤 `[ENS]`、`[NOISE]` 等特殊标记，以及源目标完全相同的句子
3. **GPU 内存**: mT5-small 约需 2GB 显存，mT5-base 约需 4GB
4. **推理速度**: 翻译模块使用 beam search (num_beams=4)，可调整以平衡质量和速度
