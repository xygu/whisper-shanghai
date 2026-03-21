# Whisper 微调指南

本目录包含使用上海方言数据集微调 Whisper 模型的完整代码。

## 目录结构

```
find_tune/
├── make_data.py              # 从 TXT 和 WAV 生成 JSONL 文件
├── load_data.py              # 创建 HuggingFace 数据集
├── train_whisper.py          # 主训练脚本
├── inference.py              # 推理脚本
├── prepare_data.sh           # 数据准备脚本
├── train.sh                  # 训练启动脚本
└── requirements_finetune.txt # 依赖包列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r find_tune/requirements_finetune.txt
```

### 2. 准备数据

确保你的数据集结构如下：
```
dataset/shanghai/
├── TXT/        # 文本标注文件
├── TXT_CN/     # 中文文本（如果有）
└── WAV/        # 音频文件
```

运行数据准备脚本：
```bash
bash find_tune/prepare_data.sh
```

或者手动执行：
```bash
# 生成 JSONL 文件
python find_tune/make_data.py

# 创建 HuggingFace 数据集
python find_tune/load_data.py
```

### 3. 开始训练

使用默认配置训练：
```bash
bash find_tune/train.sh
```

或者直接运行 Python 脚本：
```bash
python find_tune/train_whisper.py
```

### 4. 使用微调后的模型

```bash
python find_tune/inference.py --audio_file path/to/audio.wav
```

## 训练配置

在 `train_whisper.py` 中可以修改以下参数：

### 模型配置
- `model_name`: 预训练模型大小
  - `openai/whisper-tiny` (39M 参数)
  - `openai/whisper-base` (74M 参数)
  - `openai/whisper-small` (244M 参数，推荐)
  - `openai/whisper-medium` (769M 参数)
  - `openai/whisper-large` (1550M 参数)

### 训练参数
- `num_train_epochs`: 训练轮数（默认: 10）
- `per_device_train_batch_size`: 每个设备的批次大小（默认: 8）
- `gradient_accumulation_steps`: 梯度累积步数（默认: 2）
- `learning_rate`: 学习率（默认: 1e-5）
- `warmup_steps`: 预热步数（默认: 500）

### 硬件要求

| 模型大小 | 显存需求 | 推荐 GPU |
|---------|---------|----------|
| tiny    | ~2 GB   | GTX 1060+ |
| base    | ~3 GB   | GTX 1660+ |
| small   | ~6 GB   | RTX 2060+ |
| medium  | ~12 GB  | RTX 3090+ |
| large   | ~20 GB  | A100 |

## 训练监控

训练过程中会生成 TensorBoard 日志：

```bash
tensorboard --logdir ./whisper-finetuned-shanghai/runs
```

## 评估指标

训练过程中会计算 WER (Word Error Rate，词错误率)：
- WER 越低越好
- 0% 表示完美转录
- 通常目标是 WER < 10%

## 推理示例

### Python 代码
```python
from transformers import pipeline

# 加载微调后的模型
pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-finetuned-shanghai"
)

# 转录音频
result = pipe("audio.wav")
print(result["text"])
```

### 命令行
```bash
python find_tune/inference.py \
    --model_path ./whisper-finetuned-shanghai \
    --audio_file test_audio.wav \
    --language Chinese
```

## 常见问题

### 1. CUDA Out of Memory
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用更小的模型

### 2. 训练速度慢
- 确保使用 GPU: `export CUDA_VISIBLE_DEVICES=0`
- 启用混合精度训练: `fp16=True`
- 减少 `eval_steps` 和 `save_steps`

### 3. WER 不下降
- 增加训练轮数
- 调整学习率
- 检查数据质量
- 使用更大的模型

## 高级用法

### 多 GPU 训练

```bash
# 使用 accelerate
accelerate launch find_tune/train_whisper.py

# 或使用 torchrun
torchrun --nproc_per_node=4 find_tune/train_whisper.py
```

### 从检查点恢复训练

在 `train_whisper.py` 中设置：
```python
training_args = Seq2SeqTrainingArguments(
    ...
    resume_from_checkpoint="./whisper-finetuned-shanghai/checkpoint-1000"
)
```

## 参考资料

- [Whisper 论文](https://arxiv.org/abs/2212.04356)
- [Hugging Face Whisper 文档](https://huggingface.co/docs/transformers/model_doc/whisper)
- [微调教程](https://huggingface.co/blog/fine-tune-whisper)
