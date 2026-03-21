# 上海方言 Whisper 微调完整指南

本指南提供了使用上海方言数据集微调 Whisper 模型的完整流程。

## 📁 数据集结构

确保你的数据集结构如下：

```
dataset/shanghai/
├── TXT/        # 上海方言文本标注（带时间戳）
├── TXT_CN/     # 普通话翻译文本（带时间戳）
└── WAV/        # 音频文件
```

每个文件的格式示例：
- **WAV**: `A0002_S0003_0_G0003_G0004.wav`
- **TXT**: `[3.080,7.060] G0003 male 阿拉两个拧来聊聊金融方面呃`
- **TXT_CN**: `[3.080,7.060] G0003 male 我们两个人来聊聊金融方面的`

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r find_tune/requirements_finetune.txt
```

主要依赖：
- transformers >= 4.30.0
- datasets
- librosa
- evaluate
- jiwer
- tensorboard

### 2. 数据准备

运行数据准备脚本：

```bash
bash find_tune/prepare_shanghai.sh
```

或手动执行：

```bash
# 步骤 1: 生成 JSONL 文件
python find_tune/make_data_shanghai.py

# 步骤 2: 创建 HuggingFace 数据集
python find_tune/load_data_shanghai.py
```

这将：
- 从 TXT_CN 和 WAV 文件生成 `shanghai_hf_data.jsonl`
- 创建训练集和测试集（8:2 划分）
- 保存到 `dataset/shanghai/shanghai_dataset/`

### 3. 开始训练

**方式一：使用脚本（推荐）**

```bash
bash find_tune/train_shanghai.sh
```

**方式二：直接运行 Python**

```bash
python find_tune/train_shanghai.py
```

**方式三：自定义参数**

编辑 `find_tune/train_shanghai.py` 中的配置：

```python
# 模型配置
model_name = "openai/whisper-small"  # tiny, base, small, medium, large

# 训练配置
num_train_epochs = 15
per_device_train_batch_size = 8
learning_rate = 1e-5
```

### 4. 使用微调后的模型

**Python 代码：**

```python
from transformers import pipeline

# 加载模型
pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-shanghai-finetuned"
)

# 转录音频
result = pipe("test_audio.wav")
print(result["text"])
```

**命令行：**

```bash
python find_tune/inference_shanghai.py \
    --audio_file test_audio.wav \
    --model_path ./whisper-shanghai-finetuned
```

## 📊 训练监控

### TensorBoard

```bash
tensorboard --logdir ./whisper-shanghai-finetuned/runs
```

然后在浏览器中打开 `http://localhost:6006`

### 评估指标

训练过程中会计算 **WER (Word Error Rate)**：
- WER 越低越好
- 0% = 完美转录
- 目标：WER < 15%（对于方言任务）

## ⚙️ 配置说明

### 模型选择

| 模型 | 参数量 | 显存需求 | 推荐场景 |
|------|--------|----------|----------|
| whisper-tiny | 39M | ~2 GB | 快速测试 |
| whisper-base | 74M | ~3 GB | 轻量部署 |
| whisper-small | 244M | ~6 GB | **推荐** |
| whisper-medium | 769M | ~12 GB | 高精度 |
| whisper-large | 1550M | ~20 GB | 最高精度 |

### 训练参数

```python
# 基础配置
num_train_epochs = 15              # 训练轮数
per_device_train_batch_size = 8   # 批次大小
gradient_accumulation_steps = 2    # 梯度累积
learning_rate = 1e-5               # 学习率

# 评估和保存
eval_steps = 500                   # 每 500 步评估一次
save_steps = 500                   # 每 500 步保存一次
save_total_limit = 3               # 最多保存 3 个检查点
```

### 硬件要求

**最低配置：**
- GPU: GTX 1660 (6GB)
- 模型: whisper-small
- 批次大小: 4

**推荐配置：**
- GPU: RTX 3090 (24GB)
- 模型: whisper-small/medium
- 批次大小: 8-16

## 🔧 常见问题

### 1. CUDA Out of Memory

**解决方案：**
```python
# 减小批次大小
per_device_train_batch_size = 4

# 增加梯度累积
gradient_accumulation_steps = 4

# 使用更小的模型
model_name = "openai/whisper-base"
```

### 2. 训练速度慢

**解决方案：**
```bash
# 确保使用 GPU
export CUDA_VISIBLE_DEVICES=0

# 启用混合精度训练（默认已启用）
fp16 = True

# 减少评估频率
eval_steps = 1000
```

### 3. WER 不下降

**可能原因和解决方案：**
- **数据质量问题**：检查文本标注是否准确
- **训练不足**：增加 `num_train_epochs`
- **学习率不合适**：尝试 `1e-4` 或 `5e-6`
- **模型太小**：使用更大的模型

### 4. 数据集加载失败

**检查：**
```bash
# 确认文件存在
ls dataset/shanghai/TXT/
ls dataset/shanghai/TXT_CN/
ls dataset/shanghai/WAV/

# 检查文件数量是否匹配
ls dataset/shanghai/TXT/ | wc -l
ls dataset/shanghai/TXT_CN/ | wc -l
ls dataset/shanghai/WAV/ | wc -l
```

## 📈 高级用法

### 多 GPU 训练

```bash
# 使用 accelerate
accelerate launch find_tune/train_shanghai.py

# 或使用 torchrun
torchrun --nproc_per_node=4 find_tune/train_shanghai.py
```

### 从检查点恢复训练

在 `train_shanghai.py` 中添加：

```python
training_args = Seq2SeqTrainingArguments(
    ...
    resume_from_checkpoint="./whisper-shanghai-finetuned/checkpoint-1000"
)
```

### 使用上海方言而非普通话标注

修改 `make_data_shanghai.py`：

```python
# 使用上海方言文本
process_shanghai_dataset(use_chinese=False)
```

## 📝 文件说明

```
find_tune/
├── make_data_shanghai.py      # 数据预处理（生成 JSONL）
├── load_data_shanghai.py      # 创建 HuggingFace 数据集
├── train_shanghai.py          # 训练脚本
├── inference_shanghai.py      # 推理脚本
├── prepare_shanghai.sh        # 数据准备脚本
├── train_shanghai.sh          # 训练启动脚本
└── README_SHANGHAI.md         # 本文档
```

## 🎯 预期效果

经过充分训练后（15-20 轮），你应该能看到：
- **训练集 WER**: 5-10%
- **测试集 WER**: 10-20%
- **推理速度**: ~0.5-2 秒/音频片段（取决于硬件）

## 📚 参考资料

- [Whisper 论文](https://arxiv.org/abs/2212.04356)
- [Hugging Face Whisper 文档](https://huggingface.co/docs/transformers/model_doc/whisper)
- [微调教程](https://huggingface.co/blog/fine-tune-whisper)

## 💡 提示

1. **首次训练**：建议先用 `whisper-small` 和少量数据快速验证流程
2. **数据质量**：确保文本标注准确，音频清晰
3. **定期保存**：训练过程中会自动保存检查点
4. **监控训练**：使用 TensorBoard 实时查看训练进度
5. **耐心等待**：完整训练可能需要数小时到数天

祝训练顺利！🎉
