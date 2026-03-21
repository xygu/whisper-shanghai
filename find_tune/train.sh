#!/bin/bash
# Whisper 微调训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 指定使用的 GPU，多卡可设置为 0,1,2,3

# 训练参数（可根据需要修改）
MODEL_NAME="openai/whisper-small"  # 模型大小: tiny, base, small, medium, large
OUTPUT_DIR="./whisper-finetuned-shanghai"
NUM_EPOCHS=10
BATCH_SIZE=8
GRADIENT_ACCUMULATION=2
LEARNING_RATE=1e-5

echo "Starting Whisper fine-tuning..."
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo ""

# 运行训练脚本
python find_tune/train_whisper.py

echo ""
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
