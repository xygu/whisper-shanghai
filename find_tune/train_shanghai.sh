#!/bin/bash

# 上海方言 Whisper 微调训练脚本

echo "=========================================="
echo "上海方言 Whisper 微调 - 开始训练"
echo "=========================================="

# 设置 GPU（如果有多个 GPU，可以指定使用哪个）
# export CUDA_VISIBLE_DEVICES=0

# 配置参数
MODEL_SIZE=${MODEL_SIZE:-medium}  # 可选: tiny, base, small, medium, large
echo "模型大小: $MODEL_SIZE"

# 生成时间戳（与 WandB 运行记录保持一致）
TIMESTAMP=$(date +%y%m%d-%H%M%S)
export EXPERIMENT_TIMESTAMP=$TIMESTAMP

# 检查是否使用 WandB 版本
USE_WANDB=${USE_WANDB:-true}

if [ "$USE_WANDB" = "true" ]; then
    echo "使用 WandB 版本进行训练..."
    echo "实验时间戳: $TIMESTAMP"
    python find_tune/train_shanghai_wandb.py --model_size $MODEL_SIZE
else
    echo "使用标准版本进行训练..."
    python find_tune/train_shanghai.py --model_size $MODEL_SIZE
fi

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 训练完成!"
echo "=========================================="
echo ""
echo "查看训练日志:"
if [ "$USE_WANDB" = "true" ]; then
    echo "  WandB: 查看终端输出的链接"
fi
echo "  TensorBoard: tensorboard --logdir ./whisper-shanghai-finetuned/runs"
echo ""
echo "使用模型进行推理:"
echo "  python find_tune/inference_shanghai.py --audio_file your_audio.wav"
