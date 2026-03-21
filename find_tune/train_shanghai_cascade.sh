#!/bin/bash
# 上海方言级联微调训练脚本
# 任务：上海话语音 -> 上海话文本（需配合翻译模型）
# 使用统一的训练脚本 train_shanghai_wandb.py，通过 --mode cascade 切换模式

set -e

echo "=============================================="
echo "上海方言级联微调训练"
echo "任务：上海话语音 -> 上海话文本"
echo "模型：Whisper Medium（默认）"
echo "硬件配置：自动检测"
echo "=============================================="

# 1. 准备级联数据集
echo ""
echo "[Step 1] 准备级联数据集..."
python find_tune/make_data_shanghai.py
python find_tune/load_data_shanghai.py

# 2. 开始训练（使用统一脚本，--mode cascade 表示级联模式）
echo ""
echo "[Step 2] 开始级联全参数微调训练（Whisper Medium）..."
python find_tune/train_shanghai_wandb.py \
    --mode cascade \
    --model_size medium \
    --learning_rate 1e-5 \
    --num_train_epochs 30 \
    --warmup_steps 500 \
    --eval_steps 500 \
    --save_steps 500

echo ""
echo "=============================================="
echo "级联训练完成！"
echo "=============================================="
echo ""
echo "注意：级联模式输出的是上海话文本，需要配合翻译模型使用："
echo "  python find_tune/train_translation.py"
