#!/bin/bash
# 上海方言端到端微调训练脚本
# 任务：上海话语音 -> 普通话文本（跳过上海话文本中间步骤）
# 使用统一的训练脚本 train_shanghai_wandb.py，通过 --mode e2e 切换模式

set -e

echo "=============================================="
echo "上海方言端到端微调训练"
echo "任务：上海话语音 -> 普通话文本"
echo "模型：Whisper Medium（默认）"
echo "硬件配置：自动检测"
echo "=============================================="

# 1. 准备端到端数据集
echo ""
echo "[Step 1] 准备端到端数据集..."
python find_tune/make_data_shanghai_e2e.py
python find_tune/load_data_shanghai_e2e.py

# 2. 开始训练（使用统一脚本，--mode e2e 表示端到端模式）
echo ""
echo "[Step 2] 开始端到端全参数微调训练（Whisper Medium）..."
python find_tune/train_shanghai_wandb.py \
    --mode e2e \
    --model_size medium \
    --learning_rate 1e-5 \
    --num_train_epochs 30 \
    --warmup_steps 500 \
    --eval_steps 500 \
    --save_steps 500

echo ""
echo "=============================================="
echo "端到端训练完成！"
echo "=============================================="
