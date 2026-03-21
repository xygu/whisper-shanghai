#!/bin/bash
# 级联联合训练脚本：Whisper ASR + mT5 翻译模型联合训练
# 任务：上海话语音 -> 上海话文本 -> 普通话文本
# 两个 loss 加权平均作为最终 loss

set -e

echo "=============================================="
echo "级联联合训练：Whisper ASR + mT5 翻译"
echo "任务：上海话语音 -> 上海话文本 -> 普通话文本"
echo "=============================================="

# 检查数据集
if [ ! -d "dataset/shanghai/shanghai_unified_dataset" ]; then
    echo ""
    echo "[Step 1] 准备数据集..."
    python find_tune/make_data_shanghai.py
    python find_tune/load_data_shanghai.py
else
    echo "[Step 1] 数据集已存在，跳过准备步骤"
fi

# 检查翻译模型
TRANSLATION_MODEL="/mnt/workspace/workgroup/qq/ts/whisper/exp/translation-mt5-small-260320-231422/final_model"
if [ ! -d "$TRANSLATION_MODEL" ]; then
    echo ""
    echo "⚠️ 翻译模型不存在: $TRANSLATION_MODEL"
    echo "请先训练翻译模型:"
    echo "  python find_tune/train_translation.py"
    exit 1
fi

echo ""
echo "[Step 2] 开始级联联合训练..."
echo ""

# 检测 GPU 数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "检测到 ${GPU_COUNT} 个 GPU"

# 默认配置：使用 LoRA 微调 Whisper，同时微调翻译模型
# ASR loss 权重 0.5，翻译 loss 权重 0.5

if [ "$GPU_COUNT" -gt 1 ]; then
    # 多 GPU：使用 torchrun 启动 DDP 分布式训练（避免 DataParallel 的兼容性问题）
    echo "使用 DDP 分布式训练 (${GPU_COUNT} GPUs)..."
    torchrun --nproc_per_node=$GPU_COUNT find_tune/train_cascade_joint.py \
        --whisper_model openai/whisper-medium \
        --translation_model "$TRANSLATION_MODEL" \
        --fp16 \
        "$@"
else
    # 单 GPU：直接启动
    echo "使用单 GPU 训练..."
    python find_tune/train_cascade_joint.py \
        --whisper_model openai/whisper-medium \
        --translation_model "$TRANSLATION_MODEL" \
        --fp16 \
        "$@"
fi

echo ""
echo "=============================================="
echo "级联联合训练完成！"
echo "=============================================="
