#!/bin/bash
# LoRA Cascade Finetuning 定时任务脚本
# 任务：上海话语音 -> 上海话文本（LoRA 微调）

set -e

cd /mnt/workspace/workgroup/qq/ts/whisper

# 激活虚拟环境
source /mnt/workspace/workgroup/qq/qq-env/whisper/bin/activate

echo "=============================================="
echo "LoRA Cascade Finetuning 开始"
echo "时间: $(date)"
echo "任务：上海话语音 -> 上海话文本 (LoRA)"
echo "=============================================="

# 确保数据集已准备好
if [ ! -d "dataset/shanghai/shanghai_unified_dataset" ]; then
    echo "[Step 1] 准备级联数据集..."
    python find_tune/make_data_shanghai.py
    python find_tune/load_data_shanghai.py
else
    echo "[Step 1] 数据集已存在，跳过准备步骤"
fi

# 开始 LoRA 训练
# LoRA 显存占用远小于全参数微调，可以增大 batch_size
# V100 32GB × 2 配置下，LoRA 可以使用 batch_size=8
# 参数配置：
#   - batch_size: 8 (LoRA 优化)
#   - gradient_accumulation_steps: 8 (保持与之前一致)
#   - effective_batch_size: 8 × 8 × 2 = 128
#   - eval_steps: 100 (更密集的验证)
#   - dataloader_num_workers: 2
#   - num_proc: 4
echo ""
echo "[Step 2] 开始 LoRA Cascade 微调训练..."
python find_tune/train_shanghai_wandb.py \
    --mode cascade \
    --finetune_method lora \
    --model_size medium \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --warmup_steps 500 \
    --eval_steps 100 \
    --save_steps 500 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_proc 4 \
    --dataloader_num_workers 2

echo ""
echo "=============================================="
echo "LoRA Cascade 训练完成！"
echo "时间: $(date)"
echo "=============================================="
