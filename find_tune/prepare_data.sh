#!/bin/bash
# 数据准备脚本
# 用于生成 JSONL 文件并创建 HuggingFace 数据集

echo "Step 1: Generating JSONL file from TXT and WAV files..."
python find_tune/make_data.py

echo ""
echo "Step 2: Creating HuggingFace dataset..."
python find_tune/load_data.py

echo ""
echo "Data preparation completed!"
echo "You can now run the training script:"
echo "  bash find_tune/train.sh"
