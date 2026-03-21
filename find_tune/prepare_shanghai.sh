#!/bin/bash

# 上海方言数据集准备脚本

echo "=========================================="
echo "上海方言 Whisper 微调 - 数据准备"
echo "=========================================="

# 1. 生成 JSONL 文件
echo ""
echo "步骤 1/2: 生成 JSONL 文件..."
python find_tune/make_data_shanghai.py

if [ $? -ne 0 ]; then
    echo "❌ 生成 JSONL 文件失败"
    exit 1
fi

# 2. 创建 HuggingFace 数据集
echo ""
echo "步骤 2/2: 创建 HuggingFace 数据集..."
python find_tune/load_data_shanghai.py

if [ $? -ne 0 ]; then
    echo "❌ 创建数据集失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 数据准备完成!"
echo "=========================================="
echo ""
echo "下一步: 开始训练"
echo "  bash find_tune/train_shanghai.sh"
echo "或"
echo "  python find_tune/train_shanghai.py"
