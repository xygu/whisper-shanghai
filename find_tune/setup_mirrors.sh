#!/bin/bash

# 配置国内镜像脚本

echo "=========================================="
echo "配置 pip 镜像（用于安装依赖）"
echo "=========================================="

# 配置 pip 镜像
echo ""
echo "配置 pip 镜像..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
echo "✓ pip 镜像: https://mirrors.aliyun.com/pypi/simple/"

echo ""
echo "=========================================="
echo "✓ 镜像配置完成！"
echo "=========================================="
echo ""
echo "说明："
echo "  - HuggingFace 镜像已在训练脚本中自动配置"
echo "  - WandB 镜像已在训练脚本中自动配置"
echo "  - pip 镜像用于加速依赖包安装"
echo ""
echo "开始训练："
echo "  bash find_tune/train_shanghai.sh"
