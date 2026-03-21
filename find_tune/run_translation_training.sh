#!/bin/bash
# 上海话到普通话翻译模型训练脚本
# 使用 mT5 模型进行 Seq2Seq 翻译任务
#
# 使用方法：
#   直接运行: bash find_tune/run_translation_training.sh
#   自定义参数: NUM_EPOCHS=50 LEARNING_RATE=1e-4 bash find_tune/run_translation_training.sh
#   传递额外参数: bash find_tune/run_translation_training.sh --output_dir ./my_exp

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}上海话到普通话翻译模型训练${NC}"
echo -e "${GREEN}开始时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}============================================================${NC}"

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}项目根目录: $PROJECT_ROOT${NC}"

# 默认参数（可通过环境变量覆盖）
MODEL_NAME="${MODEL_NAME:-google/mt5-small}"
DATASET_PATH="${DATASET_PATH:-dataset/shanghai/translation_dataset}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-64}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-64}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
EARLY_STOPPING="${EARLY_STOPPING:-5}"

# 检查数据集是否存在
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${YELLOW}⚠ 数据集未找到: $DATASET_PATH${NC}"
    echo -e "${YELLOW}正在准备数据集...${NC}"
    python find_tune/load_data_translation.py \
        --root_dir dataset/shanghai/ \
        --output_dir "$DATASET_PATH" \
        --test_size 0.1 \
        --seed 42
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 数据准备失败！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 数据准备完成${NC}"
fi

# 打印配置
echo ""
echo -e "${GREEN}训练配置:${NC}"
echo "  模型: $MODEL_NAME"
echo "  数据集: $DATASET_PATH"
echo "  训练轮数: $NUM_EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LEARNING_RATE"
echo "  源文本最大长度: $MAX_SOURCE_LENGTH"
echo "  目标文本最大长度: $MAX_TARGET_LENGTH"
echo "  预热比例: $WARMUP_RATIO"
echo "  评估间隔: $EVAL_STEPS 步"
echo "  保存间隔: $SAVE_STEPS 步"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo "  权重衰减: $WEIGHT_DECAY"
echo "  早停耐心: $EARLY_STOPPING"
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU 信息:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# 开始训练
echo -e "${GREEN}开始训练...${NC}"
echo ""

python find_tune/train_translation.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_source_length "$MAX_SOURCE_LENGTH" \
    --max_target_length "$MAX_TARGET_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --weight_decay "$WEIGHT_DECAY" \
    --early_stopping_patience "$EARLY_STOPPING" \
    "$@"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 模型训练失败！${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}训练完成!${NC}"
echo -e "${GREEN}结束时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}============================================================${NC}"
