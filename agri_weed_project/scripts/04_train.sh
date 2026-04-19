#!/bin/bash
# YOLO训练脚本

# 激活环境（如果需要）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate agri

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_YAML="$PROJECT_ROOT/output/yolo_dataset/data.yaml"

echo "=========================================="
echo "农业杂草检测 - YOLO训练"
echo "=========================================="
echo "数据配置: $DATA_YAML"
echo ""

# 检查数据
if [ ! -f "$DATA_YAML" ]; then
    echo "❌ 错误: 未找到数据配置文件！"
    echo "请先运行: python scripts/01_prepare_data.py"
    exit 1
fi

# 训练
yolo detect train \
    data="$DATA_YAML" \
    model=yolov8m.pt \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=0 \
    project="$PROJECT_ROOT/output" \
    name=weed_detection \
    exist_ok=True

echo ""
echo "✅ 训练完成！模型保存在: $PROJECT_ROOT/output/weed_detection"