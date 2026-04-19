#!/usr/bin/env python3
"""数据准备主脚本"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import AgriConfig, CropType
from src.trainer import AgriYOLOTrainer

def main():
    # 配置（修改这里！）
    config = AgriConfig(
        crop_type=CropType.RICE,  # ← 改成你的作物
        # img_size=640,            # 如需修改尺寸
    )
    
    # 准备数据集
    trainer = AgriYOLOTrainer(config)
    trainer.prepare("output/yolo_dataset")

if __name__ == "__main__":
    main()