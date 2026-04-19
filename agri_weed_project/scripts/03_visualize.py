#!/usr/bin/env python3
"""可视化检查数据"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import AgriConfig, CropType
from src.meta_manager import AgriMetaManager

def visualize():
    config = AgriConfig()
    manager = AgriMetaManager(config)
    
    for pair in manager.pairs[:3]:  # 只看前3张
        print(f"\n{'='*50}")
        print(f"图像: {pair['stem']}")
        
        # 加载meta
        if pair['has_meta']:
            meta = manager.load_meta(pair)
            print(f"环境数据:")
            for k, v in meta.raw.items():
                print(f"  {k}: {v}")
            print(f"生长阶段: {meta.stage.value}")
            print(f"环境评分: {meta.env_score:.2f}")
        else:
            print("无环境数据")
        
        # 显示图像（缩小显示）
        img = cv2.imread(pair['image'])
        img = cv2.resize(img, (640, 480))
        cv2.imshow(f"Preview: {pair['stem']}", img)
    
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize()