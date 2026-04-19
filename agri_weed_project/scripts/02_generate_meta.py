#!/usr/bin/env python3
"""批量生成环境数据文件"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import AgriConfig

def generate_meta_for_all():
    """为所有没有meta的图像生成默认meta"""
    config = AgriConfig()
    
    rgb_dir = Path(config.get_abs_path(config.rgb_path))
    meta_dir = Path(config.get_abs_path(config.meta_path))
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 默认环境参数（根据实际情况修改！）
    default_values = {
        "distance": 1.5,      # 相机高度1.5米
        "humidity": 65.0,     # 湿度65%
        "light": 60000.0,     # 光照60000 lux（晴天）
        "temperature": 25.0,  # 温度25°C
        "soil_moisture": 55.0,# 土壤湿度55%
        "growth_days": 20.0,  # 生长20天
        "wind_speed": 1.0,    # 风速1m/s
    }
    
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    generated = 0
    
    for img_file in rgb_dir.iterdir():
        if img_file.suffix.lower() not in img_exts:
            continue
        
        meta_file = meta_dir / f"{img_file.stem}.json"
        
        if meta_file.exists():
            continue  # 已有meta，跳过
        
        # 生成（可在这里根据文件名规则自定义）
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(default_values, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 生成: {meta_file.name}")
        generated += 1
    
    print(f"\n总共生成 {generated} 个meta文件")
    print("⚠️  请检查并根据实际情况修改每个文件的数值！")

if __name__ == "__main__":
    generate_meta_for_all()