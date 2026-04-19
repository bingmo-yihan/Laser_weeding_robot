"""YOLO数据集构建"""

import os
import random
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset

from .config import AgriConfig
from .meta_manager import AgriMetaManager, MetaData
from .image_processor import AgriImageProcessor

class AgriWeedDataset(Dataset):
    def __init__(self, config: AgriConfig, pairs: List[Dict], mode: str = "train"):
        self.config = config
        self.pairs = pairs
        self.mode = mode
        self.processor = AgriImageProcessor(config, mode)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 加载图像
        img_data = self.processor.preprocess(pair["image"])
        
        # 加载meta（如果有）
        meta = None
        if pair["has_meta"]:
            from .meta_manager import AgriMetaManager
            manager = AgriMetaManager(self.config)
            meta = manager.load_meta(pair)
        
        return {
            "image": img_data["tensor"],
            "meta": meta.normalized if meta else torch.zeros(7),
            "meta_info": meta,
            "path": pair["image"],
            "stem": pair["stem"]
        }

def split_dataset(pairs: List[Dict], train_ratio=0.8, val_ratio=0.1):
    """划分训练/验证/测试集"""
    n = len(pairs)
    indices = list(range(n))
    random.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        "train": [pairs[i] for i in indices[:train_end]],
        "val": [pairs[i] for i in indices[train_end:val_end]],
        "test": [pairs[i] for i in indices[val_end:]]
    }