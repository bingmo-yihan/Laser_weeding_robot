"""图像预处理"""

import cv2
import torch
import numpy as np
from typing import Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import AgriConfig

class AgriImageProcessor:
    def __init__(self, config: AgriConfig, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.img_size = config.img_size
        self._build_transform()
    
    def _build_transform(self):
        """构建数据增强管道"""
        if self.mode == "train":
            self.transform = A.Compose([
                A.RandomResizedCrop(self.img_size, self.img_size, scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30),
                ], p=0.6),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.ISONoise(intensity=(0.1, 0.5)),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def preprocess(self, img_path: str) -> Dict:
        """预处理单张图像"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        
        # 应用变换
        transformed = self.transform(image=img)
        tensor = transformed["image"]
        
        return {
            "tensor": tensor,
            "original_shape": original_shape,
            "path": img_path
        }
    
    def save_tensor(self, tensor: torch.Tensor, save_path: str):
        """将tensor保存为图像文件"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)