#!/usr/bin/env python3
"""
农业杂草检测 - 数据准备脚本（单文件版）
"""

import os
import sys
import cv2
import json
import shutil
import random
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ====================== 配置 ======================
class CropType(Enum):
    RICE = "水稻"
    WHEAT = "小麦"
    CORN = "玉米"
    SOYBEAN = "大豆"
    COTTON = "棉花"

@dataclass
class AgriConfig:
    img_size: int = 640
    crop_type: CropType = CropType.RICE
    
    rgb_path: str = "data/agri/rgb"
    meta_path: str = "data/agri/meta"
    label_path: str = "data/agri/labels"
    output_path: str = "output/yolo_dataset"
    
    meta_keys: Tuple[str, ...] = (
        "distance", "humidity", "light", "temperature", 
        "soil_moisture", "growth_days", "wind_speed"
    )
    
    meta_norm: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "distance": (1.5, 2.0),
        "humidity": (60.0, 40.0),
        "light": (50000.0, 50000.0),
        "temperature": (25.0, 15.0),
        "soil_moisture": (50.0, 30.0),
        "growth_days": (30.0, 20.0),
        "wind_speed": (2.0, 3.0),
    })
    
    def get_abs_path(self, rel_path: str) -> str:
        script_dir = Path(__file__).parent.resolve()
        return str(script_dir / rel_path)

# ====================== 元数据管理 ======================
class AgriMetaManager:
    def __init__(self, config: AgriConfig):
        self.config = config
        self.pairs: List[Dict] = []
        self._build_index()
    
    def _build_index(self):
        rgb_dir = Path(self.config.get_abs_path(self.config.rgb_path))
        meta_dir = Path(self.config.get_abs_path(self.config.meta_path))
        
        if not rgb_dir.exists():
            raise FileNotFoundError(f"❌ RGB目录不存在: {rgb_dir}\n请创建文件夹并放入图片！")
        
        meta_dir.mkdir(parents=True, exist_ok=True)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for img_file in sorted(rgb_dir.iterdir()):
            if img_file.suffix.lower() not in img_exts:
                continue
            
            stem = img_file.stem
            meta_file = None
            for ext in ['.json', '.npz', '_meta.json', '_meta.npz']:
                cand = meta_dir / f"{stem}{ext}"
                if cand.exists():
                    meta_file = str(cand)
                    break
            
            self.pairs.append({
                "image": str(img_file),
                "meta": meta_file,
                "stem": stem,
                "has_meta": meta_file is not None
            })
        
        with_meta = sum(1 for p in self.pairs if p["has_meta"])
        print(f"🌾 找到 {len(self.pairs)} 张图像，{with_meta} 张有meta数据")
        if len(self.pairs) == 0:
            raise ValueError("没有找到任何图片！请检查 data/agri/rgb/ 目录")
    
    def load_meta(self, pair: Dict) -> Tuple[np.ndarray, Dict]:
        if pair["has_meta"]:
            raw = self._parse_file(pair["meta"])
        else:
            raw = self._default_meta()
            print(f"⚠️  {pair['stem']} 使用默认meta")
        
        vec = np.zeros(len(self.config.meta_keys))
        for i, key in enumerate(self.config.meta_keys):
            mean, std = self.config.meta_norm[key]
            val = raw[key]
            vec[i] = (val - mean) / std if std > 0 else (val - mean)
        
        return np.clip(vec, -5, 5), raw
    
    def _parse_file(self, path: str) -> Dict[str, float]:
        suffix = Path(path).suffix.lower()
        if suffix == '.npz':
            data = np.load(path)
            return {k: float(data[k]) for k in self.config.meta_keys}
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {k: float(data.get(k, 0)) for k in self.config.meta_keys}
        else:
            raise ValueError(f"不支持的格式: {suffix}")
    
    def _default_meta(self) -> Dict[str, float]:
        return {
            "distance": 1.5, "humidity": 65.0, "light": 60000.0,
            "temperature": 25.0, "soil_moisture": 55.0,
            "growth_days": 25.0, "wind_speed": 1.5,
        }
    
    def get_pairs(self) -> List[Dict]:
        return self.pairs

# ====================== 图像处理 ======================
class AgriImageProcessor:
    def __init__(self, config: AgriConfig):
        self.config = config
    
    def preprocess(self, img_path: str):
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}\n请检查文件是否存在，且不要用中文文件名！")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        
        img = self._letterbox(img, self.config.img_size)
        
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        tensor = img.transpose(2, 0, 1)
        
        return {
            "tensor": tensor,
            "original_shape": original_shape,
            "path": img_path
        }
    
    def _letterbox(self, img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return canvas
    
    def save_tensor(self, tensor: np.ndarray, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = tensor.transpose(1, 2, 0) * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)

# ====================== YOLO训练数据生成 ======================
class AgriYOLOTrainer:
    WEED_CLASSES = {
        CropType.RICE: ["rice", "barnyard_grass", "monochoria", "lobelia", "other_weed"],
        CropType.WHEAT: ["wheat", "wild_oat", "green_foxtail", "cleavers", "other_weed"],
        CropType.CORN: ["corn", "cocklebur", "velvetleaf", "pigweed", "other_weed"],
        CropType.SOYBEAN: ["soybean", "waterhemp", "lambsquarters", "ragweed", "other_weed"],
        CropType.COTTON: ["cotton", "morning_glory", "nutsedge", "palmer_amaranth", "other_weed"],
    }
    
    def __init__(self, config: AgriConfig):
        self.config = config
        self.classes = self.WEED_CLASSES.get(config.crop_type, ["crop", "weed"])
        print(f"🌱 作物类型: {config.crop_type.value}")
        print(f"   检测类别: {self.classes}")
    
    def prepare(self):
        out_path = Path(self.config.get_abs_path(self.config.output_path))
        
        if out_path.exists():
            shutil.rmtree(out_path)
        
        for split in ["train", "val", "test"]:
            (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out_path / "meta" / split).mkdir(parents=True, exist_ok=True)
        
        manager = AgriMetaManager(self.config)
        pairs = manager.get_pairs()
        
        random.seed(42)
        indices = list(range(len(pairs)))
        random.shuffle(indices)
        
        n = len(pairs)
        split_idx = {
            "train": indices[:int(n*0.8)],
            "val": indices[int(n*0.8):int(n*0.9)],
            "test": indices[int(n*0.9):]
        }
        
        processor = AgriImageProcessor(self.config)
        
        for split_name, idxs in split_idx.items():
            print(f"\n📂 处理 {split_name}: {len(idxs)} 张")
            
            for idx in idxs:
                pair = pairs[idx]
                try:
                    self._process_one(pair, processor, manager, out_path, split_name)
                except Exception as e:
                    print(f"   ❌ 跳过 {pair['stem']}: {e}")
        
        self._create_yaml(out_path)
        
        print(f"\n{'='*50}")
        print(f"✅ 数据集准备完成！")
        print(f"   输出目录: {out_path}")
        print(f"   训练命令:")
        print(f'   yolo detect train data="{out_path}/data.yaml" model=yolov8m.pt epochs=100 imgsz=640')
    
    def _process_one(self, pair, processor, manager, out_path, split):
        stem = pair["stem"]
        
        img_data = processor.preprocess(pair["image"])
        
        img_save = out_path / "images" / split / f"{stem}.jpg"
        processor.save_tensor(img_data["tensor"], str(img_save))
        
        if pair["has_meta"]:
            meta_vec, meta_raw = manager.load_meta(pair)
            meta_save = out_path / "meta" / split / f"{stem}.npz"
            np.savez(meta_save, normalized=meta_vec, raw=list(meta_raw.values()))
        
        label_src = Path(self.config.get_abs_path(self.config.label_path)) / f"{stem}.txt"
        label_dst = out_path / "labels" / split / f"{stem}.txt"
        
        if label_src.exists():
            shutil.copy(label_src, label_dst)
        else:
            label_dst.touch()
    
    def _create_yaml(self, out_path: Path):
        yaml_content = f"""# 农业杂草检测数据集
path: {out_path.absolute().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(self.classes)}
names: {self.classes}

# 多模态配置
meta_path: meta
meta_dim: 7
crop_type: {self.config.crop_type.name}
"""
        with open(out_path / "data.yaml", "w", encoding='utf-8') as f:
            f.write(yaml_content)

# ====================== 主函数 ======================
def main():
    config = AgriConfig(
        crop_type=CropType.RICE,
    )
    
    rgb_abs = config.get_abs_path(config.rgb_path)
    if not os.path.exists(rgb_abs):
        print(f"❌ 图片目录不存在: {rgb_abs}")
        print("请先创建文件夹并放入图片：")
        print(f'  mkdir -p "{rgb_abs}"')
        return
    
    trainer = AgriYOLOTrainer(config)
    trainer.prepare()

if __name__ == "__main__":
    main()