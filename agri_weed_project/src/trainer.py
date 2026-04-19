"""YOLO训练管理器"""

import os
import shutil
from pathlib import Path
from typing import Dict

from .config import AgriConfig, CropType
from .dataset import AgriWeedDataset, split_dataset
from .image_processor import AgriImageProcessor

class AgriYOLOTrainer:
    """准备YOLO格式的训练数据"""
    
    WEED_CLASSES = {
        CropType.RICE: ["rice", "barnyard_grass", "monochoria", "lobelia", "other_weed"],
        CropType.WHEAT: ["wheat", "wild_oat", "green_foxtail", "cleavers", "other_weed"],
        CropType.CORN: ["corn", "cocklebur", "velvetleaf", "pigweed", "other_weed"],
        CropType.SOYBEAN: ["soybean", "waterhemp", "lambsquarters", "ragweed", "other_weed"],
    }
    
    def __init__(self, config: AgriConfig):
        self.config = config
        self.classes = self.WEED_CLASSES.get(config.crop_type, ["crop", "weed"])
        print(f"🌱 作物: {config.crop_type.value}")
        print(f"   类别: {self.classes}")
    
    def prepare(self, output_dir: str = "output/yolo_dataset"):
        """准备标准YOLO数据集"""
        from .meta_manager import AgriMetaManager
        
        out_path = Path(self.config.get_abs_path(output_dir))
        
        # 清理并创建目录
        if out_path.exists():
            shutil.rmtree(out_path)
        for split in ["train", "val", "test"]:
            (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out_path / "meta" / split).mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        manager = AgriMetaManager(self.config)
        pairs = manager.get_all_pairs()
        
        if len(pairs) == 0:
            raise ValueError("没有找到任何图像数据！")
        
        # 划分数据集
        splits = split_dataset(pairs)
        
        # 处理每个集
        processor = AgriImageProcessor(self.config, mode="val")  # 用val模式（无增强）
        
        for split_name, split_pairs in splits.items():
            print(f"\n处理 {split_name}: {len(split_pairs)} 张")
            
            for pair in split_pairs:
                self._process_one(pair, processor, manager, out_path, split_name)
        
        # 生成yaml
        self._create_yaml(out_path)
        
        print(f"\n✅ 数据集准备完成: {out_path}")
        print(f"   下一步运行: yolo detect train data={out_path}/data.yaml model=yolov8m.pt epochs=100")
    
    def _process_one(self, pair, processor, manager, out_path, split):
        """处理单个样本"""
        stem = pair["stem"]
        
        # 保存图像
        img_data = processor.preprocess(pair["image"])
        img_save = out_path / "images" / split / f"{stem}.jpg"
        processor.save_tensor(img_data["tensor"], str(img_save))
        
        # 保存meta
        if pair["has_meta"]:
            meta = manager.load_meta(pair)
            meta_save = out_path / "meta" / split / f"{stem}.npz"
            import numpy as np
            np.savez(meta_save, 
                    normalized=meta.normalized,
                    raw=list(meta.raw.values()))
        
        # 查找标注（如果有）
        label_src = Path(self.config.get_abs_path(self.config.label_path)) / f"{stem}.txt"
        label_dst = out_path / "labels" / split / f"{stem}.txt"
        
        if label_src.exists():
            shutil.copy(label_src, label_dst)
        else:
            # 创建空标注文件（YOLO需要）
            label_dst.touch()
    
    def _create_yaml(self, out_path: Path):
        """生成YOLO配置文件"""
        yaml_content = f"""# 农业杂草检测数据集
path: {out_path.absolute()}
train: images/train
val: images/val
test: images/test

# 类别
nc: {len(self.classes)}
names: {self.classes}

# 多模态配置（供后续使用）
meta_path: meta
meta_dim: 7
crop_type: {self.config.crop_type.name}
"""
        with open(out_path / "data.yaml", "w", encoding='utf-8') as f:
            f.write(yaml_content)