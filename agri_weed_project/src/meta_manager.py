"""环境数据管理器"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .config import AgriConfig, CropType

class WeedStage(Enum):
    SEEDLING = "幼苗期"
    VEGETATIVE = "营养期"
    FLOWERING = "开花期"

@dataclass
class MetaData:
    raw: Dict[str, float]          # 原始值
    normalized: np.ndarray         # 标准化后
    stage: WeedStage               # 生长阶段
    env_score: float               # 环境评分
    source: str                    # "file" 或 "default"

class AgriMetaManager:
    def __init__(self, config: AgriConfig):
        self.config = config
        self.pairs: List[Dict] = []
        self._build_index()
    
    def _build_index(self):
        """建立图像-meta配对索引"""
        rgb_dir = Path(self.config.get_abs_path(self.config.rgb_path))
        meta_dir = Path(self.config.get_abs_path(self.config.meta_path))
        
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB目录不存在: {rgb_dir}")
        
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for img_file in sorted(rgb_dir.iterdir()):
            if img_file.suffix.lower() not in img_exts:
                continue
            
            # 查找对应meta
            stem = img_file.stem
            meta_file = self._find_meta_file(meta_dir, stem)
            
            self.pairs.append({
                "image": str(img_file),
                "meta": str(meta_file) if meta_file else None,
                "stem": stem,
                "has_meta": meta_file is not None
            })
        
        with_meta = sum(1 for p in self.pairs if p["has_meta"])
        print(f"🌾 找到 {len(self.pairs)} 张图像，{with_meta} 张有meta数据")
    
    def _find_meta_file(self, meta_dir: Path, stem: str) -> Optional[Path]:
        """查找meta文件（支持多种命名）"""
        candidates = [
            meta_dir / f"{stem}.npz",
            meta_dir / f"{stem}.json",
            meta_dir / f"{stem}_meta.npz",
            meta_dir / f"{stem}_meta.json",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return None
    
    def load_meta(self, pair: Dict) -> MetaData:
        """加载并解析meta数据"""
        if pair["has_meta"]:
            raw = self._parse_file(pair["meta"])
            source = "file"
        else:
            raw = self._default_meta()
            source = "default"
            print(f"⚠️ {pair['stem']} 使用默认meta")
        
        # 推断生长阶段
        stage = self._infer_stage(raw["growth_days"])
        
        # 标准化
        norm = self._normalize(raw)
        
        # 环境评分
        score = self._calc_env_score(raw)
        
        return MetaData(
            raw=raw,
            normalized=norm,
            stage=stage,
            env_score=score,
            source=source
        )
    
    def _parse_file(self, path: str) -> Dict[str, float]:
        """解析meta文件"""
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
        """默认农业环境参数"""
        return {
            "distance": 1.5,
            "humidity": 65.0,
            "light": 60000.0,
            "temperature": 25.0,
            "soil_moisture": 55.0,
            "growth_days": 25.0,
            "wind_speed": 1.5,
        }
    
    def _infer_stage(self, days: float) -> WeedStage:
        """根据生长天数推断阶段"""
        if days < 15:
            return WeedStage.SEEDLING
        elif days < 40:
            return WeedStage.VEGETATIVE
        else:
            return WeedStage.FLOWERING
    
    def _normalize(self, raw: Dict[str, float]) -> np.ndarray:
        """标准化为向量"""
        vec = np.zeros(len(self.config.meta_keys))
        for i, key in enumerate(self.config.meta_keys):
            mean, std = self.config.meta_norm[key]
            val = raw[key]
            vec[i] = (val - mean) / std if std > 0 else (val - mean)
        return np.clip(vec, -5, 5)
    
    def _calc_env_score(self, raw: Dict[str, float]) -> float:
        """计算环境适合度"""
        light_score = min(raw["light"] / 50000, 1.0)
        temp_score = 1.0 - abs(raw["temperature"] - 25) / 15
        wind_score = max(0, 1.0 - raw["wind_speed"] / 5)
        return (light_score + temp_score + wind_score) / 3
    
    def get_all_pairs(self) -> List[Dict]:
        """获取所有图像-meta配对"""
        return self.pairs