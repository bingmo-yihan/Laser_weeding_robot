"""农业杂草识别 - 配置文件"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
from enum import Enum
import os

class CropType(Enum):
    RICE = "水稻"
    WHEAT = "小麦"
    CORN = "玉米"
    SOYBEAN = "大豆"
    COTTON = "棉花"

@dataclass
class AgriConfig:
    # 基础
    img_size: int = 640
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    # 路径（相对项目根目录）
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(__file__)))
    rgb_path: str = "data/agri/rgb"
    meta_path: str = "data/agri/meta"
    label_path: str = "data/agri/labels"
    output_path: str = "output"
    
    # 作物类型（关键！）
    crop_type: CropType = CropType.RICE
    
    # 环境参数键名
    meta_keys: Tuple[str, ...] = (
        "distance",      # 相机高度(m)
        "humidity",      # 空气湿度(%)
        "light",         # 光照强度(lux)
        "temperature",   # 温度(°C)
        "soil_moisture", # 土壤湿度(%)
        "growth_days",   # 生长天数
        "wind_speed",    # 风速(m/s)
    )
    
    # 归一化参数 (mean, std)
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
        """获取绝对路径"""
        return os.path.join(self.project_root, rel_path)