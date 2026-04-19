"""工具函数"""

import os

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def get_stem(filename: str) -> str:
    """获取文件名（无扩展名）"""
    return os.path.splitext(os.path.basename(filename))[0]