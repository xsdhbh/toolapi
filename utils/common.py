# 定义基础目录
import os
import uuid

BASE_DIR = "uploads"


def ensure_directory_exists(path: str):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def generate_random_filename(original_filename: str) -> str:
    """生成随机文件名，保留原始文件的扩展名"""
    ext = os.path.splitext(original_filename)[1]  # 获取文件扩展名
    random_name = str(uuid.uuid4())  # 生成随机文件名
    return f"{random_name}{ext}"
