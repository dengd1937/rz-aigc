"""
工具函数
"""

import json
import random
import string
from typing import List, Dict


def generate_unique_id(prefix: str = "personahub") -> str:
    """
    生成唯一ID

    Args:
        prefix: 前缀

    Returns:
        唯一ID
    """
    random_part = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(24))
    return f"{prefix}_{random_part}"


def clean_instruction(text: str) -> str:
    """
    清理指令文本，移除前缀

    Args:
        原始文本

    Returns:
        清理后的文本
    """
    prefixes = ["用户指令：", "User instruction:", "用户指令:", "指令：", "指令:"]
    result = text.strip()
    for prefix in prefixes:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
            break
    return result


def save_jsonl(data: List[Dict], filepath: str):
    """
    保存数据为 JSONL 格式

    Args:
        data: 数据列表
        filepath: 文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict]:
    """
    加载 JSONL 数据

    Args:
        filepath: 文件路径

    Returns:
        数据列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def print_progress(current: int, total: int, message: str = ""):
    """
    打印进度

    Args:
        current: 当前进度
        total: 总进度
        message: 消息
    """
    percent = (current / total) * 100 if total > 0 else 0
    print(f"[{current}/{total}] {percent:.1f}% - {message}")