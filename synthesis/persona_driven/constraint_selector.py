"""
约束条件选择器
从约束库中智能选择并组合约束条件
"""

import random
import sys
import os
from typing import List, Tuple, Dict

try:
    from .constraints import CONSTRAINTS, CONSTRAINT_CATEGORIES
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from constraints import CONSTRAINTS, CONSTRAINT_CATEGORIES


class ConstraintSelector:
    """约束条件选择器类"""

    def __init__(self):
        self.constraints = CONSTRAINTS
        self.categories = CONSTRAINT_CATEGORIES

    def select_constraints(self, num_constraints: int = 2) -> Tuple[List[str], str]:
        """
        选择指定数量的约束条件

        Args:
            num_constraints: 要选择的约束数量，默认为2

        Returns:
            (约束列表, 约束描述字符串)
        """
        # 获取所有约束键
        all_constraints = list(self.constraints.keys())
        
        # 随机选择约束
        selected = random.sample(all_constraints, min(num_constraints, len(all_constraints)))
        
        # 生成描述
        descriptions = []
        for key in selected:
            desc = random.choice(self.constraints[key])
            descriptions.append(f"{key}: {desc}")
        
        return selected, "，".join(descriptions)

    def get_constraint_categories(self) -> List[str]:
        """获取所有约束类别"""
        return list(self.categories.keys())

    def get_constraints_by_category(self, category: str) -> List[str]:
        """获取指定类别的所有约束"""
        return [k for k in self.constraints.keys() if k.startswith(category)]

    def get_constraint_example(self, constraint_key: str) -> str:
        """获取指定约束的示例"""
        if constraint_key in self.constraints:
            return random.choice(self.constraints[constraint_key])
        return ""

    def select_constraints_for_prompt(self, num_constraints: int = 2) -> Dict[str, str]:
        """
        为 prompt 构造选择约束条件

        Args:
            num_constraints: 约束数量

        Returns:
            包含约束键和描述的字典
        """
        constraints, description = self.select_constraints(num_constraints)

        # 提取约束类别（用于重写时使用）
        categories = []
        for constraint in constraints:
            category = constraint.split(":")[0]
            if category not in categories:
                categories.append(category)

        return {
            "constraint_keys": constraints,
            "constraint_description": description,
            "constraint_categories": categories
        }
