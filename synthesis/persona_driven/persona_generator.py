"""
Persona 生成器
基于分类定义生成多样化的人物角色
"""

import random
import sys
import os
from typing import List, Dict

# 支持直接运行和作为包导入
try:
    from .persona_categories import CATEGORIES, PERSONA_TEMPLATES, ROLES, CHARACTERISTICS, CATEGORY_TO_ROLES
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from persona_categories import CATEGORIES, PERSONA_TEMPLATES, ROLES, CHARACTERISTICS, CATEGORY_TO_ROLES


class PersonaGenerator:
    """Persona 生成器类"""

    def __init__(self):
        self.categories = CATEGORIES
        self.templates = PERSONA_TEMPLATES
        self.roles = ROLES
        self.characteristics = CHARACTERISTICS
        self.category_to_roles = CATEGORY_TO_ROLES

    def generate_persona(self, category: str, detail_level: str = "medium") -> str:
        """
        生成指定分类的 persona

        Args:
            category: 分类名称
            detail_level: 详细程度，可选 small/medium/full

        Returns:
            生成的 persona 描述
        """
        if category not in self.categories:
            category = "其他分类"

        # 选择模板
        template = random.choice(self.templates)

        # 根据分类选择合适的角色类型（关键改进）
        if category in self.category_to_roles:
            available_roles = self.category_to_roles[category]
            role = random.choice(available_roles)
        else:
            # 如果分类不在映射中，使用通用角色
            role = random.choice(self.roles)

        # 生成特征描述
        characteristics = self._generate_characteristics(detail_level)

        # 根据模板格式化
        persona = template.format(
            category=category,
            description=characteristics,
            role=role,
            detail=characteristics,
            background=characteristics,
            experience=characteristics,
            specialty=characteristics
        )

        return persona

    def _generate_characteristics(self, detail_level: str) -> str:
        """
        生成角色特征描述

        Args:
            detail_level: 详细程度

        Returns:
            特征描述字符串
        """
        parts = []

        if detail_level == "small":
            # 简短描述
            keys = random.sample(list(self.characteristics.keys()), 2)
            for key in keys:
                parts.append(random.choice(self.characteristics[key]))
        elif detail_level == "medium":
            # 中等描述
            keys = random.sample(list(self.characteristics.keys()), 3)
            for key in keys:
                parts.append(random.choice(self.characteristics[key]))
        else:  # full
            # 完整描述
            for key in self.characteristics.keys():
                parts.append(random.choice(self.characteristics[key]))

        return "，".join(parts)

    def generate_personas_batch(self, categories: List[str], num_per_category: int = 5,
                               detail_level: str = "medium") -> List[Dict[str, str]]:
        """
        批量生成多个分类的 personas

        Args:
            categories: 分类列表
            num_per_category: 每个分类生成的数量
            detail_level: 详细程度

        Returns:
            persona 列表，每个元素包含 category 和 persona
        """
        personas = []

        for category in categories:
            for _ in range(num_per_category):
                persona = self.generate_persona(category, detail_level)
                personas.append({
                    "category": category,
                    "persona": persona,
                    "category_description": self.categories.get(category, "")
                })

        return personas

    def get_all_categories(self) -> List[str]:
        """获取所有分类名称"""
        return list(self.categories.keys())

    def get_category_description(self, category: str) -> str:
        """获取分类的详细描述"""
        return self.categories.get(category, "未知分类")
