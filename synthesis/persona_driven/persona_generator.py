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

        # 特殊处理"其他分类" - 避免使用矛盾的persona描述
        if category == "其他分类":
            # 为其他分类生成通用的观察者persona，避免使用"其他分类领域"这种矛盾描述
            role = random.choice(self.category_to_roles["其他分类"])
            characteristics = self._generate_characteristics(detail_level)
            # 使用更自然的描述，避免"其他分类领域"这种表述
            persona = f"一位{role}，{characteristics}"
            return persona

        # 正常分类的处理逻辑
        template = random.choice(self.templates)

        # 根据分类选择合适的角色类型（关键改进）
        if category in self.category_to_roles:
            available_roles = self.category_to_roles[category]
            role = random.choice(available_roles)
        else:
            # 如果分类不在映射中，使用通用角色
            role = random.choice(self.roles)

        # 生成特征描述 - 增强分类相关性
        characteristics = self._generate_characteristics(detail_level)

        # 在特征描述中强制加入分类关键词，增强锚定
        category_keywords = self._get_category_keywords(category)
        if category_keywords:
            characteristics = f"{characteristics}，{category_keywords}"

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

    def _get_category_keywords(self, category: str) -> str:
        """
        根据分类生成相关的关键词，增强分类锚定

        Args:
            category: 分类名称

        Returns:
            分类相关的关键词描述
        """
        # 分类到关键词的映射
        category_keyword_map = {
            "文学小说": "擅长文学创作与小说分析",
            "青春文学": "专注青春题材创作与研究",
            "亲子育儿": "精通育儿方法与亲子关系",
            "科普读物": "专注科学知识普及与传播",
            "动漫幽默": "擅长动漫分析与幽默创作",
            "人文社科": "专长人文社会科学研究",
            "艺术收藏": "精通艺术鉴赏与收藏",
            "古籍地理": "擅长古籍整理与地理研究",
            "旅游休闲": "专注旅游规划与休闲指导",
            "经济管理": "精通经济分析与管理策略",
            "励志成长": "擅长个人成长与激励指导",
            "外语学习": "专注外语教学与学习方法",
            "法律哲学": "专长法律与哲学研究",
            "政治军事": "精通政治军事分析",
            "自然科学": "专注自然科学研究",
            "家庭教育": "擅长家庭教育与理念指导",
            "两性关系": "专注两性关系研究",
            "孕产育儿": "精通孕产期与婴幼儿护理",
            "家居生活": "擅长家居设计与生活指导",
            "生活时尚": "专注生活方式与时尚品味",
            "其他分类": "专长跨领域观察与分析"
        }

        return category_keyword_map.get(category, "")

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
