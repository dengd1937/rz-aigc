"""
Prompt 构建器
基于 persona 和约束条件构建 LLM prompt
"""

import sys
import os

try:
    from .prompt_templates import instruction_following
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prompt_templates import instruction_following


class PromptBuilder:
    """Prompt 构建器类"""

    def __init__(self):
        self.instruction_template = instruction_following
        self.instruction_template_other = None  # 延迟加载

    def build_instruction_prompt(self, persona: str, example: str, constraints_description: str, category: str = None) -> str:
        """
        构建指令生成的 prompt

        Args:
            persona: 人物角色描述
            example: 约束示例
            constraints_description: 约束条件描述
            category: 分类名称（用于特殊处理"其他分类"）

        Returns:
            构建好的 prompt
        """
        # 特殊处理"其他分类" - 通过persona内容或category参数检测
        is_other_category = False
        if category == "其他分类":
            is_other_category = True
        elif persona and "其他分类" in persona:
            is_other_category = True

        if is_other_category:
            if self.instruction_template_other is None:
                try:
                    from .prompt_templates import instruction_following_other
                except ImportError:
                    from prompt_templates import instruction_following_other
                self.instruction_template_other = instruction_following_other

            prompt = self.instruction_template_other.format(
                persona=persona,
                example=example,
                constraints=constraints_description
            )
        else:
            # 确保传递category参数给模板
            prompt = self.instruction_template.format(
                persona=persona,
                example=example,
                constraints=constraints_description,
                category=category if category else "指定分类"
            )
        return prompt

    def build_solution_prompt(self, instruction: str) -> str:
        """
        构建响应生成的 prompt

        Args:
            instruction: 指令内容

        Returns:
            构建好的 prompt
        """
        try:
            from .prompt_templates import instruction_following_solution
        except ImportError:
            from prompt_templates import instruction_following_solution
        return instruction_following_solution.format(instruction=instruction)

    def build_rewrite_prompt(self, instruction: str, constraints: str, category: str) -> str:
        """
        构建重写指令的 prompt

        Args:
            instruction: 原始指令
            constraints: 约束描述
            category: 要放松的约束类别

        Returns:
            构建好的 prompt
        """
        try:
            from .prompt_templates import rewrite_if_prompt
        except ImportError:
            from prompt_templates import rewrite_if_prompt
        return rewrite_if_prompt.format(
            instruction=instruction,
            constraints=constraints,
            category=category
        )