"""
LLM 客户端
封装 LLM 调用逻辑，支持重试和错误处理
"""

import time
from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class LLMClient:
    """LLM 客户端类"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o"):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    def _call_api(self, messages: list, temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        调用 LLM API

        Args:
            messages: 消息列表
            temperature: 温度参数
            top_p: top-p 参数

        Returns:
            LLM 返回的文本内容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()

    def generate_text(self, system_prompt: str, user_prompt: str,
                     temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        生成文本

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数
            top_p: top-p 参数

        Returns:
            生成的文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages, temperature, top_p)

    def generate_instruction(self, user_prompt: str, temperature: float = 0.7) -> str:
        """
        生成指令

        Args:
            user_prompt: 用户提示
            temperature: 温度参数

        Returns:
            生成的指令
        """
        system_prompt = "你是一个指令生成专家，能够根据角色描述和约束条件创建高质量、可验证的指令。"
        return self.generate_text(system_prompt, user_prompt, temperature)

    def generate_solution(self, user_prompt: str, temperature: float = 0.5) -> str:
        """
        生成解决方案

        Args:
            user_prompt: 用户提示
            temperature: 温度参数

        Returns:
            生成的解决方案
        """
        system_prompt = "你是一个专业的助手，能够精确遵循指令并满足所有约束条件。"
        return self.generate_text(system_prompt, user_prompt, temperature)

    def rewrite_instruction(self, user_prompt: str, temperature: float = 0.7) -> str:
        """
        重写指令

        Args:
            user_prompt: 用户提示
            temperature: 温度参数

        Returns:
            重写后的指令
        """
        system_prompt = "你是一个指令优化专家，能够重写指令并移除其中一个约束条件，同时保持指令的连贯性。"
        return self.generate_text(system_prompt, user_prompt, temperature)

    def get_usage_info(self) -> dict:
        """
        获取使用信息（占位方法）

        Returns:
            使用信息字典
        """
        return {
            "model": self.model,
            "status": "ready"
        }