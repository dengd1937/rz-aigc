import json
import os
from typing import List, Dict, Generator, Union

import requests
from dotenv import load_dotenv
# OpenAI 兼容 SDK
from openai import OpenAI
# 各厂商官方 SDK
from dashscope import Generation
from zai import ZhipuAiClient
from anthropic import Anthropic


class LLMClient:
    """
    统一大模型调用封装类
    支持：GLM4 / Qwen3 / DeepSeek / Claude
    支持两种调用方式：
        1.call_openai_style：OpenAI兼容API
        2.call_sdk_style：厂商官方SDK
    """
    BASE_URLS = {
        "glm": "https://open.bigmodel.cn/api/paas/v4/",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "claude": "https://api.anthropic.com/v1",
    }

    def __init__(self):
        load_dotenv()
        self.api_keys = {
            "glm": os.getenv("ZAI_API_KEY"),
            "qwen": os.getenv("DASHSCOPE_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "claude": os.getenv("ANTHROPIC_API_KEY"),
        }

    # -----------------------------------------------------------
    # 方式一：OpenAI 风格（统一调用）
    # -----------------------------------------------------------
    def call_openai_style(
            self,
            provider: str,
            model: str,
            messages: List[Dict],
            stream: bool = False,
            **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        使用 OpenAI 风格接口调用
        """
        base_url = self.BASE_URLS.get(provider)
        api_key = self.api_keys.get(provider)
        if not base_url or not api_key:
            raise ValueError(f"Provider '{provider}' 未配置正确 base_url 或 api_key")
        client = OpenAI(api_key=api_key, base_url=base_url)

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.8,
            "max_tokens": 1024,
            "top_p": 0.9,
            **kwargs,  # 调用端可强制覆盖
        }

        if not stream:
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content
        else:
            def _stream_gen():
                for chunk in client.chat.completions.create(**payload):
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return _stream_gen()

    # -----------------------------------------------------------
    # 方式二：各厂商 SDK 调用（原生）
    # -----------------------------------------------------------
    def call_sdk_style(
            self,
            provider: str,
            model: str,
            messages: List[Dict],
            stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        使用厂商官方SDK调用
        """
        if provider == "glm":
            return self._call_glm(model, messages, stream)
        elif provider == "qwen":
            return self._call_qwen(model, messages, stream)
        elif provider == "deepseek":
            return self._call_deepseek(model, messages, stream)
        elif provider == "claude":
            return self._call_claude(model, messages, stream)
        else:
            raise ValueError(f"不支持的 provider: {provider}")

    # -------------------------
    # GLM
    # -------------------------
    def _call_glm(self, model, messages, stream):
        client = ZhipuAiClient(api_key=os.getenv("ZAI_API_KEY"))
        if not stream:
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content
        else:
            def _stream_gen():
                for chunk in client.chat.completions.create(model=model, messages=messages):
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            return _stream_gen()

    # -------------------------
    # Qwen
    # -------------------------
    def _call_qwen(self, model, messages, stream):
        if not stream:
            resp = Generation.call(api_key=os.getenv("DASHSCOPE_API_KEY"), model=model, messages=messages,
                                   result_format="message")
            return resp.output.choices[0].message.content
        else:
            def _stream_gen():
                for resp in Generation.call(api_key=os.getenv("DASHSCOPE_API_KEY"),
                                            model=model, messages=messages,
                                            result_format="message", stream=stream,
                                            incremental_output=True):
                    yield resp.output.choices[0].message.content

            return _stream_gen()
    # -------------------------
    # DeepSeek
    # -------------------------
    def _call_deepseek(self, model, messages, stream):
        if stream:
            return f"deepseek原生API不支持流式输出"

        url = "https://api.deepseek.com/chat/completions"
        payload = json.dumps({
            "messages": messages,
            "model": model
        })
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {os.getenv("DEEPSEEK_API_KEY")}'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        completion = json.loads(response.text)
        return completion['choices'][0]['message']['content']

    # -------------------------
    # Claude
    # -------------------------
    def _call_claude(self, model, messages, stream):
        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com"
        )

        if not stream:
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=1024
            )
            return response.content
        else:
            def _stream_gen():
                response = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    stream=stream
                )
                for resp in response:
                    yield resp.type
            return _stream_gen()

if __name__ == '__main__':
    client = LLMClient()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "1+1等于几？"},
    ]
    print("==============openai风格非流式输出==============")
    res = client.call_openai_style('deepseek', 'deepseek-reasoner', messages)
    print(res + "\n")

    print("==============openai风格流式输出==============")
    res = client.call_openai_style('deepseek', 'deepseek-reasoner', messages, stream=True)
    for chunk in res:
        print(chunk, end="", flush=True)
    print("\n")

    print("==============原生SDK风格非流式输出==============")
    res = client.call_sdk_style('deepseek', 'deepseek-chat', messages)
    print(res)

    print("==============原生SDK风格流式输出==============")
    res = client.call_sdk_style('deepseek', 'deepseek-chat', messages, stream=True)
    for chunk in res:
        print(chunk, end="", flush=True)
