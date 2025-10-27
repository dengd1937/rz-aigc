import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

import requests
from anthropic import Anthropic
from dashscope import Generation
from dotenv import load_dotenv
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from zai import ZhipuAiClient


class LLMClient(ABC):
    """LLM客户端抽象基类"""

    def __init__(self, cfg: Dict[str, Any]):
        # 把 JSON 里所有字段都存下来，后续想加 max_tokens、top_n 直接读即可
        self.cfg = cfg

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """聊天完成接口"""
        pass

class OllamaClient(LLMClient):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.base_url = cfg["base_url"].rstrip("/")

        # 设置重试策略
        retry = Retry(total=2, connect=2, read=2, status=2, redirect=0, raise_on_status=True)
        adapter = HTTPAdapter(max_retries=retry)

        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        options = {
            "temperature": self.cfg.get("temperature", 0.8),
            "top_p": self.cfg.get("top_p", 0.9),
            "num_predict": self.cfg.get("max_tokens", 1024)
        }

        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")

        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": stream,
            "options": options
        }
        payload.update(kwargs)
        if stream:
            return self._stream_request("/api/chat", payload)
        else:
            return self._json_request("/api/chat", payload)

    def _stream_request(self, endpoint: str, payload: Dict) -> Iterator[Dict]:
        """流式请求"""
        print(f"请求参数：{payload}")
        with self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines(delimiter=b"\n"):
                if not line:
                    continue
                part = json.loads(line)
                yield part["message"]["content"]

    def _json_request(self, endpoint: str, payload: Dict) -> Dict:
        print(f"请求参数：{payload}")
        resp = self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=60)
        resp.raise_for_status()  # 重试 3 次后仍失败直接抛错
        return resp.json()["message"]["content"]

class OpenAIClient(LLMClient):
    """OpenAI API 客户端"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv(cfg["api_key_name"]),
            base_url=cfg["base_url"]
        )

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": stream,
            "temperature": self.cfg.get("temperature", 0.8),
            "max_tokens": self.cfg.get("max_tokens", 1024),
            "top_p": self.cfg.get("top_p", 0.9),
            **kwargs,  # 调用端可强制覆盖
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        print(f"请求参数：{payload}")

        if stream:
            return self._stream_chat(payload)
        else:
            return self._non_stream_chat(payload)

    def _stream_chat(self, payload):
        """流式聊天实现"""
        response = self.client.chat.completions.create(**payload)
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _non_stream_chat(self, payload):
        """非流式聊天实现"""
        resp = self.client.chat.completions.create(**payload)
        return resp.choices[0].message.content

class GLMNativeClient(LLMClient):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        load_dotenv()
        self.client = ZhipuAiClient(api_key=os.getenv("ZAI_API_KEY"))

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": stream,
            "temperature": self.cfg.get("temperature", 0.8),
            "max_tokens": self.cfg.get("max_tokens", 1024),
            **kwargs,  # 调用端可强制覆盖
        }
        print(f"请求参数：{payload}")
        if stream:
            def _stream_gen():
                for chunk in self.client.chat.completions.create(**payload):
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return _stream_gen()
        else:
            resp = self.client.chat.completions.create(**payload)
            return resp.choices[0].message.content

class QwenNativeClient(LLMClient):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        load_dotenv()

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": stream,
            "temperature": self.cfg.get("temperature", 0.8),
            "max_tokens": self.cfg.get("max_tokens", 1024),
            "enable_thinking": self.cfg.get("enable_thinking", False),
            "result_format": "message",
            "incremental_output": False,
            **kwargs,  # 调用端可强制覆盖
        }
        print(f"请求参数：{payload}")
        if not stream:
            resp = Generation.call(api_key=os.getenv("DASHSCOPE_API_KEY"), **payload)
            return resp.output.choices[0].message.content
        else:
            payload["incremental_output"] = True
            def _stream_gen():
                for resp in Generation.call(api_key=os.getenv("DASHSCOPE_API_KEY"),**payload):
                    yield resp.output.choices[0].message.content

            return _stream_gen()

class ClaudeNativeClient(LLMClient):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        load_dotenv()
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com"
        )

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": stream,
            "temperature": self.cfg.get("temperature", 0.8),
            "max_tokens": self.cfg.get("max_tokens", 1024),
            **kwargs,  # 调用端可强制覆盖
        }
        print(f"请求参数：{payload}")
        if not stream:
            response = self.client.messages.create(**payload)
            return response.content
        else:
            def _stream_gen():
                for resp in self.client.messages.create(**payload):
                    yield resp.type
            return _stream_gen()


CLIENT_REGISTRY: Dict[str, type[LLMClient]] = {
    "openai": OpenAIClient,
    "ollama": OllamaClient,
    "glm_native": GLMNativeClient,
    "qwen_native": QwenNativeClient,
    "claude_native": ClaudeNativeClient
}

def build_client(config_path: str) -> LLMClient:
    with open(config_path, encoding="utf-8") as f:
        full_cfg = json.load(f)

    default = full_cfg["default_client"]
    print(f"当前使用 {default} client")
    client_cfg = full_cfg["clients"][default]

    cls = CLIENT_REGISTRY[default]
    return cls(client_cfg)

if __name__ == "__main__":
    llm = build_client("config.json")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "strawberry中有几个r？"},
    ]

    print("==============非流式输出==============")
    answer = llm.chat(messages, max_tokens=1024, temperature=0.8, stream=False)
    print(answer)

    print("==============流式输出==============")
    for chunk in llm.chat(messages, max_tokens=1024, temperature=0.8, stream=True):
        print(chunk, end='', flush=True)