import os
import json
import httpx
import logging
import uvicorn
import prompts
from typing import Any
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from openai import AsyncOpenAI
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Server("ai-extractor")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

async def fetch_article_content(url: str) -> str:
    """
    异步获取微信文章的正文内容
    
    :param url: 微信文章的 URL
    :return: 文章的正文内容
    """
    logger.info(f"开始获取文章内容: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers=headers
        ) as http_client:
            response = http_client.get(url)
            response.raise_for_status()
            html_content = response.text
            logger.info(f"成功获取 HTML 内容，长度: {len(html_content)} 字符")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            body_text = ""
            
            for tag in soup.find_all(['p', 'div', 'section']):
                if tag.name == 'p' and tag.get_text(strip=True):
                    body_text += tag.get_text(strip=True) + "\n"
                elif tag.name in ['div', 'section']:
                    class_names = tag.get('class', [])
                    if any(cls in ['rich_media_content', 'js_content', 'wx_rich_media_content'] for cls in class_names):
                        body_text += tag.get_text(strip=True) + "\n"
            
            body_text = body_text.strip()
            logger.info(f"提取正文内容，长度: {len(body_text)} 字符")
            
            return body_text
    except Exception as e:
        logger.error(f"获取文章失败: {str(e)}")
        raise Exception(f"Failed to fetch article: {str(e)}")

async def get_llm_response(messages: list[dict[str, str]], 
                           model: str = os.getenv("MODEL_NAME"), 
                           think_type: str = "disabled") -> str:
    """
    异步调用 OpenAI API 获取 LLM 响应
    
    :param messages: 包含系统和用户消息的列表
    :param model: 使用的模型名称，默认从环境变量中获取
    :param think_type: 思考类型，默认禁用
    :return: LLM 的响应内容
    """
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=8192,
        extra_body={"thinking": {"type": think_type}}
    )
    return response.choices[0].message.content

async def extract_companies_with_llm(content: str) -> list[dict[str, Any]]:
    """
    异步调用 LLM 分析文章，提取AI领域涉及融资的公司信息
    
    :param content: 文章的正文内容
    :return: 包含公司名称、融资信息和是否属于AI领域的列表
    """
    logger.info(f"开始调用 LLM 分析文章，内容长度: {len(content)} 字符")

    try:
        # 提取文本内容中所有公司
        extract_messages = [
            {"role": "system", "content": prompts.EXTRACT_COMPANIES_PROMPT},
            {"role": "user", "content": f"文本内容如下:\n{content}"}
        ]
        extract_companies = await get_llm_response(extract_messages)
        if extract_companies is None:
            logger.error("LLM 返回的内容为 None")
            raise Exception("LLM 返回的内容为空")
        
        logger.info(f"提取到的公司: {extract_companies}")
        
        # 判断公司是否属于AI领域且融资
        judge_messages = [
            {"role": "system", "content": prompts.JUDGE_AI_FUNDING_PROMPT},
            {"role": "user", "content": f"公司列表: \n{extract_companies}\n{content}"}
        ]
        judge_result = await get_llm_response(judge_messages)
        if judge_result is None:
            logger.error("LLM 返回的内容为 None")
            raise Exception("LLM 返回的内容为空")
        
        return judge_result
    except Exception as e:
        logger.error(f"LLM 分析失败: {str(e)}")
        raise Exception(f"Failed to extract companies with LLM: {str(e)}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="extract_ai_finance_companies",
            description="从指定URL中的文章中提取AI领域涉及融资的公司信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "文章URL"
                    }
                },
                "required": ["url"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    logger.info(f"调用工具函数: {name}")
    logger.info(f"参数: {arguments}")
    
    if name == "extract_ai_finance_companies":
        url = arguments.get("url")
        
        if not url:
            logger.error("URL 参数为空")
            raise ValueError("URL is required")
        
        try:
            logger.info(f"开始处理 URL: {url}")
            content = await fetch_article_content(url)
            companies = await extract_companies_with_llm(content)
            
            result = json.dumps(companies, ensure_ascii=False, indent=2)
            logger.info(f"成功提取公司信息，结果: {result}")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            error_result = json.dumps({
                "error": str(e),
                "url": url
            }, ensure_ascii=False, indent=2)
            return [TextContent(type="text", text=error_result)]
    
    logger.error(f"未知工具: {name}")
    raise ValueError(f"Unknown tool: {name}")

sse = SseServerTransport("/messages/")

async def handle_sse(request: Request):
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send
    ) as streams:
        await app.run(
            streams[0],
            streams[1],
            app.create_initialization_options()
        )

routes = [
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
]

starlette_app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000)
