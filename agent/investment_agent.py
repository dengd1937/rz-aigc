import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from agents import (
    Agent, 
    Runner, 
    function_tool, 
    InputGuardrail, 
    GuardrailFunctionOutput,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    OpenAIChatCompletionsModel
)
from agents.exceptions import InputGuardrailTripwireTriggered
from openai import AsyncOpenAI
from pydantic import BaseModel
from tavily import AsyncTavilyClient
from exa_py import Exa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DEEPSEEK_MODEL = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")


deepseek_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

set_default_openai_client(client=deepseek_client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
exa_client = Exa(api_key=EXA_API_KEY)

logger.info("所有客户端初始化完成")
logger.info("=" * 80)


def extract_json_from_markdown(text: str) -> str:
    """
    从 Markdown 代码块中提取 JSON 字符串
    
    Args:
        text: 可能包含 Markdown 代码块的文本
        
    Returns:
        提取出的 JSON 字符串
    """
    text = text.strip()
    
    if "```json" in text:
        start_idx = text.find("```json") + 7
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
    
    if "```" in text:
        start_idx = text.find("```") + 3
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            return text[start_idx:end_idx].strip()
    
    if text.startswith("{") and text.endswith("}"):
        return text
    
    try:
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
    except:
        pass
    
    return text


class InvestmentAnalysisOutput(BaseModel):
    """
    投资分析结果的结构化输出
    """
    recommendation: str
    market_outlook: str
    competitive_advantage: str
    funding_needs: str
    reasoning: str


class RiskAssessmentOutput(BaseModel):
    """
    风险评估结果的结构化输出
    """
    risk_level: str
    market_risk: str
    technical_risk: str
    financial_risk: str
    operational_risk: str
    mitigation_suggestions: str


@function_tool
async def search_investment_info(topic: str) -> str:
    """
    使用 Tavily 搜索投融资相关信息
    
    Args:
        topic: 投融资主题或公司名称
        
    Returns:
        返回投融资相关信息摘要
    """
    logger.info(f"[search_investment_info] 开始搜索: {topic}")
    
    try:
        response = await tavily_client.search(
            query=f"{topic} 融资 投资",
            search_depth="advanced",
            max_results=10,
            include_answer=True,
            include_raw_content=True
        )
        
        logger.info(f"[search_investment_info] Tavily 响应成功")
        logger.debug(f"[search_investment_info] 响应数据: {response}")
        
        results_summary = []
        if response.get("answer"):
            results_summary.append(f"答案: {response['answer']}")
        
        for result in response.get("results", []):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            results_summary.append(f"- {title}\n  {content}\n  链接: {url}")
        
        result_str = f"搜索到关于 {topic} 的投融资信息：\n" + "\n".join(results_summary)
        logger.info(f"[search_investment_info] 返回结果，长度: {len(result_str)} 字符")
        return result_str
    except Exception as e:
        error_msg = f"搜索投融资信息失败: {str(e)}"
        logger.error(f"[search_investment_info] {error_msg}")
        return error_msg


@function_tool
async def get_market_trends(industry: str) -> str:
    """
    使用 Exa 搜索行业市场趋势
    
    Args:
        industry: 行业名称
        
    Returns:
        返回行业市场趋势分析
    """
    logger.info(f"[get_market_trends] 开始搜索: {industry}")
    
    try:
        response = exa_client.search(
            query=f"{industry} industry trends market size growth rate",
            num_results=5,
            contents={"summary": True}
        )
        
        logger.info(f"[get_market_trends] Exa 响应成功")
        logger.debug(f"[get_market_trends] 响应数据: {response}")
        
        results_summary = []
        for result in response.results:
            title = result.title
            url = result.url
            content = result.text
            results_summary.append(f"- {title}\n  {content}\n  链接: {url}")
        
        result_str = f"获取到 {industry} 行业的市场趋势数据：\n" + "\n".join(results_summary)
        logger.info(f"[get_market_trends] 返回结果，长度: {len(result_str)} 字符")
        return result_str
    except Exception as e:
        error_msg = f"获取行业市场趋势失败: {str(e)}"
        logger.error(f"[get_market_trends] {error_msg}")
        return error_msg


@function_tool
async def analyze_company_financials(company: str) -> str:
    """
    使用 Tavily 搜索公司财务状况
    
    Args:
        company: 公司名称
        
    Returns:
        返回公司财务分析结果
    """
    logger.info(f"[analyze_company_financials] 开始分析: {company}")
    
    try:
        response = await tavily_client.search(
            query=f"{company} 财务数据 营收 利润 现金流 负债率",
            search_depth="advanced",
            max_results=10,
            include_answer=True,
            include_raw_content=True
        )
        
        logger.info(f"[analyze_company_financials] Tavily 响应成功")
        logger.debug(f"[analyze_company_financials] 响应数据: {response}")
        
        results_summary = []
        if response.get("answer"):
            results_summary.append(f"财务分析: {response['answer']}")
        
        for result in response.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            results_summary.append(f"- {title}\n  {content}")
        
        result_str = f"已分析 {company} 的财务状况：\n" + "\n".join(results_summary)
        logger.info(f"[analyze_company_financials] 返回结果，长度: {len(result_str)} 字符")
        return result_str
    except Exception as e:
        error_msg = f"分析公司财务状况失败: {str(e)}"
        logger.error(f"[analyze_company_financials] {error_msg}")
        return error_msg


deepseek_model = OpenAIChatCompletionsModel(
    model=DEEPSEEK_MODEL,
    openai_client=deepseek_client
)

risk_assessment_agent = Agent(
    name="Risk Assessment Agent",
    handoff_description="专业风险评估专家，负责分析评估投资项目的潜在风险",
    instructions="""你是一个专业的风险评估专家。你的任务是：

1. 分析投资项目的潜在市场风险、技术风险、财务风险、运营风险等
2. 评估风险等级（低、中、高）
3. 给出风险缓解建议

请基于提供的信息进行深入分析，给出专业、客观的风险评估报告。

重要：你必须以 JSON 格式输出，包含以下字段：
{
    "risk_level": "风险等级（低/中/高）",
    "market_risk": "市场风险分析",
    "technical_risk": "技术风险分析",
    "financial_risk": "财务风险分析",
    "operational_risk": "运营风险分析",
    "mitigation_suggestions": "风险缓解建议"
}

只输出 JSON 对象，不要输出任何其他内容。""",
    model=deepseek_model,
    tools=[
        get_market_trends,
        analyze_company_financials,
    ],
)

investment_analysis_agent = Agent(
    name="Investment Analysis Agent",
    handoff_description="专业投融资分析师，负责评估投资价值和市场前景",
    instructions="""你是一个专业的投融资分析师。你的任务是：

1. 分析投融资主题的市场前景和投资价值
2. 评估行业竞争格局和公司竞争优势
3. 分析融资需求和资金用途合理性
4. 提供投资建议（推荐、观望、不推荐）

请基于提供的信息进行深入分析，给出专业、客观的投资分析报告。

重要：你必须以 JSON 格式输出，包含以下字段：
{
    "recommendation": "投资建议（推荐/观望/不推荐）",
    "market_outlook": "市场前景分析",
    "competitive_advantage": "竞争优势分析",
    "funding_needs": "融资需求分析",
    "reasoning": "分析推理过程"
}

只输出 JSON 对象，不要输出任何其他内容。

注意：完成投资分析后，如果需要进一步的风险评估，可以将任务移交给风险评估专家。""",
    model=deepseek_model,
    tools=[
        search_investment_info,
        get_market_trends,
        analyze_company_financials,
    ],
    handoffs=[risk_assessment_agent],
)


class TopicValidationOutput(BaseModel):
    """
    主题验证结果
    """
    is_valid: bool
    topic_type: str
    reasoning: str


topic_validation_agent = Agent(
    name="Topic Validation Agent",
    instructions="""验证用户输入的投融资主题是否有效。
    
判断标准：
1. 是否是有效的公司名称或行业领域
2. 是否与投融资相关
3. 是否有足够的信息进行分析

重要：你必须以 JSON 格式输出，包含以下字段：
{
    "is_valid": true/false,
    "topic_type": "主题类型（公司/行业/其他）",
    "reasoning": "验证理由"
}

只输出 JSON 对象，不要输出任何其他内容。""",
    model=deepseek_model,
)


async def topic_guardrail(ctx, agent, input_data):
    """
    输入防护：验证投融资主题是否有效
    """
    logger.info(f"[topic_guardrail] 开始验证主题: {input_data}")
    
    try:
        result = await Runner.run(topic_validation_agent, input_data, context=ctx.context)
        output_text = result.final_output
        
        logger.info(f"[topic_guardrail] 原始输出: {output_text}")
        
        try:
            json_text = extract_json_from_markdown(output_text)
            output_json = json.loads(json_text)
            is_valid = output_json.get("is_valid", False)
            
            logger.info(f"[topic_guardrail] 验证结果: is_valid={is_valid}")
            logger.debug(f"[topic_guardrail] 验证输出详情: {output_json}")
            
            return GuardrailFunctionOutput(
                output_info=output_json,
                tripwire_triggered=not is_valid,
            )
        except json.JSONDecodeError as e:
            logger.error(f"[topic_guardrail] JSON 解析失败: {e}")
            logger.error(f"[topic_guardrail] 原始输出: {output_text}")
            
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reasoning": "输出格式错误"},
                tripwire_triggered=True,
            )
    except Exception as e:
        logger.error(f"[topic_guardrail] 验证失败: {e}")
        raise


orchestrator_agent = Agent(
    name="Investmentment Orchestrator",
    instructions="""你是投融资与风险判断的主协调者。你的任务是：

1. 理解用户的投融资主题或公司
2. 协调使用投资分析和风险评估专家
3. 综合投资分析和风险评估结果
4. 提供全面的投融资决策建议

工作流程：
1. 首先进行投资分析，评估投资价值
2. 然后进行风险评估，识别潜在项目风险
3. 综合两个方面的分析结果
4. 给出最终的投资建议和风险提示

请确保分析全面、客观、专业。""",
    model=deepseek_model,
    handoffs=[
        investment_analysis_agent,
        risk_assessment_agent,
    ],
    input_guardrails=[
        InputGuardrail(guardrail_function=topic_guardrail),
    ],
)


async def analyze_investment_topic(topic: str) -> dict:
    """
    分析投融资主题
    
    Args:
        topic: 投融资主题或公司名称
        
    Returns:
        返回分析结果，包含投资分析和风险评估
    """
    logger.info("=" * 80)
    logger.info(f"开始分析投融资主题: {topic}")
    logger.info("=" * 80)
    
    query = f"请对 '{topic}' 进行全面的投融资与风险分析，包括投资价值评估和风险评估。"
    
    try:
        logger.info("[analyze_investment_topic] 开始调用 Runner.run")
        result = await Runner.run(
            orchestrator_agent,
            input=query,
        )
        
        logger.info("[analyze_investment_topic] Runner.run 完成")
        
        output_text = result.final_output
        logger.info(f"[analyze_investment_topic] 原始输出: {output_text}")
        
        try:
            json_text = extract_json_from_markdown(output_text)
            output_json = json.loads(json_text)
            logger.info(f"[analyze_investment_topic] JSON 解析成功")
            
            return {
                "topic": topic,
                "analysis": output_json,
                "raw_result": result,
                "success": True,
            }
        except json.JSONDecodeError as e:
            logger.warning(f"[analyze_investment_topic] JSON 解析失败，返回原始文本: {e}")
            
            return {
                "topic": topic,
                "analysis": output_text,
                "raw_result": result,
                "success": True,
            }
    except InputGuardrailTripwireTriggered as e:
        logger.error(f"[analyze_investment_topic] 主题验证失败: {e}")
        return {
            "topic": topic,
            "error": "主题验证失败，请提供有效的投融资主题或公司名称",
            "success": False,
        }
    except Exception as e:
        logger.error(f"[analyze_investment_topic] 分析失败: {e}")
        logger.exception(f"[analyze_investment_topic] 异常详情: {e}")
        return {
            "topic": topic,
            "error": str(e),
            "success": False,
        }


async def main():
    """
    主函数，用于测试投融资 agent
    """
    topic = "2026年小米汽车"
    
    logger.info("=" * 80)
    logger.info("开始主函数")
    logger.info("=" * 80)
    
    result = await analyze_investment_topic(topic)
    
    print("\n" + "=" * 80)
    print("投融资投融资风险分析报告")
    print("=" * 80)
    print(f"分析主题: {result['topic']}")
    
    if result['success']:
        print("\n分析结果:")
        print(result['analysis'])
    else:
        print(f"\n错误: {result.get('error', '未知错误')}")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
