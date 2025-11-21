from enum import Enum

from dotenv import load_dotenv
from langchain_classic.output_parsers import DatetimeOutputParser, EnumOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatTongyi(model="qwen3-30b-a3b-instruct-2507")

class ColorEnum(Enum):
    RED = "红色"
    GREEN = "绿色"
    BLUE = "蓝色"
    YELLOW = "黄色"
    PURPLE = "紫色"

class TaskOutputParser(BaseOutputParser):
    """自定义输出解析器，用于解析任务列表"""

    def parse(self, text: str) -> str:
        """
        输出格式: [任务名称] - [开始时间] - [截止时间] - [优先级]
        """
        return text

    def get_format_instructions(self) -> str:
        """返回格式说明"""
        return """请使用以下格式输出任务列表：
    [任务 1] - [开始时间] - [截止时间] - [优先级]
    [任务 2] - [开始时间] - [截止时间] - [优先级]
    [任务 3] - [开始时间] - [截止时间] - [优先级]
    ...
    例如：
    [完成项目报告] - [2023-10-01] - [2023-10-15] - [高]
    [准备会议材料] - [2023-10-10] - [2023-10-12] - [中]
    [回复邮件] - [2023-10-02] - [2023-10-03] - [低]"""


def create_chain(parser, template):
    """创建处理链的通用函数"""
    format_instructions = parser.get_format_instructions()

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
    )

    return prompt_template | model | parser

def datetime_parser_example(query):
    datetime_parser = DatetimeOutputParser(format="%Y-%m-%d %H:%M:%S")
    chain = create_chain(datetime_parser, "回答用户问题：{input}\n{format_instructions}")
    result = chain.invoke({"input": query})
    return result

def enum_parser_example(query):
    enum_parser = EnumOutputParser(enum=ColorEnum)
    chain = create_chain(enum_parser, "请回答以下问题，{format_instructions}。只返回选项值，不要添加解释或其他内容。\n\n问题：{input}")
    result = chain.invoke({"input": query})
    return result.value

def task_list_parser_example(query):
    task_parser = TaskOutputParser()
    chain = create_chain(task_parser, "{format_instructions}\n\n请根据以下要求生成任务列表：{input}")
    result = chain.invoke({"input": query})
    return result

def main():
    query = "为一个软件开发项目制定一个任务计划"
    result = task_list_parser_example(query)
    print(result)

if __name__ == "__main__":
    main()

