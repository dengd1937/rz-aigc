from typing import Any, Dict, List
from uuid import UUID

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate


class CustomCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器"""

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID,
                            parent_run_id: UUID | None = None, tags: list[str] | None = None,
                            metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """Chat LLM开始处理时调用，打印各个参数的值"""
        print(f"LLM开始处理，参数如下：")
        print(f"serialized: {serialized}")
        print(f"messages: {messages}")
        print(f"run_id: {run_id}")
        print(f"parent_run_id: {parent_run_id}")
        print(f"tags: {tags}")
        print(f"metadata: {metadata}")
        print("=" * 50)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """LLM处理结束时调用，打印各个参数的值"""
        print(f"LLM处理结束，参数如下：")
        print(f"response: {response}")
        print(f"run_id: {run_id}")
        print(f"parent_run_id: {parent_run_id}")
        print("=" * 50)

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """Chain开始时调用，打印各个参数"""
        print(f"Chain开始处理，参数如下：")
        print(f"serialized: {serialized}")
        print(f"inputs: {inputs}")
        print(f"run_id: {run_id}")
        print(f"parent_run_id: {parent_run_id}")
        print(f"tags: {tags}")
        print(f"metadata: {metadata}")
        print("=" * 50)

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None,
                     **kwargs: Any) -> Any:
        """Chain结束时调用，打印各个参数"""
        print(f"Chain处理结束，参数如下：")
        print(f"outputs: {outputs}")
        print(f"run_id: {run_id}")
        print(f"parent_run_id: {parent_run_id}")
        print("=" * 50)

def main():
    load_dotenv()

    model = ChatTongyi(model="qwen3-30b-a3b-instruct-2507")
    prompt = PromptTemplate(
        input_variables=["company"],
        template="{company}的创始人是谁?"
    )
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"company": "小米"}, config={"callbacks": [CustomCallbackHandler()]}))


if __name__ == '__main__':
    main()






