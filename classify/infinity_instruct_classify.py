import json
import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def extract_conversations(dataset_folder):
    """
    提取conversations中第一个from等于"human"的vale值

    Args:
        dataset_folder (str): JSONL文件的目录
    """
    for file_name in os.listdir(dataset_folder):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(dataset_folder, file_name)
            # 构造输出文件路径
            output_file_name = "infinity_instruct.jsonl"
            output_file_path = os.path.join(dataset_folder, output_file_name)

            with open(file_path, "r", encoding="utf-8") as infile, \
                    open(output_file_path, "w", encoding="utf-8") as outfile:
                for line in infile:
                    data = json.loads(line)
                    conversations = data["conversations"]
                    first_human_value = None

                    # 查找第一个from等于"human"的对话
                    for conversation in conversations:
                        if conversation["from"] == "human":
                            first_human_value = conversation["value"]
                            break

                    # 创建新的数据结构，只包含id和提取的值
                    extracted_data = {
                        "id": data.get("id", None),
                        "instruction": first_human_value
                    }

                    # 写入到输出文件
                    outfile.write(json.dumps(extracted_data, ensure_ascii=False) + "\n")

def classify_instruction(instruction: str, model_name: str) -> str:
    """
    使用Langchain的LCEL表达式对单条指令进行分类

    Args:
        instruction: 需要分类的指令文本

    Returns:
        分类标签
    """
    classification_template = """
    请根据以下指令内容对其进行分类。分类应该基于指令的主要目的或主题。

    可能的分类标签包括：
    - 数据处理：涉及数据清洗、转换、分析等任务
    - 代码生成：要求生成或修改代码的任务
    - 问题解答：寻求特定问题答案的任务
    - 文本创作：创作、润色或修改文本内容的任务
    - 其他：不属于以上类别的任务

    请仅回复最适合的分类标签，不要包含其他解释或说明。

    指令内容：
    {instruction}

    分类标签：
    """

    prompt = PromptTemplate.from_template(classification_template)
    model = ChatTongyi(model=model_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    classification = chain.invoke({"instruction": instruction})
    return classification.strip()

def main():
    # dataset_folder = "./datasets/infinity_instruct/extracted_data"
    # extract_conversations(dataset_folder)

    # 使用langchain的LECL表达式让大模型对数据进行分类标签



    pass


if __name__ == '__main__':
    main()