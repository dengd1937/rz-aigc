import json
import os
import random
import sys
import time

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

def classify_instruction(instruction: str, model_name: str, max_retries: int = 3) -> str:
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
    for attempt in range(max_retries):
        try:
            # 执行分类
            result = chain.invoke({"instruction": instruction})
            return result.strip()
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试分类失败: {e}")
            if attempt < max_retries - 1:
                # 指数退避策略，随机等待一段时间再重试
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"等待 {wait_time:.2f} 秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)
            else:
                print("达到最大重试次数，分类失败")
                raise e

def classify_dataset(dataset_path: str, model_name: str, output_path: str):
    """
    对整个数据集进行分类

    Args:
        dataset_path: 包含指令的JSONL文件路径
        model_name: 模型名称
        output_path: 输出分类结果的文件路径
    """
    start_line = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            start_line = sum(1 for _ in f)
    print(f"从第 {start_line + 1} 行开始处理")

    base_name = os.path.splitext(output_path)[0]
    error_path = f"{base_name}_errors.jsonl"

    with open(dataset_path, "r", encoding="utf-8") as infile, \
            open(output_path, "a", encoding="utf-8") as outfile, \
            open(error_path, "a", encoding="utf-8") as errorfile:
        # 跳过已经处理过的行
        for _ in range(start_line):
            next(infile)

        for line in infile:
            data = json.loads(line)
            instruction = data["instruction"]
            id = data.get("id")
            try:
                # 仅测试使用
                if id == 10:
                    raise Exception("测试异常抛出")

                classification = classify_instruction(instruction, model_name)
                data["classification"] = classification

                # 仅测试使用
                if id == 15:
                    sys.exit(1)

                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                outfile.flush()

            except Exception as e:
                error_data = {
                    "id": id,
                    "instruction": instruction,
                    "error": str(e)
                }
                errorfile.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                errorfile.flush()

def main():
    # dataset_folder = "./datasets/infinity_instruct/extracted_data"
    # extract_conversations(dataset_folder)

    # 使用langchain的LECL表达式让大模型对数据进行分类标签
    input_path = "./datasets/infinity_instruct/extracted_data/infinity_instruct.jsonl"
    output_path = "./datasets/infinity_instruct/infinity_instruct_classify.jsonl"
    qwen_model = "qwen3-235b-a22b-instruct-2507"
    classify_dataset(input_path, qwen_model, output_path)


if __name__ == '__main__':
    main()