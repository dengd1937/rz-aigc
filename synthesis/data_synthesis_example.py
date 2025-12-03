import json
import os
import random

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def read_txt_file_by_paragraphs(file_path, num_samples=20):
    """
    读取txt文件，并按段落拆分，随机取指定数量的分片

    Args:
        file_path (str): 文件路径
        num_samples (int): 需要随机采样的段落数量，默认为20

    Returns:
        list: 包含段落内容的列表，每个元素为一个段落
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 按段落拆分（使用两个换行符作为段落分隔符）
        paragraphs = content.split('\n\n')

        # 过滤掉空段落
        paragraphs = [para.strip() for para in paragraphs if para.strip()]

        # 如果段落数量少于需要的样本数，则返回所有段落
        if len(paragraphs) <= num_samples:
            sampled_paragraphs = paragraphs
        else:
            # 随机采样指定数量的段落
            sampled_paragraphs = random.sample(paragraphs, num_samples)

        return sampled_paragraphs

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

def _get_response(model:str, system_prompt:str, prompt: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_question(model: str, content: str) -> str:
    """
    生成问题

    :param model: 大模型名称
    :param content: 段落内容
    :return 问题
    """
    system_prompt = """你是一位问题生成专家，擅长分析输入文本，能基于文本信息生成理解与细节类问题。
    你具有如下技能：
    1. 分析文本内容
        -在提供的文本中确定关键主题、论点和细节。
        -识别文本中隐含的信息、上下文和潜在的空白。
    2. 确保问题质量
        -使问题清晰、简洁，并与文本直接相关。
        -避免引导性或带有偏见的措辞；保持中立。
    
    Constraints：
        -严格根据提供的文本回答所有问题；不要引入外部信息。
        -根据文本的内容和目的，生成平衡的问题类型组合。
        -用与输入文本相同的语言表达问题。
        -每次只生成一个问题，不生成除问题之外任何文本。
    """
    prompt = f"文本内容：\n{content}"
    return _get_response(model=model, system_prompt=system_prompt, prompt=prompt)

def generate_answer(model: str, content: str, question: str) -> str:
    """
    生成选项和答案选项

    :param model: 大模型
    :param content: 文本内容
    :param question: 问题
    :return: 选项和答案选项
    """
    system_prompt = """你是一个专业的考试题目生成器，能够根据用户提供的文本和问题，生成对应的A、B、C、D四个选项以及正确答案。
    你具有如下技能:
    1. 理解文本和问题
        - 仔细阅读用户提供的文本和问题，确保完全理解其内容和意图。
        - 识别问题的类型（如事实细节、推理判断、词汇理解等）。
    2. 生成选项
        - 基于文本内容，生成四个合理的选项（A、B、C、D）。
        - 确保其中三个选项为干扰项，一个为正确答案。
        - 干扰项应具有迷惑性，但必须与文本内容有明显区别。
        - 正确答案必须严格依据文本内容，准确无误。
    3. 标注正确答案
        - 明确标注哪个选项是正确答案（例如：ans：C）。
        - 确保答案选项与生成的选项一致。
    
    Constraints:
        - 只基于用户提供的文本生成选项和答案，不引入外部知识。
        - 选项语言需与文本语言一致。
        - 每个选项应简洁明了，避免冗长。
        - 确保生成的选项覆盖问题所问的不同方面，但只有一个符合文本原意。
    
    输出格式如下所示: 
    A、xxx\nB、xxx\nC、xxx\nD、xxx\nans:B
    """
    prompt = f"文本内容：\n{content}\n问题：\n{question}"
    return _get_response(model=model, system_prompt=system_prompt, prompt=prompt)

def main():
    question_model = "gpt-4.1-nano"
    ans_model = "o4-mini"

    # 读取txt文件，并按段落拆分，随机取20个分片
    file_path = "./data/三国演义.txt"
    paragraphs = read_txt_file_by_paragraphs(file_path)

    # 生成问题、选项和答案
    qa_pairs = []
    for i, paragraph in enumerate(paragraphs):
        question = generate_question(question_model, paragraph)
        answer = generate_answer(ans_model, paragraph, question)
        print(f"段落内容: {paragraph}")
        print(f"问题: {question}")
        print(f"选项: {answer}")

        qa_pair = [
            {"role": "user", "content": question},
            {"role": "gpt", "content": answer}
        ]
        qa_pairs.append(qa_pair)

    # 将数据写入文件
    output_file = "./outputs/qa_pairs.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa_pair in qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    print(f"已将 {len(qa_pairs)} 个QA对保存到 {output_file}")


if __name__ == '__main__':
    main()
