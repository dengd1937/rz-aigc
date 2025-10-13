import os
import time
from typing import List

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from eval_prompts import *

load_dotenv()

# 初始化客户端
client = OpenAI()

# 模型列表
MODEL_NAMES = ["gpt-3.5-turbo", "gpt-4.1-mini"]
# MODEL_NAMES = ['qwen2.5:8b']

def parse_choices(choices_str: str) -> List[str]:
    """解析选项字符串"""
    try:
        choices = eval(choices_str)
        return choices
    except:
        return []

def transform_answer(answer: int) -> str:
    """答案从数字0、1、2、3转成A、B、C、D字母"""
    answer_map = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D'
    }
    return answer_map.get(answer, '')

def format_question(question: str, choices: List[str]) -> str:
    """格式化问题（用户提示词）"""
    prompt = f"""问题：{question}

选项：
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

请回答选项字母（A、B、C或D）"""
    return prompt

def call_api(system_prompt: str, prompt: str, model_name: str, max_tokens: int, max_retries: int = 1) -> str:
    """调用API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""
    return ""

def call_api_zero_shot(prompt: str, model_name: str, max_retries: int = 1) -> str:
    """调用API - Zero Shot模式"""
    return call_api(ZERO_SHOT_PROMPT, prompt, model_name, 10, max_retries)

def call_api_zero_shot_cot(prompt: str, model_name: str, max_retries: int = 1) -> str:
    """调用API - Zero Shot CoT模式"""
    return call_api(ZERO_SHOT_COT_PROMPT, prompt, model_name, 2048, max_retries)

def call_api_one_shot(prompt: str, model_name: str, max_retries: int = 1) -> str:
    """调用API - Few Shot模式"""
    return call_api(ONE_SHOT_PROMPT, prompt, model_name, 10, max_retries)

def call_api_one_shot_cot(prompt: str, model_name: str, max_retries: int = 1) -> str:
    """调用API - Few Shot模式"""
    return call_api(ONE_SHOT_COT_PROMPT, prompt, model_name, 2048, max_retries)

def call_api_three_shot(prompt: str, model_name: str, max_retries: int = 1) -> str:
    """调用API - Few Shot模式"""
    return call_api(THREE_SHOT_PROMPT, prompt, model_name, 10, max_retries)

def call_api_three_shot_cot(prompt: str, model_name: str, max_retries: int = 1) -> str:
    """调用API - Few Shot模式"""
    return call_api(THREE_SHOT_COT_PROMPT, prompt, model_name, 2048, max_retries)

def extract_answer(response: str) -> str:
    """提取答案字母"""
    response = response.strip().upper()
    for char in ['A', 'B', 'C', 'D']:
        if char in response:
            return char
    return ""

def extract_answer_and_process(response: str) -> tuple:
    """从CoT响应中提取答案和推理过程"""
    # 提取推理过程
    process = ""
    answer = ""

    if "推理过程" in response and "最终答案" in response:
        parts = response.split("最终答案")
        process = parts[0].replace("推理过程", "").replace("：", "").replace(":", "").strip()
        answer_part = parts[1]
        answer = extract_answer(answer_part)
    else:
        # 如果格式不标准，整个作为process，尝试提取答案
        process = response
        answer = extract_answer(response)

    return answer, process

def main():
    dataset_folder = "./datasets/chinese_data"

    # 读取文件夹下所有CSV文件并合并
    print(f"读取数据集文件夹: {dataset_folder}")
    csv_files = []
    for file in os.listdir(dataset_folder):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(dataset_folder, file))

    print(f"找到 {len(csv_files)} 个数据集文件:")
    dataframes = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"已加载 {os.path.basename(file_path)}: {len(df)} 条数据")
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")

    df = pd.concat(dataframes, ignore_index=True)
    total = len(df)
    print(f"\n共 {total} 条数据")

    # dataset_path = "./datasets/chinese_data/chinese-college-chemistry-test-00000-of-00001.csv"
    # # 读取数据
    # print(f"读取数据集: {dataset_path}")
    # df = pd.read_csv(dataset_path)
    # total = len(df)
    # print(f"共 {total} 条数据")

    # 为每个模型准备统计结果
    model_stats = {}

    for model in MODEL_NAMES:
        print(f"开始评测模型: {model}")

        # 初始化统计数据
        stats = {
            'total': 0,
            'zero_shot_correct': 0,
            'cot_correct': 0,
            'one_shot_correct': 0,
            'one_shot_cot_correct': 0,
            'three_shot_correct': 0,
            'three_shot_cot_correct': 0
        }

        for idx, row in tqdm(df.iterrows(), total=total, desc="处理中"):
            question = row['question']
            subject = row['subject']
            choices_str = row['choices']
            answer_int = row['answer']

            choices = parse_choices(choices_str)
            if len(choices) != 4:
                print(f"警告: 第 {idx} 行选项 {choices_str} 数量不是4，跳过")
                continue

            answer = transform_answer(answer_int)
            if answer == '':
                print(f"警告: 第 {idx} 行答案 {answer_int} 索引转换失败，跳过")
                continue

            prompt = format_question(question, choices)

            # zero_shot
            zero_shot_response = call_api_zero_shot(prompt, model)
            zero_shot_answer = extract_answer(zero_shot_response)
            time.sleep(0.5)  # 避免请求过快

            # zero_shot_cot
            cot_response = call_api_zero_shot_cot(prompt, model)
            cot_answer, cot_process = extract_answer_and_process(cot_response)
            time.sleep(0.5)  # 避免请求过快

            # one_shot
            one_shot_response = call_api_one_shot(prompt, model)
            one_shot_answer = extract_answer(one_shot_response)
            time.sleep(0.5)  # 避免请求过快

            # one_shot_cot
            one_shot_cot_response = call_api_one_shot_cot(prompt, model)
            one_shot_cot_answer = extract_answer(one_shot_cot_response)
            time.sleep(0.5)  # 避免请求过快

            # three_shot
            three_shot_response = call_api_three_shot(prompt, model)
            three_shot_answer = extract_answer(three_shot_response)
            time.sleep(0.5)  # 避免请求过快

            # three_shot_cot
            three_shot_cot_response = call_api_three_shot_cot(prompt, model)
            three_shot_cot_answer = extract_answer(three_shot_cot_response)
            time.sleep(0.5) # 避免请求过快

            # 更新统计数据
            stats['total'] += 1
            # print(f"{idx + 1}条数据处理完毕，原本答案是：{answer}，zero_shot是{zero_shot_answer}，zero_shot_cot是{cot_answer}\n")
            if zero_shot_answer == answer:
                stats['zero_shot_correct'] += 1
            if cot_answer == answer:
                stats['cot_correct'] += 1
            if one_shot_answer == answer:
                stats['one_shot_correct'] += 1
            if one_shot_cot_answer == answer:
                stats['one_shot_cot_correct'] += 1
            if three_shot_answer == answer:
                stats['three_shot_correct'] += 1
            if three_shot_cot_answer == answer:
                stats['three_shot_cot_correct'] += 1

        # 保存模型统计数据
        model_stats[model] = stats

    print("\n=================== 所有模型总评估结果 ===================")
    for model, stats in model_stats.items():
        zero_shot_acc = stats['zero_shot_correct'] / stats['total']
        cot_acc = stats['cot_correct'] / stats['total']
        one_shot_acc = stats['one_shot_correct'] / stats['total']
        one_shot_cot_acc = stats['one_shot_cot_correct'] / stats['total']
        three_shot_acc = stats['three_shot_correct'] / stats['total']
        three_shot_cot_acc = stats['three_shot_cot_correct'] / stats['total']
        print(f"模型 {model}:")
        print(f"总题目数: {stats['total']}")
        print(f"Zero-Shot 正确数: {stats['zero_shot_correct']} ({zero_shot_acc:.2%})")
        print(f"Zero-Shot CoT 正确数: {stats['cot_correct']} ({cot_acc:.2%})")
        print(f"One-Shot 正确数: {stats['one_shot_correct']} ({one_shot_acc:.2%})")
        print(f"One-Shot CoT 正确数: {stats['one_shot_cot_correct']} ({one_shot_cot_acc:.2%})")
        print(f"Three-Shot 正确数: {stats['three_shot_correct']} ({three_shot_acc:.2%})")
        print(f"Three-Shot CoT 正确数: {stats['three_shot_cot_correct']} ({three_shot_cot_acc:.2%})")
        print('-' * 50)

if __name__ == '__main__':
    main()