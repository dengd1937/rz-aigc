import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from eval.mmlu_evaluator_prompts import *

load_dotenv()

# client = OpenAI()
# MODEL_NAMES = ["gpt-3.5-turbo", "gpt-4.1-mini"]
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL_NAMES = ['qwen2.5-32b-instruct', 'qwen3-30b-a3b-instruct-2507']

NUMBER_PATTERN = r'<answer>(\d+)</answer>'
THINK_PATTERN = r'<think>(.*?)</think>'
OUTPUT_FILE_PATH = "./results/mmlu_evaluation_results.csv"

def extract_answer(response: str, pattern: str) -> str:
    """从模型响应中提取"""
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def zero_shot(model:str, question: str, choices: str, max_tokens:int = 10):
    """零样本评测"""
    prompt = ZERO_SHOT_PROMPT.format(question=question, choices=choices)
    res = get_response(model, "", prompt, max_tokens)
    return extract_answer(res, NUMBER_PATTERN)

def zero_shot_cot(model:str, question: str, choices: str):
    """零样本cot评测"""
    system_prompt = "请进行逐步推理分析"
    prompt = ZERO_SHOT_COT_PROMPT.format(question=question, choices=choices)
    res = get_response(model, system_prompt, prompt)
    thinking = extract_answer(res, THINK_PATTERN)
    answer = extract_answer(res, NUMBER_PATTERN)
    return thinking, answer

def one_shot(model:str, question: str, choices: str, max_tokens:int = 10):
    """ one-shot 评测 """
    prompt = ONE_SHOT_PROMPT.format(question=question, choices=choices)
    res = get_response(model, "", prompt, max_tokens)
    return extract_answer(res, NUMBER_PATTERN)

def one_shot_cot(model:str, question: str, choices: str):
    """one-shot cot评测"""
    system_prompt = "请进行逐步推理分析"
    prompt = ONE_SHOT_COT_PROMPT.format(question=question, choices=choices)
    res = get_response(model, system_prompt, prompt)
    thinking = extract_answer(res, THINK_PATTERN)
    answer = extract_answer(res, NUMBER_PATTERN)
    return thinking, answer

def three_shot(model:str, question: str, choices: str, max_tokens:int = 10):
    """ three-shot 评测 """
    prompt = THREE_SHOT_PROMPT.format(question=question, choices=choices)
    res = get_response(model, "", prompt, max_tokens)
    return extract_answer(res, NUMBER_PATTERN)

def three_shot_cot(model:str, question: str, choices: str):
    """one-shot cot评测"""
    system_prompt = "请进行逐步推理分析"
    prompt = THREE_SHOT_COT_PROMPT.format(question=question, choices=choices)
    res = get_response(model, system_prompt, prompt)
    thinking = extract_answer(res, THINK_PATTERN)
    answer = extract_answer(res, NUMBER_PATTERN)
    return thinking, answer

def get_response(model:str, system_prompt:str, prompt: str, max_tokens:int = 8192):
    # print(f"system_prompt: {system_prompt}\nprompt:{prompt}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def load_sample(files_path: str, n=3):
    df = load_dataset(files_path)
    return df.sample(n)

def load_dataset(files_path: str):
    # 读取文件夹下所有CSV文件并合并
    print(f"读取数据集文件夹: {files_path}")
    csv_files = []
    for file in os.listdir(files_path):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(files_path, file))

    print(f"找到 {len(csv_files)} 个数据集文件:")
    dataframes = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"已加载 {os.path.basename(file_path)}: {len(df)} 条数据")
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
    return pd.concat(dataframes, ignore_index=True)

def save_results(results: list, output_file: str):
    results_df = pd.DataFrame(results)
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除已存在的文件: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"详细评估结果已保存至{output_file}")

def print_stats(model_stats):
    print("\n=================== 所有模型总评估结果 ===================")
    for model, stats in model_stats.items():
        zero_shot_acc = stats['zero_shot_correct'] / stats['total']
        zero_shot_cot_acc = stats['zero_shot_cot_correct'] / stats['total']
        one_shot_acc = stats['one_shot_correct'] / stats['total']
        one_shot_cot_acc = stats['one_shot_cot_correct'] / stats['total']
        three_shot_acc = stats['three_shot_correct'] / stats['total']
        three_shot_cot_acc = stats['three_shot_cot_correct'] / stats['total']
        print(f"模型 {model}:")
        print(f"总题目数: {stats['total']}")
        print(f"Zero-Shot 正确数: {stats['zero_shot_correct']} ({zero_shot_acc:.2%})")
        print(f"Zero-Shot CoT 正确数: {stats['zero_shot_cot_correct']} ({zero_shot_cot_acc:.2%})")
        print(f"One-Shot 正确数: {stats['one_shot_correct']} ({one_shot_acc:.2%})")
        print(f"One-Shot CoT 正确数: {stats['one_shot_cot_correct']} ({one_shot_cot_acc:.2%})")
        print(f"Three-Shot 正确数: {stats['three_shot_correct']} ({three_shot_acc:.2%})")
        print(f"Three-Shot CoT 正确数: {stats['three_shot_cot_correct']} ({three_shot_cot_acc:.2%})")

def main():
    data_path = "./datasets/chinese_data"
    df = load_dataset(data_path)

    model_stats = {}
    results = []
    for model in MODEL_NAMES:
        print(f"开始评测模型: {model}")
        stats = {
            'total': 0,
            'zero_shot_correct': 0,
            'zero_shot_cot_correct': 0,
            'one_shot_correct': 0,
            'one_shot_cot_correct': 0,
            'three_shot_correct': 0,
            'three_shot_cot_correct': 0
        }
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
            question = row['question']
            subject = row['subject']
            choices = row['choices']
            answer = str(row['answer'])

            # 评测
            zero_shot_answer = zero_shot(model, question, choices)
            one_shot_answer = one_shot(model, question, choices)
            three_shot_answer = three_shot(model, question, choices)
            zero_shot_cot_think, zero_shot_cot_answer = zero_shot_cot(model, question, choices)
            one_shot_cot_think, one_shot_cot_answer = one_shot_cot(model, question, choices)
            three_shot_cot_think, three_shot_cot_answer = three_shot_cot(model, question, choices)


            stats['total'] += 1
            if zero_shot_answer == answer:
                stats['zero_shot_correct'] += 1
            if one_shot_answer == answer:
                stats['one_shot_correct'] += 1
            if three_shot_answer == answer:
                stats['three_shot_correct'] += 1
            if zero_shot_cot_answer == answer:
                stats['zero_shot_cot_correct'] += 1
            if one_shot_cot_answer == answer:
                stats['one_shot_cot_correct'] += 1
            if three_shot_cot_answer == answer:
                stats['three_shot_cot_correct'] += 1

            results.append({
                'model': model,
                'question': question,
                'subject': subject,
                'choices': choices,
                'answer': answer,
                'zero_shot_answer': zero_shot_answer,
                'zero_shot_cot_think': zero_shot_cot_think,
                'zero_shot_cot_answer': zero_shot_cot_answer,
                'one_shot_answer': one_shot_answer,
                'one_shot_cot_think': one_shot_cot_think,
                'one_shot_cot_answer': one_shot_cot_answer,
                'three_shot_answer': three_shot_answer,
                'three_shot_cot_think': three_shot_cot_think,
                'three_shot_cot_answer': three_shot_cot_answer
            })

        model_stats[model] = stats

    save_results(results, OUTPUT_FILE_PATH)
    print_stats(model_stats)

if __name__ == '__main__':
    main()