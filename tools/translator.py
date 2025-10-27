import csv
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

system_prompt_question = """你是一位专业的学术翻译专家，精通英文和中文。
你的任务是将MMLU（Massive Multitask Language Understanding）基准测试数据集中的内容翻译成中文。

翻译要求：
1. 保持学术严谨性和专业性，使用准确的学术术语
2. 对于专业术语，优先使用该领域通用的中文翻译
3. 保持原文的语气和风格（如疑问句保持疑问，陈述句保持陈述）
4. 翻译要自然流畅，符合中文表达习惯
5. 对于人名、地名等专有名词，使用常见的中文译名
6. 保持选项的简洁性和准确性
7. 不要添加或删减原文中的信息
8. 对于科学符号、公式、单位等，保持原样或使用国际通用表达

请仅返回翻译后的文本，不要包含任何解释或额外说明。"""

system_prompt_choices = """你是一位专业的学术翻译专家，精通英文和中文。
你的任务是将MMLU（Massive Multitask Language Understanding）基准测试数据集中的选项翻译成中文。

翻译要求：
1. 保持学术严谨性和专业性，使用准确的学术术语
2. 对于专业术语，优先使用该领域通用的中文翻译
3. 翻译要自然流畅，符合中文表达习惯
4. 对于人名、地名等专有名词，使用常见的中文译名
5. 不要添加或删减原文中的信息
6. 对于科学符号、公式、单位等，保持原样或使用国际通用表达

重要说明：
- 输入是一个包含多个选项的列表
- 每个选项可能包含多个完形填空答案（用逗号分隔）
- 你需要翻译每个选项内部的内容，但保持选项之间的分隔
- 必须严格按照输入的格式返回，保持列表结构

返回格式：['选项1翻译', '选项2翻译', '选项3翻译', '选项4翻译']
请仅返回翻译后的列表，不要包含任何解释或额外说明。"""

client = OpenAI()

def translate_text(client, text, system_prompt,max_retries=3):
    """
    使用OpenAI API翻译文本
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请将以下英文翻译成中文：\n\n{text}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"翻译失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return text  # 如果所有重试都失败，返回原文
    return text


def translate_csv(input_file, output_file):
    """
    翻译CSV文件中的内容
    """
    print(f"正在处理文件: {input_file}")

    # 读取CSV文件
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        print(f"警告: {input_file} 是空文件")
        return

    translated_rows = []
    total = len(rows)

    for idx, row in enumerate(rows, 1):
        print(f"进度: {idx}/{total}")
        translated_row = {}

        # 翻译问题
        if 'question' in row:
            print(f"  翻译问题...")
            translated_row['question'] = translate_text(client, row['question'], system_prompt_question)

        # 保持subject不变或翻译
        if 'subject' in row:
            translated_row['subject'] = row['subject']

        # 翻译选项
        if 'choices' in row:
            print(f"  翻译选项...")
            translated_row['choices'] = translate_text(client, row['choices'], system_prompt_choices)

        # 保持答案索引不变
        if 'answer' in row:
            translated_row['answer'] = row['answer']

        # 添加其他字段
        for field in fieldnames:
            if field not in translated_row:
                translated_row[field] = row[field]

        translated_rows.append(translated_row)

        # 添加短暂延迟，避免API限流
        time.sleep(0.5)

    # 写入翻译后的CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translated_rows)

    print(f"翻译完成! 输出文件: {output_file}")


if __name__ == "__main__":
    input_dir = '/Users/dengdi/MySpace/PycharmProjects/rz-aigc/eval/datasets/extracted_data/college-computer-science-test-00000-of-00001.csv'
    output_dir = '/eval/datasets/chinese_data/chinese-college-computer-science-test-00000-of-00001.csv'
    translate_csv(input_dir, output_dir)