"""
Persona-driven 数据合成主流程脚本
整合所有组件，基于分类定义生成高质量的指令和响应数据

使用方法：
python run.py --categories 文学小说 青春文学 --num 3 --type preference --output output.jsonl
"""

import argparse
import json
import os
import sys
import random
import string
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

# 动态导入组件，支持直接运行
try:
    from persona_generator import PersonaGenerator
    from constraint_selector import ConstraintSelector
    from prompt_builder import PromptBuilder
    from llm_client import LLMClient
    from utils import generate_unique_id, clean_instruction, save_jsonl, print_progress
except ImportError:
    # 如果导入失败，尝试从当前目录导入
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from persona_generator import PersonaGenerator
    from constraint_selector import ConstraintSelector
    from prompt_builder import PromptBuilder
    from llm_client import LLMClient
    from utils import generate_unique_id, clean_instruction, save_jsonl, print_progress

load_dotenv()

class PersonaDrivenSynthesizer:
    """Persona-driven 数据合成器 - 整合所有组件"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o"):
        """
        初始化合成器

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
        """
        self.persona_gen = PersonaGenerator()
        self.constraint_selector = ConstraintSelector()
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient(api_key, base_url, model)

    def synthesize_preference_pair(self, persona: str) -> Dict:
        """
        合成一个完整的偏好数据对 (chosen/rejected)

        流程：
        1. 生成 persona
        2. 选择约束条件
        3. 构建指令生成 prompt
        4. 调用 LLM 生成指令
        5. 生成 chosen 响应
        6. 重写指令生成 rejected 版本
        7. 生成 rejected 响应
        """
        # 1. 选择约束条件
        constraint_info = self.constraint_selector.select_constraints_for_prompt(
            num_constraints=2
        )

        # 2. 获取约束示例
        example = self.constraint_selector.get_constraint_example(
            constraint_info["constraint_keys"][0]
        )

        # 3. 构建指令生成 prompt
        prompt = self.prompt_builder.build_instruction_prompt(
            persona=persona,
            example=example,
            constraints_description=constraint_info["constraint_description"]
        )

        # 4. 生成指令
        raw_instruction = self.llm_client.generate_instruction(prompt)
        instruction = clean_instruction(raw_instruction)

        # 5. 生成 chosen 响应
        chosen_response = self.llm_client.generate_solution(
            self.prompt_builder.build_solution_prompt(instruction)
        )

        # 6. 重写指令生成 rejected 版本
        rewrite_prompt = self.prompt_builder.build_rewrite_prompt(
            instruction=instruction,
            constraints=constraint_info["constraint_description"],
            category=constraint_info["constraint_categories"][0]
        )
        rewritten_raw = self.llm_client.rewrite_instruction(rewrite_prompt)
        rewritten_instruction = clean_instruction(rewritten_raw)

        # 7. 生成 rejected 响应
        rejected_response = self.llm_client.generate_solution(
            self.prompt_builder.build_solution_prompt(rewritten_instruction)
        )

        return {
            "id": generate_unique_id(),
            "persona": persona,
            "constraints": constraint_info["constraint_keys"],
            "chosen": {
                "instruction": instruction,
                "response": chosen_response
            },
            "rejected": {
                "instruction": rewritten_instruction,
                "response": rejected_response,
                "relaxed_category": constraint_info["constraint_categories"][0]
            }
        }

    def synthesize_sft_data(self, persona: str) -> Dict:
        """
        合成 SFT 数据 (instruction + response)

        流程：
        1. 生成 persona
        2. 选择约束条件
        3. 构建指令生成 prompt
        4. 调用 LLM 生成指令
        5. 生成响应
        """
        # 1. 选择约束条件
        constraint_info = self.constraint_selector.select_constraints_for_prompt(
            num_constraints=2
        )

        # 2. 获取约束示例
        example = self.constraint_selector.get_constraint_example(
            constraint_info["constraint_keys"][0]
        )

        # 3. 构建指令生成 prompt
        prompt = self.prompt_builder.build_instruction_prompt(
            persona=persona,
            example=example,
            constraints_description=constraint_info["constraint_description"]
        )

        # 4. 生成指令
        raw_instruction = self.llm_client.generate_instruction(prompt)
        instruction = clean_instruction(raw_instruction)

        # 5. 生成响应
        response = self.llm_client.generate_solution(
            self.prompt_builder.build_solution_prompt(instruction)
        )

        return {
            "id": generate_unique_id(),
            "persona": persona,
            "constraints": constraint_info["constraint_keys"],
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
        }

    def batch_synthesize(self, categories: List[str], num_per_category: int,
                        data_type: str, output_file: str) -> List[Dict]:
        """
        批量合成数据

        Args:
            categories: 分类列表
            num_per_category: 每个分类生成的数量
            data_type: 数据类型 (preference/sft)
            output_file: 输出文件路径

        Returns:
            合成的数据列表
        """
        # 1. 生成所有 personas
        personas = self.persona_gen.generate_personas_batch(
            categories=categories,
            num_per_category=num_per_category,
            detail_level="medium"
        )

        print(f"开始合成 {len(personas)} 条数据")
        print(f"数据类型: {data_type}")
        print(f"分类: {', '.join(categories)}")
        print("-" * 50)

        # 2. 合成数据
        results = []
        failed = 0

        for idx, persona_data in enumerate(tqdm(personas)):
            try:
                if data_type == "preference":
                    result = self.synthesize_preference_pair(persona_data["persona"])
                else:  # sft
                    result = self.synthesize_sft_data(persona_data["persona"])

                if result:
                    result["category"] = persona_data["category"]
                    result["category_description"] = persona_data["category_description"]
                    results.append(result)
                else:
                    failed += 1

                # 每 5 条打印一次进度
                if (idx + 1) % 5 == 0:
                    print_progress(idx + 1, len(personas), f"成功: {len(results)}, 失败: {failed}")

            except Exception as e:
                print(f"第 {idx + 1} 条数据生成失败: {e}")
                failed += 1
                continue

        # 3. 保存结果
        if output_file:
            save_jsonl(results, output_file)
            print(f"\n数据已保存到: {output_file}")
            print(f"成功生成: {len(results)} 条数据")
            print(f"失败: {failed} 条")

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Persona-driven 数据合成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成偏好数据
  python run.py --categories 文学小说 青春文学 --num 3 --type preference --output preference.jsonl

  # 生成 SFT 数据
  python run.py --categories 科普读物 人文社科 --num 5 --type sft --output sft.jsonl

  # 列出所有分类
  python run.py --list
        """
    )

    # API 配置
    parser.add_argument("--api_key", default=os.getenv("MIMO_API_KEY"), type=str, help="API 密钥")
    parser.add_argument("--base_url", default="https://api.xiaomimimo.com/v1", type=str, help="API 基础 URL")
    parser.add_argument("--model", type=str, default="mimo-v2-flash", help="模型名称，默认: mimo-v2-flash")

    # 数据生成配置
    parser.add_argument("--categories", nargs="+", default=None, help="要生成的分类列表，默认: 所有分类")
    parser.add_argument("--num", type=int, default=5, help="每个分类生成的数量，默认: 5")
    parser.add_argument("--type", type=str, default="sft", choices=["preference", "sft"],
                       help="数据类型: preference (偏好对) 或 sft (监督微调)，默认: sft")
    parser.add_argument("--output", type=str, default="./output/persona_driven_sft.jsonl", help="输出文件路径，默认: ./output/persona_driven_sft.jsonl")

    # 辅助功能
    parser.add_argument("--list", action="store_true", help="列出所有可用分类")
    parser.add_argument("--info", action="store_true", help="显示组件信息")

    args = parser.parse_args()

    # 列出分类
    if args.list:
        try:
            from persona_categories import CATEGORIES
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from persona_categories import CATEGORIES
        print("=" * 60)
        print("可用分类:")
        print("=" * 60)
        for i, (cat, desc) in enumerate(CATEGORIES.items(), 1):
            print(f"{i:2d}. {cat}: {desc}")
        return

    # 显示组件信息
    if args.info:
        print("=" * 60)
        print("Persona-driven 数据合成系统组件")
        print("=" * 60)
        print("组件:")
        print("  - PersonaGenerator: 生成多样化的人物角色")
        print("  - ConstraintSelector: 选择可验证的约束条件")
        print("  - PromptBuilder: 构建 LLM prompt")
        print("  - LLMClient: 封装 LLM 调用")
        print("  - DataSynthesizer: 整合所有组件")
        print("\n流程:")
        print("  1. 生成 persona → 2. 选择约束 → 3. 构建 prompt →")
        print("  4. 生成指令 → 5. 生成响应 → 6. 重写指令 → 7. 生成拒绝响应")
        return

    # 检查必要参数，如果未指定则使用所有分类
    if not args.categories:
        try:
            from persona_categories import CATEGORIES
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from persona_categories import CATEGORIES
        args.categories = list(CATEGORIES.keys())
        print(f"未指定分类，自动使用所有分类（共 {len(args.categories)} 个）")

    # 初始化合成器
    try:
        synthesizer = PersonaDrivenSynthesizer(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
    except Exception as e:
        print(f"初始化合成器失败: {e}")
        print("请确保已安装 openai 和 tenacity 库")
        print("pip install openai tenacity")
        return

    # 执行批量合成
    print("\n开始数据合成...")
    results = synthesizer.batch_synthesize(
        categories=args.categories,
        num_per_category=args.num,
        data_type=args.type,
        output_file=args.output
    )

    # 打印统计信息
    print("\n" + "=" * 60)
    print("合成统计:")
    print("=" * 60)
    print(f"总数据量: {len(results)}")

    if results:
        # 分类分布
        categories_count = {}
        for item in results:
            cat = item.get("category", "未知")
            categories_count[cat] = categories_count.get(cat, 0) + 1

        print("\n分类分布:")
        for cat, count in categories_count.items():
            print(f"  {cat}: {count} 条")

        # 数据示例
        if len(results) > 0:
            print("\n数据示例:")
            print("-" * 60)
            sample = results[0]
            if args.type == "preference":
                print(f"ID: {sample['id']}")
                print(f"分类: {sample['category']}")
                print(f"Persona: {sample['persona'][:80]}...")
                print(f"\nChosen Instruction: {sample['chosen']['instruction'][:80]}...")
                print(f"Chosen Response: {sample['chosen']['response'][:80]}...")
                print(f"\nRejected Instruction: {sample['rejected']['instruction'][:80]}...")
                print(f"Rejected Response: {sample['rejected']['response'][:80]}...")
            else:
                print(f"ID: {sample['id']}")
                print(f"分类: {sample['category']}")
                print(f"Persona: {sample['persona'][:80]}...")
                print(f"\nMessages:")
                for msg in sample['messages']:
                    print(f"  [{msg['role']}]: {msg['content'][:60]}...")


if __name__ == "__main__":
    main()