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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

    def synthesize_preference_pair(self, persona: str, category: str = None) -> Dict:
        """
        合成一个完整的偏好数据对 (chosen/rejected)

        流程：
        1. 生成 persona
        2. 选择约束条件
        3. 构建指令生成 prompt
        4. 调用 LLM 生成指令
        5. 生成 chosen 响应（确保不超过500字）
        6. 重写指令生成 rejected 版本
        7. 生成 rejected 响应（确保不超过500字）
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
            constraints_description=constraint_info["constraint_description"],
            category=category
        )

        # 4. 生成指令
        raw_instruction = self.llm_client.generate_instruction(prompt)
        instruction = clean_instruction(raw_instruction)

        # 5. 生成 chosen 响应（带重试，确保不超过500字）
        chosen_response = self._generate_response_with_length_check(instruction)
        if chosen_response is None:
            return None  # 响应验证失败，返回None

        # 6. 重写指令生成 rejected 版本
        rewrite_prompt = self.prompt_builder.build_rewrite_prompt(
            instruction=instruction,
            constraints=constraint_info["constraint_description"],
            category=constraint_info["constraint_categories"][0]
        )
        rewritten_raw = self.llm_client.rewrite_instruction(rewrite_prompt)
        rewritten_instruction = clean_instruction(rewritten_raw)

        # 7. 生成 rejected 响应（带重试，确保不超过500字）
        rejected_response = self._generate_response_with_length_check(rewritten_instruction)
        if rejected_response is None:
            return None  # 响应验证失败，返回None

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

    def _generate_response_with_length_check(self, instruction: str, category: str = None) -> str:
        """
        生成响应并确保不超过500字，且响应完整

        Args:
            instruction: 指令内容
            category: 分类名称（用于调试）

        Returns:
            符合长度要求的完整响应
        """
        max_retries = 5  # 增加重试次数
        response = None

        for attempt in range(max_retries):
            response = self.llm_client.generate_solution(
                self.prompt_builder.build_solution_prompt(instruction)
            )

            # 验证响应长度和完整性
            if self._validate_response(response, instruction):
                break
            else:
                # 如果验证失败，重新生成
                if attempt < max_retries - 1:
                    # 根据失败原因定制不同的重试策略
                    if len(response) > 500:
                        # 长度超标：使用更强的约束和更低的温度
                        strict_prompt = self.prompt_builder.build_solution_prompt(instruction) + "\n\n【关键警告】响应长度必须≤500字！当前要求：精简表达，只保留核心内容，删除所有冗余描述。请重新生成。"
                        response = self.llm_client.generate_solution(strict_prompt, temperature=0.1)
                    elif not response.endswith(('。', '！', '？', '!', '?')):
                        # 结尾错误：强调完整性要求
                        strict_prompt = self.prompt_builder.build_solution_prompt(instruction) + "\n\n【关键警告】必须以中文句号（。）、感叹号（！）或问号（？）结尾！不能有截断！请重新生成完整响应。"
                        response = self.llm_client.generate_solution(strict_prompt, temperature=0.2)
                    elif '...' in response[-10:]:
                        # 包含截断标记：强调完整性
                        strict_prompt = self.prompt_builder.build_solution_prompt(instruction) + "\n\n【关键警告】不能有省略号截断！必须提供完整的结论。请重新生成。"
                        response = self.llm_client.generate_solution(strict_prompt, temperature=0.2)
                    else:
                        # 其他问题：通用严格模式
                        strict_prompt = self.prompt_builder.build_solution_prompt(instruction) + "\n\n【严格要求】必须满足：1.字数≤500字 2.以句号/感叹号/问号结尾 3.无截断 4.无多余格式。请重新生成。"
                        response = self.llm_client.generate_solution(strict_prompt, temperature=0.1)

        # 如果最终验证失败，尝试优雅截断作为最后手段
        if not self._validate_response(response, instruction):
            if response and len(response) > 500:
                # 尝试优雅截断
                truncated = self._truncate_response_gracefully(response)
                if self._validate_response(truncated, instruction):
                    print(f"[INFO-{category}] 使用优雅截断处理超长响应 ({len(response)}→{len(truncated)}字)")
                    return truncated

            # 仍然失败，记录调试信息
            if category:
                self._validate_response_with_debug(response, instruction, category)
            return None

        return response

    def _validate_response(self, response: str, instruction: str) -> bool:
        """
        验证响应是否符合要求

        Args:
            response: 响应内容
            instruction: 对应的指令

        Returns:
            是否符合要求
        """
        if not response:
            return False

        # 1. 长度验证
        if len(response) > 500:
            return False

        # 2. 完整性验证（检查是否以句号、感叹号、问号结尾）
        if not response.endswith(('。', '！', '？', '!', '?')):
            return False

        # 3. 最小长度验证（避免空响应）
        if len(response.strip()) < 10:
            return False

        # 4. 检查是否包含截断标记
        if '...' in response[-10:]:  # 结尾有省略号可能是截断
            return False

        return True

    def _validate_response_with_debug(self, response: str, instruction: str, category: str) -> bool:
        """
        带调试信息的验证函数
        """
        if not response:
            print(f"[DEBUG-{category}] 失败: 响应为空")
            return False

        if len(response) > 500:
            print(f"[DEBUG-{category}] 失败: 长度超标 ({len(response)}字)")
            return False

        if not response.endswith(('。', '！', '？', '!', '?')):
            print(f"[DEBUG-{category}] 失败: 结尾不符，结尾='{response[-20:]}'")
            return False

        if len(response.strip()) < 10:
            print(f"[DEBUG-{category}] 失败: 长度过短 ({len(response.strip())}字)")
            return False

        if '...' in response[-10:]:
            print(f"[DEBUG-{category}] 失败: 包含截断标记")
            return False

        return True

    def synthesize_sft_data(self, persona: str, category: str = None) -> Dict:
        """
        合成 SFT 数据 (instruction + response)

        流程：
        1. 生成 persona
        2. 选择约束条件
        3. 构建指令生成 prompt
        4. 调用 LLM 生成指令
        5. 生成响应（确保不超过500字）
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
            constraints_description=constraint_info["constraint_description"],
            category=category
        )

        # 4. 生成指令
        raw_instruction = self.llm_client.generate_instruction(prompt)
        instruction = clean_instruction(raw_instruction)

        # 5. 生成响应（确保不超过500字）
        response = self._generate_response_with_length_check(instruction, category)
        if response is None:
            return None  # 响应验证失败，返回None

        return {
            "id": generate_unique_id(),
            "persona": persona,
            "constraints": constraint_info["constraint_keys"],
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
        }

    def _truncate_response_gracefully(self, text: str) -> str:
        """
        优雅地截断响应，确保不超过500字

        优先级：
        1. 在最后一个完整句子处截断
        2. 在最后一个段落处截断
        3. 在最后一个词处截断
        """
        if len(text) <= 500:
            return text

        # 截取前500字符
        truncated = text[:500]

        # 1. 尝试在最后一个句号处截断
        for marker in ['。', '！', '？', '.', '!', '?']:
            last_pos = truncated.rfind(marker)
            if last_pos > 10:  # 至少保留10个字符
                return truncated[:last_pos + 1]

        # 2. 尝试在最后一个换行处截断
        last_newline = truncated.rfind('\n')
        if last_newline > 10:
            return truncated[:last_newline]

        # 3. 尝试在最后一个空格处截断
        last_space = truncated.rfind(' ')
        if last_space > 10:
            return truncated[:last_space]

        # 4. 都不行，直接截断并添加省略号
        return truncated + "..."

    def _synthesize_single_item(self, persona_data: Dict, data_type: str) -> Tuple[Dict, bool]:
        """单个数据合成任务（用于并行执行）"""
        try:
            category = persona_data["category"]
            if data_type == "preference":
                result = self.synthesize_preference_pair(persona_data["persona"], category)
            else:
                result = self.synthesize_sft_data(persona_data["persona"], category)

            if result:
                result["category"] = category
                result["category_description"] = persona_data["category_description"]
                return result, True
            else:
                return None, False
        except Exception as e:
            return None, False

    def batch_synthesize(self, categories: List[str], num_per_category: int,
                        data_type: str, output_file: str, workers: int = 1) -> List[Dict]:
        """
        批量合成数据 - 支持并行处理

        Args:
            categories: 分类列表
            num_per_category: 每个分类生成的数量
            data_type: 数据类型 (preference/sft)
            output_file: 输出文件路径
            workers: 并行 worker 数量 (1=串行, >1=并行)

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
        print(f"处理模式: {'并行' if workers > 1 else '串行'} (workers={workers})")
        print("-" * 50)

        results = []
        failed = 0

        if workers > 1:
            # 并行处理模式
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(self._synthesize_single_item, persona_data, data_type): idx
                    for idx, persona_data in enumerate(personas)
                }

                # 收集结果（使用 tqdm 显示进度）
                for future in tqdm(as_completed(future_to_idx), total=len(personas), desc="生成中"):
                    result, success = future.result()
                    if success and result:
                        results.append(result)
                    else:
                        failed += 1
        else:
            # 串行处理模式（原逻辑）
            for idx, persona_data in enumerate(tqdm(personas)):
                try:
                    category = persona_data["category"]
                    if data_type == "preference":
                        result = self.synthesize_preference_pair(persona_data["persona"], category)
                    else:  # sft
                        result = self.synthesize_sft_data(persona_data["persona"], category)

                    if result:
                        result["category"] = category
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
  # 生成 SFT 数据（串行处理，默认）
  python run.py --categories 文学小说 青春文学 --num 5 --type sft --output sft.jsonl

  # 生成 SFT 数据（并行处理，5个worker，速度更快）
  python run.py --categories 文学小说 青春文学 --num 20 --type sft --output sft.jsonl --workers 5

  # 生成偏好数据（并行处理）
  python run.py --categories 科普读物 人文社科 --num 10 --type preference --output preference.jsonl --workers 8

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
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数量，设置 >1 启用并行处理，默认: 1")

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
        output_file=args.output,
        workers=args.workers
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