import pandas as pd
import os
from pathlib import Path


def _process_parquet_files(dataset_dir, output_dir, n_rows=None, output_format='jsonl', filter_conditions=None):
    """
    处理parquet文件的核心函数

    Args:
        dataset_dir: parquet文件所在的目录
        output_dir: 输出文件的目录
        n_rows: 指定提取的数据行数，如果为None则提取全部数据
        output_format: 输出格式 ('csv' 或 'jsonl')
        filter_conditions: 筛选条件字典，例如 {'lang': 'zh'} 来筛选中文数据
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取dataset目录的路径
    dataset_path = Path(dataset_dir)

    # 查找所有parquet文件
    parquet_files = list(dataset_path.glob('*.parquet'))

    if not parquet_files:
        print(f"在 {dataset_dir} 目录中没有找到parquet文件")
        return

    print(f"找到 {len(parquet_files)} 个parquet文件")

    # 处理每个parquet文件
    for parquet_file in parquet_files:
        print(f"\n处理文件: {parquet_file.name}")

        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_file)

            # 应用筛选条件
            if filter_conditions:
                original_len = len(df)
                for column, value in filter_conditions.items():
                    if column in df.columns:
                        df = df[df[column] == value]
                        print(f"  - 根据 '{column}'='{value}' 筛选后剩余 {len(df)} 行")
                    else:
                        print(f"  - 警告: 列 '{column}' 不存在于数据中")
                print(f"  - 筛选前行数: {original_len}, 筛选后行数: {len(df)}")

            # 如果指定了n_rows参数，则只提取前n行
            if n_rows is not None:
                df = df.head(n_rows)
                print(f"  - 总行数: {len(df)}, 提取行数: {len(df)}")
            else:
                print(f"  - 行数: {len(df)}")

            # 处理已存在的'id'列
            if 'id' in df.columns:
                # 如果已存在'id'列，将其重命名为'original_id'
                df.rename(columns={'id': 'original_id'}, inplace=True)

            # 添加自增ID列
            df.insert(0, 'id', range(1, len(df) + 1))

            # 显示基本信息
            print(f"  - 列数: {len(df.columns)}")
            print(f"  - 列名: {list(df.columns)}")

            # 确定输出文件扩展名和路径
            if output_format == 'csv':
                output_filename = parquet_file.stem + '.csv'
                output_path = Path(output_dir) / output_filename
                # 保存为CSV文件
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif output_format == 'jsonl':
                output_filename = parquet_file.stem + '.jsonl'
                output_path = Path(output_dir) / output_filename
                # 保存为JSONL文件
                df.to_json(output_path, orient='records', lines=True, force_ascii=False)

            print(f"  - 已保存到: {output_path}")

            # 显示前几行数据作为预览
            print("\n前3行数据预览:")
            if output_format == 'csv':
                print(df.head(3))
            elif output_format == 'jsonl':
                print(df.head(3).to_json(orient='records', lines=True))
            print("-" * 80)

        except Exception as e:
            print(f"  - 处理文件时出错: {e}")

    print(f"\n所有文件处理完成！输出目录: {output_dir}")


def extract_parquet_to_csv(dataset_dir, output_dir, n_rows=None, filter_conditions=None):
    """
    读取datasets文件夹中的所有parquet文件并转换为CSV格式

    Args:
        dataset_dir: parquet文件所在的目录
        output_dir: 输出CSV文件的目录
        n_rows: 指定提取的数据行数，如果为None则提取全部数据
        filter_conditions: 筛选条件字典，例如 {'lang': 'zh'} 来筛选中文数据
    """
    _process_parquet_files(dataset_dir, output_dir, n_rows, 'csv', filter_conditions)


def extract_parquet_to_jsonl(dataset_dir, output_dir, n_rows=None, filter_conditions=None):
    """
    读取指定目录下的所有parquet文件并转换为JSONL格式

    Args:
        dataset_dir (str): parquet文件所在的目录
        output_dir (str): 输出JSONL文件的目录
        n_rows (int, optional): 指定提取的数据行数，如果为None则提取全部数据
        filter_conditions (dict, optional): 筛选条件字典，例如 {'lang': 'zh'} 来筛选中文数据
    """
    _process_parquet_files(dataset_dir, output_dir, n_rows, 'jsonl', filter_conditions)


if __name__ == "__main__":
    # 默认：使用原始函数
    dataset_dir = '../classify/datasets/infinity_instruct'
    output_dir = dataset_dir + '/extracted_data'
    # 示例：提取100条中文数据
    extract_parquet_to_jsonl(dataset_dir=dataset_dir, output_dir=output_dir, n_rows=100, filter_conditions={'langdetect': 'zh-cn'})