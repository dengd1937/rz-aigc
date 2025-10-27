import pandas as pd
import os
from pathlib import Path


def extract_parquet_files(dataset_dir, output_dir, n_rows=None):
    """
    读取datasets文件夹中的所有parquet文件并转换为CSV格式

    Args:
        dataset_dir: parquet文件所在的目录
        output_dir: 输出CSV文件的目录
        n_rows: 指定提取的数据行数，如果为None则提取全部数据
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

            # 如果指定了n_rows参数，则只提取前n行
            if n_rows is not None:
                df = df.head(n_rows)
                print(f"  - 总行数: {len(pd.read_parquet(parquet_file))}, 提取行数: {len(df)}")
            else:
                print(f"  - 行数: {len(df)}")

            # 显示基本信息
            print(f"  - 列数: {len(df.columns)}")
            print(f"  - 列名: {list(df.columns)}")

            # 生成输出文件名（将.parquet替换为.csv）
            output_filename = parquet_file.stem + '.csv'
            output_path = Path(output_dir) / output_filename

            # 保存为CSV文件
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"  - 已保存到: {output_path}")

            # 显示前几行数据作为预览
            print("\n前3行数据预览:")
            print(df.head(3))
            print("-" * 80)

        except Exception as e:
            print(f"  - 处理文件时出错: {e}")

    print(f"\n所有文件处理完成！输出目录: {output_dir}")


if __name__ == "__main__":
    # 默认：使用原始函数
    dataset_dir = '../eval/datasets'
    output_dir = dataset_dir + '/extracted_data'
    extract_parquet_files(dataset_dir=dataset_dir, output_dir=output_dir, n_rows=10)