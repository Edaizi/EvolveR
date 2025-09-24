import pandas as pd

def convert_jsonl_to_parquet(jsonl_file_path, parquet_file_path):
    """
    将JSONL文件转换为Parquet文件。

    参数：
        jsonl_file_path (str): 输入的JSONL文件路径。
        parquet_file_path (str): 输出的Parquet文件路径。
    """
    # 使用pandas逐行读取JSONL文件，并将其转换为DataFrame
    df = pd.read_json(jsonl_file_path, lines=True)

    # 将DataFrame写入Parquet文件
    # `engine='pyarrow'` 指定使用pyarrow引擎
    # `index=False` 表示在Parquet文件中不包含DataFrame的索引
    df.to_parquet(parquet_file_path, engine='pyarrow', index=False)

    print(f"成功将 '{jsonl_file_path}' 转换为 '{parquet_file_path}'")


input_file = 'data/exp-rl/nq_hotpotqa_train-new/new_searchr1_train_data.jsonl'
output_file = 'data/exp-rl/nq_hotpotqa_train-new/new_searchr1_train_data.parquet'

# 调用函数进行转换
convert_jsonl_to_parquet(input_file, output_file)