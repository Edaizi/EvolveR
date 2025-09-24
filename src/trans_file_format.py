import json
import os
import pandas as pd

def convert_parquet_to_jsonl(parquet_file, output_jsonl_file, num_rows=-1):
    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_file)
        if num_rows != -1:
            # Select the specified number of rows
            df_subset = df.head(num_rows)
        else:
            df_subset = df
        
        # Convert to JSONL format
        df_subset.to_json(output_jsonl_file, orient='records', lines=True)
        print(f"Successfully converted {df_subset.shape[0]} rows to {output_jsonl_file}.")
    
    except Exception as e:
        print(f"Error: {e}")



def parquet_to_jsonl(input_dir, output_file, filter_keyword=None):
    # 找到目录下所有的 .parquet 文件
    parquet_files = [os.path.join(input_dir, file) for file in sorted(os.listdir(input_dir)) if file.endswith('.parquet')]
    
    # 如果指定了过滤条件，只保留包含过滤关键词的文件
    if filter_keyword:
        parquet_files = [file for file in parquet_files if filter_keyword in os.path.basename(file)]
    
    # 检查是否找到符合条件的文件
    if not parquet_files:
        print(f"未找到符合条件的 .parquet 文件（过滤条件: {filter_keyword}）")
        return

    print(f"找到 {len(parquet_files)} 个 .parquet 文件，开始合并...")

    # 读取所有的 parquet 文件并合并
    df_list = []
    for parquet_file in parquet_files:
        print(f"正在处理文件: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        df_list.append(df)
    
    # 合并所有 DataFrame
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # 保存为 JSON Lines (.jsonl) 格式
    df_combined.to_json(output_file, orient='records', lines=True, force_ascii=False)

    print(f"合并后的数据已保存到 {output_file}")

if __name__ == "__main__":
    input_path = 'Grpo/data/gsm8k/main'  # 替换为你的 .parquet 文件所在的目录
    filter_keyword = 'test'  # 过滤条件，如果不需要过滤，可以设置为 None
    output_path = input_path + f'/{filter_keyword}.jsonl'  # 输出的 jsonl 文件路径
    
    # 合并所有文件
    # parquet_to_jsonl(input_path, output_path, filter_keyword)
    convert_parquet_to_jsonl("data/exp-rl/nq_hotpotqa_train-20250903/train.parquet", "data/exp-rl/nq_hotpotqa_train-20250903/train.jsonl")
    convert_parquet_to_jsonl("data/exp-rl/nq_hotpotqa_train-20250903/test.parquet", "data/exp-rl/nq_hotpotqa_train-20250903/test.jsonl")
