import json
import os
import pandas as pd
import random

sample_ratio = 0.1
all_path = "data/exp-rl/nq_hotpotqa_train-20250903/test.parquet"
sample_path = f"data/exp-rl/nq_hotpotqa_train-20250903/test-sample_{sample_ratio}.parquet"


df = pd.read_parquet(all_path)

# 根据data_source字段，统计所有数据，打印每一种数据源的个数
data_source_counts = df['data_source'].value_counts()
print("All testset data source counts:")
print(data_source_counts)

no_sample_class = "bamboogle"

# 按类别进行随机采样
def sample_by_category(df, sample_ratio):
    sampled_data = []
    for category, group in df.groupby('data_source'):
        if category != no_sample_class:
            sample_size = max(1, int(len(group) * sample_ratio))  # 确保至少采样一个
            sampled_group = group.sample(n=sample_size, random_state=42)
        else:
            sampled_group = group
        sampled_data.append(sampled_group)
    return pd.concat(sampled_data, ignore_index=True)


sampled_df = sample_by_category(df, sample_ratio)

# 保存采样后的数据，两种格式（parquet、jsonl）
sampled_df.to_parquet(sample_path, index=False)
sampled_jsonl_path = sample_path.replace('.parquet', '.jsonl')
sampled_df.to_json(sampled_jsonl_path, orient='records', lines=True, force_ascii=False)


# 打印采样后的数据源分布
sampled_data_source_counts = sampled_df['data_source'].value_counts()
print("Sampled testset data source counts:")
print(sampled_data_source_counts)
