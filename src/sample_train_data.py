import pandas as pd
import os

data_path = "data/exp-rl/nq_hotpotqa_train-20250903/train.parquet"
dataframe = pd.read_parquet(data_path)
dataframe = dataframe.sample(n=1200, random_state=42)

sample_path = "data/exp-rl/nq_hotpotqa_train-20250903/train-sample_1200.parquet"
dataframe.to_parquet(sample_path, index=False)

print(f"Sampled {len(dataframe)} rows to {sample_path}")