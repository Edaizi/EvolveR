import json
import pandas as pd
import os

# Search-R1 template
def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


# def modify_prompt_content(parquet_file, output_file):
#     """
#     修改parquet文件中所有数据的prompt content字段
#     """
#     prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information or get new experience principles. \
# After reasoning, you can search for past experiences by <search_experience> query </search_experience> to get relevant past experience principles (may be guilding or warning principles) and it will return the top searched results between <experience> and </experience>. You can use these principles which you think is helpful to help you answer the question. \
# If you find you lack some knowledge, you can call a search engine by <search_knowledge> query </search_knowledge> and it will return the top searched results between <information> and </information>. \
# You can search knowledge and experience as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>"""

#     # 读取parquet文件
#     df = pd.read_parquet(parquet_file)
#     print(f"成功读取文件: {parquet_file}")
#     print(f"数据行数: {len(df)}")
    
#     # 修改每一行的prompt content
#     modified_count = 0
#     for idx, row in df.iterrows():
#         # print(f"row: {row}")
#         original_content = row['prompt'][0]['content']
        
#         # 提取问题部分
#         question_part = original_content.split("Question:")[1].strip()
#         new_content = prefix + " Question: " + question_part + "\n"
        
#         # 更新content
#         df.at[idx, 'prompt'][0]['content'] = new_content
#         modified_count += 1
    
#     # 保存修改后的文件
#     df.to_parquet(output_file, index=False)
#     print(f"成功修改了 {modified_count} 行数据")
#     print(f"修改后的文件已保存到: {output_file}")


def modify_prompt_content(parquet_file, output_file):
    """
    修改parquet文件中所有数据的prompt content字段
    """

    # 读取parquet文件
    df = pd.read_parquet(parquet_file)
    print(f"成功读取文件: {parquet_file}")
    print(f"数据行数: {len(df)}")
    
    # 修改每一行的prompt content
    modified_count = 0
    for idx, row in df.iterrows():
        # print(f"row: {row}")
        original_content = row['prompt'][0]['content']
        
        # 提取问题部分
        question_part = original_content.split("Question:")[1].strip()
        new_content = " Question: " + question_part + "\n"
        
        # 更新content
        df.at[idx, 'prompt'][0]['content'] = new_content
        modified_count += 1

    # 创建目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存修改后的文件
    df.to_parquet(output_file, index=False)
    print(f"成功修改了 {modified_count} 行数据")
    print(f"修改后的文件已保存到: {output_file}")



if __name__ == "__main__":
    # 示例用法
    input_file = "data/exp-rl/nq_hotpotqa_train/test.parquet"
    output_file = "data/exp-rl/nq_hotpotqa_train-20250903/test.parquet"
    
    
    # 执行修改
    modify_prompt_content(input_file, output_file)