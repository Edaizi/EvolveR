from dataclasses import dataclass, field
import requests
from typing import List, Dict
from unittest.mock import patch

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
p_project_root = os.path.dirname(project_root) 
sys.path.extend([project_root, p_project_root])

from verl.utils.reward_score.qa_em import extract_solution
from .similarity_utils import EmbeddingClient, get_similarity_score


@dataclass
class InfoGainRewardOutput:
    """统一信息增益奖励的输出结构。"""
    reward: float
    metrics: dict = field(default_factory=dict)

def _flatten_docs(nested_list: List) -> List[Dict]:
    """递归地将嵌套列表中的文档字典展平。"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(_flatten_docs(item))
        elif isinstance(item, dict) and 'title' in item and 'text' in item:
            flat_list.append(item)
    return flat_list

def _normalize_title(title: str) -> str:
    """Removes leading/trailing whitespace and quotes from a title."""
    return title.strip().strip('\'"')

# def info_gain_reward_fn(
#     retrieved_docs: List[dict], 
#     golden_docs: List[dict],
#     answer: str,
#     embedding_client: 'EmbeddingClient',
#     config: Dict = None
# ) -> InfoGainRewardOutput:
#     """
#     计算信息增益奖励。
#     这个奖励由三部分组成：
#     1. 核心目标奖励 (R_core): 检查答案是否在任何一个检索到的文档中。
#     2. 额外精准奖励 (R_bonus): 检查是否有任何检索到的文档是官方的"黄金文档"。
#     3. 稠密语义相似性奖励 (R_dense): 计算检索到的文档与黄金文档之间的最大语义相似度。
#     """

#     weights = {
#         "core": 0.3,
#         "bonus": 0.5,
#         "dense": 0.2
#     }
    
#     # 1. --- 数据预处理 ---
#     # 展平检索到的文档和黄金文档，并提取标题和文本
#     all_retrieved_docs_flat = _flatten_docs(retrieved_docs)
    
#     if not all_retrieved_docs_flat:
#         return InfoGainRewardOutput(reward=0.0, metrics={'reason': 'no retrieved docs'})
    


#     retrieved_texts = [doc.get('text', '') for doc in all_retrieved_docs_flat]
#     retrieved_titles = {_normalize_title(doc.get('title', '')) for doc in all_retrieved_docs_flat}
    
#     # answer 是一个<answer>...</answer> 包裹的答案，需要先提取出来
#     answer = extract_solution(answer)

#     # 2. --- 奖励计算 ---
#     # 2.1 R_core: 答案是否存在于任何检索到的文档文本中
#     r_core = 1.0 if any(answer.lower() in text.lower() for text in retrieved_texts) else 0.0

#     # 如果没有黄金文档，则只能计算核心奖励
#     if not golden_docs:
#         return InfoGainRewardOutput(
#             reward=weights["core"] * r_core,
#             metrics={"r_core": r_core, "r_bonus": 0.0, "r_dense": 0.0, "reason": "no golden docs", 
#             'final_reward': r_core, 'precision': 0, 'recall': 0, 'f1_score': 0}
#         )

#     # 提取黄金文档的文本和标题
#     all_golden_docs_flat = _flatten_docs(golden_docs)

#     golden_texts = [doc.get('text', '') for doc in all_golden_docs_flat]
#     golden_titles = {_normalize_title(doc.get('title', '')) for doc in all_golden_docs_flat}

#     # 检索文档和黄金文档的文本里是有重复的，所以需要去重
#     retrieved_texts = list(set(retrieved_texts))
#     golden_texts = list(set(golden_texts))

    
#     # 2.2 R_bonus: 检索到的标题与黄金标题是否有交集
#     r_bonus = 1.0 if not retrieved_titles.isdisjoint(golden_titles) else 0.0

#     # 2.3 R_dense: 检索到的文本和黄金文本之间的语义相似度
#     r_dense = 0.0
#     if golden_texts and retrieved_texts and embedding_client:
#         # r_dense = get_similarity_score(
#         #     text1=retrieved_texts,
#         #     text2=golden_texts,
#         #     embedding_client=embedding_client
#         # )

#         r_dense = 0
#     # --- 最终奖励和指标 ---
#     total_reward = (weights["core"] * r_core + 
#                     weights["bonus"] * r_bonus + 
#                     weights["dense"] * r_dense)
    
#     metrics = {
#         'r_core': r_core,
#         'r_bonus': r_bonus,
#         'r_dense': r_dense,
#         'final_reward': total_reward,
#         'precision': 0,
#         'recall': 0,
#         'f1_score': 0,
#     }

#     return InfoGainRewardOutput(reward=total_reward, metrics=metrics)


def info_gain_reward_fn(
    retrieved_docs: List[dict], 
    golden_docs: List[dict],
    answer: str,
    embedding_client: 'EmbeddingClient',
    config: Dict = None
) -> InfoGainRewardOutput:
    """
    计算基于标题 F1 分数的信息增益奖励。
    这个奖励只包含 R_core，即检索到的文档标题和黄金文档标题之间的 F1 分数。
    R_bonus 和 R_dense 始终为 0。
    """
    
    # 1. --- 数据预处理 ---
    all_retrieved_docs_flat = _flatten_docs(retrieved_docs)
    all_golden_docs_flat = _flatten_docs(golden_docs)

    retrieved_titles = {_normalize_title(doc.get('title', '')) for doc in all_retrieved_docs_flat}
    golden_titles = {_normalize_title(doc.get('title', '')) for doc in all_golden_docs_flat}
    
    # 2. --- R_core (F1 Score) 计算 ---
    len_retrieved = len(retrieved_titles)
    len_golden = len(golden_titles)

    if len_retrieved == 0 and len_golden == 0:
        # 如果没有检索到任何文档，也没有黄金文档，可以认为任务是空的，
        # 在信息增益场景下，没有检索到信息，增益为0更合理。
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        intersection_count = 0
    else:
        intersection = retrieved_titles.intersection(golden_titles)
        intersection_count = len(intersection)
        
        precision = intersection_count / len_retrieved if len_retrieved > 0 else 0.0
        recall = intersection_count / len_golden if len_golden > 0 else 0.0
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
    r_core = f1_score
    
    # --- 最终奖励和指标 ---
    metrics = {
        'r_core': r_core,
        'r_bonus': 0.0,
        'r_dense': 0.0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'retrieved_titles_count': len_retrieved,
        'golden_titles_count': len_golden,
        'intersection_count': intersection_count
    }

    return InfoGainRewardOutput(reward=r_core, metrics=metrics)


