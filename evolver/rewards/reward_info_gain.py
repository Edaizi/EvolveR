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
    reward: float
    metrics: dict = field(default_factory=dict)

def _flatten_docs(nested_list: List) -> List[Dict]:
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


def info_gain_reward_fn(
    retrieved_docs: List[dict], 
    golden_docs: List[dict],
    answer: str,
    embedding_client: 'EmbeddingClient',
    config: Dict = None
) -> InfoGainRewardOutput:
    
    all_retrieved_docs_flat = _flatten_docs(retrieved_docs)
    all_golden_docs_flat = _flatten_docs(golden_docs)

    retrieved_titles = {_normalize_title(doc.get('title', '')) for doc in all_retrieved_docs_flat}
    golden_titles = {_normalize_title(doc.get('title', '')) for doc in all_golden_docs_flat}
    
    len_retrieved = len(retrieved_titles)
    len_golden = len(golden_titles)

    if len_retrieved == 0 and len_golden == 0:
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


