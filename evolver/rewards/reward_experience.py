from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import requests
from unittest.mock import patch


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
p_project_root = os.path.dirname(project_root)
sys.path.extend([project_root, p_project_root])

from .similarity_utils import EmbeddingClient, get_similarity_score

@dataclass
class ExperienceRewardOutput:
    reward: float
    metrics: dict = field(default_factory=dict)

def experience_reward_fn(
    query: str,
    experience_events: List[Dict[str, Any]],
    embedding_client: EmbeddingClient,
    config: Optional[Dict] = None,
    scaling_factor: float = 1.0,
    mode: str = 'max' # 'max' or 'mean'
) -> ExperienceRewardOutput:
    """
    Args:
        experience_events (List[Dict[str, Any]]): A list of events, where each event
            corresponds to a <search_experience> action and contains the query
            and the principles that were matched.
        embedding_client: A shared client for calculating embeddings.
        config (Optional[Dict]): Configuration dictionary, e.g., for API URLs.
        scaling_factor (float): A constant scaling factor for the reward.

    Returns:
        ExperienceRewardOutput: 
    """
    ##
    
    max_reward = 0.0
    best_metrics = {
        'matched_rule_merit_score': 0.0,
        'retrieval_similarity_score': 0.0,
        'scaling_factor': scaling_factor,
        'has_matched': False  
    }

    if not experience_events:
        return ExperienceRewardOutput(reward=0.0, metrics=best_metrics)


    for event in experience_events:
        matched_principles = event["matched_principles"]

        if not query or not matched_principles:
            continue
        # For each event, find the best matching principle
        for principle in matched_principles:
            description = principle["description"]
            merit_score = principle["metric_score"]

            if not description:
                continue

            # Calculate the dynamic relevance score
            relevance_score = get_similarity_score(
                text1=query,
                text2=description,
                embedding_client=embedding_client
            )
            
            # Calculate the potential reward for this specific principle match
            current_reward = scaling_factor * merit_score * relevance_score

            if current_reward > max_reward:
                max_reward = current_reward
                best_metrics = {
                    'matched_rule_merit_score': merit_score,
                    'retrieval_similarity_score': relevance_score,
                    'scaling_factor': scaling_factor,
                    'has_matched': True
                }

    return ExperienceRewardOutput(
        reward=max_reward,
        metrics=best_metrics
    )

