from dataclasses import dataclass, field
import os
import random
import re
import requests
from typing import Union, List, Dict

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
p_project_root = os.path.dirname(project_root)
sys.path.extend([project_root, p_project_root])


from verl.utils.reward_score.qa_em import em_check
from verl.utils.reward_score.qa_em_format import compute_score_em as compute_score_em_format
from verl.utils.reward_score.qa_f1 import f1_score_o2 as compute_f1_score
from evolver.rewards.config import *


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    if not matches:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


@dataclass
class OutcomeRewardOutput:
    reward: float
    metrics: dict = field(default_factory=dict)

def dv_reward_fn(queries: List[str], api_url: str = diversity_api_url, do_print=False) -> float:
    payload = {
        "queries": queries
    }
    try:
        output = requests.post(api_url, json=payload).json()
        if do_print: print(output)
        reward = output['overall_independence_score']
        return reward
    except Exception as e:
        print(f"[WARNING] Independence score error! {str(e)}")
        return 0.0

def f1_reward_fn(solution_str: str, ground_truth, api_url: str = f1_api_url, do_print=False) -> float:
    payload = {
        "generated_text": solution_str,
        "reference_points": ground_truth,
        "threshold": 0.75
    }
    try:
        output = requests.post(api_url, json=payload).json()
        if do_print: print(output)
        reward = output['f1']
        return reward
    except Exception as e:
        print(f"[WARNING] F1 score error! {str(e)}")
        return 0.0


def outcome_reward_fn(
    queries: List[str],
    solution_str: str, 
    ground_truth: Union[str, List[str], Dict], 
    question_type: str = 'closed',
    config: Dict = None
) -> OutcomeRewardOutput:


    f1_api_url = config.get('f1_api_url') if config else None
    diversity_api_url = config.get('diversity_api_url') if config else None
    weights = config.get('weights', {}) if config else outcome_weights
    
    metrics = {}
    
    answer = extract_solution(solution_str=solution_str)
    if answer is not None:
        f1, precision, recall = compute_f1_score(answer, ground_truth['target'])
        metrics['f1_score'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall

    if question_type == 'closed':
        em_score_format = compute_score_em_format(
            solution_str, 
            ground_truth
            )

        em_score = compute_score_em(solution_str, ground_truth)

        metrics['em_score'] = em_score
        metrics['em_score_format'] = em_score_format
        
        # total_reward = em_score_format
        total_reward = em_score
        
    else: 
        if f1_api_url and weights.get('f1', 0) > 0:
            f1_score = f1_reward_fn(solution_str, ground_truth, f1_api_url)
            metrics['f1_score'] = f1_score
        else:
            f1_score = 0.0

        if diversity_api_url and weights.get('diversity', 0) > 0:
            diversity_score = dv_reward_fn(queries, diversity_api_url)
            metrics['diversity_score'] = diversity_score
        else:
            diversity_score = 0.0
        
        total_reward = (
            weights.get('f1', 0) * f1_score + 
            weights.get('diversity', 0) * diversity_score
        )
    
    return OutcomeRewardOutput(
        reward=total_reward,
        metrics=metrics
    )




