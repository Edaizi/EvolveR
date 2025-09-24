from dataclasses import dataclass, field
import re

@dataclass
class FormatRewardOutput:
    reward: float
    metrics: dict = field(default_factory=dict)



def format_reward_fn(solution_str: str, experience_enabled: bool = True) -> FormatRewardOutput:
    think_contents = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    think_count = len([c for c in think_contents if len(c.strip()) >= 3])

    search_experience_contents = re.findall(r'<search_experience>(.*?)</search_experience>', solution_str, re.DOTALL)
    search_experience_count = len([c for c in search_experience_contents if len(c.strip()) >= 3])

    search_knowledge_contents = re.findall(r'<search_knowledge>(.*?)</search_knowledge>', solution_str, re.DOTALL)
    search_knowledge_count = len([c for c in search_knowledge_contents if len(c.strip()) >= 3])

    answer_contents = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    has_answer = any(len(c.strip()) >= 3 for c in answer_contents)
    
    if not experience_enabled:
        hallucinated_experience_search = search_experience_count > 0
        total_search_count = search_knowledge_count
        format_valid = (think_count > 0) and (search_knowledge_count > 0) and has_answer and (not hallucinated_experience_search)
        reward = 1.0 if format_valid else 0.0
        return FormatRewardOutput(
            reward=reward,
            metrics={
                'think_count': think_count,
                'search_experience_count': search_experience_count,
                'search_knowledge_count': search_knowledge_count,
                'total_search_count': total_search_count,
                'has_answer': has_answer,
            }
        )

    think_score_mapping = {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1}
    think_score = think_score_mapping.get(think_count, 0.0)
    if think_count > 8:
        think_score = 0.5

    search_score = 0.0
    total_search_count = search_experience_count + search_knowledge_count
    has_search_experience = search_experience_count > 0
    has_search_knowledge = search_knowledge_count > 0
    
    if has_search_experience and has_search_knowledge:
        diversity_score = 0.5 
    elif has_search_experience or has_search_knowledge:
        diversity_score = 0.2 
    else:
        diversity_score = 0.0
        
    count_bonus = min((total_search_count - 1) * 0.1, 0.5) if total_search_count > 1 else 0.0
    search_score = diversity_score + count_bonus

    answer_score = 1.0 if has_answer else 0.0

    is_complete = think_count > 0 and total_search_count > 0 and has_answer
    
    if not is_complete:
        reward = 0.0
    else:
        think_weight = 1.0
        search_weight = 1.0
        
        total_score = (think_score * think_weight + 
                       search_score * search_weight)
        
        total_weight = think_weight + search_weight
        
        reward = total_score / total_weight
    
    return FormatRewardOutput(
        reward=reward,
        metrics={
            'think_count': think_count,
            'search_experience_count': search_experience_count,
            'search_knowledge_count': search_knowledge_count,
            'total_search_count': total_search_count,
            'has_answer': has_answer,
            'think_score': think_score,
            'search_score': search_score,
            'answer_score': answer_score
        }
    )

