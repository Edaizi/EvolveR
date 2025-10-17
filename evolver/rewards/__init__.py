from .reward_format import format_reward_fn
from .reward_outcome import outcome_reward_fn
from .reward_experience import experience_reward_fn
from .reward_info_gain import info_gain_reward_fn


from typing import Union, List, Dict, Any


def exp_rl_reward_fn(
    solution_str: str, 
    ground_truth: Union[str, List[str]], 
    data_item: Dict[str, Any], 
    format_score: float = 0.0,
    retrieved_docs: List[dict] = None,
    golden_docs: List[dict] = None,
    item_index: int = None,
    embedding_client: 'EmbeddingClient' = None,
    reward_weights: Dict[str, float] = None,
    experience_enabled: bool = True,
    **kwargs
):
    """
    Main reward function interface for Exp-RL that combines multiple reward components.
    
    This function acts as a dispatcher, retrieving necessary information from the 
    data_item dictionary and passing it to specialized reward functions.
    
    Args:
        solution_str: The generated solution string.
        ground_truth: The ground truth answer(s).
        data_item: A dictionary containing rich trajectory information, 
                   including 'experience_results'.
        format_score: Legacy format correctness score.
        golden_docs: The ground truth documents from the dataset.
        item_index: The index of the current item in the batch.
        embedding_client: A shared client for calculating embeddings.
        **kwargs: Catches any other potential arguments.
        
    Returns:
        A tuple containing the total reward and a dictionary of decomposed metrics.
    """

    queries = data_item.non_tensor_batch['question']
    
    class RewardOutputMock:
        def __init__(self, reward=0.0, metrics=None):
            from collections import defaultdict
            self.reward = reward
            self.metrics = defaultdict(lambda: 0.0)
            if metrics:
                self.metrics.update(metrics)

    # Initialize all reward components with mock objects.
    # If a component's weight is > 0, this mock will be replaced by the actual output.
    format_reward_output = RewardOutputMock()
    outcome_reward = RewardOutputMock()
    info_gain_reward_output = RewardOutputMock()
    exp_reward = RewardOutputMock()

    # Component 1: Format reward
    if reward_weights.get("format", 0.0) > 0:
        format_reward_output = format_reward_fn(solution_str, experience_enabled=experience_enabled)
    
    # Component 2: Outcome reward (EM score)
    if reward_weights.get("outcome", 0.0) > 0:
        outcome_reward = outcome_reward_fn(queries, solution_str, ground_truth)
    
    # Component 3: Information gain reward
    if reward_weights.get("info_gain", 0.0) > 0:
        info_gain_reward_output = info_gain_reward_fn(
            retrieved_docs=retrieved_docs,
            golden_docs=golden_docs,
            answer=solution_str,
            embedding_client=embedding_client
        )

    # Component 4: Experience reward
    if reward_weights.get("experience", 0.0) > 0:
        all_experience_events = data_item.meta_info.get('experience_events', [])
        
        experience_events = []
        if item_index is not None and all_experience_events and item_index < len(all_experience_events):
            experience_events = all_experience_events[item_index]

        exp_reward = experience_reward_fn(
            query=queries,
            experience_events=experience_events,
            embedding_client=embedding_client
        )
    
    # Combine rewards with weights
    total_reward = 0.0
    total_weight = 0.0

    # A dictionary to hold the weighted rewards for clarity
    weighted_rewards = {
        "format": reward_weights.get("format", 0.0) * format_reward_output.reward,
        "outcome": reward_weights.get("outcome", 0.0) * outcome_reward.reward,
        "info_gain": reward_weights.get("info_gain", 0.0) * info_gain_reward_output.reward,
        "experience": reward_weights.get("experience", 0.0) * exp_reward.reward,
    }

    for component, weight in reward_weights.items():
        if weight > 0 and component in weighted_rewards:
            total_reward += weighted_rewards[component]
            total_weight += weight

    # Normalize the total reward by the sum of active weights
    if total_weight > 0:
        total_reward /= total_weight
    else:
        total_reward = 0.0
    
    reward_metrics = {
        "reward/score_em": outcome_reward.metrics['em_score'],
        "reward/score_em_format": outcome_reward.metrics['em_score_format'],
        "reward/format": format_reward_output.reward,
        "reward/outcome": outcome_reward.reward,
        "reward/info_gain": info_gain_reward_output.reward,
        "reward/info_gain/core": info_gain_reward_output.metrics['r_core'],
        "reward/info_gain/bonus": info_gain_reward_output.metrics['r_bonus'],
        "reward/info_gain/dense": info_gain_reward_output.metrics['r_dense'],
        "reward/info_gain/precision": info_gain_reward_output.metrics['precision'],
        "reward/info_gain/recall": info_gain_reward_output.metrics['recall'],
        "reward/info_gain/f1_score": info_gain_reward_output.metrics['f1_score'],
        "reward/experience": exp_reward.reward,
        "reward/experience/principle_merit": exp_reward.metrics['matched_rule_merit_score'],
        "reward/experience/retrieval_similarity": exp_reward.metrics['retrieval_similarity_score'],
        "reward/experience/scaling_factor": exp_reward.metrics['scaling_factor'],
        "reward/experience/has_matched": 1 if exp_reward.metrics['has_matched'] else 0,
        "reward/total_weighted": total_reward,
    }
    
    # Add detailed format metrics for logging
    for key, value in format_reward_output.metrics.items():
        reward_metrics[f"reward/format/{key}"] = value

    # Add F1/Precision/Recall if they exist
    if 'f1_score' in outcome_reward.metrics:
        reward_metrics["reward/outcome/f1_score"] = outcome_reward.metrics['f1_score']
    if 'precision' in outcome_reward.metrics:
        reward_metrics["reward/outcome/precision"] = outcome_reward.metrics['precision']
    if 'recall' in outcome_reward.metrics:
        reward_metrics["reward/outcome/recall"] = outcome_reward.metrics['recall']

    return total_reward, reward_metrics

__all__ = ["exp_rl_reward_fn"]

