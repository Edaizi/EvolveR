# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter, defaultdict

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(normalized_prediction, normalized_ground_truth):
    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def f1_score_o2(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    
    final_f1, final_precision, final_recall = 0, 0, 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        f1, precision, recall = f1_score(normalized_prediction, golden_answer)
        if final_f1 < f1:
            final_f1 = f1
        if final_precision < precision:
            final_precision = precision
        if final_recall < recall:
            final_recall = recall

    return final_f1, final_precision, final_recall


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
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_f1(solution_str, ground_truth, queries=None, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0.0, 0.0, 0.0
    else:
        return f1_score_o2(answer, ground_truth['target'])


