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


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    """
    Checks if the text sequence follows the expected format:
    (<think> -> <search_knowledge> -> <information> or <think> -> <search_experience> -> <experience>)* -> <think> -> <answer>
    """
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False

    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search_knowledge", "information", "search_experience", "experience", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False
    
    # Check for proper sequence pattern and no extraneous content
    split_pattern = r"(</?(?:think|search_knowledge|information|search_experience|experience|answer)>)"
    parts = re.split(split_pattern, content)
    
    state = "start"
    
    for part in parts:
        if not part.strip():
            continue
            
        is_tag = re.match(r"</?(?:think|search_knowledge|information|search_experience|experience|answer)>", part)
        
        if is_tag:
            # State transitions based on tags
            if part == "<think>" and state in ["start", "information", "experience"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search_knowledge>" and state == "after_think":
                state = "in_search_knowledge"
            elif part == "</search_knowledge>" and state == "in_search_knowledge":
                state = "after_search_knowledge"
            elif part == "<information>" and state == "after_search_knowledge":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<search_experience>" and state == "after_think":
                state = "in_search_experience"
            elif part == "</search_experience>" and state == "in_search_experience":
                state = "after_search_experience"
            elif part == "<experience>" and state == "after_search_experience":
                state = "in_experience"
            elif part == "</experience>" and state == "in_experience":
                state = "experience"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False
        else:
            # Content validation based on state
            if state in ["in_think", "in_search_knowledge", "in_search_experience", "in_information", "in_experience", "in_answer"]:
                pass  # Content is allowed inside these tags
            elif part.strip(): # Any non-whitespace content is invalid between tags
                return False
    
    return state == "end"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_retrieval_blocks(text: str) -> list[str]:
    """Extracts content from both <information> and <experience> tags."""
    info_pattern = r"<information>(.*?)</information>"
    exp_pattern = r"<experience>(.*?)</experience>"
    
    info_matches = re.findall(info_pattern, text, re.DOTALL)
    exp_matches = re.findall(exp_pattern, text, re.DOTALL)
    
    all_matches = info_matches + exp_matches
    return [match.strip() for match in all_matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    seqs = extract_retrieval_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0.2, final_format_score=0.1, retrieval_score=0.1, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {solution_str}")
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return final_format_score # 0.1