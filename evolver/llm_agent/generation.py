import random
import torch
import re
from collections import defaultdict
import os
import copy
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

from evolver.experience.experience_manager import RetrievedExperiencePackage




@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    retrieve_component: Dict[str, bool] = None
    mask_sections: List[str] = None
    experience_enabled: bool = True

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        self.experience_manager = None

    def _build_prompt_with_experience(
        self,
        query: str,
        packages: List[RetrievedExperiencePackage],
        max_length: int
    ) -> str:
        """
        Strategically builds a prompt from retrieved experience packages to fit within a token budget.
        """
        # Get component flags from config safely. Defaults to all True if not provided.
        flags = self.config.retrieve_component or {}
        include_principle = flags.get("principle", True)
        include_structure = flags.get("structure", True)
        include_success = flags.get("success_trajectory", True)
        include_failure = flags.get("failure_trajectory", True)

        base_prompt = f"Question: {query}\n\n"
        base_prompt=""
        
        # Helper to format a trajectory
        def format_trajectory(traj, traj_type):
            outcome = "Success" if traj_type == "positive" else "Failure"

            if traj.log and isinstance(traj.log, list) and len(traj.log) > 0 and "content" in traj.log[0]:
                log_str = traj.log[0]["content"]
            else:
                log_str = str(traj.log)
            
            if len(log_str) > self.config.max_obs_length:
                log_str = log_str[:self.config.max_obs_length] + "..."

            return f"--- Example of {outcome} ---\n{log_str}\n"

        # Prioritize components: 1. Principles, 2. One pos/neg example, 3. More examples
        components = []
        for i, pkg in enumerate(packages):
            if include_principle:
                # Priority 1: Principle description, type, metric_score
                components.append(f"[Principle {i}], type: {pkg.principle.type}, metric score: {pkg.principle.metric_score:.2f}, description: {pkg.principle.description}\n")
            
            if include_structure and pkg.principle.structure:
                try:
                    structure_str = "\n".join([f"({s}, {p}, {o})" for s, p, o in pkg.principle.structure])
                
                except:
                    structure_str = "\n".join([str(data) for data in pkg.principle.structure])

                components.append(f"[Principle {i} structure subgraph]: \n[{structure_str}]\n")
            
            # Priority 2: One positive and one negative example
            if include_success and pkg.positive_examples:
                components.append(format_trajectory(pkg.positive_examples[0], "positive"))
            if include_failure and pkg.negative_examples:
                components.append(format_trajectory(pkg.negative_examples[0], "negative"))

            # Priority 3: Additional examples
            remaining_pos = pkg.positive_examples[1:] if include_success else []
            remaining_neg = pkg.negative_examples[1:] if include_failure else []
            
            # Alternate adding positive and negative examples
            i, j = 0, 0
            while i < len(remaining_pos) or j < len(remaining_neg):
                if i < len(remaining_pos):
                    components.append(format_trajectory(remaining_pos[i], "positive"))
                    i += 1
                if j < len(remaining_neg):
                    components.append(format_trajectory(remaining_neg[j], "negative"))
                    j += 1
        
        # Build the prompt by adding components until the budget is reached
        final_prompt = base_prompt
        for comp in components:
            # Use character length as a proxy for token count
            if len(final_prompt) + len(comp) <= max_length:
                final_prompt += comp
            else:
                do_print = random.randint(1, 100) == 1
                if do_print:
                    print("Prompt budget reached. Truncating experience components. [final_prompt length]:", len(final_prompt))
                    print("Final prompt so far:", final_prompt)
                break
        return final_prompt

    def set_experience_manager(self, experience_manager):
        self.experience_manager = experience_manager

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [
            resp.split('</search_experience>')[0] + '</search_experience>'
            if '</search_experience>' in resp
            else resp.split('</search_knowledge>')[0] + '</search_knowledge>'
            if '</search_knowledge>' in resp
            else resp.split('</answer>')[0] + '</answer>'
            if '</answer>' in resp
            else resp
            for resp in responses_str
        ]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        #####
        # add chat template here
        next_obs_with_chat_template = []
        for no in next_obs:
            no_ = '<|im_start|>user\n'+no+'<|im_end|>\n<|im_start|>assistant\n' if no != '' else no
            next_obs_with_chat_template.append(no_)
        #####
        next_obs_ids = self.tokenizer(
            next_obs_with_chat_template, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")           
            next_obs_ids = torch.cat([next_obs_ids[:, :self.config.max_obs_length-10], next_obs_ids[:, -10:]], dim=1)

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                masked_info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information or experience block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            if masked_info is not None:
                tensors_with_mask.append(masked_info)
            else:
                info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
                tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info


    def _update_right_side(self, right_side: Dict,
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          masked_info: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    masked_info=masked_info,
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
                
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        # for key in active_batch.batch.keys():
        #     active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output


    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool) 
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) 
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) 
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Initialize accumulators for each item in the batch
        batch_size = gen_batch.batch['input_ids'].shape[0]
        accumulated_experience_events = [[] for _ in range(batch_size)]
        accumulated_retrieved_docs = [[] for _ in range(batch_size)]

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, experience_events_for_meta, retrieved_docs_for_meta = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            # Accumulate retrieved principles and docs
            active_indices = torch.where(active_mask)[0]
            for i, idx in enumerate(active_indices):
                if experience_events_for_meta[i]:
                    accumulated_experience_events[idx.item()].extend(experience_events_for_meta[i])
                if retrieved_docs_for_meta[i]:
                    accumulated_retrieved_docs[idx.item()].extend(retrieved_docs_for_meta[i])

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Make a version of next_obs_ids with sections masked out (if configured)
            masked_info = None
            if self.config.mask_sections is not None and len(self.config.mask_sections) > 0:
                pad_id = self.tokenizer.pad_token_id
                masked_info = next_obs_ids.clone()
                flags = []
                for obs in next_obs:
                    need_mask = False
                    for sec in self.config.mask_sections:
                        end_tag = f'</{sec}>'
                        if end_tag in obs:
                            need_mask = True
                            break
                    flags.append(need_mask)
                if any(flags):
                    mask_tensor = torch.tensor(flags, dtype=torch.bool, device=masked_info.device)
                    masked_info[mask_tensor] = pad_id
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                masked_info=masked_info
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info = rollings.meta_info

            gen_output = self._generate_with_gpu_padding(rollings_active)


            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, _, _ = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)


            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        
        final_output.meta_info['full_dialogue_responses'] = original_right_side['responses']
        final_output.meta_info['experience_events'] = accumulated_experience_events
        final_output.meta_info['retrieved_docs'] = accumulated_retrieved_docs
        if not self.is_validation:
            final_output.meta_info['golden_docs'] = gen_batch.meta_info['golden_docs']
        else:
            final_output.meta_info['golden_docs'] = []
        return final_output


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""

        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        experience_queries = [(i, content) for i, (action, content) in enumerate(zip(cur_actions, contents)) if action == 'search_experience']
        knowledge_queries = [content for action, content in zip(cur_actions, contents) if action == 'search_knowledge']
        
        # Initialize results with empty strings
        experience_results = {i: "" for i, _ in experience_queries}
        experience_events_for_meta = [[] for _ in range(len(cur_actions))]

        if experience_queries and self.experience_manager and self.config.experience_enabled:
            query_indices, queries = zip(*experience_queries)
            
            # Batch retrieve packages
            batch_packages = [self.experience_manager.retrieve(q) for q in queries]
            # print(f"[DEBUG] batch_packages: {batch_packages}")

            for i, packages, query in zip(query_indices, batch_packages, queries):
                prompt_piece = self._build_prompt_with_experience(
                    query=query, # The original query is already part of the history
                    packages=packages, 
                    max_length=self.config.max_obs_length
                )
                experience_results[i] = prompt_piece
                
                # Store detailed experience event for reward calculation
                matched_principles_data = []
                for pkg in packages:
                    matched_principles_data.append({
                        "description": pkg.principle.description,
                        "metric_score": pkg.principle.metric_score,
                        "similarity_score": pkg.similarity_score,
                        "principle_id": pkg.principle.principle_id
                    })
                
                experience_events_for_meta[i].append({
                    "query": query,
                    "matched_principles": matched_principles_data
                })


        if do_search and knowledge_queries:
            knowledge_results, original_knowledge_results = self.batch_search(knowledge_queries)
            assert len(knowledge_results) == sum([1 for action in cur_actions if action == 'search_knowledge'])
        else:
            knowledge_results = [''] * sum([1 for action in cur_actions if action == 'search_knowledge'])
            original_knowledge_results = [{}] * sum([1 for action in cur_actions if action == 'search_knowledge'])

        # Combine results back into next_obs
        result_idx = 0

        full_experience_events = [[] for _ in range(len(active_mask))] # For storing experience events
        full_retrieved_docs = [[] for _ in range(len(active_mask))] # For storing docs for active items

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search_knowledge':
                    retrieved_doc_content = knowledge_results.pop(0)
                    full_retrieved_docs[i].append(original_knowledge_results.pop(0))
                    next_obs.append(f'\n\n<information>{retrieved_doc_content.strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                elif action == 'search_experience':
                    result_str = experience_results[i].strip()
                    if experience_events_for_meta[i]:
                        full_experience_events[i].extend(experience_events_for_meta[i])

                    next_obs.append(f'\n\n<experience>{result_str}</experience>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    if self.config.experience_enabled:
                        next_obs.append(f'\nMy previous action is invalid. \
    If I want to search knowledge, I should put the query between <search_knowledge> and </search_knowledge>. \
    If I want to search experience, I should put the query between <search_experience> and </search_experience>. \
    If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    else:
                        next_obs.append(f'\nMy previous action is invalid. \
    If I want to search knowledge, I should put the query between <search_knowledge> and </search_knowledge>. \
    If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
        
        return next_obs, dones, valid_action, is_search, full_experience_events, full_retrieved_docs


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str):
                actionable_pattern = r'<(search_knowledge|search_experience|answer)>(.*?)</\1>'
                matches = list(re.finditer(actionable_pattern, prediction, re.DOTALL))
                if matches:
                    last = matches[-1]
                    action = last.group(1)
                    content = last.group(2).strip()
                else:
                    think_match = re.search(r'<think>(.*?)</think>', prediction, re.DOTALL)
                    if think_match:
                        action = 'think'
                        content = think_match.group(1).strip()
                    else:
                        action = None
                        content = ''
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents


    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results], [self._passages2dict(result) for result in results]


    def _batch_search(self, queries):
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        return requests.post(self.config.search_url, json=payload, proxies={"http": None, "https": None}).json()


    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    
    def _passages2dict(self, retrieval_result):
        format_reference = []
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])

            format_reference.append({
                "title": title,
                "text": text
            })

        return format_reference