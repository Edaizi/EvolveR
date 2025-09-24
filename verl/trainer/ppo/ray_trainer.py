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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
import time

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance


from evolver.llm_agent.generation import LLMGenerationManager, GenerationConfig
from evolver.experience.experience_manager import ExperienceManager, VectorDBClient
import uuid # Make sure uuid is imported
from evolver.experience.experience_manager import Trajectory

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1, # 单机多进程模式
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value    # KL coefficient
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    if 'experience_events' in batch.meta_info and batch.meta_info['experience_events']:
        total_searches = sum(len(sample_events) for sample_events in batch.meta_info['experience_events'])
        metrics['experience/total_searches_per_batch'] = total_searches
        
        num_samples = len(batch.meta_info['experience_events'])
        if num_samples > 0:
            metrics['experience/avg_searches_per_sample'] = total_searches / num_samples
    else:
        metrics['experience/total_searches_per_batch'] = 0
        metrics['experience/avg_searches_per_sample'] = 0

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)
    
        self._create_dataloader()
        self._init_logger()

        self.experience_manager = None
        if self.config.experience.get('enable', True):
            vector_db_client = VectorDBClient(base_url=self.config.experience.vdb_server_url)
            self.experience_manager = ExperienceManager(
                vector_db_client=vector_db_client,
                tokenizer=self.tokenizer,
                config=self.config
            )
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # --- Choose system_prompt based on experience.enable ---
        with open_dict(self.config):
            if hasattr(self.config, 'experience') and self.config.experience.get('enable', True):
                self.config.data.system_prompt = self.config.data.system_prompt_experience
                print("Experience is enabled. Using experience-aware system prompt.")
            else:
                self.config.data.system_prompt = self.config.data.system_prompt_no_experience
                print("Experience is disabled. Using experience-unaware system prompt.")
        
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         system_prompt=self.config.data.get('system_prompt', None))
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       system_prompt=self.config.data.get('system_prompt', None))
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """

        import torch
        reward_tensor_lst = []
        data_source_lst = []
        val_reward_metrics = defaultdict(list)
        outcome_precision_list = []
        outcome_recall_list = []
        outcome_f1_list = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            retrieve_component=getattr(self.config.experience, 'retrieve_component', None),
            mask_sections=getattr(self.config.algorithm.state_masking, 'mask_sections', None),
            experience_enabled=self.config.experience.get('enable', True)
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if self.experience_manager:
            generation_manager.set_experience_manager(self.experience_manager)

        if not self.config.do_search:
            for test_data in tqdm(self.val_dataloader, desc="Validating"):
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.trainer.get('val_do_sample', False),
                    'temperature': self.config.trainer.get('val_temperature', 1.0),
                    'validate': True,
                }
                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                total_reward_tensor, em_reward_tensor, em_reward_format_tensor, reward_metrics_batch = self.val_reward_fn(test_batch)
                for k, v in reward_metrics_batch.items():
                    # Store per-batch scalar means to avoid shape mismatch when averaging later
                    val_reward_metrics[k].append(float(np.mean(v)))

                reward_tensor_lst.append(em_reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * em_reward_tensor.shape[0]))

                outcome_precision_list.append(reward_metrics_batch['reward/outcome/precision'])
                outcome_recall_list.append(reward_metrics_batch['reward/outcome/recall'])
                outcome_f1_list.append(reward_metrics_batch['reward/outcome/f1_score'])
        else:
            for batch_dict in tqdm(self.val_dataloader, desc="Validating"):
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.trainer.get('val_do_sample', False),
                    'temperature': self.config.trainer.get('val_temperature', 1.0),
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    total_reward_tensor, em_reward_tensor, em_reward_format_tensor, reward_metrics_batch = self.val_reward_fn(test_batch)
                    for k, v in reward_metrics_batch.items():
                        # Store per-batch scalar means to avoid shape mismatch when averaging later
                        val_reward_metrics[k].append(float(np.mean(v)))

                    reward_tensor_lst.append(em_reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(em_reward_tensor)))

                    outcome_precision_list.append(reward_metrics_batch['reward/outcome/precision'])
                    outcome_recall_list.append(reward_metrics_batch['reward/outcome/recall'])
                    outcome_f1_list.append(reward_metrics_batch['reward/outcome/f1_score'])

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        outcome_precisions = np.concatenate(outcome_precision_list, axis=0)
        outcome_recalls = np.concatenate(outcome_recall_list, axis=0)
        outcome_f1s = np.concatenate(outcome_f1_list, axis=0)
        data_sources = np.concatenate(data_source_lst, axis=0)
 
        # evaluate test_score based on data source
        data_source_reward = defaultdict(list)
        data_source_outcome_precision = defaultdict(list)
        data_source_outcome_recall = defaultdict(list)
        data_source_outcome_f1 = defaultdict(list)
 
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_outcome_precision[data_source].append(outcome_precisions[i])
            data_source_outcome_recall[data_source].append(outcome_recalls[i])
            data_source_outcome_f1[data_source].append(outcome_f1s[i])
            
        metric_dict = {}
        em_benchs = []

        for data_source, rewards in data_source_reward.items():
            em_bench = np.mean(rewards)
            em_benchs.append(em_bench)
            metric_dict[f'val/test_score/{data_source}'] = em_bench
            metric_dict[f'val/test_precision/{data_source}'] = np.mean(data_source_outcome_precision[data_source])
            metric_dict[f'val/test_recall/{data_source}'] = np.mean(data_source_outcome_recall[data_source])
            metric_dict[f'val/test_f1/{data_source}'] = np.mean(data_source_outcome_f1[data_source])
         
        metric_dict["val/test_score/report_em"] = np.mean(em_benchs)
        metric_dict["val/test_score/report_precision"] = np.mean(outcome_precisions)
        metric_dict["val/test_score/report_recall"] = np.mean(outcome_recalls)
        metric_dict["val/test_score/report_f1"] = np.mean(outcome_f1s)
            
        for key, values in val_reward_metrics.items():
            metric_dict[f'val/{key}'] = np.mean(values)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
        


    def _store_trajectories(self, original_batch_dict: dict, batch: DataProto, reward_metrics: dict, 
                            choice_ratio: float = 0.25,
                            selection_strategy: str = 'balanced'):
        """
        Processes and stores trajectories from a completed batch.
        
        Args:
            selection_strategy (str): 'random' or 'balanced'.
                                      'balanced' tries to pick half successful, half failed.
        """
        assert selection_strategy in ['random', 'balanced'], f"Invalid selection strategy: {selection_strategy}"
        # We need to un-repeat the batch to get original samples before saving
        # n_agent is the number of trajectories generated per prompt
        n_agent = self.config.actor_rollout_ref.rollout.n_agent
        original_batch_size = len(original_batch_dict['question'])

        run_id = batch.meta_info.get('run_id', None)
        if not run_id:
            run_id = f"gs{self.global_steps}_{uuid.uuid4().hex[:6]}"
 
        for i in range(original_batch_size):
            num_to_select = max(1, int(n_agent * choice_ratio))
            
            if selection_strategy == 'balanced':
                outcomes = np.array([reward_metrics['reward/outcome'][i * n_agent + j] == 1.0 for j in range(n_agent)])
                successful_indices = np.where(outcomes)[0]
                failed_indices = np.where(~outcomes)[0]

                num_succ_to_pick = min(len(successful_indices), num_to_select // 2)
                num_fail_to_pick = min(len(failed_indices), num_to_select - num_succ_to_pick)
                
                # If one group is short, take more from the other
                if num_succ_to_pick < num_to_select // 2:
                    num_fail_to_pick = min(len(failed_indices), num_to_select - num_succ_to_pick)

                succ_picked = np.random.choice(successful_indices, size=num_succ_to_pick, replace=False)
                fail_picked = np.random.choice(failed_indices, size=num_fail_to_pick, replace=False)
                selected_agent_indices = np.concatenate([succ_picked, fail_picked]).astype(int)

            else: # Default to 'random'
                selected_agent_indices = np.random.choice(n_agent, size=num_to_select, replace=False)

            for agent_j in selected_agent_indices:
                idx = i * n_agent + agent_j

                # *** Use the original, un-shuffled batch_dict to get question and golden_answer ***
                question = original_batch_dict['question'][i]
                golden_answer = original_batch_dict['golden_answers'][i]

                # Safely extract retrieved principles from experience_events
                experience_events = batch.meta_info.get('experience_events')
                retrieved_principles = {}
                if experience_events and len(experience_events) > idx:
                    sample_events = experience_events[idx]
                    for event in sample_events:
                        for principle in event.get('matched_principles', []):
                            p_id = principle.get('principle_id')
                            sim_score = principle.get('similarity_score')
                            if p_id and sim_score is not None:
                                retrieved_principles[p_id] = sim_score

                full_dialogue_ids = batch.meta_info['full_dialogue_responses'][idx]
                full_dialogue_ids = full_dialogue_ids.long()

                # The full dialogue is also repeated
                full_dialogue = self.tokenizer.decode(
                    full_dialogue_ids,
                    skip_special_tokens=True
                )
                final_outcome = reward_metrics['reward/outcome'][idx] == 1.0

                if isinstance(golden_answer, np.ndarray):
                    golden_answer = golden_answer.tolist()
                golden_answers_str = '\t'.join(map(str, golden_answer)) if isinstance(golden_answer, list) else str(golden_answer)

                trajectory_data = {
                    "trajectory_id": f"traj_{run_id}_{i}_{agent_j}",
                    "question": question,
                    "log": [{"event": "dialogue", "content": full_dialogue}],
                    "final_outcome": final_outcome,
                    "retrieved_principles": retrieved_principles,
                    "golden_answer": golden_answers_str
                }
                # Create trajectory object first to pass to buffer
                trajectory_obj = Trajectory.from_dict(trajectory_data)

                # Step 1: Archive trajectory to VDB immediately
                try:
                    log_str = json.dumps(trajectory_obj.log, ensure_ascii=False)
                except Exception:
                    log_str = str(trajectory_obj.log)
                
                # Defer DB insert to reflection; buffer the trajectory only
                pass

                # Step 2: Add the object to the in-memory buffer for reflection
                if self.experience_manager:
                    self.experience_manager.add_trajectory(trajectory_obj)
                
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        if self.experience_manager:
            status = self.experience_manager.vector_db_client.get_vdb_status()
            print(f"Initial VDB status: {status}")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate() 
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            retrieve_component=getattr(self.config.experience, 'retrieve_component', None),
            mask_sections=getattr(self.config.algorithm.state_masking, 'mask_sections', None),
            experience_enabled=self.config.experience.get('enable', True)
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        if self.experience_manager:
            generation_manager.set_experience_manager(self.experience_manager)

        # start training loop
        for epoch in tqdm(range(self.config.trainer.total_epochs), desc="Total epochs"):
            for batch_dict in tqdm(self.train_dataloader, desc="Training"):
                print(f'epoch: {epoch}, step: {self.global_steps}')
                metrics = {}
                timing_raw = {}

                all_golden_docs = []
                if 'metadata' in batch_dict and batch_dict['metadata'] is not None:
                    for metadata_item in batch_dict['metadata']:
                        item_golden_docs = []
                        
                        if metadata_item is None:
                            all_golden_docs.append(item_golden_docs)
                            continue

                        supporting_facts = metadata_item.get('supporting_facts', {})
                        context = metadata_item.get('context', {})

                        supporting_facts_titles = supporting_facts.get('title', [])

                        context_titles = context.get('title', [])
                        context_sentences = context.get('sentences', [])

                        if len(supporting_facts_titles) > 0 and len(context_titles) > 0 and len(context_sentences) > 0:
                            context_title_to_idx = {title: i for i, title in enumerate(context_titles)}
                            for sf_title in supporting_facts_titles:
                                if sf_title in context_title_to_idx:
                                    doc_idx = context_title_to_idx[sf_title]
                                    if doc_idx < len(context_sentences):
                                        doc_sentences_list = context_sentences[doc_idx]
                                        item_golden_docs.append({"title": sf_title, "text": " ".join(doc_sentences_list)})

                        all_golden_docs.append(item_golden_docs)
                

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
         
                gen_batch.meta_info = {
                    'question': batch.non_tensor_batch['question'],
                    'golden_answers': batch.non_tensor_batch['golden_answers'],
                    'data_source': batch.non_tensor_batch['data_source'],
                    'ability': batch.non_tensor_batch['ability'],
                    'extra_info': batch.non_tensor_batch['extra_info'],
                    'index': batch.non_tensor_batch['index'],
                    'golden_docs': all_golden_docs,
                }

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        
                        # we combine with rule-based rm
                        reward_tensor, em_reward_tensor, em_reward_format_tensor, reward_metrics = self.reward_fn(batch)

                        # Create a separate dictionary for logging with averaged metrics
                        logging_reward_metrics = {key: np.mean(values) for key, values in reward_metrics.items()}
                        metrics.update(logging_reward_metrics)
                        
                        # Store trajectories after rewards are calculated, passing the per-sample metrics
                        if self.experience_manager:
                            self._store_trajectories(batch_dict, batch, reward_metrics, choice_ratio=self.config.experience.trajectory_choice_ratio)
                        
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)



                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            try:
                                self._save_checkpoint()
                            except Exception as e:
                                print(f"Error saving checkpoint at step [{self.global_steps}] with error: {e}")
                    
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.experience_manager and self.global_steps % self.config.experience.organize_interval == 0:
                    print(f"Organizing experience at step {self.global_steps}")

                    do_export = (self.global_steps % self.config.experience.export_interval == 0)
                    self.experience_manager.reflection(self.actor_rollout_wg, export=do_export)

                    if do_export:
                        status = self.experience_manager.vector_db_client.get_vdb_status()
                        logger.log({
                            "experience/trajectories_in_vdb": status.get('data_status', {}).get('trajectories_count', -1),
                            "experience/principles_in_vdb": status.get('data_status', {}).get('principles_count', -1),
                            "experience/merged_principles": getattr(self.experience_manager, 'merged_count', 0),
                            "experience/deduplicated_principles": getattr(self.experience_manager, 'deduplicated_principles_count', 0),
                        }, step=self.global_steps)

                    # Optional: clean low-metric principles
                    threshold = getattr(self.config.experience, 'clean_low_metric_threshold', None)
                    clean_interval = getattr(self.config.experience, 'clean_interval', 0)
                    if threshold is not None and clean_interval and self.global_steps % clean_interval == 0:
                        start_time = time.time()
                        try:
                            delete_count = self.experience_manager.vector_db_client.clean_low_metric_principles(threshold)
                            if delete_count >= 0:
                                print(f"Cleaned {delete_count} low-metric principles with threshold {threshold}")
                                logger.log({"experience/cleaned_principles_count": delete_count}, step=self.global_steps)
                            else:
                                print(f"Failed to clean low-metric principles with threshold {threshold}")
                        except Exception as e:
                            print(f"Error cleaning low-metric principles: {e}")
                        end_time = time.time()
                        print(f"clean low-metric principles time: {end_time - start_time:.2f} seconds")

                        status = self.experience_manager.vector_db_client.get_vdb_status()
                        logger.log({
                            "experience/trajectories_in_vdb": status.get('data_status', {}).get('trajectories_count', -1),
                            "experience/principles_in_vdb": status.get('data_status', {}).get('principles_count', -1),
                            "experience/merged_principles": getattr(self.experience_manager, 'merged_count', 0),
                        }, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
