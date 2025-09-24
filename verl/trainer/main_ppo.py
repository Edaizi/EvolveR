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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from collections import defaultdict

from evolver.rewards import exp_rl_reward_fn
from evolver.rewards.similarity_utils import EmbeddingClient



def _select_rm_score_fn(data_source, experience_enabled=True):
    # Fallback to the custom Exp-RL reward function for other data sources
    return exp_rl_reward_fn



class RewardManager():
    """The reward manager.
    """

    def __init__(self, config, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.embedding_client = EmbeddingClient(api_url=config.experience.embedding_api_url, 
                                                api_key=config.experience.embedding_api_key,
                                                model_name=config.experience.embedding_model)
        self.experience_enabled = config.experience.enable
        self.reward_weights = config.rewards.weights
        print(f"Reward manager's embedding client initialized with api_url: {config.experience.embedding_api_url}, api_key: {config.experience.embedding_api_key}, model_name: {config.experience.embedding_model}")
        print(f"Reward manager's reward weights: {self.reward_weights}")

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            # To maintain a consistent return type, return an empty metrics dict
            return data.batch['rm_scores'], {}

        total_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        em_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        em_reward_format_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # This will aggregate the decomposed metrics from each sample in the batch
        batch_reward_metrics = defaultdict(list)
        already_print_data_sources = {}
        

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences) 

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, experience_enabled=self.experience_enabled)
            golden_docs = data_item.meta_info.get('golden_docs', {})
            retrieved_docs = data_item.meta_info['retrieved_docs']

            # Unpack the tuple returned by the reward function
            score, metrics_dict = compute_score_fn(
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                format_score=self.format_score,
                data_item=data_item,
                retrieved_docs=retrieved_docs,
                golden_docs=golden_docs,
                item_index=i,
                embedding_client=self.embedding_client,
                reward_weights=self.reward_weights
            )

            # Aggregate the decomposed metrics
            for key, value in metrics_dict.items():
                batch_reward_metrics[key].append(value)
            
            # Assign the scalar score to the reward tensor
            total_reward_tensor[i, valid_response_length - 1] = score
            em_reward_tensor[i, valid_response_length - 1] = metrics_dict['reward/score_em']
            em_reward_format_tensor[i, valid_response_length - 1] = metrics_dict['reward/score_em_format']
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        

        # Return the tuple that RayPPOTrainer expects
        return total_reward_tensor, em_reward_tensor, em_reward_format_tensor, batch_reward_metrics


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }


    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    
    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(config=config, tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(config=config, tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
