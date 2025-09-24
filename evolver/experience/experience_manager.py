import os
import json
import uuid
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, asdict, field
from collections import deque
import sys
import requests
import time
import random
import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import verl.utils.torch_functional as verl_F
import re
from tqdm import tqdm
from collections import defaultdict
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上1级目录
p_project_root = os.path.dirname(project_root)  # 上2级目录
sys.path.extend([project_root, p_project_root])

from evolver.experience import config as exp_config
from evolver.experience import prompts as exp_prompts


from tools.logger_factory import setup_logger
logger = setup_logger("experience_manager")



class VectorDBClient:
    """A client that communicates with the vector database via HTTP API."""
    def __init__(self, base_url: Optional[str] = None):
        if not base_url:
            base_url = os.environ.get("VDB_SERVER_URL", "http://127.0.0.1:8080")
        self.base_url = base_url
        logger.info(f"Initialized VectorDBClient with API base URL: {base_url}")

    def add(self, principle_id: str, description: str, structure: List[Dict[str, str]],
            principle_type: str, metric_score: float, usage_count: int,
            success_count: int, successful_trajectory_ids: "deque[str]",
            failed_trajectory_ids: "deque[str]") -> None:
        """Add a principle to the database via HTTP API."""
        url = f"{self.base_url}/principles/"
        payload = {
            "principle_id": principle_id,
            "description": description,
            "structure": structure,
            "principle_type": principle_type,
            "metric_score": metric_score,
            "usage_count": usage_count,
            "success_count": success_count,
            "successful_trajectory_ids": list(successful_trajectory_ids),
            "failed_trajectory_ids": list(failed_trajectory_ids)
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logger.debug(f"Successfully added principle {principle_id} to VDB")
        except requests.RequestException as e:
            logger.error(f"Failed to add principle {principle_id} to VDB: {e}")

    def update_principle(self, principle_id: str, **kwargs) -> bool:
        """Update a principle in the database via HTTP API."""
        url = f"{self.base_url}/principles/"
        payload = {"principle_id": principle_id}
        payload.update(kwargs)
        
        try:
            response = requests.put(url, json=payload)
            response.raise_for_status()
            logger.debug(f"Successfully updated principle {principle_id} to VDB")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to update principle {principle_id} to VDB: {e}")
            return False

    def search_principle(self, question: str, top_k: int) -> List["tuple[str, float, Dict]"]:
        """Search for principles based on a question via HTTP API."""
        url = f"{self.base_url}/search/"
        payload = {
            "query": question,
            "top_k": top_k
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            results = response.json()
            
            converted_results = []
            for item in results:
                principle_id = item["principle_id"]
                similarity = item["similarity_score"]
                entity_data = {
                    "principle_id": principle_id,
                    "type": item["principle_type"],
                    "description": item["description"],
                    "structure": item["structure"],
                    "metric_score": item["metric_score"],
                    "usage_count": item["usage_count"],
                    "success_count": item["success_count"],
                    "successful_trajectory_ids": item["successful_trajectory_ids"],
                    "failed_trajectory_ids": item["failed_trajectory_ids"]
                }
                converted_results.append((principle_id, similarity, ExperiencePrinciple.from_dict(entity_data)))
            
            do_print = random.randint(1, 1000) == 1
            if do_print:
                logger.info(f"Search experience returned {len(converted_results)} results")
            return converted_results

        except requests.RequestException as e:
            logger.error(f"Failed to search principles from VDB: {e}")
            return []

    def get_trajectory(self, trajectory_id: str) -> Optional["Trajectory"]:
        """Search for a trajectory by its ID via HTTP API."""
        url = f"{self.base_url}/trajectories/{trajectory_id}"
        
        try:
            start_time = time.time()
            response = requests.get(url)
            response.raise_for_status()
            trajectory_data = response.json()
            do_print = random.randint(1, 1000) == 1

            if do_print:
                # 检索耗时
                logger.info(f"Successfully retrieved trajectory {trajectory_id} from VDB")
                logger.info(f"Trajectory {trajectory_id} retrieval time: {time.time() - start_time} seconds")

            return Trajectory.from_dict({"trajectory_id": trajectory_id, "question": trajectory_data["query"],
                                        "log": trajectory_data["log"], "final_outcome": trajectory_data["final_outcome"], "retrieved_principles": trajectory_data["retrieved_principles"], "golden_answer": trajectory_data["golden_answer"]})

        except requests.RequestException as e:
            logger.error(f"Failed to retrieve trajectory {trajectory_id}, error: {e}")
            return None

    def add_trajectories_batch(self, trajectories: List["Trajectory"]) -> bool:
        """Batch add trajectories via HTTP API."""
        url = f"{self.base_url}/trajectories/batch"
        payload = {
            "items": []
        }
        for traj in trajectories:
            try:
                try:
                    log_str = json.dumps(traj.log, ensure_ascii=False)
                except Exception:
                    log_str = str(traj.log)
                payload["items"].append({
                    "trajectory_id": traj.trajectory_id,
                    "query": traj.question,
                    "log": log_str,
                    "final_outcome": traj.final_outcome,
                    "retrieved_principles": list(traj.retrieved_principles.keys()) if isinstance(traj.retrieved_principles, dict) else traj.retrieved_principles,
                    "golden_answer": traj.golden_answer or ""
                })
            except Exception as e:
                logger.error(f"Failed to serialize trajectory {getattr(traj, 'trajectory_id', 'unknown')}: {e}")
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to batch add trajectories to VDB: {e}")
            return False

    def get_trajectories(self, trajectory_ids: List[str]) -> List["Trajectory"]:
        """Batch get trajectories by IDs via HTTP API."""
        if not trajectory_ids:
            return []
        url = f"{self.base_url}/trajectories/batch_get"
        try:
            resp = requests.post(url, json={"ids": trajectory_ids})
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data:
                results.append(Trajectory.from_dict({
                    "trajectory_id": item.get("id"),
                    "question": item.get("query", ""),
                    "log": item.get("log", ""),
                    "final_outcome": item.get("final_outcome", False),
                    "retrieved_principles": item.get("retrieved_principles", []),
                    "golden_answer": item.get("golden_answer", "")
                }))
            return results
        except requests.RequestException as e:
            logger.error(f"Failed to batch get trajectories from VDB: {e}")
            return []

    def update_trajectory(self, trajectory_id: str, **kwargs) -> bool:
        """Update a trajectory in the database via HTTP API."""
        url = f"{self.base_url}/trajectories/"
        payload = {"trajectory_id": trajectory_id}
        payload.update(kwargs)
        
        try:
            response = requests.put(url, json=payload)
            response.raise_for_status()
            logger.debug(f"Successfully updated trajectory {trajectory_id}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to update trajectory {trajectory_id}, error: {e}")
            return False
    
    def get_vdb_status(self) -> dict:
        url = f"{self.base_url}/initial/data-status"
        try:
            response = requests.get(url)
            response.raise_for_status()
            status = response.json()

            return status
            
        except requests.RequestException as e:
            logger.error(f"Failed to get VDB status: {e}")
            return {}

    def get_principles_batch(self, principle_ids: List[str]) -> List["ExperiencePrinciple"]:
        """Batch get principles by IDs via HTTP API."""
        if not principle_ids:
            return []
        url = f"{self.base_url}/principles/batch_get"
        try:
            resp = requests.post(url, json={"ids": principle_ids})
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data:
                norm = item.copy()
                if "principle_id" not in norm:
                    norm["principle_id"] = norm.get("id", "")
                if "type" not in norm:
                    norm["type"] = norm.get("principle_type", "")
                if isinstance(norm.get("successful_trajectory_ids", []), list):
                    norm["successful_trajectory_ids"] = list(norm.get("successful_trajectory_ids", []))
                if isinstance(norm.get("failed_trajectory_ids", []), list):
                    norm["failed_trajectory_ids"] = list(norm.get("failed_trajectory_ids", []))
                results.append(ExperiencePrinciple.from_dict(norm))
            return results
        except requests.RequestException as e:
            logger.error(f"Failed to batch get principles from VDB: {e}")
            return []

    def update_principles_batch(self, principles_data: List[Dict]) -> bool:
        """Batch update principles via HTTP API."""
        if not principles_data:
            return True
        url = f"{self.base_url}/principles/batch_update"
        payload = {"items": principles_data}
        try:
            response = requests.put(url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully batch updated {len(principles_data)} principles in VDB.")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to batch update principles in VDB: {e}")
            return False

    def delete_principle(self, principle_id: str) -> bool:
        """Delete a principle by its ID via HTTP API."""
        url = f"{self.base_url}/principles/"
        payload = {"id": principle_id} # Match the DeleteRequest model in the router
        try:
            response = requests.delete(url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully requested deletion for principle {principle_id} from VDB.")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to delete principle {principle_id} from VDB: {e}")
            return False

    def delete_principles_batch(self, principle_ids: List[str]) -> bool:
        """Batch delete principles by their IDs via HTTP API."""
        if not principle_ids:
            return True
        url = f"{self.base_url}/principles/delete_batch"
        payload = {"ids": principle_ids}
        try:
            response = requests.delete(url, json=payload)
            response.raise_for_status()
            do_print = random.randint(1, 100) == 1
            if do_print:
                logger.info(f"Successfully requested batch deletion for {len(principle_ids)} principles from VDB.")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to batch delete principles from VDB: {e}")
            return False

    def clean_low_metric_principles(self, threshold: float) -> int:
        """Call VDB API to clean principles with metric_score < threshold."""
        url = f"{self.base_url}/principles/clean_low_metric"
        try:
            resp = requests.delete(url, json={"threshold": threshold})
            resp.raise_for_status()
            data = resp.json()
            return data.get("delete_count", 0)
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Failed to clean low-metric principles in VDB: {e}")
            return -1


@dataclass
class ExperiencePrinciple:
    """
    The core, distilled unit of knowledge in the experience base.
    Represents a piece of guiding or cautionary advice.
    """
    principle_id: str
    type: str  # 'guiding' or 'cautionary'
    description: str  # Natural language description for semantic retrieval
    structure: List[Dict[str, str]]  # Structured triplets for precise logic
    metric_score: float = 0.5      # Laplace smoothing
    usage_count: int = 0
    success_count: int = 0
    # Use deque for efficient appends and pops from the left
    successful_trajectory_ids: "deque[str]" = field(default_factory=lambda: deque(maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE))
    failed_trajectory_ids: "deque[str]" = field(default_factory=lambda: deque(maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE))
    
    # Custom method to handle deque for JSON serialization
    def to_dict(self):
        d = asdict(self)
        d['successful_trajectory_ids'] = list(self.successful_trajectory_ids)
        d['failed_trajectory_ids'] = list(self.failed_trajectory_ids)
        return d

    @classmethod
    def from_dict(cls, d):
        d['successful_trajectory_ids'] = deque(d.get('successful_trajectory_ids', []), maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE)
        d['failed_trajectory_ids'] = deque(d.get('failed_trajectory_ids', []), maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE)
        # Handle potential key errors for backward compatibility if new fields are added
        fields = cls.__dataclass_fields__.keys()
        filtered_d = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered_d)


@dataclass
class Trajectory:
    """
    The raw material for experience distillation, representing a full interaction episode.
    """
    trajectory_id: str
    question: str
    log: List[Dict[str, Any]] 
    final_outcome: bool  # True for success, False for failure
    retrieved_principles: Dict[str, float]  # {principle_id: similarity_score}
    golden_answer: str

    def to_dict(self):
        d = asdict(self)
        d['log'] = [step.copy() for step in self.log]
        # Convert deque to list for JSON serialization
        d['retrieved_principles'] = dict(self.retrieved_principles)
    
        return d
    
    @classmethod
    def from_dict(cls, d: dict):
        # Make a copy to work with, preserving the original dict
        init_data = d.copy()

        # If 'log' is a string, parse it from JSON.
        log = init_data.get('log', [])
        if isinstance(log, str):
            try:
                init_data['log'] = json.loads(log)
            except (json.JSONDecodeError, TypeError):
                init_data['log'] = [] # On error, default to an empty list

        # If 'retrieved_principles' is a list, convert to dict.
        principles = init_data.get('retrieved_principles', {})
        if isinstance(principles, list):
            init_data['retrieved_principles'] = {pid: 1.0 for pid in principles}

        # Filter for known fields and create the dataclass instance
        known_fields = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in init_data.items() if k in known_fields}
        
        return cls(**filtered_data)


@dataclass
class RetrievedExperiencePackage:
    """
    A rich package returned to the agent, containing not just a principle
    but also concrete positive and negative examples.
    """
    principle: ExperiencePrinciple
    similarity_score: float
    positive_examples: List[Trajectory]
    negative_examples: List[Trajectory] 



class ExperienceManager:
    """
    A unified manager for an agent's long-term experience, capable of
    learning from both successes and failures to build a robust knowledge base.
    """
    def __init__(self, vector_db_client: VectorDBClient, tokenizer: Any, config: Any):
        self.vector_db_client = vector_db_client
        self.tokenizer = tokenizer
        self.actor_rollout_wg = None

        self.trajectory_buffer: List[Trajectory] = []

        self.config = config
        self.collapsed_principle_threshold = 5
        self.merged_count = 0
        self.deduplicated_principles_count = 0

        self.experience_data_dir = config.experience.experience_data_dir
        os.makedirs(self.experience_data_dir, exist_ok=True)
        self.principle_path = os.path.join(self.experience_data_dir, "principles.json")
        self.trajectory_path = os.path.join(self.experience_data_dir, "trajectories.json")

        self.use_external_model = (
            hasattr(self.config.experience, 'summary_llm_api') and
            self.config.experience.summary_llm_api and
            self.config.experience.summary_llm_api.get('url')
        )
        logger.info(f"Use external LLM for summary: {self.config.experience.summary_llm_api}")


    # --- Public API ---
    def add_trajectory_from_dict(self, traj_data: dict):
        """Adds a completed trajectory from a dictionary to the processing buffer."""
        # Potentially validate the dictionary keys here
        try:
            # Ensure retrieved_principles is a dict
            if 'retrieved_principles' not in traj_data or not isinstance(traj_data['retrieved_principles'], dict):
                traj_data['retrieved_principles'] = {}
            trajectory = Trajectory(**traj_data)
            self.add_trajectory(trajectory)
        except TypeError as e:
            logger.error(f"Failed to create Trajectory from dict due to TypeError: {e}. Data: {traj_data.keys()}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while creating Trajectory from dict: {e}. Data: {traj_data.keys()}")

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Adds a completed trajectory to the processing buffer."""
        self.trajectory_buffer.append(trajectory)
        do_print = random.randint(1, 1000) == 1
        if do_print:
            logger.info(f"Added trajectory {trajectory.trajectory_id} to buffer. Buffer size: {len(self.trajectory_buffer)}.")

    def retrieve(self, question: str, top_k: int = exp_config.TOP_K_PRINCIPLES) -> List[RetrievedExperiencePackage]:
        """
        Retrieves relevant experience packages for a given question, including
        principles and their associated positive/negative examples.
        """
        start_time = time.time()
        vdb_status = self.vector_db_client.get_vdb_status()
        if not vdb_status or vdb_status.get("data_status", {}).get("principles_count") in ("N/A", 0) or not question:
            return []

        search_results = self.vector_db_client.search_principle(question, top_k=top_k)
        search_principle_time = time.time() - start_time

        packages = []
        seen_descriptions = set()
        ids_to_delete = []
        special_tokens = [
            "<think>", "</think>", "<search_experience>", "</search_experience>",
            "<search_knowledge>", "</search_knowledge>", "<answer>", "</answer>",
            "<experience>", "</experience>", "<information>", "</information>"
        ]

        for p_id, sim_score, item in search_results:     
            # 1. Skip principles with special tokens
            if any(token in item.description for token in special_tokens):
                do_print = random.randint(1, 100) == 1
                if do_print:
                    logger.warning(f"Skipping retrieved principle {p_id} containing special tokens.")
                ids_to_delete.append(p_id)
                continue

            # 2. Skip principles with duplicate descriptions within this batch
            clean_description = item.description.strip().strip('"').strip("'")
            if clean_description in seen_descriptions:
                do_print = random.randint(1, 100) == 1
                if do_print:
                    logger.warning(f"Marking for deletion principle {p_id} due to duplicate description.")
                ids_to_delete.append(p_id)
                continue
            seen_descriptions.add(clean_description)

            pos_examples_ids = list(item.successful_trajectory_ids)[-exp_config.RETRIEVE_K_SUCCESS_TRAJECTORIES:]
            neg_examples_ids = list(item.failed_trajectory_ids)[-exp_config.RETRIEVE_K_FAILED_TRAJECTORIES:]
            
            # detect collapsed principles like "!!!!!" or "!!!!!..."
            if isinstance(item.description, str) and item.description.startswith("!"):
                pattern = f"^!{{{self.collapsed_principle_threshold},}}"
                if re.match(pattern, item.description):
                    do_print = random.randint(1, 100) == 1
                    if do_print:
                        logger.warning(f"Collapsed principle {item.principle_id} found and marked for deletion.")
                    ids_to_delete.append(item.principle_id)
                    continue

            # Batch get trajectories for this principle
            wanted_ids = pos_examples_ids + neg_examples_ids
            id_to_traj = {t.trajectory_id: t for t in self.vector_db_client.get_trajectories(wanted_ids)}

            pos_examples: List[Trajectory] = []
            for t_id in pos_examples_ids:
                traj = id_to_traj.get(t_id)
                if traj:
                    pos_examples.append(self._compact_trajectory(traj))

            neg_examples: List[Trajectory] = []
            for t_id in neg_examples_ids:
                traj = id_to_traj.get(t_id)
                if traj:
                    neg_examples.append(self._compact_trajectory(traj))
            
            packages.append(RetrievedExperiencePackage(
                principle=item,
                similarity_score=sim_score,
                positive_examples=pos_examples,
                negative_examples=neg_examples,
            ))
        
        if ids_to_delete:
            self.vector_db_client.delete_principles_batch(ids_to_delete)

        do_print = random.randint(1, 1000) == 1
        if do_print:
            logger.info(f"Retrieve question: {question}")
            for i, p in enumerate(packages):
                logger.info(f"Principle [{i}], type: {p.principle.type}, metric_score: {p.principle.metric_score:.2f}, description: {p.principle.description}")
            logger.info(f"Total Retrieval time: {time.time() - start_time:.2f} seconds, search principle time: {search_principle_time:.2f} seconds")
        return packages


    def reflection(self, actor_rollout_wg, export: bool = True) -> None:
        """
        The core offline process. It updates metrics based on recent trajectories
        and distills new principles from them.
        """
        self.actor_rollout_wg = actor_rollout_wg
        if not self.trajectory_buffer:
            logger.info("Reflection skipped: trajectory buffer is empty.")
            return

        # Flush buffered trajectories to VDB first to avoid missing links
        try:
            self.vector_db_client.add_trajectories_batch(self.trajectory_buffer)
            logger.info(f"Flushed {len(self.trajectory_buffer)} buffered trajectories to VDB before reflection")
        except Exception as e:
            logger.error(f"Failed to flush trajectories before reflection: {e}")

        logger.info(f"Starting reflection on {len(self.trajectory_buffer)} trajectories...")
        
        start_time = time.time()

        self._update_metric_scores()

        logger.info(f"Update metric scores time: {time.time() - start_time:.2f} seconds")
        
        self.merged_count, self.deduplicated_principles_count = self._distill_new_experiences()

        # Log a summary for periodic visibility instead of doing a full, heavy export
        self._log_reflection_summary(self.trajectory_buffer, self.merged_count, self.deduplicated_principles_count)

        # Periodic export snapshot
        if export:
            # Optional: brief wait to reduce export race with recent inserts
            try:
                status = self.vector_db_client.get_vdb_status()
                logger.info(f"VDB status before export: {status.get('data_status', {}) if status else 'N/A'}")
            except Exception as e:
                logger.warning(f"Failed to fetch VDB status before export: {e}")

            time.sleep(3)  # sleep to allow DB to settle

            self._save_data()

        logger.info("Reflection finished. Clearing trajectory buffer.")
        self.trajectory_buffer.clear()

    def _deduplicate_potential_principles(self, principles: List[Dict]) -> List[Dict]:
        """
        Deduplicates a list of potential principles using LLM-based similarity checks.
        It forms a graph of similar principles and picks one representative from each connected component.
        This is intended for principles generated from the same question.
        """
        if len(principles) <= 1:
            return principles

        # Fallback to simple description matching if actor is not available for LLM calls
        if self.actor_rollout_wg is None:
            logger.warning("actor_rollout_wg is not set. Falling back to exact description match for deduplication.")
            unique_principles = []
            seen_descriptions = set()
            for p in principles:
                if p['description'] not in seen_descriptions:
                    unique_principles.append(p)
                    seen_descriptions.add(p['description'])
            return unique_principles

        prompts = []
        # Store indices of principles in pairs
        all_pairs = []
        for i, j in itertools.combinations(range(len(principles)), 2):
            # Only compare within the same question and same type to avoid over-merging
            qi = getattr(principles[i].get('source_traj'), 'question', None)
            qj = getattr(principles[j].get('source_traj'), 'question', None)
            ti = principles[i].get('type')
            tj = principles[j].get('type')
            if qi != qj or ti != tj:
                continue
            all_pairs.append((i, j))

        for (i, j) in all_pairs:
            desc1 = principles[i]['description']
            desc2 = principles[j]['description']
            # To have a canonical order and potentially better matching, put the shorter description first.
            if len(desc1) > len(desc2):
                desc1, desc2 = desc2, desc1
            
            prompt = exp_prompts.MATCH_PRINCIPLE_PROMPT.format(
                summary=desc1,
                existing_principle_description=desc2
            )
            prompts.append(prompt)
        
        if not prompts:
            return principles
            
        responses = self._batch_llm_call(
            prompts, 
            max_new_tokens=8, 
            batch_size=self.config.experience.summary_experience_batch_size, 
            state="dedup_principles",
            show_progress=False
        )
        
        if responses is None or len(responses) != len(prompts):
            logger.error(f"LLM call for deduplication failed or returned unexpected number of responses. Got {len(responses) if responses else 'None'}, expected {len(prompts)}. Skipping deduplication.")
            return principles

        # Build adjacency list for the graph using indices
        adj = defaultdict(list)
        for idx, response in enumerate(responses):
            if response and "yes" in response.lower():
                i, j = all_pairs[idx]
                adj[i].append(j)
                adj[j].append(i)

        # Find connected components to identify clusters of duplicates
        visited = set()
        representatives = []
        for i in range(len(principles)):
            if i not in visited:
                # This principle is a representative of a new component
                representatives.append(principles[i])
                # Mark all principles in this component as visited (using BFS)
                q = [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
        
        return representatives


    def _log_reflection_summary(self, processed_trajectories: List[Trajectory], merged_count: int, deduplicate_count: int):
        """Logs a summary of the reflection cycle for periodic visibility."""
        logger.info(f"Reflection Cycle Summary: Processed trajectories: {len(processed_trajectories)}; Merged into existing principles: {merged_count}; Unique principles distilled: {deduplicate_count}; Total principles in DB: {self.vector_db_client.get_vdb_status().get('data_status', {}).get('principles_count', 'N/A')}; Total trajectories in DB: {self.vector_db_client.get_vdb_status().get('data_status', {}).get('trajectories_count', 'N/A')}")


    def _save_data(self, overwrite: bool = True):
        """Saves principles and trajectory store via VDB export API."""
        try:
            url = f"{self.vector_db_client.base_url}/export/"
            
            # The root directory is now directly from the config
            output_root_dir = self.experience_data_dir
            
            exp_name = os.environ.get("EXPERIMENT_NAME") or getattr(self.config.trainer, "experiment_name", "exp")
            
            payload = {
                "collections": ["principles", "trajectories"],
                "format": "jsonl",
                "output_root_dir": output_root_dir,
                "include_metadata": True,
                "experiment_name": exp_name
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            info = response.json()
            # Construct the final path for logging, matching the server's logic
            final_dir = os.path.join(output_root_dir, exp_name, "db_exports")
            logger.info(f"Exported experience to {final_dir}. Files: {info.get('files_created', [])}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")



    def _update_metric_scores(self):
        """Updates the metric scores of principles based on trajectory outcomes using batch operations."""
        if not self.trajectory_buffer:
            return

        # Step 1: Aggregate updates for all principles in the buffer
        principle_updates = defaultdict(lambda: {"usage_delta": 0, "success_delta": 0})
        for traj in self.trajectory_buffer:
            for p_id in traj.retrieved_principles.keys():
                principle_updates[p_id]["usage_delta"] += 1
                if traj.final_outcome:
                    principle_updates[p_id]["success_delta"] += 1
        
        if not principle_updates:
            return
            
        p_ids_to_update = list(principle_updates.keys())
        
        # Step 2: Batch fetch all unique principles that need updating
        try:
            existing_principles = self.vector_db_client.get_principles_batch(p_ids_to_update)
            if not existing_principles:
                logger.warning("Could not fetch principles for metric update. Skipping.")
                return
        except Exception as e:
            logger.error(f"Failed to batch fetch principles: {e}. Skipping metric update.")
            return

        # Step 3: Calculate new scores and prepare batch update payload
        update_payload = []
        for p in existing_principles:
            if p.principle_id not in principle_updates:
                continue

            updates = principle_updates[p.principle_id]
            
            new_usage_count = p.usage_count + updates["usage_delta"]
            new_success_count = p.success_count + updates["success_delta"]
            new_metric_score = (new_success_count + 1) / (new_usage_count + 2)  # Laplace smoothing

            update_payload.append({
                "principle_id": p.principle_id,
                "usage_count": new_usage_count,
                "success_count": new_success_count,
                "metric_score": new_metric_score
            })
            
            do_print = random.randint(1, 1000) == 1
            if do_print:
                logger.info(f"Principle {p.principle_id} metric will be updated to {new_metric_score:.3f}")

        # Step 4: Perform batch update
        if update_payload:
            self.vector_db_client.update_principles_batch(update_payload)

    def _batch_llm_call(self, prompts: List[str], max_new_tokens: int, batch_size: int = 32, state="", show_progress: bool = True) -> Optional[List[str]]:
        """Dispatcher to use either external API or internal actor for LLM calls."""
        if not isinstance(batch_size, int):
            logger.warning(f"Batch size is not an integer: {batch_size}. Converting to integer.")
            try:
                batch_size = int(batch_size)
            except ValueError:
                logger.error(f"Failed to convert batch size to integer: {batch_size}. Using default value 32.")
                batch_size = 32

        if self.use_external_model:
            return self._batch_llm_call_with_api(prompts, max_new_tokens, batch_size, state, show_progress)
        else:
            return self._batch_llm_call_with_actor(prompts, max_new_tokens, batch_size, state, show_progress)

    def _batch_llm_call_with_api(self, prompts: List[str], max_new_tokens: int, batch_size: int = 32, state="", show_progress: bool = True) -> Optional[List[str]]:
        """Uses an external LLM API to get responses for a batch of prompts."""
        api_config = self.config.experience.summary_llm_api
        api_url = api_config.get('url')
        api_key = api_config.get('api_key', 'EMPTY')
        model = api_config.get('model')
        proxy_url = api_config.get('proxy_url')

        if api_url:
            endpoint = api_url.rstrip("/")
            if endpoint.endswith("/chat/completions"):
                pass
            elif endpoint.endswith("/v1") or endpoint.endswith("/v1beta"):
                endpoint = endpoint + "/chat/completions"
            else:
                endpoint = endpoint + "/v1/chat/completions"
            api_url = endpoint

        if proxy_url:
            proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
        else:
            proxies = None

        if not api_url or not model:
            logger.error("Summary LLM API URL or model not configured. Cannot perform API call.")
            return None
        if not prompts:
            return []

        import concurrent.futures as _fut
        from functools import partial as _partial

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        def _one_request(idx: int, prompt: str) -> "tuple[int, str]":
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 1.0,  # Match actor rollout config
            }
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=120, proxies=proxies)
                resp.raise_for_status()
                data = resp.json()
                content = data['choices'][0]['message']['content']
                return idx, str(content).strip().strip('"').strip("'")
            except requests.RequestException as e:
                logger.error(f"API request error [{idx}] for prompt '{prompt[:20]}...': {e}")
                return idx, ""
            except (KeyError, IndexError, TypeError, ValueError) as e:
                body = resp.text if 'resp' in locals() else 'N/A'
                logger.error(f"API parse error [{idx}] for prompt '{prompt[:20]}...': {e}. Response: {body}")
                return idx, ""

        results: List[str] = [""] * len(prompts)
        indices = list(range(len(prompts)))
        iterator = range(0, len(prompts), max(1, batch_size))

        if show_progress:
            iterator = tqdm(iterator, desc=f"batch distill({state}) experiences with API...")

        for start in iterator:
            end = min(start + max(1, batch_size), len(prompts))
            chunk_idxs = indices[start:end]
            max_workers = min(batch_size, max(1, len(chunk_idxs)))
            with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_one_request, i, prompts[i]) for i in chunk_idxs]
                for fut in _fut.as_completed(futs):
                    idx, content = fut.result()
                    results[idx] = content

        return results

    def _batch_llm_call_with_actor(self, prompts: List[str], max_new_tokens: int, batch_size: int = 32, state="", show_progress: bool = True) -> Optional[List[str]]:
        """Uses the actor_rollout_wg to get responses for a batch of prompts, following verl's tokenization logic."""
        if self.actor_rollout_wg is None:
            logger.error("actor_rollout_wg is not set in ExperienceManager. Cannot perform LLM call.")
            return None
        if not prompts:
            return []

        all_responses = []
        
        # Process prompts in batches
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"batch distill({state}) experiences...")
        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]
            
            try:
                # Step 1: Apply chat template to all prompts in this batch
                prompts_with_template = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
                    ) for p in batch_prompts
                ]

                # Step 2: Tokenize each prompt individually using the verl helper function
                tokenized_outputs = [
                    verl_F.tokenize_and_postprocess_data(
                        prompt=p,
                        tokenizer=self.tokenizer,
                        max_length=4096,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,  # Keep left_pad=True to match original
                        truncation='error'
                    ) for p in prompts_with_template
                ]

                # Step 3: Extract and stack the tensors
                all_input_ids = [out[0] for out in tokenized_outputs]  # Keep the batch dimension
                all_attention_masks = [out[1] for out in tokenized_outputs]

                # Concatenate along batch dimension
                input_ids = torch.cat(all_input_ids, dim=0)
                attention_mask = torch.cat(all_attention_masks, dim=0)
                position_ids = compute_position_id_with_mask(attention_mask)

                # Step 4: Create DataProto and call the actor
                batch_data: DataProto = DataProto.from_dict({
                    'input_ids': input_ids, 
                    'attention_mask': attention_mask, 
                    'position_ids': position_ids
                })

                gen_batch = batch_data.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'do_sample': True,
                    'temperature': 1.0,
                    'response_length': max_new_tokens,
                    'recompute_log_prob': False,
                }

                # Pad to be divisible by world_size
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

                responses_str = self.tokenizer.batch_decode(
                    output_gen_batch.batch['responses'], 
                    skip_special_tokens=True
                )

                all_responses.extend([r.strip().strip('"').strip("'") for r in responses_str])

            except Exception as e:
                logger.error(f"Error during batch LLM call with actor_rollout_wg (batch {i//batch_size}): {e}", exc_info=True)

        return all_responses

    def _parse_summarization_response(self, response: str, trajectory: Trajectory, should_generate_structure: bool = False) -> Optional["tuple[str, List[Dict], str]"]:
        """Parses the raw LLM response from summarization into a structured format"""
        if not response:
            return None

        if not should_generate_structure:
            # When no structure is requested, the response is expected to be a single sentence principle.
            description = response.strip()
            if description.startswith("[Output]"):
                # The prompt templates end with "[Output]:", so we split on that.
                description = description.split("[Output]", 1)[-1].strip().lstrip(':').strip()

            # Clean up common markdown/LLM artifacts
            description = re.sub(r'```.*?\n|```', '', description, flags=re.DOTALL).strip()
            
            if not description:
                logger.warning(f"Could not parse description from LLM output (no structure mode). Output: {response[:100]}")
                return None
            
            structure = []
            p_type = 'guiding' if trajectory.final_outcome else 'cautionary'

            do_print = random.randint(1, 1000) == 1
            if do_print:
                logger.info(f"Successfully parsed trajectory {trajectory.trajectory_id} into a potential '{p_type}' principle (no structure mode).")
            
            return description, structure, p_type

        # --- Original logic for when structure is expected ---
        description = ""
        structure_str = ""
        response = response.strip().strip("\n")
        if response.startswith("[Output]"):
            response = response.split("[Output]")[1].strip().strip("\n")
        # Use regex to robustly find description and structure parts
        desc_match = re.search(r"\[DESCRIPTION\]:(.*?)(?=\[STRUCTURE\]:|$)", response, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip().strip("\n")

        struct_match = re.search(r"\[STRUCTURE\]:(.*)", response, re.DOTALL)
        if struct_match:
            structure_str = struct_match.group(1).strip()
        
        # Clean up common markdown/LLM artifacts
        structure_str = re.sub(r'```json|```', '', structure_str).strip()

        structure = []
        if structure_str:
            try:
                # First, try to parse as valid JSON. This is the ideal case.
                parsed_json = json.loads(structure_str)
                if isinstance(parsed_json, list) and all(isinstance(item, dict) for item in parsed_json):
                    structure = parsed_json
                elif isinstance(parsed_json, list) and all(isinstance(item, (list, tuple)) and len(item) == 3 for item in parsed_json):
                    structure = [{"subject": str(s), "predicate": str(p), "object": str(o)} for s, p, o in parsed_json]
            except json.JSONDecodeError:
                # If JSON parsing fails, use regex to find (s, p, o) triplets
                triplets = re.findall(r'[\(\[]\s*["\']?([^,]+?)["\']?\s*,\s*["\']?([^,]+?)["\']?\s*,\s*["\']?([^,\]\)]+?)["\']?\s*[\)\]]', structure_str)
                for s, p, o in triplets:
                    structure.append( (s.strip().strip('"\'').strip("(").strip("[").strip(")").strip("]"), p.strip().strip('"\'').strip("(").strip("[").strip(")").strip("]"), o.strip().strip('"\'').strip("(").strip("[").strip(")").strip("]")) )
        
        if not description and not structure:
            logger.warning(f"Could not parse either description or structure from LLM output. Output: {response[:100]}")
            return None

        if not description and structure:
            # If description is missing, generate one from the structured triplets
            # This logic must handle both list of dicts (from JSON) and list of tuples (from regex)
            triplet_strs = []
            for t in structure:
                if isinstance(t, dict):
                    # Handles JSON parsing result
                    s = t.get("subject", "")
                    p = t.get("predicate", "")
                    o = t.get("object", "")
                    triplet_strs.append(f"{s}, {p}, {o}")
                elif isinstance(t, (list, tuple)) and len(t) == 3:
                    # Handles regex fallback result
                    triplet_strs.append(f"{t[0]}, {t[1]}, {t[2]}")

            description = "Principle based on relations: " + "; ".join(triplet_strs)

            do_print = random.randint(1, 1000) == 1
            if do_print:
                logger.info(f"Generated description from structure: '{description}'")

        p_type = 'guiding' if trajectory.final_outcome else 'cautionary'
        
        do_print = random.randint(1, 1000) == 1
        if do_print:
            logger.info(f"Successfully parsed trajectory {trajectory.trajectory_id} into a potential '{p_type}' principle.")
        return description, structure, p_type


    def _batch_find_matching_principles(self, potential_principles: List[Dict], match_batch_size: int = 128) -> List[List[ExperiencePrinciple]]:
        """
        Finds matching principles for a batch of potential principles using batch LLM calls.
        It internally chunks the matching prompts to avoid overloading the LLM.
        """
        all_prompts = []
        # metadata to map responses back to their original potential_principle and candidate
        prompts_meta = []
        
        for i, p_principle in enumerate(potential_principles):
            description = p_principle["description"]
            # Step 1: Get all candidates for the current potential principle
            candidates = self.vector_db_client.search_principle(description, top_k=exp_config.TOP_K_PRINCIPLES)
            if not candidates:
                continue

            # Step 2: Prepare prompts for all valid candidates
            for candidate_id, similarity, principle  in candidates:
                if similarity < exp_config.SIMILARITY_THRESHOLD:
                    continue

                prompt = exp_prompts.MATCH_PRINCIPLE_PROMPT.format(
                    summary=description,
                    existing_principle_description=principle.description
                )
                all_prompts.append(prompt)
                prompts_meta.append({
                    "potential_principle_index": i,
                    "candidate_principle": principle
                })

        if not all_prompts:
            return [[] for _ in potential_principles]

        # Process all prompts in manageable chunks
        all_responses = []
        logger.info(f"Checking {len(all_prompts)} potential principle matches in chunks of {match_batch_size}...")
        
            
        all_responses = self._batch_llm_call(all_prompts, max_new_tokens=8, state="match_principle", batch_size=self.config.experience.summary_experience_batch_size)
            
        # Process all collected responses
        match_results = [[] for _ in potential_principles]
        for i, response in enumerate(all_responses):
            if response and "yes" in response.lower():
                meta = prompts_meta[i]
                p_principle_index = meta["potential_principle_index"]
                candidate = meta["candidate_principle"]
                match_results[p_principle_index].append(candidate)
                
                do_print = random.randint(1, 1000) == 1
                if do_print:
                    logger.info(f"Found a match for new summary (index {p_principle_index}) with existing principle {candidate.principle_id}")

        return match_results


    def _distill_new_experiences(self) -> int:
        """
        Iterates through the buffer to distill and integrate new principles using batch processing.
        Returns the number of trajectories that were merged into existing principles.
        """

        logger.info("Distilling new experiences using batch processing...")
        if not self.trajectory_buffer:
            return 0

        # --- Step 1: Batch Summarize Trajectories ---
        logger.info(f"Summarizing {len(self.trajectory_buffer)} trajectories in buffer")

        should_generate_structure = self.config.experience.retrieve_component.get("structure", False)

        summarization_prompts = []
        for traj in self.trajectory_buffer:
            if should_generate_structure:
                prompt_template = (exp_prompts.SUMMARIZE_SUCCESSFUL_TRAJECTORY_PROMPT
                                   if traj.final_outcome
                                   else exp_prompts.SUMMARIZE_FAILED_TRAJECTORY_PROMPT)
            else:
                prompt_template = (exp_prompts.SUMMARIZE_SUCCESSFUL_TRAJECTORY_NO_STRUCTURE_PROMPT
                                   if traj.final_outcome
                                   else exp_prompts.SUMMARIZE_FAILED_TRAJECTORY_NO_STRUCTURE_PROMPT)

            trajectory_log_str = json.dumps(traj.log, indent=2)
            trajectory_log_str = '[Question]: ' + traj.question + '\n' + '[Golden Answer]: ' + traj.golden_answer + '\n' + trajectory_log_str
            if len(trajectory_log_str) > 4096:
                trajectory_log_str = trajectory_log_str[:4096] + "..."
            prompt = prompt_template.format(trajectory_log=trajectory_log_str)
            summarization_prompts.append(prompt)
        
        summarization_responses = self._batch_llm_call(summarization_prompts, max_new_tokens=512, batch_size=self.config.experience.summary_experience_batch_size, state="summarize_experience")
        if not summarization_responses:
            logger.error("Batch summarization failed. Aborting distillation.")
            return 0

        potential_principles = []
        for i, response in enumerate(summarization_responses):
            traj = self.trajectory_buffer[i]
            parsed_result = self._parse_summarization_response(response, traj, should_generate_structure)
            if parsed_result:
                desc, struct, p_type = parsed_result
                potential_principles.append({
                    "description": desc,
                    "structure": struct,
                    "type": p_type,
                    "source_traj": traj
                })


        logger.info(f"Successfully parsed {len(potential_principles)} potential principles from trajectories.")
        
        # --- Step 2: Global deduplication in one batch (restricted within same question and type) ---
        if potential_principles:
            before_cnt = len(potential_principles)
            potential_principles = self._deduplicate_potential_principles(potential_principles)
            logger.info(f"Total unique potential principles after deduplication: {len(potential_principles)} (from {before_cnt})")


        if not potential_principles:
            return 0, 0

        # --- Step 3: Batch find matching principles ---
        all_matches = self._batch_find_matching_principles(potential_principles, match_batch_size=self.config.experience.summary_experience_batch_size)

        # --- Step 4: Process each potential principle with its matches ---
        logger.info("Processing potential principles...")
        merged_count = 0
        for i, p_principle in enumerate(potential_principles):
            matching_principles = all_matches[i]
            
            if matching_principles:
                merged_count += 1
                for principle in matching_principles:
                    self._merge_into_existing_principle(principle, p_principle["source_traj"])
            else:
                self._create_new_principle(
                    p_principle["description"],
                    p_principle["structure"],
                    p_principle["type"],
                    p_principle["source_traj"]
                )
        
        return merged_count, len(potential_principles)


    def _create_new_principle(self, desc: str, struct: List[Dict], p_type: str, source_traj: Trajectory):
        """Creates and stores a new ExperiencePrinciple."""
        special_tokens = [
            "<think>", "</think>", "<search_experience>", "</search_experience>",
            "<search_knowledge>", "</search_knowledge>", "<answer>", "</answer>",
            "<experience>", "</experience>", "<information>", "</information>"
        ]
        
        # 1. Check for special tokens
        if any(token in desc for token in special_tokens):
            logger.warning(f"Rejecting new principle due to special tokens: '{desc[:20]}...'")
            return

        new_id = f"p_{uuid.uuid4().hex[:8]}"
        new_principle = ExperiencePrinciple(
            principle_id=new_id,
            type=p_type,
            description=desc,
            structure=struct,
            metric_score=0.5, # initial metric score
            usage_count=0,
            success_count=0
        )

        url = f"{self.vector_db_client.base_url}/principles/"

        try:
            # Use client helper to ensure correct payload keys (principle_type, etc.)
            self.vector_db_client.add(
                principle_id=new_principle.principle_id,
                description=new_principle.description,
                structure=new_principle.structure,
                principle_type=new_principle.type,
                metric_score=new_principle.metric_score,
                usage_count=new_principle.usage_count,
                success_count=new_principle.success_count,
                successful_trajectory_ids=new_principle.successful_trajectory_ids,
                failed_trajectory_ids=new_principle.failed_trajectory_ids,
            )
            do_print = random.randint(1, 1000) == 1
            if do_print:
                logger.info(f"Created new {p_type} principle {new_id}: '{desc[:20]}...'")
        except Exception as e:
            logger.error(f"Failed to create new principle {new_id}, error: {e}")

        self._link_trajectory_to_principle(source_traj, new_principle)


    def _merge_into_existing_principle(self, principle: ExperiencePrinciple, source_traj: Trajectory):
        """Merges a new trajectory's insights into an existing principle using VectorDBClient."""
        logger.info(f"Merging trajectory {source_traj.trajectory_id} into existing principle {principle.principle_id}")
        
        # Link the trajectory and update the principle in VectorDB
        self._link_trajectory_to_principle(source_traj, principle)
        
        # Update the principle's metric score based on this trajectory
        principle.usage_count += 1
        if source_traj.final_outcome:
            principle.success_count += 1
        try:
            # Update the principle in VectorDB with new metric scores
            self.vector_db_client.update_principle(
                principle_id=principle.principle_id,
                usage_count=principle.usage_count,
                success_count=principle.success_count,
                metric_score=principle.metric_score
            )
        except Exception as e:
            logger.error(f"Failed to update principle {principle.principle_id}, skip, error: {e}")
        
        do_print = random.randint(1, 1000) == 1
        if do_print:
            logger.info(f"Successfully merged trajectory {source_traj.trajectory_id} into principle {principle.principle_id}")

    def _link_trajectory_to_principle(self, trajectory: Trajectory, principle: ExperiencePrinciple):
        """Links a trajectory to a principle after the trajectory has already been archived."""
        
        # Update the principle with the new trajectory ID
        if trajectory.final_outcome:
            principle.successful_trajectory_ids.append(trajectory.trajectory_id)
        else:
            principle.failed_trajectory_ids.append(trajectory.trajectory_id)
        
        # Update the principle in the vector database
        self.vector_db_client.update_principle(
            principle_id=principle.principle_id,
            successful_trajectory_ids=list(principle.successful_trajectory_ids),
            failed_trajectory_ids=list(principle.failed_trajectory_ids)
        )
        
        do_print = random.randint(1, 1000) == 1
        if do_print:
            logger.info(f"Linked trajectory {trajectory.trajectory_id} to principle {principle.principle_id}")


    def _compact_trajectory(self, trajectory: Trajectory, structured: bool = False) -> Trajectory:
        """
        Creates a compacted version of a trajectory for agent consumption.
        - If `structured` is False (default): keep original single-string log format, only truncate
          contents within <think> and <information> tags in-place.
        - If `structured` is True: parse raw log into a structured list of events.
        """
        compacted_traj_data = asdict(trajectory)

        # Determine if the log is the raw single-string format
        raw_single = (
            trajectory.log
            and isinstance(trajectory.log, list)
            and len(trajectory.log) == 1
            and "content" in trajectory.log[0]
        )

        # If not raw single string, just truncate fields in-place and keep existing structure
        if not raw_single:
            compacted_log = []
            for step in trajectory.log:
                new_step = step.copy()
                if "thought" in new_step and isinstance(new_step["thought"], str):
                    if len(new_step["thought"]) > exp_config.MAX_THOUGHT_CHARS:
                        new_step["thought"] = new_step["thought"][:exp_config.MAX_THOUGHT_CHARS] + "..."
                if "observation" in new_step and isinstance(new_step["observation"], str):
                    if len(new_step["observation"]) > exp_config.MAX_DOC_CHARS:
                        new_step["observation"] = new_step["observation"][:exp_config.MAX_DOC_CHARS] + "..."
                compacted_log.append(new_step)
            compacted_traj_data['log'] = compacted_log
            return Trajectory(**compacted_traj_data)

        # It is the raw single-string format; truncate in-place first
        full_dialogue = trajectory.log[0]['content']

        def _truncate(content: str, max_len: int) -> str:
            return content[:max_len] + "..." if len(content) > max_len else content

        # Truncate <think> blocks
        full_dialogue = re.sub(
            r'<think>(.*?)</think>',
            lambda m: f"<think>{_truncate(m.group(1), exp_config.MAX_THOUGHT_CHARS)}</think>",
            full_dialogue,
            flags=re.DOTALL
        )

        # Truncate <information> blocks
        full_dialogue = re.sub(
            r'<information>(.*?)</information>',
            lambda m: f"<information>{_truncate(m.group(1), exp_config.MAX_DOC_CHARS)}</information>",
            full_dialogue,
            flags=re.DOTALL
        )

        # Default behavior: keep single-string log
        if not structured:
            compacted_traj_data['log'] = [{"event": "dialogue", "content": full_dialogue}]
            return Trajectory(**compacted_traj_data)

        # Structured=True: parse into structured list of events
        pattern = r'<(think|search|search_knowledge|search_experience|answer|information|experience)>(.*?)</\1>'
        parsed_log = []
        for match in re.finditer(pattern, full_dialogue, re.DOTALL):
            tag = match.group(1)
            content = match.group(2).strip()
            
            # Normalize 'search' to 'search_knowledge' for consistent processing
            if tag == "search":
                tag = "search_knowledge"

            # Now, apply compaction (truncation) and structure the log
            if tag == "think":
                if len(content) > exp_config.MAX_THOUGHT_CHARS:
                    content = content[:exp_config.MAX_THOUGHT_CHARS] + "..."
                parsed_log.append({"thought": content})
            
            elif tag in ["search_knowledge", "search_experience", "answer"]:
                # Structure actions more explicitly
                parsed_log.append({"action": {"type": tag, "input": content}})

            elif tag in ["information", "experience"]:
                if len(content) > exp_config.MAX_DOC_CHARS:
                    content = content[:exp_config.MAX_DOC_CHARS] + "..."
                # Use a consistent "observation" key
                parsed_log.append({"observation": content})

        compacted_traj_data['log'] = parsed_log
        return Trajectory(**compacted_traj_data)

