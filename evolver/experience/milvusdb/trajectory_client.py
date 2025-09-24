import json
import time
from typing import List, Dict, Optional
from pymilvus import FieldSchema, DataType, CollectionSchema, MilvusClient
from config import BaseVectorDBClient, exp_config, logger

class TrajectoryVectorDBClient(BaseVectorDBClient):
    def __init__(self,
                 collection_name: str = None,
                 experiment_name: str = None,
                 db_path: str = exp_config.MILVUS_DB_PATH,
                 api_url: str = "http://127.0.0.1:8000/v1",
                 api_key: str = "empty",
                 model_name: str = "bge_m3"):

        if experiment_name:
            experiment_name = experiment_name.replace(".", "_")
            
        if experiment_name:
            self.collection_name = f"exp_trajectories_{experiment_name}"
        elif collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = exp_config.MILVUS_TRAJECTORY_COLLECTION_NAME
            
        super().__init__(db_path, api_url, api_key, model_name)
        self._create_collection_if_not_exist()

    def _create_collection_if_not_exist(self):
        if self.collection_name not in self.client.list_collections():
            logger.info(f"[trajectory_client] Creating trajectories collection '{self.collection_name}'...")
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="query_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="log", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="final_outcome", dtype=DataType.BOOL),
                FieldSchema(name="retrieved_principles", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="golden_answer", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields=fields, description="Agent trajectories with query embeddings")
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                dimension=self.dim,
                auto_id=False,
            )
            logger.info(f"[trajectory_client] Successfully created trajectories collection '{self.collection_name}'")

    def add_trajectory(self, trajectory_id: str, query: str, log: str, 
                      final_outcome: bool, retrieved_principles: List[str], golden_answer: str) -> None:

        query_vector = self._get_embedding(query)
        current_time = int(time.time())
        data = {
            "id": trajectory_id,
            "query_vector": query_vector,
            "query": query,
            "log": log,
            "final_outcome": final_outcome,
            "retrieved_principles": json.dumps(retrieved_principles),
            "golden_answer": golden_answer,
            "created_at": current_time,
            "updated_at": current_time,
        }
        res = self.client.insert(collection_name=self.collection_name, data=[data])
        logger.info(f"[Milvus] Inserted trajectory {trajectory_id} into collection '{self.collection_name}': {res}")

    def add_trajectories_batch(self, items: List[Dict]):
        """Batch insert trajectories with batch embedding."""
        if not items:
            return
        
        # 1. Batch fetch embeddings for all queries
        queries = [it.get("query", "") for it in items]
        try:
            query_vectors_np = self.embedding_provider.get_embeddings(queries)
            query_vectors = query_vectors_np.tolist()
            if len(query_vectors) != len(items):
                logger.error(f"Embedding service returned {len(query_vectors)} vectors for {len(items)} trajectories. Aborting.")
                return
        except Exception as e:
            logger.error(f"Failed to fetch embeddings for trajectory batch: {e}")
            return

        # 2. Prepare entities for batch insertion
        current_time = int(time.time())
        data = []
        for i, it in enumerate(items):
            data.append({
                "id": it["trajectory_id"],
                "query_vector": query_vectors[i],
                "query": it.get("query", ""),
                "log": it.get("log", ""),
                "final_outcome": it.get("final_outcome", False),
                "retrieved_principles": json.dumps(it.get("retrieved_principles", [])),
                "golden_answer": it.get("golden_answer", ""),
                "created_at": current_time,
                "updated_at": current_time,
            })
            
        # 3. Batch insert into Milvus
        try:
            res = self.client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"[Milvus] Batch inserted {len(data)} trajectories into collection '{self.collection_name}': {res}")
        except Exception as e:
            logger.error(f"Failed to batch insert trajectories: {e}")

    def get_trajectories_by_ids(self, ids: List[str]) -> List[Dict]:
        if not ids:
            return []
        # Build filter expression: id in ["a","b"]
        quoted = ",".join([f'"{i}"' for i in ids])
        expr = f"id in [{quoted}]"
        res = self.client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["*"],
            limit=len(ids)
        )
        return res

    def delete_trajectory(self, trajectory_id: Optional[str] = None, before_timestamp: Optional[int] = None):

        if trajectory_id:
            expr = f'id == "{trajectory_id}"'
        elif before_timestamp:
            expr = f'created_at < {before_timestamp}'
        else:
            raise ValueError("[trajectory_client] Must provide trajectory_id or before_timestamp")

        self.client.delete(collection_name=self.collection_name, filter=expr)
        logger.info(f"[trajectory_client] Deleted trajectory(s) from collection '{self.collection_name}' with condition: {expr}")

    def get_all_trajectories(self, limit: int = 16384, offset: int = 0) -> List[Dict]:
        res = self.client.query(
            collection_name=self.collection_name,
            filter="",
            limit=limit,
            offset=offset,
        )
        return res

    def drop_collection(self):
        collection_name = self.collection_name
        super().drop_collection(self.collection_name)
        logger.info(f"Dropped trajectories collection '{collection_name}'")

    def update_trajectory(self, trajectory_id: str, **kwargs) -> bool:
        backup_data = None
        temp_id = None
        
        try:
            existing = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{trajectory_id}"',
                output_fields=["*"]
            )
            
            if not existing:
                logger.warning(f"[trajectory_client] Trajectory {trajectory_id} not found in collection '{self.collection_name}'")
                return False
            
            backup_data = existing[0].copy()
            update_data = backup_data.copy()
            
            logger.info(f"[trajectory_client] Starting update for trajectory {trajectory_id} in collection '{self.collection_name}'")
            logger.debug(f"[trajectory_client] Original data fields: {list(backup_data.keys())}")
            
            need_re_embed = False
            updated_fields = []
            
            for field, value in kwargs.items():
                if field in ['query', 'log', 'final_outcome', 'retrieved_principles', 'merged_trajectory_ids', 'golden_answer']:

                    if field in backup_data:
                        if field == 'query' and value != backup_data.get('query'):
                            need_re_embed = True
                            update_data['query'] = value
                            updated_fields.append(f"query: {value}")
                            
                        elif field in ['retrieved_principles', 'merged_trajectory_ids']:
                            if isinstance(value, list):
                                update_data[field] = json.dumps(value)
                            else:
                                update_data[field] = value
                            updated_fields.append(f"{field}: {value}")
                        
                        elif field == 'golden_answer':
                            update_data['golden_answer'] = value
                            updated_fields.append("golden_answer: <updated>")
                        
                        else:
                            update_data[field] = value
                            updated_fields.append(f"{field}: {value}")
                    else:
                        logger.warning(f"[trajectory_client] Field '{field}' not found in collection schema, skipping")
                else:
                    logger.warning(f"[trajectory_client] Field '{field}' is not updatable, skipping")
            
            if not updated_fields:
                logger.info(f"[trajectory_client] No valid fields to update for trajectory {trajectory_id} in collection '{self.collection_name}'")
                return True
            
            if need_re_embed and 'query_vector' in backup_data:
                logger.info(f"Re-generating embedding for updated query")
                update_data['query_vector'] = self._get_embedding(update_data['query'])
            
            update_data['updated_at'] = int(time.time())
            
            logger.info(f"[trajectory_client] Updated fields: {', '.join(updated_fields)}")
            
            temp_id = f"{trajectory_id}_temp_{int(time.time())}"
            temp_data = update_data.copy()
            temp_data['id'] = temp_id
            
            logger.debug(f"[trajectory_client] Step 1: Inserting temporary data with id {temp_id}")
            self.client.insert(collection_name=self.collection_name, data=[temp_data])

            temp_check = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{temp_id}"',
                output_fields=["id"]
            )
            
            if not temp_check:
                raise Exception("[trajectory_client] Failed to insert temporary data")
            
            logger.debug(f"[trajectory_client] Step 2: Deleting original data with id {trajectory_id}")
            self.client.delete(collection_name=self.collection_name, ids=[trajectory_id])
            
            logger.debug(f"[trajectory_client] Step 3: Inserting final updated data with id {trajectory_id}")
            self.client.insert(collection_name=self.collection_name, data=[update_data])

            final_check = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{trajectory_id}"',
                output_fields=["id"]
            )
            
            if not final_check:
                raise Exception("[trajectory_client] Failed to insert final updated data")

            logger.debug(f"[trajectory_client] Step 4: Cleaning up temporary data with id {temp_id}")
            self.client.delete(collection_name=self.collection_name, ids=[temp_id])
            
            logger.info(f"[trajectory_client] Successfully updated trajectory {trajectory_id} in collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"[trajectory_client] Error updating trajectory {trajectory_id} in collection '{self.collection_name}': {e}")
            
            try:
                if temp_id:
                    logger.info(f"[trajectory_client] Cleaning up temporary data {temp_id} due to error")
                    try:
                        self.client.delete(collection_name=self.collection_name, ids=[temp_id])
                    except:
                        pass
                
                existing_check = self.client.query(
                    collection_name=self.collection_name,
                    filter=f'id == "{trajectory_id}"',
                    output_fields=["id"]
                )
                
                if not existing_check and backup_data:
                    logger.info(f"[trajectory_client] Attempting to restore backup data for trajectory {trajectory_id}")
                    self.client.insert(collection_name=self.collection_name, data=[backup_data])
                    
                    restore_check = self.client.query(
                        collection_name=self.collection_name,
                        filter=f'id == "{trajectory_id}"',
                        output_fields=["id"]
                    )
                    
                    if restore_check:
                        logger.info(f"[trajectory_client] Successfully restored backup data for trajectory {trajectory_id}")
                    else:
                        logger.error(f"[trajectory_client] Failed to restore backup data for trajectory {trajectory_id}")
                
            except Exception as restore_error:
                logger.error(f"[trajectory_client] Error during recovery process: {restore_error}")
            
            return False
