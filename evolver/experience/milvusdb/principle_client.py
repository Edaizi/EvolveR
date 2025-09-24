from typing import List, Dict, Tuple, Optional
from pymilvus import FieldSchema, DataType, CollectionSchema, MilvusClient
from dataclasses import dataclass, field
from collections import deque
import json
import time
from datetime import datetime
from config import BaseVectorDBClient, exp_config, logger
import numpy as np
from pymilvus import utility, Collection, connections, DataType, FieldSchema, CollectionSchema, AnnSearchRequest, RRFRanker


class PrincipleVectorDBClient(BaseVectorDBClient):
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
            self.collection_name = f"exp_principles_{experiment_name}"
        elif collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = exp_config.MILVUS_COLLECTION_NAME
            
        super().__init__(db_path, api_url, api_key, model_name)
        self._create_collection_if_not_exist()

    def _create_collection_if_not_exist(self):
        if self.collection_name not in self.client.list_collections():
            logger.info(f"Creating principles collection '{self.collection_name}'...")
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="structure", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="principle_type", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="metric_score", dtype=DataType.FLOAT),
                FieldSchema(name="usage_count", dtype=DataType.INT64),
                FieldSchema(name="success_count", dtype=DataType.INT64),
                FieldSchema(name="successful_trajectory_ids", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="failed_trajectory_ids", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64), 
            ]
            schema = CollectionSchema(fields=fields, description="Experience principles with embeddings")
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                dimension=self.dim,
                auto_id=False,
            )
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(field_name="description_vector", index_type="FLAT", metric_type="COSINE")
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)
            logger.info(f"Successfully created principles collection '{self.collection_name}'")

    def _sanitize_entity(self, data: dict) -> dict:
        import math as _math
        import json as _json
        d = dict(data or {})
        # 长度裁剪与默认值
        def _s(v, max_len):
            s = "" if v is None else str(v)
            return s[:max_len] if max_len and len(s) > max_len else s
        def _j(v, max_len):
            if isinstance(v, str):
                s = v
            elif v is None:
                s = "[]"
            else:
                try:
                    s = _json.dumps(v, ensure_ascii=False)
                except Exception:
                    s = "[]"
            return s[:max_len] if max_len and len(s) > max_len else s
        def _f(v):
            try:
                x = float(0.0 if v is None else v)
            except Exception:
                x = 0.0
            if not _math.isfinite(x):
                x = 0.0
            return x
        def _i(v):
            try:
                return int(0 if v is None else v)
            except Exception:
                return 0
        # 字段级别
        if 'id' in d:
            d['id'] = _s(d.get('id'), 128)
        if 'description' in d:
            d['description'] = _s(d.get('description'), 2048)
        if 'structure' in d:
            d['structure'] = _j(d.get('structure'), 8192)
        if 'principle_type' in d:
            d['principle_type'] = _s(d.get('principle_type') or 'guiding', 16)
        if 'metric_score' in d:
            d['metric_score'] = _f(d.get('metric_score'))
        if 'usage_count' in d:
            d['usage_count'] = _i(d.get('usage_count'))
        if 'success_count' in d:
            d['success_count'] = _i(d.get('success_count'))
        if 'successful_trajectory_ids' in d:
            d['successful_trajectory_ids'] = _j(d.get('successful_trajectory_ids'), 16384)
        if 'failed_trajectory_ids' in d:
            d['failed_trajectory_ids'] = _j(d.get('failed_trajectory_ids'), 16384)
        if 'created_at' in d:
            d['created_at'] = _i(d.get('created_at'))
        if 'updated_at' in d:
            d['updated_at'] = _i(d.get('updated_at'))
        if 'description_vector' in d:
            vec = d.get('description_vector')
            if not isinstance(vec, (list, tuple)):
                vec = [0.0] * self.dim
            else:
                v2 = []
                for x in vec:
                    try:
                        fx = float(x)
                    except Exception:
                        fx = 0.0
                    if not _math.isfinite(fx):
                        fx = 0.0
                    v2.append(fx)
                if len(v2) < self.dim:
                    v2.extend([0.0] * (self.dim - len(v2)))
                elif len(v2) > self.dim:
                    v2 = v2[:self.dim]
                vec = v2
            d['description_vector'] = vec
        return d

    def add_principle(self, principle_id: str, description: str, structure: List[Dict[str, str]],
            principle_type: str, metric_score: float, usage_count: int,
            success_count: int, successful_trajectory_ids: deque[str],
            failed_trajectory_ids: deque[str]) -> None:

        description_vector = self._get_embedding(description)
        current_time = int(time.time())
        data = {
            "id": principle_id,
            "description_vector": description_vector,
            "description": description,
            "structure": json.dumps(structure),
            "principle_type": principle_type,
            "metric_score": metric_score,
            "usage_count": usage_count,
            "success_count": success_count,
            "successful_trajectory_ids": json.dumps(list(successful_trajectory_ids)),
            "failed_trajectory_ids": json.dumps(list(failed_trajectory_ids)),
            "created_at": current_time,
            "updated_at": current_time,
        }
        res = self.client.insert(collection_name=self.collection_name, data=[data])
        logger.info(f"[Milvus] Inserted principle {principle_id} into collection '{self.collection_name}': {res}")

    def add_principles_batch(self, principles: List[Dict], embedding_batch_size: int = 256, vdb_insert_batch_size: int = 1000):
        """
        Adds a batch of principles to the Milvus collection.
        It fetches their embeddings in smaller, manageable batches to avoid overwhelming the embedding service,
        and then inserts all principles into Milvus in a single large batch.
        """
        if not principles:
            return

        # 1. Batch fetch embeddings in smaller chunks
        all_vectors = []
        num_principles = len(principles)
        logger.info(f"Starting to fetch embeddings for {num_principles} principles in batches of {embedding_batch_size}...")
        
        for i in range(0, num_principles, embedding_batch_size):
            batch_principles = principles[i:i + embedding_batch_size]
            descriptions = [p['description'] for p in batch_principles]
            
            try:
                logger.info(f"Fetching embeddings for batch {i//embedding_batch_size + 1}/{(num_principles + embedding_batch_size - 1)//embedding_batch_size}...")
                # Corrected the method call to use the correct provider and method name
                batch_vectors_np = self.embedding_provider.get_embeddings(descriptions)
                batch_vectors = batch_vectors_np.tolist()

                if len(batch_vectors) != len(descriptions):
                    logger.error(f"Embedding service returned {len(batch_vectors)} vectors for {len(descriptions)} descriptions. Aborting.")
                    return
                all_vectors.extend(batch_vectors)
            except Exception as e:
                logger.error(f"Failed to fetch embeddings for a batch: {e}")
                return # Abort the entire process if any embedding batch fails

        if len(all_vectors) != num_principles:
            logger.error("Final vector count does not match principle count. Aborting insertion.")
            return

        # 2. Prepare entities for a single large batch insertion
        entities = []
        for i, p in enumerate(principles):
            successful_ids = deque(p.get('successful_trajectory_ids', []), maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE)
            failed_ids = deque(p.get('failed_trajectory_ids', []), maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE)
            
            entities.append({
                "id": p['principle_id'],
                "description": p['description'],
                "description_vector": all_vectors[i],
                "structure": json.dumps(p.get('structure', [])),
                "principle_type": p.get('principle_type', 'guiding'),
                "metric_score": p.get('metric_score', 0.5),
                "usage_count": p.get('usage_count', 0),
                "success_count": p.get('success_count', 0),
                "successful_trajectory_ids": json.dumps(list(successful_ids)),
                "failed_trajectory_ids": json.dumps(list(failed_ids)),
                "created_at": int(time.time()),
                "updated_at": int(time.time())
            })
        
        # 3. Batch insert all entities into Milvus in chunks
        logger.info(f"Starting to insert {len(entities)} entities into Milvus in batches of {vdb_insert_batch_size}...")
        for i in range(0, len(entities), vdb_insert_batch_size):
            batch_entities = entities[i:i + vdb_insert_batch_size]
            try:
                result = self.client.insert(collection_name=self.collection_name, data=batch_entities)
                logger.info(f"[Milvus] Batch inserted {result['insert_count']} principles into collection '{self.collection_name}'.")
            except Exception as e:
                logger.error(f"Failed to batch insert a chunk of principles into Milvus: {e}")


    def search_principles(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        query_vector = self._get_embedding(query)
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["description", "structure", "metric_score", "principle_type", "usage_count", "success_count", "created_at", "updated_at", "successful_trajectory_ids", "failed_trajectory_ids"],
            anns_field="description_vector",
        )
        results = []
        if res and len(res) > 0 and len(res[0]) > 0:
            for hit in res[0]:
                principle_id = str(hit['id'])
                distance = hit.get('distance', 0.0)
                similarity_score = 1 - (distance / 2.0)
                entity = hit.get('entity', {})
                entity_data = {
                    "description": entity.get('description', ''),
                    "structure": entity.get('structure', ''),
                    "metric_score": entity.get('metric_score', 0.0),
                    "principle_type": entity.get('principle_type', ''),
                    "usage_count": entity.get('usage_count', 0),
                    "success_count": entity.get('success_count', 0),
                    "successful_trajectory_ids": entity.get('successful_trajectory_ids', '[]'),
                    "failed_trajectory_ids": entity.get('failed_trajectory_ids', '[]'),
                    "created_at": entity.get('created_at', 0),
                    "updated_at": entity.get('updated_at', 0),
                }
                results.append((principle_id, similarity_score, entity_data))
        return results
    
    def get_all_principles(self, limit: int = 16384, offset: int = 0) -> List[Dict]:
        res = self.client.query(
            collection_name=self.collection_name,
            filter="",
            limit=limit,
            offset=offset,
        )
        return res
    
    def delete_principles_batch(self, principle_ids: List[str]) -> int:
        """
        Deletes multiple principles from the collection by their IDs.
        Returns the number of deleted entities.
        """
        if not principle_ids:
            logger.warning("delete_principles_batch called with an empty list of IDs.")
            return 0

        # Milvus expects a string list for the 'in' operator, so we need to quote each ID
        quoted_ids = [f'"{pid}"' for pid in principle_ids]
        expr = f'id in [{",".join(quoted_ids)}]'
        
        try:
            res = self.client.delete(collection_name=self.collection_name, filter=expr)
            delete_count = res.delete_count if hasattr(res, 'delete_count') else len(principle_ids)
            logger.info(f"[principle_client] Deleted {delete_count} principle(s) from collection '{self.collection_name}' with condition: {expr}")
            return delete_count
        except Exception as e:
            logger.error(f"Failed to batch delete principles with expr '{expr}': {e}")
            raise

    def delete_principle(self, principle_id: Optional[str] = None, before_timestamp: Optional[int] = None):
        logger.info(f"delete_principle called with principle_id={principle_id}, before_timestamp={before_timestamp}")
        if principle_id:
            self.client.delete(
                collection_name=self.collection_name,
                ids=[principle_id] 
            )
            logger.info(f"[principle_client] Deleted principle by ID: {principle_id} from collection '{self.collection_name}'")
        elif before_timestamp:
            expr = f'created_at < {before_timestamp}'
            self.client.delete(
                collection_name=self.collection_name,
                filter=expr
            )
            logger.info(f"[principle_client] Deleted principles created before {before_timestamp} from collection '{self.collection_name}'")
        else:
            raise ValueError("[principle_client] Must provide principle_id or before_timestamp")

    def clean_low_metric_principles(self, threshold: float) -> int:
        expr = f"metric_score < {threshold}"
        res = self.client.delete(collection_name=self.collection_name, filter=expr)
        # The result of a successful delete operation is a list of deleted primary keys.
        delete_count = len(res) if isinstance(res, list) else 0
        logger.info(f"[principle_client] Cleaned {delete_count} principles with metric_score < {threshold} from collection '{self.collection_name}'")
        return delete_count

    def drop_collection(self):
        collection_name = self.collection_name
        super().drop_collection(self.collection_name)
        logger.info(f"Dropped principles collection '{collection_name}'")
    

    def update_principle(self, principle_id: str, **kwargs) -> bool:
        backup_data = None
        temp_id = None
        
        try:
            # 先查询现有数据
            existing = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{principle_id}"',
                output_fields=["*"]
            )
            
            if not existing:
                logger.warning(f"Principle {principle_id} not found in collection '{self.collection_name}'")
                return False
            
            backup_data = existing[0].copy()
            update_data = backup_data.copy()
            
            logger.info(f"[principle_client] Starting update for principle {principle_id} in collection '{self.collection_name}'")
            logger.debug(f"[principle_client] Original data fields: {list(backup_data.keys())}")
            
            need_re_embed = False
            updated_fields = []
            
            for field, value in kwargs.items():
                if field in ['description', 'structure', 'principle_type', 'metric_score', 
                        'usage_count', 'success_count', 'successful_trajectory_ids', 'failed_trajectory_ids']:
                    
                    if field in backup_data:
                        if field == 'description' and value != backup_data.get('description'):
                            need_re_embed = True
                            update_data['description'] = value
                            updated_fields.append(f"description: {value}")
                            
                        elif field == 'structure':
                            if isinstance(value, (list, dict)):
                                update_data['structure'] = json.dumps(value)
                            else:
                                update_data['structure'] = value
                            updated_fields.append(f"structure: {value}")
                            
                        elif field in ['successful_trajectory_ids', 'failed_trajectory_ids']:
                            if isinstance(value, (list, deque)):
                                update_data[field] = json.dumps(list(value))
                            else:
                                update_data[field] = value
                            updated_fields.append(f"{field}: {value}")
                            
                        else:
                            update_data[field] = value
                            updated_fields.append(f"{field}: {value}")
                    else:
                        logger.warning(f"[principle_client] Field '{field}' not found in collection schema, skipping")
                else:
                    logger.warning(f"[principle_client] Field '{field}' is not updatable, skipping")
            
            if not updated_fields:
                logger.info(f"[principle_client] No valid fields to update for principle {principle_id} in collection '{self.collection_name}'")
                return True
            
            if need_re_embed and 'description_vector' in backup_data:
                logger.info(f"[principle_client] Re-generating embedding for updated description")
                update_data['description_vector'] = self._get_embedding(update_data['description'])
            
            update_data['updated_at'] = int(time.time())  
            logger.info(f"[principle_client] Updated fields: {', '.join(updated_fields)}")
            
            temp_id = f"{principle_id}_temp_{int(time.time())}"
            temp_data = update_data.copy()
            temp_data['id'] = temp_id
            temp_data = self._sanitize_entity(temp_data)
            
            logger.debug(f"[principle_client] Step 1: Inserting temporary data with id {temp_id}")
            self.client.insert(collection_name=self.collection_name, data=[temp_data])
            
            temp_check = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{temp_id}"',
                output_fields=["id"]
            )
            
            if not temp_check:
                raise Exception("Failed to insert temporary data")
            
            logger.debug(f"[principle_client] Step 2: Deleting original data with id {principle_id}")
            self.client.delete(collection_name=self.collection_name, ids=[principle_id])
            
            logger.debug(f"[principle_client] Step 3: Inserting final updated data with id {principle_id}")
            self.client.insert(collection_name=self.collection_name, data=[self._sanitize_entity(update_data)])
            
            final_check = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{principle_id}"',
                output_fields=["id"]
            )
            
            if not final_check:
                raise Exception("Failed to insert final updated data")
            
            logger.debug(f"[principle_client] Step 4: Cleaning up temporary data with id {temp_id}")
            self.client.delete(collection_name=self.collection_name, ids=[temp_id])
            
            logger.info(f"[principle_client] Successfully updated principle {principle_id} in collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"[principle_client] Error updating principle {principle_id} in collection '{self.collection_name}': {e}")

            try:
                if temp_id:
                    logger.info(f"[principle_client] Cleaning up temporary data {temp_id} due to error")
                    try:
                        self.client.delete(collection_name=self.collection_name, ids=[temp_id])
                    except:
                        pass

                existing_check = self.client.query(
                    collection_name=self.collection_name,
                    filter=f'id == "{principle_id}"',
                    output_fields=["id"]
                )
                
                if not existing_check and backup_data:
                    logger.info(f"[principle_client] Attempting to restore backup data for principle {principle_id}")
                    self.client.insert(collection_name=self.collection_name, data=[self._sanitize_entity(backup_data)])
                    
                    # 验证恢复成功
                    restore_check = self.client.query(
                        collection_name=self.collection_name,
                        filter=f'id == "{principle_id}"',
                        output_fields=["id"]
                    )
                    
                    if restore_check:
                        logger.info(f"[principle_client] Successfully restored backup data for principle {principle_id}")
                    else:
                        logger.error(f"[principle_client] Failed to restore backup data for principle {principle_id}")
                
            except Exception as restore_error:
                logger.error(f"[principle_client] Error during recovery process: {restore_error}")
            
            return False

    def get_principles_batch(self, ids: List[str]) -> List[dict]:
        """Batch get principles by a list of IDs."""
        if not ids:
            return []
        try:
            filter_expr = f'id in {json.dumps(ids)}'
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["*"]
            )
            return results
        except Exception as e:
            logger.error(f"Error batch getting principles: {e}")
            return []

    def update_principles_batch(self, principles_data: List[dict]) -> bool:
        """
        Batch update principles using a safe "read-modify-write" strategy.
        This avoids the destructive nature of upsert by preserving all existing fields,
        handles re-embedding for description changes, and chunks operations for stability.
        """
        if not principles_data:
            return True

        ids_to_update = [p['principle_id'] for p in principles_data]
        
        # --- Step 1: Batch Read ---
        existing_principles_list = self.get_principles_batch(ids_to_update)
        if not existing_principles_list:
            logger.warning(f"None of the principles to update were found: {ids_to_update}")
            return False
        
        existing_map = {p['id']: p for p in existing_principles_list}
        updates_map = {p['principle_id']: p for p in principles_data}

        final_data_to_insert = []
        
        # --- Step 2: Modify in Memory ---
        for p_id in ids_to_update:
            existing_data = existing_map.get(p_id)
            update_kwargs = updates_map.get(p_id)
            if not existing_data or not update_kwargs:
                logger.warning(f"Skipping update for '{p_id}', either no existing data found or no update data provided.")
                continue
            
            updated_data = existing_data.copy()
            
            # Apply updates and check for description change
            new_description = update_kwargs.get('description')
            if new_description and new_description != existing_data.get('description'):
                logger.info(f"Re-embedding needed for principle {p_id} due to description change.")
                updated_data['description'] = new_description
                if 'description_vector' in updated_data:
                    updated_data['description_vector'] = self._get_embedding(new_description)

            for field, value in update_kwargs.items():
                if field != 'description' and field in updated_data: # Description is already handled
                    updated_data[field] = value
            
            updated_data['updated_at'] = int(time.time())
            final_data_to_insert.append(updated_data)

        if not final_data_to_insert:
            logger.info("No valid data to update after processing.")
            return True
            
        # --- Step 3: Batch Write (Delete then Insert) in Chunks ---
        CHUNK_SIZE = 1000  # Safe chunk size to avoid Milvus limits
        try:
            for i in range(0, len(final_data_to_insert), CHUNK_SIZE):
                chunk = final_data_to_insert[i:i + CHUNK_SIZE]
                ids_in_chunk = [p['id'] for p in chunk]
                
                logger.info(f"Processing chunk {i//CHUNK_SIZE + 1}: {len(chunk)} principles.")
                
                self.client.delete(collection_name=self.collection_name, ids=ids_in_chunk)
                sanitized_chunk = [self._sanitize_entity(p) for p in chunk]
                self.client.insert(collection_name=self.collection_name, data=sanitized_chunk)

            logger.info(f"Successfully batch updated a total of {len(final_data_to_insert)} principles.")
            return True
        except Exception as e:
            logger.error(f"Error during chunked batch update: {e}")
            return False

