from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
import json
import os
import time
import logging
from datetime import datetime
from collections import deque
from models import (ExportRequest, ExportResponse, StatusResponse, 
                   PrincipleCreateRequest, TrajectoryCreateRequest)
from dependencies import get_principle_client, get_trajectory_client
from config import BASE_DIR, exp_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export-import"])


@router.post("/", response_model=ExportResponse)
async def export_data(
    request: ExportRequest,
    principle_client=Depends(get_principle_client),
    trajectory_client=Depends(get_trajectory_client)
):

    try:
        if request.output_root_dir and request.experiment_name:
            output_dir = Path(request.output_root_dir) / request.experiment_name / "db_exports"
        elif request.output_root_dir:
            output_dir = Path(request.output_root_dir)
        else:
            output_dir = Path(BASE_DIR) / "exports"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        skipped = []

        if request.format not in ["jsonl", "json"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'jsonl' or 'json'")
        
        if request.mode not in ["overwrite", "append"]:
            raise HTTPException(status_code=400, detail="Mode must be 'overwrite' or 'append'")
        if request.format not in ["jsonl", "json"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'jsonl' or 'json'")
        
        if "principles" in request.collections:
            principles_file = await _export_collection(
                client=principle_client,
                collection_type="principles",
                output_dir=output_dir,
                exp_name=request.experiment_name,
                format=request.format,
                include_metadata=request.include_metadata
            )
            if principles_file:
                created_files.append(str(principles_file))
                logger.info(f"Exported principles to: {principles_file}")
            else:
                logger.warning("Principles collection is empty. Skipped export.")
                skipped.append("principles")

        if "trajectories" in request.collections:
            trajectories_file = await _export_collection(
                client=trajectory_client,
                collection_type="trajectories",
                output_dir=output_dir,
                exp_name=request.experiment_name,
                format=request.format,
                include_metadata=request.include_metadata
            )
            if trajectories_file:
                created_files.append(str(trajectories_file))
                logger.info(f"Exported trajectories to: {trajectories_file}")
            else:
                logger.warning("Trajectories collection is empty. Skipped export.")
                skipped.append("trajectories")

        if created_files:
            msg = f"Successfully exported {len(created_files)} files"
            if skipped:
                msg += f"; skipped empty collections: {', '.join(skipped)}"
        else:
            msg = "No data found in selected collections. Nothing exported."
        
        return ExportResponse(
            status="success",
            message=msg,
            files_created=created_files,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.post("/import", response_model=StatusResponse)
async def import_data(
    file_path: str, 
    collection_type: str, 
    format: str = "jsonl",
    principle_client=Depends(get_principle_client),
    trajectory_client=Depends(get_trajectory_client)
):
    try:
        if collection_type not in ["principles", "trajectories"]:
            raise HTTPException(status_code=400, detail="collection_type must be 'principles' or 'trajectories'")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        imported_count = 0
        errors = []
        
        # Batch processing setup
        batch = []
        batch_size = 1000  # Records per batch
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if format == "jsonl":
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        if "_metadata" in record:
                            continue
                        batch.append(record)
                        
                        if len(batch) >= batch_size:
                            imported_count += await _import_batch(batch, collection_type, principle_client, trajectory_client)
                            batch.clear()
                            
                    except json.JSONDecodeError as e:
                        error_msg = f"Skipping invalid JSON on line {line_num}: {e}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Error importing record on line {line_num}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                # Import any remaining records in the last batch
                if batch:
                    imported_count += await _import_batch(batch, collection_type, principle_client, trajectory_client)
                    batch.clear()

            else:  # json format, less common for large datasets
                data = json.load(f)
                for i, record in enumerate(data):
                    if "_metadata" in record:
                        continue
                    try:
                        if collection_type == "principles":
                            await _import_principle_record(record, principle_client)
                        else:
                            await _import_trajectory_record(record, trajectory_client)
                        imported_count += 1
                    except Exception as e:
                        error_msg = f"Error importing record {i}: {record.get('id', 'unknown')} - {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
        
        message = f"Successfully imported {imported_count} {collection_type} records from {file_path}"
        if errors:
            message += f". {len(errors)} errors occurred."
        
        return StatusResponse(
            status="success",
            message=message,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error importing data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import data: {str(e)}")


async def _import_batch(batch: List[dict], collection_type: str, p_client, t_client):
    """Imports a batch of records and returns the count of successful imports."""
    try:
        if collection_type == "principles":
            # Convert record keys to match what add_principles_batch expects
            items_to_add = [
                {
                    "principle_id": rec.get('id', ''),
                    "description": rec.get('description', ''),
                    "structure": rec.get('structure', []),
                    "principle_type": rec.get('principle_type', 'guiding'),
                    "metric_score": rec.get('metric_score', 0.5),
                    "usage_count": rec.get('usage_count', 0),
                    "success_count": rec.get('success_count', 0),
                    "successful_trajectory_ids": rec.get('successful_trajectory_ids', []),
                    "failed_trajectory_ids": rec.get('failed_trajectory_ids', [])
                }
                for rec in batch
            ]
            p_client.add_principles_batch(items_to_add)
            return len(items_to_add)
        elif collection_type == "trajectories":
            items_to_add = [
                {
                    "trajectory_id": rec.get('id', ''),
                    "query": rec.get('query', ''),
                    "log": rec.get('log', ''),
                    "final_outcome": rec.get('final_outcome', False),
                    "retrieved_principles": rec.get('retrieved_principles', []),
                    "golden_answer": rec.get('golden_answer', "")
                }
                for rec in batch
            ]
            t_client.add_trajectories_batch(items_to_add)
            return len(items_to_add)
    except Exception as e:
        logger.error(f"Error during batch import for {collection_type}: {e}")
        # In case of batch failure, we don't know which one failed, so we count 0
    return 0


async def _export_collection(client, collection_type: str, output_dir: Path, exp_name: str, format: str, include_metadata: bool) -> Path:
    try:
        sanitized_exp_name = exp_name.replace('-', '_') if exp_name else "export"
        for old_file in output_dir.glob(f"{collection_type}_{sanitized_exp_name}_*.{format}"):
            try:
                old_file.unlink()
                logger.info(f"Removed old export file: {old_file}")
            except OSError as e:
                logger.error(f"Error removing old file {old_file}: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{collection_type}_{sanitized_exp_name}_{timestamp}.{format}"
        filepath = output_dir / filename
        
        # --- High-Performance, Chunked Export Logic ---
        CHUNK_SIZE = 10000  # Process N records at a time
        MILVUS_MAX_FETCH_LIMIT = 16384 # Milvus has a hard limit for offset + topk
        total_exported = 0

        with open(filepath, 'w', encoding='utf-8') as f:
            if format == "json":
                f.write("[\n")  # Start of JSON array

            offset = 0
            while True:
                # Adjust limit to not exceed Milvus's max fetch size and respect the offset
                limit = min(CHUNK_SIZE, MILVUS_MAX_FETCH_LIMIT - offset)

                if limit <= 0:
                    logger.warning(
                        f"Approaching Milvus fetch limit of {MILVUS_MAX_FETCH_LIMIT} for '{collection_type}'. "
                        f"Stopping export for this collection to prevent errors. Export may be incomplete."
                    )
                    break

                logger.info(f"Fetching chunk for '{collection_type}' collection (offset: {offset}, limit: {limit})...")
                if collection_type == "principles":
                    records_chunk = client.get_all_principles(limit=limit, offset=offset)
                elif collection_type == "trajectories":
                    records_chunk = client.get_all_trajectories(limit=limit, offset=offset)
                else:
                    break  # Should not happen

                if not records_chunk:
                    logger.info(f"No more records found for '{collection_type}'. Export finished for this collection.")
                    break
                
                logger.info(f"Processing {len(records_chunk)} records from chunk...")
                
                processed_chunk = [_process_record(rec, collection_type) for rec in records_chunk]

                if format == "jsonl":
                    for record in processed_chunk:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                elif format == "json":
                    # For all but the first chunk, prepend a comma
                    if total_exported > 0:
                        f.write(",\n")
                    
                    # Write all items except the last with a trailing comma
                    for i, record in enumerate(processed_chunk):
                        f.write(json.dumps(record, ensure_ascii=False, indent=2))
                        if i < len(processed_chunk) - 1:
                            f.write(",\n")

                total_exported += len(records_chunk)
                offset += len(records_chunk)  # Use actual returned count for the next offset

                # If we received fewer records than we requested, we are at the end
                if len(records_chunk) < limit:
                    break
            
            if format == "json":
                f.write("\n]\n")  # End of JSON array

        if total_exported == 0:
            logger.warning(f"No records found in '{collection_type}'; skipping export.")
            try:
                filepath.unlink() # Clean up empty file
            except OSError:
                pass
            return None

        logger.info(f"Successfully exported {total_exported} records for '{collection_type}' to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error exporting collection '{collection_type}': {e}")
        raise

def _process_record(record_data: dict, collection_type: str) -> dict:
    """Helper to process a single record for export."""
    if collection_type == "principles":
        record_data.pop('description_vector', None)
        record_data['structure'] = _safe_json_loads(record_data.get('structure', '[]'), [])
        record_data['successful_trajectory_ids'] = _safe_json_loads(record_data.get('successful_trajectory_ids', '[]'), [])
        record_data['failed_trajectory_ids'] = _safe_json_loads(record_data.get('failed_trajectory_ids', '[]'), [])
        whitelisted_fields = ["id", "created_at", "description", "failed_trajectory_ids", "metric_score", "principle_type", "structure", "success_count", "successful_trajectory_ids", "updated_at", "usage_count"]
    
    elif collection_type == "trajectories":
        record_data.pop('query_vector', None)
        record_data['retrieved_principles'] = _safe_json_loads(record_data.get('retrieved_principles', '[]'), [])
        whitelisted_fields = ["id", "created_at", "final_outcome", "golden_answer", "log", "query", "retrieved_principles", "updated_at"]
    
    return {k: record_data[k] for k in whitelisted_fields if k in record_data}


async def _import_principle_record(record: dict, client):
    try:
        successful_ids = deque(
            record.get('successful_trajectory_ids', []), 
            maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE
        )
        failed_ids = deque(
            record.get('failed_trajectory_ids', []), 
            maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE
        )
        
        client.add_principle(
            principle_id=record.get('id', ''),
            description=record.get('description', ''),
            structure=record.get('structure', []),
            principle_type=record.get('principle_type', 'guiding'),
            metric_score=record.get('metric_score', 0.5),
            usage_count=record.get('usage_count', 0),
            success_count=record.get('success_count', 0),
            successful_trajectory_ids=successful_ids,
            failed_trajectory_ids=failed_ids
        )
    except Exception as e:
        logger.error(f"Error importing principle record {record.get('id', 'unknown')}: {e}")
        raise

async def _import_trajectory_record(record: dict, client):
    try:
        client.add_trajectory(
            trajectory_id=record.get('id', ''),
            query=record.get('query', ''),
            log=record.get('log', ''),
            final_outcome=record.get('final_outcome', False),
            retrieved_principles=record.get('retrieved_principles', []),
            golden_answer=record.get('golden_answer', "")
        )
    except Exception as e:
        logger.error(f"Error importing trajectory record {record.get('id', 'unknown')}: {e}")
        raise

def _safe_json_loads(json_str: str, default_value):
    try:
        return json.loads(json_str) if json_str else default_value
    except (json.JSONDecodeError, TypeError):
        return default_value
