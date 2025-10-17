from typing import List
from fastapi import APIRouter, HTTPException, Depends
from collections import deque
import json
import logging
from datetime import datetime

from models import (PrincipleCreateRequest, PrincipleUpdateRequest, 
                   PrincipleResult, DeleteRequest, StatusResponse, 
                   SearchRequest, SearchResult, PrincipleBatchGetRequest, PrincipleBatchUpdateRequest, CleanLowMetricRequest,
                   PrincipleBatchDeleteRequest)
from dependencies import get_principle_client
from config import exp_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/principles", tags=["principles"])


@router.post("/", response_model=StatusResponse)
async def create_principle(
    request: PrincipleCreateRequest,
    client=Depends(get_principle_client)
):
    try:
        successful_ids = deque(
            request.successful_trajectory_ids, 
            maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE
        )
        failed_ids = deque(
            request.failed_trajectory_ids, 
            maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE
        )
        
        client.add_principle(
            principle_id=request.principle_id,
            description=request.description,
            structure=request.structure,
            principle_type=request.principle_type,
            metric_score=request.metric_score,
            usage_count=request.usage_count,
            success_count=request.success_count,
            successful_trajectory_ids=successful_ids,
            failed_trajectory_ids=failed_ids
        )
        
        logger.info(f"Successfully created principle: {request.principle_id}")
        return StatusResponse(
            status="success",
            message=f"Principle {request.principle_id} created successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating principle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create principle: {str(e)}")




@router.get("/", response_model=List[PrincipleResult])
async def get_all_principles(client=Depends(get_principle_client)):
    try:
        principles = client.get_all_principles()
        
        results = []
        for principle in principles:
            try:
                structure = json.loads(principle.get('structure', '[]'))
            except json.JSONDecodeError:
                structure = []
            try:
                successful_trajectory_ids = json.loads(principle.get('successful_trajectory_ids', '[]'))
            except json.JSONDecodeError:
                successful_trajectory_ids = []
                
            try:
                failed_trajectory_ids = json.loads(principle.get('failed_trajectory_ids', '[]'))
            except json.JSONDecodeError:
                failed_trajectory_ids = []
                
            results.append(PrincipleResult(
                id=principle.get('id', ''),
                description=principle.get('description', ''),
                structure=structure,
                principle_type=principle.get('principle_type', ''),
                metric_score=principle.get('metric_score', 0.0),
                usage_count=principle.get('usage_count', 0),
                success_count=principle.get('success_count', 0),
                successful_trajectory_ids=successful_trajectory_ids,  
                failed_trajectory_ids=failed_trajectory_ids,     
                created_at=principle.get('created_at', 0),
                updated_at=principle.get('updated_at', 0) 
            ))
        
        logger.info(f"Retrieved {len(results)} principles")
        return results
        
    except Exception as e:
        logger.error(f"Error getting all principles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get principles: {str(e)}")


@router.get("/{principle_id}", response_model=PrincipleResult)
async def get_principle_by_id(
    principle_id: str, 
    client=Depends(get_principle_client)
):
    try:
        result = client.client.query(
            collection_name=client.collection_name,
            filter=f'id == "{principle_id}"',
            output_fields=["*"],
            limit=1
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Principle {principle_id} not found")
        
        principle = result[0]
        try:
            structure = json.loads(principle.get('structure', '[]'))
        except json.JSONDecodeError:
            structure = []
        
        try:
            successful_trajectory_ids = json.loads(principle.get('successful_trajectory_ids', '[]'))
        except json.JSONDecodeError:
            successful_trajectory_ids = []
                
        try:
            failed_trajectory_ids = json.loads(principle.get('failed_trajectory_ids', '[]'))
        except json.JSONDecodeError:
            failed_trajectory_ids = []
        
        return PrincipleResult(
            id=principle.get('id', ''),
            description=principle.get('description', ''),
            structure=structure,
            principle_type=principle.get('principle_type', ''),
            metric_score=principle.get('metric_score', 0.0),
            usage_count=principle.get('usage_count', 0),
            success_count=principle.get('success_count', 0),
            successful_trajectory_ids=successful_trajectory_ids, 
            failed_trajectory_ids=failed_trajectory_ids,       
            created_at=principle.get('created_at', 0),
            updated_at=principle.get('updated_at', 0) 
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting principle {principle_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get principle: {str(e)}")


@router.put("/", response_model=StatusResponse)
async def update_principle(
    request: PrincipleUpdateRequest,
    client=Depends(get_principle_client)
):
    try:
        update_data = {}
        if request.description is not None:
            update_data['description'] = request.description
        if request.structure is not None:
            update_data['structure'] = request.structure
        if request.principle_type is not None:
            update_data['principle_type'] = request.principle_type
        if request.metric_score is not None:
            update_data['metric_score'] = request.metric_score
        if request.usage_count is not None:
            update_data['usage_count'] = request.usage_count
        if request.success_count is not None:
            update_data['success_count'] = request.success_count
        if request.successful_trajectory_ids is not None:
            update_data['successful_trajectory_ids'] = deque(
                request.successful_trajectory_ids, 
                maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE
            )
        if request.failed_trajectory_ids is not None:
            update_data['failed_trajectory_ids'] = deque(
                request.failed_trajectory_ids, 
                maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE
            )
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update provided")
        
        success = client.update_principle(request.principle_id, **update_data)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Principle {request.principle_id} not found")
        
        logger.info(f"Successfully updated principle: {request.principle_id}")
        return StatusResponse(
            status="success",
            message=f"Principle {request.principle_id} updated successfully",
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating principle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update principle: {str(e)}")


@router.delete("/", response_model=StatusResponse)
async def delete_principle(
    request: DeleteRequest,
    client=Depends(get_principle_client)
):
    try:
        if not request.id and not request.before_timestamp:
            raise HTTPException(status_code=400, detail="Must provide either id or before_timestamp")
        
        client.delete_principle(
            principle_id=request.id,
            before_timestamp=request.before_timestamp
        )
        
        message = f"Deleted principle by ID: {request.id}" if request.id else f"Deleted principles before timestamp: {request.before_timestamp}"
        logger.info(message)
        return StatusResponse(
            status="success",
            message=message,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error deleting principle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete principle: {str(e)}")


@router.delete("/delete_batch", response_model=StatusResponse)
async def delete_principles_batch(
    request: PrincipleBatchDeleteRequest,
    client=Depends(get_principle_client)
):
    """Batch delete principles by a list of IDs."""
    try:
        if not request.ids:
            raise HTTPException(status_code=400, detail="No IDs provided for batch deletion.")
        
        count = client.delete_principles_batch(principle_ids=request.ids)
        
        message = f"Batch delete request for {len(request.ids)} principles processed. Deleted: {count}."
        logger.info(message)
        return StatusResponse(
            status="success",
            message=message,
            timestamp=datetime.now().isoformat(),
            delete_count=count
        )
    except Exception as e:
        logger.error(f"Error during batch principle deletion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to batch delete principles: {str(e)}")


@router.delete("/clean_low_metric", response_model=StatusResponse)
async def clean_low_metric_principles(request: CleanLowMetricRequest,
                                      client=Depends(get_principle_client)):
    try:
        delete_count = client.clean_low_metric_principles(threshold=request.threshold)
        msg = f"Cleaned {delete_count} principles with metric_score < {request.threshold}"
        logger.info(msg)
        return StatusResponse(status="success", message=msg, timestamp=datetime.now().isoformat(), delete_count=delete_count)
    except Exception as e:
        logger.error(f"Error cleaning low-metric principles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clean low-metric principles: {str(e)}")


@router.post("/batch_get", response_model=List[PrincipleResult])
async def batch_get_principles(
    request: PrincipleBatchGetRequest,
    client=Depends(get_principle_client)
):
    try:
        principles = client.get_principles_batch(ids=request.ids)
        
        results = []
        for p in principles:
            try:
                structure = json.loads(p.get('structure', '[]'))
            except (json.JSONDecodeError, TypeError):
                structure = []
            try:
                successful_trajectory_ids = json.loads(p.get('successful_trajectory_ids', '[]'))
            except (json.JSONDecodeError, TypeError):
                successful_trajectory_ids = []
            try:
                failed_trajectory_ids = json.loads(p.get('failed_trajectory_ids', '[]'))
            except (json.JSONDecodeError, TypeError):
                failed_trajectory_ids = []

            results.append(PrincipleResult(
                id=p.get('id', ''),
                description=p.get('description', ''),
                structure=structure,
                principle_type=p.get('principle_type', ''),
                metric_score=p.get('metric_score', 0.0),
                usage_count=p.get('usage_count', 0),
                success_count=p.get('success_count', 0),
                successful_trajectory_ids=successful_trajectory_ids,
                failed_trajectory_ids=failed_trajectory_ids,
                created_at=p.get('created_at', 0),
                updated_at=p.get('updated_at', 0)
            ))
        
        logger.info(f"Successfully batch retrieved {len(results)} principles.")
        return results
    except Exception as e:
        logger.error(f"Error batch getting principles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to batch get principles: {str(e)}")



@router.put("/batch_update", response_model=StatusResponse)
async def batch_update_principles(
    request: PrincipleBatchUpdateRequest,
    client=Depends(get_principle_client)
):
    try:
        # Convert Pydantic models to dictionaries for the client
        update_data = [item.dict() for item in request.items]
        
        success = client.update_principles_batch(update_data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Batch update operation failed in the client.")
            
        logger.info(f"Successfully triggered batch update for {len(request.items)} principles.")
        return StatusResponse(
            status="success",
            message=f"Batch update for {len(request.items)} principles completed successfully.",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error during batch update of principles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to batch update principles: {str(e)}")


