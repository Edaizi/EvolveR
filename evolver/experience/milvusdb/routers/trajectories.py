from typing import List
from fastapi import APIRouter, HTTPException, Depends
import json
import logging
from datetime import datetime

from models import (TrajectoryCreateRequest, TrajectoryUpdateRequest,
                   TrajectoryResult, DeleteRequest, StatusResponse,
                   TrajectoryBatchCreateRequest, TrajectoryBatchGetRequest)
from dependencies import get_trajectory_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trajectories", tags=["trajectories"])


@router.post("/", response_model=StatusResponse)
async def create_trajectory(
    request: TrajectoryCreateRequest,
    client=Depends(get_trajectory_client)
):
    try:
        client.add_trajectory(
            trajectory_id=request.trajectory_id,
            query=request.query,
            log=request.log,
            final_outcome=request.final_outcome,
            retrieved_principles=request.retrieved_principles,
            golden_answer=request.golden_answer
        )
        
        logger.info(f"Successfully created trajectory: {request.trajectory_id}")
        return StatusResponse(
            status="success",
            message=f"Trajectory {request.trajectory_id} created successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating trajectory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create trajectory: {str(e)}")


@router.get("/", response_model=List[TrajectoryResult])
async def get_all_trajectories(client=Depends(get_trajectory_client)):
    try:
        trajectories = client.get_all_trajectories()
        
        results = []
        for trajectory in trajectories:
            try:
                retrieved_principles = json.loads(trajectory.get('retrieved_principles', '[]'))
            except json.JSONDecodeError:
                retrieved_principles = []
            
            results.append(TrajectoryResult(
                id=trajectory.get('id', ''),
                query=trajectory.get('query', ''),
                log=trajectory.get('log', ''),
                final_outcome=trajectory.get('final_outcome', False),
                retrieved_principles=retrieved_principles,
                golden_answer=trajectory.get('golden_answer', ''),
                created_at=trajectory.get('created_at', 0),
                updated_at=trajectory.get('updated_at', 0) 
            ))
        
        logger.info(f"Retrieved {len(results)} trajectories")
        return results
        
    except Exception as e:
        logger.error(f"Error getting all trajectories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trajectories: {str(e)}")


@router.delete("/", response_model=StatusResponse)
async def delete_trajectory(request: DeleteRequest,
                            client=Depends(get_trajectory_client)):
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="Trajectory client not initialized")
        
        if not request.id and not request.before_timestamp:
            raise HTTPException(status_code=400, detail="Must provide either id or before_timestamp")
        
        client.delete_trajectory(
            trajectory_id=request.id,
            before_timestamp=request.before_timestamp
        )
        
        message = f"Deleted trajectory by ID: {request.id}" if request.id else f"Deleted trajectories before timestamp: {request.before_timestamp}"
        logger.info(message)
        return StatusResponse(
            status="success",
            message=message,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error deleting trajectory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete trajectory: {str(e)}")


@router.put("/", response_model=StatusResponse)
async def update_trajectory(request: TrajectoryUpdateRequest,
                            client=Depends(get_trajectory_client)):
    try:
        update_data = {}
        if request.query is not None:
            update_data['query'] = request.query
        if request.log is not None:
            update_data['log'] = request.log
        if request.final_outcome is not None:
            update_data['final_outcome'] = request.final_outcome
        if request.retrieved_principles is not None:
            update_data['retrieved_principles'] = request.retrieved_principles
        if request.golden_answer is not None:
            update_data['golden_answer'] = request.golden_answer
   
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update provided")
        
        success = client.update_trajectory(request.trajectory_id, **update_data)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Trajectory {request.trajectory_id} not found")
        
        logger.info(f"Successfully updated trajectory: {request.trajectory_id}")
        return StatusResponse(
            status="success",
            message=f"Trajectory {request.trajectory_id} updated successfully",
            timestamp=datetime.now().isoformat()
        )   
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating trajectory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update trajectory: {str(e)}")


@router.get("/{trajectory_id}", response_model=TrajectoryResult)
async def get_trajectory_by_id(trajectory_id: str,
                               client=Depends(get_trajectory_client)):
    try:
        result = client.client.query(
            collection_name=client.collection_name,
            filter=f'id == "{trajectory_id}"',
            output_fields=["*"],
            limit=1
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Trajectory {trajectory_id} not found")
        
        trajectory = result[0]
        try:
            retrieved_principles = json.loads(trajectory.get('retrieved_principles', '[]'))
        except json.JSONDecodeError:
            retrieved_principles = []
        
        return TrajectoryResult(
            id=trajectory.get('id', ''),
            query=trajectory.get('query', ''),
            log=trajectory.get('log', ''),
            final_outcome=trajectory.get('final_outcome', False),
            retrieved_principles=retrieved_principles,
            golden_answer=trajectory.get('golden_answer', ''),
            created_at=trajectory.get('created_at', 0),
            updated_at=trajectory.get('updated_at', 0) 
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trajectory {trajectory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trajectory: {str(e)}")


@router.post("/batch", response_model=StatusResponse)
async def create_trajectories_batch(request: TrajectoryBatchCreateRequest,
                                    client=Depends(get_trajectory_client)):
    try:
        items = [
            {
                "trajectory_id": it.trajectory_id,
                "query": it.query,
                "log": it.log,
                "final_outcome": it.final_outcome,
                "retrieved_principles": it.retrieved_principles,
                "golden_answer": it.golden_answer or ""
            }
            for it in request.items
        ]
        client.add_trajectories_batch(items)
        logger.info(f"Successfully created {len(items)} trajectories (batch)")
        return StatusResponse(status="success", message=f"Inserted {len(items)} trajectories", timestamp=datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Error creating trajectories batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create trajectories batch: {str(e)}")


@router.post("/batch_get", response_model=List[TrajectoryResult])
async def get_trajectories_batch(request: TrajectoryBatchGetRequest,
                                 client=Depends(get_trajectory_client)):
    try:
        results = client.get_trajectories_by_ids(request.ids)
        formatted = []
        for trajectory in results:
            try:
                retrieved_principles = json.loads(trajectory.get('retrieved_principles', '[]'))
            except json.JSONDecodeError:
                retrieved_principles = []
            formatted.append(TrajectoryResult(
                id=trajectory.get('id', ''),
                query=trajectory.get('query', ''),
                log=trajectory.get('log', ''),
                final_outcome=trajectory.get('final_outcome', False),
                retrieved_principles=retrieved_principles,
                golden_answer=trajectory.get('golden_answer', ''),
                created_at=trajectory.get('created_at', 0),
                updated_at=trajectory.get('updated_at', 0)
            ))
        logger.info(f"Batch got {len(formatted)} trajectories")
        return formatted
    except Exception as e:
        logger.error(f"Error getting trajectories batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trajectories batch: {str(e)}")
