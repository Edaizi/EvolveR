import time
import psutil
import logging
from config import exp_config
from typing import Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
from collections import deque
from models import StatusResponse
from dependencies import get_principle_client,get_trajectory_client, initialize_clients
from routers.principles import create_principle  
from routers.trajectories import create_trajectory  

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/initial", tags=["initialization"])


@router.post("/auto-import", response_model=StatusResponse)
async def auto_import_data(
    principle_file: Optional[str] = Query(default=None, description="Path to principles file (json/jsonl)"),
    trajectory_file: Optional[str] = Query(default=None, description="Path to trajectories file (json/jsonl)"),
    format: str = Query(default="jsonl", description="json or jsonl"),
    principle_client=Depends(get_principle_client),
    trajectory_client=Depends(get_trajectory_client)
):
    try:
        from routers.export_import import import_data
        imported = []
        if principle_file:
            await import_data(principle_file, "principles", format, principle_client, trajectory_client)
            imported.append("principles")
        if trajectory_file:
            await import_data(trajectory_file, "trajectories", format, principle_client, trajectory_client)
            imported.append("trajectories")

        if not principle_file and not trajectory_file:
            import os
            experiment_name = os.environ.get("EXPERIMENT_NAME", "").replace('-', '_')
            if not experiment_name:
                return StatusResponse(status="success", message="No experiment name provided; auto-import skipped.", timestamp=datetime.now().isoformat())
            base_dir = Path("/mnt/petrelfs/wurong/workspace/evolver/data/evolver/result") / experiment_name / "db_exports"
            candidates = list(base_dir.glob("principles_*.jsonl")) + list(base_dir.glob("trajectories_*.jsonl"))
            if not candidates:
                msg = f"No export files found under {base_dir}; auto-import skipped."
                logger.warning(msg)
                return StatusResponse(status="success", message=msg, timestamp=datetime.now().isoformat())
            p_file = next((str(p) for p in candidates if Path(p).name.startswith("principles_")), None)
            t_file = next((str(p) for p in candidates if Path(p).name.startswith("trajectories_")), None)
            if p_file:
                await import_data(p_file, "principles", "jsonl", principle_client, trajectory_client)
                imported.append("principles")
            if t_file:
                await import_data(t_file, "trajectories", "jsonl", principle_client, trajectory_client)
                imported.append("trajectories")
        
        message = "Imported: " + ", ".join(imported) if imported else "Nothing imported"
        return StatusResponse(status="success", message=message, timestamp=datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Error auto importing data: {e}")
        raise HTTPException(status_code=500, detail=f"Auto import failed: {str(e)}")


@router.post("/sample-principles", response_model=StatusResponse)
async def initialize_sample_principles(client=Depends(get_principle_client)):  
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="Principle client not initialized")
        client._create_collection_if_not_exist()
        sample_principles_data = [
            {
                "principle_id": "p_001",
                "description": "Break down complex problems into smaller, manageable sub-problems to simplify analysis and execution.",
                "structure": [{"action": "decompose", "target": "complex problems", "result": "simplified"}],
                "principle_type": "guiding",
                "metric_score": 0.9,
                "usage_count": 10,
                "success_count": 9,
                "successful_trajectory_ids": deque(["traj_s_001", "traj_s_002"], maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE),
                "failed_trajectory_ids": deque([], maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE)
            },
            {
                "principle_id": "p_002",
                "description": "Avoid using hardcoded credentials directly in code; always use environment variables or a secure secret management system.",
                "structure": [{"rule": "no hardcoding", "object": "credentials", "method": "environment variables"}],
                "principle_type": "cautionary",
                "metric_score": 0.7,
                "usage_count": 5,
                "success_count": 3,
                "successful_trajectory_ids": deque(["traj_s_003"], maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE),
                "failed_trajectory_ids": deque(["traj_f_001"], maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE)
            },
            {
                "principle_id": "p_003",
                "description": "Always validate and sanitize user input to prevent injection attacks and ensure data integrity.",
                "structure": [{"rule": "validate", "target": "user input", "method": "sanitization", "purpose": "security"}],
                "principle_type": "cautionary",
                "metric_score": 0.85,
                "usage_count": 8,
                "success_count": 7,
                "successful_trajectory_ids": deque(["traj_s_004", "traj_s_005"], maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE),
                "failed_trajectory_ids": deque(["traj_f_002"], maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE)
            },
            {
                "principle_id": "p_004",
                "description": "Use consistent code formatting and meaningful variable names to enhance code readability and maintainability.",
                "structure": [{"action": "standardize", "target": "code style", "benefit": "readability"}],
                "principle_type": "guiding",
                "metric_score": 0.75,
                "usage_count": 15,
                "success_count": 13,
                "successful_trajectory_ids": deque(["traj_s_006", "traj_s_007", "traj_s_008"], maxlen=exp_config.MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE),
                "failed_trajectory_ids": deque(["traj_f_003", "traj_f_004"], maxlen=exp_config.MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE)
            }
        ]
        
        for data in sample_principles_data:
            client.add_principle(**data)
        
        logger.info("Sample principles initialized successfully")
        return StatusResponse(
            status="success",
            message="Sample principles initialized successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error initializing sample principles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize sample principles: {str(e)}")




@router.post("/sample-trajectories", response_model=StatusResponse)
async def initialize_sample_trajectories(client=Depends(get_trajectory_client)):
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="Trajectory client not initialized")
        client._create_collection_if_not_exist()
        sample_trajectories_data = [
            {
                "trajectory_id": "traj_s_001",
                "query": "How to decompose a complex sorting algorithm problem?",
                "log": """Step 1: Analyzed the problem - need to sort large dataset efficiently
                    Step 2: Decomposed into sub-problems:
                    - Data validation and preprocessing
                    - Algorithm selection (merge sort vs quick sort)
                    Final outcome: Successfully implemented efficient sorting solution""",
                "final_outcome": True,
                "retrieved_principles": ["p_001"]
            },
            {
                "trajectory_id": "traj_s_002", 
                "query": "How to design a complex API architecture?",
                "log": """Step 1: Requirements analysis - identified need for comprehensive API
                    Step 2: Applied decomposition principle:
                    - Authentication endpoints
                    Result: Well-structured, maintainable API design""",
                "final_outcome": True,
                "retrieved_principles": ["p_001"]
            },
            {
                "trajectory_id": "traj_s_003",
                "query": "How to secure database credentials in application?",
                "log": """Step 1: Code review revealed hardcoded database credentials
                    Step 2: Applied security principle - use environment variables
                    Step 3: Created .env file with sensitive credentials
                    Result: Secured credentials properly using environment variables""",
                "final_outcome": True,
                "retrieved_principles": ["p_002"]
            },
            {
                "trajectory_id": "traj_s_004",
                "query": "How to implement secure user input validation?",
                "log": """Step 1: Analyzed all user input fields in registration system
                    Step 2: Applied input validation principle
                    Result: Robust input validation preventing injection attacks""",
                "final_outcome": True,
                "retrieved_principles": ["p_003"]
            },
            
            {
                "trajectory_id": "traj_f_001",
                "query": "How to integrate third-party API securely?",
                "log": """Step 1: Added third-party API integration for payment processing
                    Result: FAILED - Credentials exposure led to security incident
                    Lesson: Should have used environment variables from the start""",
                "final_outcome": False,
                "retrieved_principles": ["p_002"] 
            },
            {
                "trajectory_id": "traj_f_002",
                "query": "How to implement user comment system quickly?",
                "log": """Step 1: Implemented user comment functionality for blog
                    Step 2: Added basic length validation only
                    Result: FAILED - Insufficient input validation led to XSS vulnerability
                    Lesson: Input validation principle should never be compromised""",
                "final_outcome": False,
                "retrieved_principles": ["p_003"]
            },
            {
                "trajectory_id": "traj_f_003",
                "query": "How to refactor legacy code quickly?",
                "log": """Step 1: Tasked with refactoring complex legacy authentication system
                    Step 2: Attempted to tackle entire system at once due to time pressure
                    Step 3: MISTAKE - Did not apply decomposition principle
                    Result: FAILED - Monolithic refactoring approach caused system instability
                    Lesson: Should have broken down the refactoring into smaller, manageable parts""",
                "final_outcome": False,
                "retrieved_principles": ["p_001"] 
            }
        ]

        for trajectory_data in sample_trajectories_data:
            client.add_trajectory(**trajectory_data)
        
        logger.info("Sample trajectories initialized successfully")
        return StatusResponse(
            status="success",
            message="Sample trajectories initialized successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error initializing sample trajectories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize sample trajectories: {str(e)}")

@router.post("/all-sample-data", response_model=StatusResponse)
async def initialize_all_sample_data(
    principle_client=Depends(get_principle_client),
    trajectory_client=Depends(get_trajectory_client)
):
    try:
        await initialize_sample_principles(principle_client)
        
        await initialize_sample_trajectories(trajectory_client)
        
        logger.info("All sample data initialized successfully")
        return StatusResponse(
            status="success",
            message="All sample data (principles and trajectories) initialized successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error initializing all sample data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize all sample data: {str(e)}")



@router.delete("/clear-principles", response_model=StatusResponse)
async def clear_principles_data(client=Depends(get_principle_client)):
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="Principle client not initialized")
        client.drop_collection()
        client._create_collection_if_not_exist()
        logger.warning("All principles data cleared and collection recreated")
        
        return StatusResponse(
            status="success",
            message="All principles data cleared successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error clearing principles data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear principles data: {str(e)}")


@router.delete("/clear-trajectories", response_model=StatusResponse)
async def clear_trajectories_data(client=Depends(get_trajectory_client)):
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="Trajectory client not initialized")
        client.drop_collection()
        client._create_collection_if_not_exist()
        logger.warning("All trajectories data cleared and collection recreated")
        
        return StatusResponse(
            status="success",
            message="All trajectories data cleared successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error clearing trajectories data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear trajectories data: {str(e)}")



@router.delete("/clear-database", response_model=StatusResponse)
async def clear_all_data(
    principle_client=Depends(get_principle_client),
    trajectory_client=Depends(get_trajectory_client)
):
    try:
        if principle_client is None or trajectory_client is None:
            raise HTTPException(status_code=500, detail="Database clients not initialized")
        
        principle_collection_name = principle_client.collection_name
        trajectory_collection_name = trajectory_client.collection_name
        
        principle_client.drop_collection()
        trajectory_client.drop_collection()
        
        logger.warning(f"All database collections dropped: '{principle_collection_name}', '{trajectory_collection_name}'")
        return StatusResponse(
            status="success", 
            message="All collections dropped successfully", 
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error dropping database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to drop database: {str(e)}")


@router.get("/data-status", response_model=dict)
async def get_data_status(
    principle_client=Depends(get_principle_client),
    trajectory_client=Depends(get_trajectory_client)
):
    try:
        def _count_rows(client, cname):
            try:
                stats = client.client.get_collection_stats(cname)
                if isinstance(stats, dict):
                    rc = stats.get('row_count') or stats.get('rowCount')
                else:
                    import json as _json
                    rc = _json.loads(stats).get('row_count')
                return int(rc) if rc is not None else "N/A"
            except Exception as e:
                logger.warning(f"Failed to get stats for {cname}: {e}. Falling back to light query.")
                try:
                    # very light probe just to ensure collection is accessible
                    res = client.client.query(collection_name=cname, filter="", limit=1)
                    return ">= 1" if res else 0
                except Exception as qe:
                    logger.error(f"Fallback probe failed for {cname}: {qe}")
                    return "N/A"

        principle_count = _count_rows(principle_client, principle_client.collection_name)
        trajectory_count = _count_rows(trajectory_client, trajectory_client.collection_name)

        return {
            "status": "success",
            "data_status": {
                "principles_count": principle_count,
                "trajectories_count": trajectory_count,
                "principles_collection_name": principle_client.collection_name if principle_client else "N/A",
                "trajectories_collection_name": trajectory_client.collection_name if trajectory_client else "N/A",
                "last_check": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting data status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data status: {str(e)}")


@router.post("/reinitialize", response_model=StatusResponse)
async def reinitialize_clients(experiment_name: Optional[str] = None):
    try:
        initialize_clients(experiment_name)
        
        message = f"VectorDB clients reinitialized successfully"
        if experiment_name:
            message += f" with experiment_name: {experiment_name}"
        
        logger.info(message)
        return StatusResponse(
            status="success",
            message=message,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error reinitializing clients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reinitialize clients: {str(e)}")
