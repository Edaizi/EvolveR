from typing import Optional
from fastapi import HTTPException, Request
from principle_client import PrincipleVectorDBClient
from trajectory_client import TrajectoryVectorDBClient
import logging

logger = logging.getLogger(__name__)

principle_client: Optional[PrincipleVectorDBClient] = None
trajectory_client: Optional[TrajectoryVectorDBClient] = None

def get_principle_client() -> PrincipleVectorDBClient:
    if principle_client is None:
        raise HTTPException(status_code=500, detail="Principle client not initialized")
    return principle_client

def get_trajectory_client() -> TrajectoryVectorDBClient:
    if trajectory_client is None:
        raise HTTPException(status_code=500, detail="Trajectory client not initialized")
    return trajectory_client

def initialize_clients(experiment_name: str = None, embedding_api_url: str = None, embedding_api_key: str = None, embedding_model: str = None):
    global principle_client, trajectory_client
    try:
        logger.info(f"Initializing VectorDB clients with experiment_name: {experiment_name}, embedding_api_url: {embedding_api_url}, embedding_model: {embedding_model}")
        
        principle_client = PrincipleVectorDBClient(experiment_name=experiment_name, api_url=embedding_api_url, api_key=embedding_api_key, model_name=embedding_model)
        
        trajectory_client = TrajectoryVectorDBClient(experiment_name=experiment_name, api_url=embedding_api_url, api_key=embedding_api_key, model_name=embedding_model)
        
        logger.info(f"VectorDB clients initialized successfully with experiment_name: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to initialize VectorDB clients: {e}")
        raise

def shutdown_clients():
    logger.info("Shutting down VectorDB clients")
