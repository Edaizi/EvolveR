from typing import List
from fastapi import APIRouter, HTTPException, Depends
import json
import logging

from models import SearchRequest, SearchResult
from dependencies import get_principle_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

@router.post("/", response_model=List[SearchResult])
async def search_principles(
    request: SearchRequest,
    client=Depends(get_principle_client)
):
    try:
        results = client.search_principles(request.query, request.top_k)
        
        search_results = []
        for principle_id, similarity_score, entity_data in results:
            try:
                structure = json.loads(entity_data.get('structure', '[]'))
            except json.JSONDecodeError:
                structure = []
            try:
                successful_trajectory_ids = json.loads(entity_data.get('successful_trajectory_ids', '[]'))
            except json.JSONDecodeError:
                successful_trajectory_ids = []
            try:
                failed_trajectory_ids = json.loads(entity_data.get('failed_trajectory_ids', '[]'))
            except json.JSONDecodeError:
                failed_trajectory_ids = []
            
            search_results.append(SearchResult(
                principle_id=principle_id,
                similarity_score=similarity_score,
                description=entity_data.get('description', ''),
                structure=structure,
                principle_type=entity_data.get('principle_type', ''),
                metric_score=entity_data.get('metric_score', 0.0),
                usage_count=entity_data.get('usage_count', 0),
                success_count=entity_data.get('success_count', 0),
                successful_trajectory_ids=successful_trajectory_ids,
                failed_trajectory_ids=failed_trajectory_ids,
                created_at=entity_data.get('created_at', 0),
                updated_at=entity_data.get('updated_at', 0)
            ))
        
        logger.info(f"Search completed: query='{request.query}', results={len(search_results)}")
        return search_results
        
    except Exception as e:
        logger.error(f"Error searching principles: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
