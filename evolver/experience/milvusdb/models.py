from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from dataclasses import dataclass


class PrincipleCreateRequest(BaseModel):
    principle_id: str
    description: str
    structure: List[Union[Dict[str, str], List[str]]] 
    principle_type: str = "guiding"
    metric_score: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    successful_trajectory_ids: List[str] = []
    failed_trajectory_ids: List[str] = []

class PrincipleUpdateRequest(BaseModel):
    principle_id: str
    description: Optional[str] = None
    structure: Optional[List[Union[Dict[str, str], List[str]]]] = None
    principle_type: Optional[str] = None
    metric_score: Optional[float] = None
    usage_count: Optional[int] = None
    success_count: Optional[int] = None
    successful_trajectory_ids: Optional[List[str]] = None
    failed_trajectory_ids: Optional[List[str]] = None

class PrincipleResult(BaseModel):
    id: str
    description: str
    structure: List[Union[Dict[str, str], List[str]]]
    principle_type: str
    metric_score: float
    usage_count: int
    success_count: int
    successful_trajectory_ids: List[str]
    failed_trajectory_ids: List[str]
    created_at: int
    updated_at: int

class PrincipleBatchGetRequest(BaseModel):
    ids: List[str]

class PrincipleBatchDeleteRequest(BaseModel):
    ids: List[str]

class PrincipleBatchUpdateRequestItem(BaseModel):
    principle_id: str
    usage_count: int
    success_count: int
    metric_score: float

class PrincipleBatchUpdateRequest(BaseModel):
    items: List[PrincipleUpdateRequest]


class TrajectoryCreateRequest(BaseModel):
    trajectory_id: str
    query: str
    log: str
    final_outcome: bool
    retrieved_principles: List[str] = []
    golden_answer: Optional[str] = ""

class TrajectoryUpdateRequest(BaseModel):
    trajectory_id: str
    query: Optional[str] = None
    log: Optional[str] = None
    final_outcome: Optional[bool] = None
    retrieved_principles: Optional[List[str]] = None
    merged_trajectory_ids: Optional[List[str]] = None
    golden_answer: Optional[str] = None

class TrajectoryResult(BaseModel):
    id: str
    query: str
    log: str
    final_outcome: bool
    retrieved_principles: List[str]
    golden_answer: str = ""
    created_at: int
    updated_at: int

class TrajectoryBatchCreateItem(BaseModel):
    trajectory_id: str
    query: str
    log: str
    final_outcome: bool
    retrieved_principles: List[str] = []
    golden_answer: Optional[str] = ""

class TrajectoryBatchCreateRequest(BaseModel):
    items: List[TrajectoryBatchCreateItem]

class TrajectoryBatchGetRequest(BaseModel):
    ids: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    principle_id: str
    similarity_score: float
    description: str
    structure: List[Union[Dict[str, str], List[str]]]
    principle_type: str
    metric_score: float
    usage_count: int
    success_count: int
    successful_trajectory_ids: List[str] = []  
    failed_trajectory_ids: List[str] = []    
    created_at: int
    updated_at: int


class DeleteRequest(BaseModel):
    id: Optional[str] = None
    before_timestamp: Optional[int] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    delete_count: Optional[int] = None

class CleanLowMetricRequest(BaseModel):
    threshold: float


class ExportRequest(BaseModel):
    collections: List[str] = ["principles", "trajectories"]
    format: str = "jsonl"
    output_root_dir: Optional[str] = None # Renamed for clarity
    include_metadata: bool = True
    experiment_name: Optional[str] = None # For server-side naming
    # The 'filename' and 'mode' fields are no longer used by the primary export logic
    # but are kept for potential backward compatibility or other uses.
    filename: Optional[str] = None
    mode: str = "overwrite"

class ExportResponse(BaseModel):
    status: str
    message: str
    files_created: List[str]
    timestamp: str
