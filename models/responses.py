from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class DatasetSummary(BaseModel):
    columns: List[str]
    rows: int
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    unique_values: Dict[str, int]
    sample_rows: int
    file_size_mb: float

class DatasetInsights(BaseModel):
    summary: DatasetSummary
    insights: List[str]
    suggested_task_type: str
    suggested_target_column: Optional[str]
    suggested_missing_strategy: str

class ModelMetrics(BaseModel):
    task_type: str
    model_type: str
    results: Dict[str, float]
    feature_importance: Optional[List[List[Any]]] = None
    training_time: Optional[float] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    files: Dict[str, DatasetInsights]

class PreprocessResponse(BaseModel):
    message: str
    files: Dict[str, Dict[str, str]]

class TrainResponse(BaseModel):
    message: str
    results: Dict[str, ModelMetrics]
