from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

class MissingStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    DROP = "drop"

class EncodingMethod(str, Enum):
    ONEHOT = "onehot"
    LABEL = "label"
    TARGET = "target"
    KFOLD = "kfold"

class PreprocessRequest(BaseModel):
    missing_strategy: MissingStrategy
    scaling: bool = True
    encoding: EncodingMethod
    target_column: Optional[str] = None
    selected_features: Optional[Dict[str, List[str]]] = None

class TrainRequest(BaseModel):
    task_type: TaskType
    model_type: Optional[str] = None
    target_columns: Optional[Dict[str, str]] = None
    
    @validator('target_columns', pre=True)
    def parse_target_columns(cls, v):
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except:
                return {}
        return v or {}
