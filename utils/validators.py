import os
from fastapi import UploadFile, HTTPException
from config.settings import settings

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    
    # Check file extension
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in settings.allowed_file_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_extensions}"
        )
    
    # Note: File size validation would need to be done after reading the file
    # This is a basic validation
    
    return True

def validate_preprocessing_params(
    missing_strategy: str,
    encoding: str,
    target_column: str = None
) -> bool:
    """Validate preprocessing parameters"""
    
    allowed_missing = ["mean", "median", "mode", "drop"]
    if missing_strategy not in allowed_missing:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid missing strategy. Allowed: {allowed_missing}"
        )
    
    allowed_encoding = ["onehot", "label", "target", "kfold"]
    if encoding not in allowed_encoding:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid encoding method. Allowed: {allowed_encoding}"
        )
    
    # Target encoding methods require target column
    if encoding in ["target", "kfold"] and not target_column:
        raise HTTPException(
            status_code=400,
            detail=f"Target column required for {encoding} encoding"
        )
    
    return True
