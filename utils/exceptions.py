from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger()

class MLPlatformException(Exception):
    """Base exception for ML platform"""
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DatasetError(MLPlatformException):
    """Dataset related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATASET_ERROR", details)

class ModelTrainingError(MLPlatformException):
    """Model training related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_TRAINING_ERROR", details)

class PreprocessingError(MLPlatformException):
    """Preprocessing related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PREPROCESSING_ERROR", details)

class ValidationError(MLPlatformException):
    """Validation related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)

# Exception handlers
async def mlplatform_exception_handler(request: Request, exc: MLPlatformException):
    logger.error(f"MLPlatform error: {exc.message}", error_code=exc.error_code, details=exc.details)
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url.path)
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}", path=str(request.url.path))
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )
