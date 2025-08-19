import os
import json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
import uvicorn

from config.settings import settings
from models.requests import PreprocessRequest, TrainRequest
from models.responses import UploadResponse, PreprocessResponse, TrainResponse
from services.file_service import FileService
from services.preprocessing_service import PreprocessingService
from services.ml_service import MLService
from services.ml_service import AsyncMLService  # New async service
from utils.validators import file_validator, validate_preprocessing_params
# from auth.security import get_current_user, User, limiter
from utils.exceptions import (
    MLPlatformException, mlplatform_exception_handler, 
    general_exception_handler, DatasetError, ModelTrainingError, ValidationError, PreprocessingError
)

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="No-Code ML Platform API",
    description="Backend API for training ML models without code",
    version="1.0.0"
)

# Add rate limiter
# app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add custom exception handlers
app.add_exception_handler(MLPlatformException, mlplatform_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
file_service = FileService()
preprocessing_service = PreprocessingService()
ml_service = MLService()
async_ml_service = AsyncMLService()

# Ensure upload directory exists
os.makedirs(settings.upload_directory, exist_ok=True)

@app.post("/upload", response_model=UploadResponse)
# @limiter.limit("10/minute")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    # current_user: User = Depends(get_current_user)
):
    """Upload and analyze datasets with enhanced security"""
    try:
        results = {}
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            # user_id=current_user.user_id)
            
            # Enhanced file validation
            await file_validator.validate_file(file)
            
            # Save file
            file_path = await file_service.save_uploaded_file(file)
                                                            #    current_user.user_id)
            
            # Analyze dataset
            analysis = file_service.analyze_dataset(file_path)
            results[file.filename] = analysis
        
        logger.info(f"Successfully processed {len(files)} files")
                    # user_id=current_user.user_id)
        return UploadResponse(message="Files uploaded successfully", files=results)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", )
        # user_id=current_user.user_id)
        if isinstance(e, HTTPException):
            raise e
        raise DatasetError(f"Failed to process uploaded files: {str(e)}")

@app.post("/preprocess", response_model=PreprocessResponse)
# @limiter.limit("5/minute")
async def preprocess_data(
    request: Request,
    files: List[UploadFile] = File(...),
    missing_strategy: str = Form(...),
    scaling: bool = Form(...),
    encoding: str = Form(...),
    target_column: str = Form(None),
    selected_features_json: str = Form(None),
    # current_user: User = Depends(get_current_user)
):
    """Preprocess uploaded datasets with enhanced validation"""
    try:
        # Validate parameters
        validate_preprocessing_params(missing_strategy, encoding, target_column)
        
        # Parse selected features
        selected_features_dict = {}
        if selected_features_json:
            try:
                selected_features_dict = json.loads(selected_features_json)
            except json.JSONDecodeError:
                raise ValidationError("Invalid selected_features_json format")
        
        results = {}
        
        for file in files:
            logger.info(f"Preprocessing file: {file.filename}")
            # user_id=current_user.user_id)
            
            # Validate file
            await file_validator.validate_file(file)
            
            # Save file temporarily
            file_path = await file_service.save_uploaded_file(file)
            # current_user.user_id)
            
            # Get selected features for this file
            selected_features = selected_features_dict.get(file.filename, None)
            
            # Preprocess
            preprocessed_path = preprocessing_service.preprocess_dataset(
                file_path=file_path,
                missing_strategy=missing_strategy,
                scaling=scaling,
                encoding=encoding,
                target_column=target_column,
                selected_features=selected_features
            )
            
            results[file.filename] = {"preprocessed_file": preprocessed_path}
        
        logger.info(f"Successfully preprocessed {len(files)} files")
        # user_id=current_user.user_id)
        return PreprocessResponse(message="Preprocessing completed", files=results)
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        # user_id=current_user.user_id)
        if isinstance(e, (HTTPException, MLPlatformException)):
            raise e
        raise PreprocessingError(f"Preprocessing failed: {str(e)}")

@app.post("/train", response_model=TrainResponse)
# @limiter.limit("3/minute")
async def train_models(
    request: Request,
    preprocessed_filenames: List[str] = Form(...),
    target_column: str = Form(None),
    task_type: str = Form(...),
    model_type: str = Form(None),
    # current_user: User = Depends(get_current_user)
):
    """Train ML models asynchronously with enhanced error handling"""
    try:
        # Parse target columns
        target_columns = {}
        if target_column:
            try:
                target_columns = json.loads(target_column)
            except json.JSONDecodeError:
                target_columns = {}
        
        results = {}
        
        for preprocessed_file in preprocessed_filenames:
            logger.info(f"Training model on: {preprocessed_file}")
                        #  user_id=current_user.user_id)
            
            if not os.path.exists(preprocessed_file):
                raise ModelTrainingError(f"Preprocessed file not found: {preprocessed_file}")
            
            # Get target column for this file
            filename = os.path.basename(preprocessed_file).replace("preprocessed_", "")
            file_target = target_columns.get(filename, target_column if isinstance(target_column, str) else None)
            
            # Train model asynchronously
            result = await async_ml_service.train_model_async(
                file_path=preprocessed_file,
                task_type=task_type,
                model_type=model_type,
                target_column=file_target
            )
            
            results[preprocessed_file] = result
        
        logger.info(f"Successfully trained models on {len(preprocessed_filenames)} files")
        # user_id=current_user.user_id)
        return TrainResponse(message="Training completed")
    # results=results)
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        # user_id=current_user.user_id)
        if isinstance(e, (HTTPException, MLPlatformException)):
            raise e
        raise ModelTrainingError(f"Model training failed: {str(e)}")

# New prediction endpoint
@app.post("/models/{model_id}/predict")
# @limiter.limit("20/minute")
async def predict(
    request: Request,
    model_id: str,
    data: Dict[str, Any],
    # current_user: User = Depends(get_current_user)
):
    """Make predictions using trained model"""
    try:
        model_path = Path(settings.upload_directory) / f"trained_model_{model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load model and make prediction
        result = await async_ml_service.predict_async(model_path, data)
        
        logger.info(f"Prediction made for model {model_id}")
        # user_id=current_user.user_id)
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # user_id=current_user.user_id)
        if isinstance(e, HTTPException):
            raise e
        raise ModelTrainingError(f"Prediction failed: {str(e)}")

@app.get("/download-model/{filename}")
async def download_model(
    filename: str,
    # current_user: User = Depends(get_current_user)
):
    """Download trained model file with user authorization"""
    try:
        model_filename = f"trained_model_{filename.replace('.csv', '')}.pkl"
        model_path = Path(settings.upload_directory) / model_filename
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        return FileResponse(
            path=model_path,
            filename=model_filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Model download error: {str(e)}")
        # user_id=current_user.user_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "No-Code ML Platform API is running",
        "version": "1.0.0",
        "timestamp": "2025-08-19T10:17:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
