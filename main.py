import os
import json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog
import uvicorn

from config.settings import settings
from models.requests import PreprocessRequest, TrainRequest
from models.responses import UploadResponse, PreprocessResponse, TrainResponse
from services.file_service import FileService
from services.preprocessing_service import PreprocessingService
from services.ml_service import MLService
from utils.validators import validate_preprocessing_params

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

# Ensure upload directory exists
os.makedirs(settings.upload_directory, exist_ok=True)

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and analyze datasets"""
    try:
        results = {}
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            
            # Save file
            file_path = await file_service.save_uploaded_file(file)
            
            # Analyze dataset
            analysis = file_service.analyze_dataset(file_path)
            results[file.filename] = analysis
        
        logger.info(f"Successfully processed {len(files)} files")
        return UploadResponse(message="Files uploaded successfully", files=results)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(
    files: List[UploadFile] = File(...),
    missing_strategy: str = Form(...),
    scaling: bool = Form(...),
    encoding: str = Form(...),
    target_column: str = Form(None),
    selected_features_json: str = Form(None)
):
    """Preprocess uploaded datasets"""
    try:
        # Validate parameters
        validate_preprocessing_params(missing_strategy, encoding, target_column)
        
        # Parse selected features
        selected_features_dict = {}
        if selected_features_json:
            try:
                selected_features_dict = json.loads(selected_features_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid selected_features_json format")
        
        results = {}
        
        for file in files:
            logger.info(f"Preprocessing file: {file.filename}")
            
            # Save file temporarily
            file_path = await file_service.save_uploaded_file(file)
            
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
        return PreprocessResponse(message="Preprocessing completed", files=results)
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainResponse)
async def train_models(
    preprocessed_filenames: List[str] = Form(...),
    target_column: str = Form(None),
    task_type: str = Form(...),
    model_type: str = Form(None)
):
    """Train ML models on preprocessed data"""
    try:
        # Parse target columns
        target_columns = {}
        if target_column:
            try:
                target_columns = json.loads(target_column)
            except json.JSONDecodeError:
                # Handle single string target column
                target_columns = {}
        
        results = {}
        
        for preprocessed_file in preprocessed_filenames:
            logger.info(f"Training model on: {preprocessed_file}")
            
            if not os.path.exists(preprocessed_file):
                raise HTTPException(status_code=404, detail=f"Preprocessed file not found: {preprocessed_file}")
            
            # Get target column for this file
            filename = os.path.basename(preprocessed_file).replace("preprocessed_", "")
            file_target = target_columns.get(filename, target_column if isinstance(target_column, str) else None)
            
            # Train model
            result = ml_service.train_model(
                file_path=preprocessed_file,
                task_type=task_type,
                model_type=model_type,
                target_column=file_target
            )
            
            results[preprocessed_file] = result
        
        logger.info(f"Successfully trained models on {len(preprocessed_filenames)} files")
        return TrainResponse(message="Training completed", results=results)
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-model/{filename}")
async def download_model(filename: str):
    """Download trained model file"""
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-preprocessed/{filename}")
async def download_preprocessed(filename: str):
    """Download preprocessed data file"""
    try:
        preprocessed_path = Path(settings.upload_directory) / f"preprocessed_{filename}"
        
        if not preprocessed_path.exists():
            raise HTTPException(status_code=404, detail="Preprocessed file not found")
        
        return FileResponse(
            path=preprocessed_path,
            filename=f"preprocessed_{filename}",
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Preprocessed file download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "No-Code ML Platform API is running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
