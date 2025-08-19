import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # File Upload Configuration
    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = [".csv", ".xlsx", ".json"]
    upload_directory: str = "uploads"
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Processing Configuration
    max_chunk_size: int = 10000
    default_test_size: float = 0.2
    random_state: int = 42
    
    class Config:
        env_file = ".env"

settings = Settings()

