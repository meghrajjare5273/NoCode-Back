import os
import hashlib
from pathlib import Path
from typing import List, Set
from fastapi import UploadFile, HTTPException
from config.settings import settings


# Allowed file extensions (now relying on extension validation only)
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.json'}

# Content type headers mapping (alternative to magic)
CONTENT_TYPE_MAPPING = {
    '.csv': ['text/csv', 'application/csv', 'text/plain'],
    '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    '.json': ['application/json', 'text/json']
}

# Malicious file signatures to block
MALICIOUS_SIGNATURES = {
    b'\x4d\x5a',  # MZ (Windows executable)
    b'\x50\x4b\x03\x04',  # ZIP file header (suspicious for CSV)
    b'\x89\x50\x4e\x47',  # PNG (for CSV files, this would be suspicious)
}


class EnhancedFileValidator:
    def __init__(self):
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.uploaded_hashes: Set[str] = set()  # Track duplicate uploads
    
    async def validate_file(self, file: UploadFile) -> bool:
        """Comprehensive file validation"""
        
        # Basic filename validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_file_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_extensions}"
            )
        
        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Check file size
        if len(content) > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Check for empty files
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file not allowed")
        
        # Validate content type from request headers (alternative to magic)
        if hasattr(file, 'content_type') and file.content_type:
            allowed_content_types = CONTENT_TYPE_MAPPING.get(file_ext, [])
            if allowed_content_types and file.content_type not in allowed_content_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Content type mismatch. Expected: {allowed_content_types}, Got: {file.content_type}"
                )
        
        # Basic content validation by file signature
        await self._validate_file_signature(content, file_ext)
        
        # Check for malicious file signatures
        for signature in MALICIOUS_SIGNATURES:
            if content.startswith(signature):
                raise HTTPException(
                    status_code=400,
                    detail="Potentially malicious file detected"
                )
        
        # Check for duplicate uploads (based on file hash)
        file_hash = hashlib.md5(content).hexdigest()
        if file_hash in self.uploaded_hashes:
            raise HTTPException(
                status_code=400,
                detail="Duplicate file detected"
            )
        self.uploaded_hashes.add(file_hash)
        
        # Validate CSV structure if it's a CSV file
        if file_ext == '.csv':
            await self._validate_csv_structure(content)
        
        return True
    
    async def _validate_file_signature(self, content: bytes, file_ext: str):
        """Basic file signature validation without magic library"""
        if file_ext == '.csv':
            # CSV files should start with printable characters
            if not content[:100].decode('utf-8', errors='ignore').isprintable():
                raise HTTPException(
                    status_code=400,
                    detail="Invalid CSV file format"
                )
        elif file_ext == '.xlsx':
            # XLSX files are ZIP archives, should start with PK
            if not content.startswith(b'PK'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid XLSX file format"
                )
        elif file_ext == '.json':
            # JSON files should be valid JSON
            try:
                import json
                json.loads(content.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON file format"
                )
    
    async def _validate_csv_structure(self, content: bytes):
        """Validate CSV file structure"""
        try:
            import pandas as pd
            import io
            
            # Try to read as CSV
            df = pd.read_csv(io.BytesIO(content))
            
            # Check minimum requirements
            if len(df.columns) == 0:
                raise HTTPException(status_code=400, detail="CSV file has no columns")
            
            if len(df) == 0:
                raise HTTPException(status_code=400, detail="CSV file has no data rows")
            
            # Check for reasonable number of columns (prevent memory issues)
            if len(df.columns) > 1000:
                raise HTTPException(status_code=400, detail="Too many columns (max: 1000)")
            
            # Check for reasonable number of rows
            if len(df) > 1000000:
                raise HTTPException(status_code=400, detail="Too many rows (max: 1,000,000)")
                
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV validation failed: {str(e)}")


# Global validator instance
file_validator = EnhancedFileValidator()


def validate_preprocessing_params(
    missing_strategy: str,
    encoding: str,
    target_column: str = None
) -> bool:
    """Enhanced parameter validation with better error messages"""
    
    allowed_missing = ["mean", "median", "mode", "drop"]
    if missing_strategy not in allowed_missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid missing strategy",
                "allowed_values": allowed_missing,
                "received": missing_strategy
            }
        )
    
    allowed_encoding = ["onehot", "label", "target", "kfold"]
    if encoding not in allowed_encoding:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid encoding method",
                "allowed_values": allowed_encoding,
                "received": encoding
            }
        )
    
    if encoding in ["target", "kfold"] and not target_column:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Target column required for {encoding} encoding",
                "required_field": "target_column"
            }
        )
    
    return True
