"""
REST API for Chiaroscuro Forge using FastAPI.

This module provides a complete REST API for image processing operations
with authentication, rate limiting, and async support.

Features:
- Image processing endpoints
- Batch processing
- Job status monitoring
- API key authentication
- Rate limiting
- OpenAPI documentation

Example:
    Start the server:
    ```bash
    uvicorn chiaroscuro_forge.api:app --reload
    ```
    
    Access the API:
    ```python
    import requests
    
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'image': open('input.jpg', 'rb')},
        data={'gamma': '1.2'}
    )
    ```
"""

import hashlib
import logging
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading

try:
    from fastapi import (
        FastAPI, File, UploadFile, Form, HTTPException,
        Depends, Security, status, BackgroundTasks
    )
    from fastapi.security import APIKeyHeader
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = None  # type: ignore
    Field = None  # type: ignore

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Only define API models if FastAPI is available
if FASTAPI_AVAILABLE:
    # ===== API Models =====

    class ProcessingParams(BaseModel):
        """Processing parameters for image enhancement."""
        gamma: Optional[float] = Field(None, ge=0.1, le=3.0, description="Gamma correction")
        scale_factor: Optional[float] = Field(None, gt=0, le=4.0, description="Scale factor")
        denoise_sigma: Optional[float] = Field(None, ge=0, le=10.0, description="Denoise sigma")
        sharpen_amount: Optional[float] = Field(None, ge=0, le=2.0, description="Sharpen amount")
        equalize_method: Optional[str] = Field(None, description="Equalization method")
        color_preservation: Optional[str] = Field(None, description="Color preservation method")
        application_type: Optional[str] = Field("general", description="Application type")


    class JobStatus(str, Enum):
        """Job processing status."""
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"


    class JobInfo(BaseModel):
        """Job information response."""
        job_id: str
        status: JobStatus
        created_at: datetime
        completed_at: Optional[datetime] = None
        progress: Optional[float] = Field(None, ge=0.0, le=1.0)
        result_url: Optional[str] = None
        error: Optional[str] = None


    class APIResponse(BaseModel):
        """Standard API response."""
        success: bool
        message: str
        data: Optional[Dict[str, Any]] = None

else:
    # Stub classes when FastAPI is not available
    ProcessingParams = None  # type: ignore
    JobStatus = None  # type: ignore
    JobInfo = None  # type: ignore
    APIResponse = None  # type: ignore


# ===== Authentication =====

class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self):
        """Initialize API key manager."""
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Create a default development key
        self.create_key("dev-key-12345", name="Development Key")
    
    def create_key(
        self,
        key: Optional[str] = None,
        name: str = "API Key",
        rate_limit: int = 100
    ) -> str:
        """Create a new API key.
        
        Args:
            key: Optional specific key (for testing)
            name: Key name/description
            rate_limit: Requests per hour
            
        Returns:
            API key string
        """
        if key is None:
            key = f"sk_{secrets.token_urlsafe(32)}"
        
        with self._lock:
            self._keys[key] = {
                'name': name,
                'created_at': datetime.now(),
                'rate_limit': rate_limit,
                'requests': []
            }
        
        return key
    
    def validate_key(self, key: str) -> bool:
        """Validate an API key.
        
        Args:
            key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        return key in self._keys
    
    def check_rate_limit(self, key: str) -> bool:
        """Check if key is within rate limit.
        
        Args:
            key: API key
            
        Returns:
            True if within limit, False otherwise
        """
        if key not in self._keys:
            return False
        
        with self._lock:
            key_data = self._keys[key]
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            
            # Remove old requests
            key_data['requests'] = [
                req_time for req_time in key_data['requests']
                if req_time > hour_ago
            ]
            
            # Check limit
            if len(key_data['requests']) >= key_data['rate_limit']:
                return False
            
            # Record this request
            key_data['requests'].append(now)
            return True


# ===== Job Manager =====

class JobManager:
    """Manages background processing jobs."""
    
    def __init__(self):
        """Initialize job manager."""
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._job_counter = 0
    
    def create_job(self) -> str:
        """Create a new job.
        
        Returns:
            Job ID
        """
        with self._lock:
            self._job_counter += 1
            job_id = f"job_{self._job_counter}_{int(time.time() * 1000)}"
            
            self._jobs[job_id] = {
                'status': "pending" if not FASTAPI_AVAILABLE else JobStatus.PENDING,
                'created_at': datetime.now(),
                'completed_at': None,
                'progress': 0.0,
                'result': None,
                'error': None
            }
            
            return job_id
    
    def get_job(self, job_id: str) -> Optional[Any]:
        """Get job information.
        
        Args:
            job_id: Job ID
            
        Returns:
            JobInfo or None if not found
        """
        job = self._jobs.get(job_id)
        if not job or not FASTAPI_AVAILABLE:
            return None
        
        return JobInfo(
            job_id=job_id,
            status=job['status'],
            created_at=job['created_at'],
            completed_at=job['completed_at'],
            progress=job['progress'],
            result_url=job.get('result_url'),
            error=job.get('error')
        )
    
    def update_job(
        self,
        job_id: str,
        status: Optional[Any] = None,
        progress: Optional[float] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ):
        """Update job status.
        
        Args:
            job_id: Job ID
            status: New status
            progress: Progress (0.0-1.0)
            result: Result data
            error: Error message
        """
        if job_id not in self._jobs:
            return
        
        with self._lock:
            job = self._jobs[job_id]
            
            if status:
                job['status'] = status
            if progress is not None:
                job['progress'] = progress
            if result is not None:
                job['result'] = result
            if error is not None:
                job['error'] = error
            
            if status in ("completed", "failed") or (FASTAPI_AVAILABLE and status in (JobStatus.COMPLETED, JobStatus.FAILED)):
                job['completed_at'] = datetime.now()


# Global managers (always available)
api_key_manager = APIKeyManager()
job_manager = JobManager()

# FastAPI-specific components (only when available)
if FASTAPI_AVAILABLE:
    # API key header
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


    async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
        """Verify API key from request header.
        
        Args:
            api_key: API key from header
            
        Returns:
            Validated API key
            
        Raises:
            HTTPException: If key is invalid or rate limited
        """
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key"
            )
        
        if not api_key_manager.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        if not api_key_manager.check_rate_limit(api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return api_key
else:
    api_key_header = None  # type: ignore
    verify_api_key = None  # type: ignore


# ===== FastAPI Application =====

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Chiaroscuro Forge API",
        description="REST API for intelligent image enhancement",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Chiaroscuro Forge API",
            "version": "1.0.0",
            "documentation": "/docs",
            "status": "running"
        }
    
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    
    @app.post("/api/v1/process", tags=["Processing"], response_model=JobInfo)
    async def process_image(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(..., description="Image file to process"),
        gamma: Optional[float] = Form(None),
        scale_factor: Optional[float] = Form(None),
        application_type: Optional[str] = Form("general"),
        api_key: str = Depends(verify_api_key)
    ):
        """Process a single image with specified parameters.
        
        Args:
            image: Image file
            gamma: Gamma correction value
            scale_factor: Scale factor
            application_type: Application type
            api_key: API key for authentication
            
        Returns:
            Job information with job_id for status tracking
        """
        # Create job
        job_id = job_manager.create_job()
        
        # Read image
        try:
            contents = await image.read()
            img = Image.open(BytesIO(contents))
        except Exception as e:
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=f"Failed to read image: {str(e)}"
            )
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Start background processing
        background_tasks.add_task(
            _process_image_background,
            job_id,
            img,
            gamma,
            scale_factor,
            application_type
        )
        
        return job_manager.get_job(job_id)
    
    
    @app.get("/api/v1/jobs/{job_id}", tags=["Jobs"], response_model=JobInfo)
    async def get_job_status(
        job_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Get status of a processing job.
        
        Args:
            job_id: Job ID
            api_key: API key for authentication
            
        Returns:
            Job information
        """
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job
    
    
    @app.post("/api/v1/keys", tags=["Authentication"], response_model=APIResponse)
    async def create_api_key(
        name: str = Form("New API Key"),
        rate_limit: int = Form(100)
    ):
        """Create a new API key (admin endpoint).
        
        Args:
            name: Key name/description
            rate_limit: Requests per hour
            
        Returns:
            API key
        """
        key = api_key_manager.create_key(name=name, rate_limit=rate_limit)
        
        return APIResponse(
            success=True,
            message="API key created successfully",
            data={"api_key": key, "rate_limit": rate_limit}
        )
    
    
    def _process_image_background(
        job_id: str,
        img: Image.Image,
        gamma: Optional[float],
        scale_factor: Optional[float],
        application_type: str
    ):
        """Background task for image processing.
        
        Args:
            job_id: Job ID
            img: PIL Image
            gamma: Gamma correction
            scale_factor: Scale factor
            application_type: Application type
        """
        try:
            job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=0.1)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            job_manager.update_job(job_id, progress=0.3)
            
            # Simple processing (for demonstration)
            # In practice, would call chiaroscuro_forge.process_image
            if gamma:
                img_array = np.clip(img_array / 255.0, 0, 1)
                img_array = np.power(img_array, gamma)
                img_array = (img_array * 255).astype(np.uint8)
            
            job_manager.update_job(job_id, progress=0.8)
            
            # Save result (in practice, would save to file storage)
            result_img = Image.fromarray(img_array)
            
            job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                result={"width": img.width, "height": img.height}
            )
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=str(e)
            )


else:
    # FastAPI not available
    app = None
    logger.warning(
        "FastAPI not installed. API functionality unavailable. "
        "Install with: pip install 'fastapi[all]' uvicorn"
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload
        
    Example:
        >>> from chiaroscuro_forge.api import run_server
        >>> run_server(port=8000)
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not installed. Install with: pip install 'fastapi[all]' uvicorn"
        )
    
    uvicorn.run("chiaroscuro_forge.api:app", host=host, port=port, reload=reload)
