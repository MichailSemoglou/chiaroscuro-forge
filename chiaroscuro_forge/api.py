"""
REST API for Chiaroscuro Forge using FastAPI.

This module provides a complete REST API for image processing operations
with authentication, rate limiting, and async support.

Features:
- Image processing endpoints
- Batch processing
- Job status monitoring with TTL cleanup
- API key authentication with key management
- Rate limiting
- Health and observability endpoints
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

import logging
import os
import secrets
import tempfile
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage import img_as_ubyte as _ubyte
from skimage import io as skio

from .config import ProcessingConfig
from .constants import MAX_DIMENSION, MAX_FILE_SIZE_MB, MAX_IMAGE_PIXELS
from .optional import optional_import
from .processing import process_image as _process

logger = logging.getLogger(__name__)

FASTAPI_AVAILABLE = (
    optional_import("fastapi", "REST API", "pip install 'fastapi[all]' uvicorn") is not None
)

if FASTAPI_AVAILABLE:
    import uvicorn
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        File,
        Form,
        HTTPException,
        Security,
        UploadFile,
        status,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field

    # ===== API Models =====

    class ProcessingParams(BaseModel):
        """Processing parameters for image enhancement."""

        application_type: Optional[str] = Field("general", description="Application type")
        scale_factor: Optional[float] = Field(None, gt=0, le=4.0, description="Scale factor")
        denoise_type: Optional[str] = Field(None, description="Denoising method")
        denoise_sigma: Optional[float] = Field(None, ge=0, le=10.0, description="Denoise sigma")
        sharpen: Optional[bool] = Field(None, description="Apply sharpening")
        sharpen_amount: Optional[float] = Field(None, ge=0, le=2.0, description="Sharpen amount")
        equalize: Optional[bool] = Field(None, description="Apply equalization")
        equalize_method: Optional[str] = Field(None, description="Equalization method")
        clip_limit: Optional[float] = Field(None, gt=0, description="CLAHE clip limit")
        clip_limit_kernel_size: Optional[int] = Field(None, gt=0, description="CLAHE kernel size")
        contrast_stretch_percentiles: Optional[Tuple[float, float]] = Field(
            None, description="Contrast stretch percentiles"
        )
        gamma: Optional[float] = Field(None, ge=0.1, le=3.0, description="Gamma correction")
        color_preservation: Optional[str] = Field(None, description="Color preservation method")
        color_preservation_strength: Optional[float] = Field(
            None, ge=0.0, le=1.0, description="Color preservation strength"
        )
        calculate_advanced_metrics: Optional[bool] = Field(
            None, description="Enable advanced metrics"
        )
        use_tiling: Optional[bool] = Field(None, description="Force tiling on/off")
        tile_size: Optional[int] = Field(None, description="Tile size")
        tile_overlap: Optional[int] = Field(None, description="Tile overlap")

    class JobStatus(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"

    class JobInfo(BaseModel):
        job_id: str
        status: JobStatus
        created_at: datetime
        completed_at: Optional[datetime] = None
        progress: Optional[float] = Field(None, ge=0.0, le=1.0)
        result: Optional[Dict[str, Any]] = None
        result_url: Optional[str] = None
        error: Optional[str] = None

    class APIResponse(BaseModel):
        success: bool
        message: str
        data: Optional[Dict[str, Any]] = None

    class HealthInfo(BaseModel):
        status: str
        timestamp: datetime
        jobs: Dict[str, int]
        uptime_seconds: float

    class HealthResponse(APIResponse):
        data: Optional[HealthInfo] = None

else:
    ProcessingParams = None  # type: ignore
    JobStatus = None  # type: ignore
    JobInfo = None  # type: ignore
    APIResponse = None  # type: ignore

    class HealthInfo:
        def __init__(
            self,
            status: str,
            timestamp: datetime,
            jobs: Dict[str, int],
            uptime_seconds: float,
        ):
            self.status = status
            self.timestamp = timestamp
            self.jobs = jobs
            self.uptime_seconds = uptime_seconds

    class HealthResponse:
        def __init__(self, success: bool, message: str, data: Optional[HealthInfo] = None):
            self.success = success
            self.message = message
            self.data = data


JOB_TTL_HOURS = 24
"""Hours after which completed/failed jobs are eligible for cleanup."""


def _build_processing_config_from_params(
    processing_params: Optional[Any],
    default_application_type: str = "general",
) -> ProcessingConfig:
    """Convert API request params into a ProcessingConfig instance."""
    config = ProcessingConfig(application_type=default_application_type)
    if processing_params is None:
        return config

    overrides: Dict[str, Any] = {}
    values = processing_params.dict(exclude_none=True)
    for key, value in values.items():
        if key == "gamma":
            overrides["gamma_correction"] = value
        elif key == "calculate_advanced_metrics":
            overrides["calculate_advanced_metrics"] = value
        else:
            overrides[key] = value

    application_type = overrides.pop("application_type", default_application_type)
    return ProcessingConfig(application_type=application_type).merge(overrides)


# ===== Authentication =====


class APIKeyManager:
    """Manages API keys for authentication.

    Supports key creation, validation, rate limiting, and revocation.
    """

    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        logger.info("APIKeyManager initialized (no default keys)")

    def create_key(
        self,
        key: Optional[str] = None,
        name: str = "API Key",
        rate_limit: int = 100,
    ) -> str:
        if key is None:
            key = f"sk_{secrets.token_urlsafe(32)}"
        with self._lock:
            self._keys[key] = {
                "name": name,
                "created_at": datetime.now(),
                "rate_limit": rate_limit,
                "requests": [],
            }
        logger.info("API key created: %s (rate_limit=%d/h)", name, rate_limit)
        return key

    def revoke_key(self, key: str) -> bool:
        with self._lock:
            if key in self._keys:
                del self._keys[key]
                logger.info("API key revoked")
                return True
            return False

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": v["name"],
                    "created_at": v["created_at"].isoformat(),
                    "rate_limit": v["rate_limit"],
                    "request_count_last_hour": len(
                        [r for r in v["requests"] if r > datetime.now() - timedelta(hours=1)]
                    ),
                }
                for v in self._keys.values()
            ]

    def validate_key(self, key: str) -> bool:
        return key in self._keys

    def check_rate_limit(self, key: str) -> bool:
        with self._lock:
            if key not in self._keys:
                return False
            key_data = self._keys[key]
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            key_data["requests"] = [
                req_time for req_time in key_data["requests"] if req_time > hour_ago
            ]
            if len(key_data["requests"]) >= key_data["rate_limit"]:
                return False
            key_data["requests"].append(now)
            return True


# ===== Job Manager =====


class JobManager:
    """Manages background processing jobs with TTL-based cleanup."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._job_counter = 0
        self._start_time = datetime.now()

    def create_job(self) -> str:
        with self._lock:
            self._job_counter += 1
            job_id = f"job_{self._job_counter}_{int(time.time() * 1000)}"
            self._jobs[job_id] = {
                "status": JobStatus.PENDING if FASTAPI_AVAILABLE else "pending",
                "created_at": datetime.now(),
                "completed_at": None,
                "progress": 0.0,
                "result": None,
                "error": None,
            }
            logger.debug("Job created: %s", job_id)
            return job_id

    def get_job(self, job_id: str) -> Optional[Any]:
        job = self._jobs.get(job_id)
        if not job or not FASTAPI_AVAILABLE:
            return None
        return JobInfo(
            job_id=job_id,
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            progress=job["progress"],
            result=job.get("result"),
            result_url=job.get("result_url"),
            error=job.get("error"),
        )

    def update_job(
        self,
        job_id: str,
        status: Optional[Any] = None,
        progress: Optional[float] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ):
        with self._lock:
            if job_id not in self._jobs:
                logger.warning("Job %s not found for update", job_id)
                return
            job = self._jobs[job_id]
            if status is not None:
                job["status"] = status
            if progress is not None:
                job["progress"] = progress
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            if status in ("completed", "failed") or (
                FASTAPI_AVAILABLE and status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ):
                job["completed_at"] = datetime.now()
                logger.info("Job %s %s", job_id, status)

    def cleanup_expired_jobs(self) -> int:
        cutoff = datetime.now() - timedelta(hours=JOB_TTL_HOURS)
        removed = 0
        with self._lock:
            stale = [
                jid
                for jid, j in self._jobs.items()
                if j["completed_at"] is not None
                and j["completed_at"] < cutoff
                and j["status"] in ("completed", "failed")
            ]
            for jid in stale:
                del self._jobs[jid]
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired jobs", removed)
        return removed

    def job_counts(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for job in self._jobs.values():
                s = job["status"]
                if hasattr(s, "value"):
                    s = s.value
                counts[s] = counts.get(s, 0) + 1
            return counts

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self._start_time).total_seconds()


# Global managers (always available)
api_key_manager = APIKeyManager()
job_manager = JobManager()

# FastAPI-specific components
if FASTAPI_AVAILABLE:
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def _bootstrap_is_authorized(api_key: str) -> bool:
        if not api_key:
            return False
        env_key = os.environ.get("CHIAROSCURO_API_KEY")
        if not env_key:
            return False
        allow_bootstrap = os.environ.get("CHIAROSCURO_ALLOW_BOOTSTRAP", "false").lower() == "true"
        if not allow_bootstrap:
            return False
        return secrets.compare_digest(api_key, env_key)

    async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
            )
        if not api_key_manager.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        if not api_key_manager.check_rate_limit(api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
            )
        return api_key

    async def verify_api_key_or_bootstrap(
        api_key: str = Security(api_key_header),
    ) -> str:
        if api_key_manager.validate_key(api_key):
            if not api_key_manager.check_rate_limit(api_key):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded.",
                )
            return api_key

        existing = api_key_manager.list_keys()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Provide X-API-Key header.",
            )

        if _bootstrap_is_authorized(api_key):
            return api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bootstrap not authorized. Set CHIAROSCURO_API_KEY and CHIAROSCURO_ALLOW_BOOTSTRAP=true, or provide a valid API key.",
        )

else:
    api_key_header = None  # type: ignore
    verify_api_key = None  # type: ignore
    verify_api_key_or_bootstrap = None  # type: ignore


# ===== FastAPI Application =====

if FASTAPI_AVAILABLE:
    from chiaroscuro_forge import __version__ as _pkg_version

    app = FastAPI(
        title="Chiaroscuro Forge API",
        description="REST API for intelligent image enhancement",
        version=_pkg_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    @app.on_event("startup")
    async def _on_startup():
        logger.info("API server starting up")
        job_manager.cleanup_expired_jobs()
        env_key = os.environ.get("CHIAROSCURO_API_KEY")
        if env_key and not api_key_manager.list_keys():
            api_key_manager.create_key(env_key, name="Environment initial key")
            logger.info("Initial API key provisioned from environment")

    @app.on_event("shutdown")
    async def _on_shutdown():
        logger.info("API server shutting down")
        job_manager.cleanup_expired_jobs()

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "Chiaroscuro Forge API",
            "version": _pkg_version,
            "documentation": "/docs",
            "status": "running",
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/api/v1/health", tags=["Health"], response_model=HealthResponse)
    async def detailed_health(api_key: str = Depends(verify_api_key)):
        return HealthResponse(
            success=True,
            message="Health check",
            data=HealthInfo(
                status="healthy",
                timestamp=datetime.now(),
                jobs=job_manager.job_counts(),
                uptime_seconds=job_manager.uptime_seconds,
            ),
        )

    @app.post("/api/v1/process", tags=["Processing"], response_model=JobInfo)
    async def process_image_endpoint(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(..., description="Image file to process"),
        gamma: Optional[float] = Form(None),
        scale_factor: Optional[float] = Form(None),
        application_type: Optional[str] = Form("general"),
        denoise_type: Optional[str] = Form(None),
        denoise_sigma: Optional[float] = Form(None),
        sharpen: Optional[bool] = Form(None),
        sharpen_amount: Optional[float] = Form(None),
        equalize: Optional[bool] = Form(None),
        equalize_method: Optional[str] = Form(None),
        clip_limit: Optional[float] = Form(None),
        clip_limit_kernel_size: Optional[int] = Form(None),
        contrast_stretch_percentiles: Optional[str] = Form(None),
        color_preservation: Optional[str] = Form(None),
        color_preservation_strength: Optional[float] = Form(None),
        calculate_advanced_metrics: Optional[bool] = Form(None),
        use_tiling: Optional[bool] = Form(None),
        tile_size: Optional[int] = Form(None),
        tile_overlap: Optional[int] = Form(None),
        api_key: str = Depends(verify_api_key),
    ):
        job_id = job_manager.create_job()
        try:
            max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            content_length = image.headers.get("content-length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError(f"Image file too large: {content_length} bytes (max {max_bytes})")

            chunks = []
            total = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = await image.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Image file too large: exceeds {max_bytes} bytes")
                chunks.append(chunk)
            contents = b"".join(chunks)

            img = Image.open(BytesIO(contents))
            img.load()
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS or width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ValueError(f"Image dimensions too large: {width}x{height} pixels")
        except Exception as exc:
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=type(exc).__name__,
            )
            logger.error("Failed to read image for job %s: %s", job_id, exc)
            raise HTTPException(
                status_code=400, detail="Invalid or unsupported image file"
            ) from exc

        parsed_percentiles = None
        if contrast_stretch_percentiles is not None:
            try:
                low, high = (
                    float(part.strip()) for part in contrast_stretch_percentiles.split(",")
                )
                parsed_percentiles = (low, high)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="contrast_stretch_percentiles must be a comma-separated pair",
                ) from exc

        processing_params = ProcessingParams(
            application_type=application_type,
            scale_factor=scale_factor,
            denoise_type=denoise_type,
            denoise_sigma=denoise_sigma,
            sharpen=sharpen,
            sharpen_amount=sharpen_amount,
            equalize=equalize,
            equalize_method=equalize_method,
            clip_limit=clip_limit,
            clip_limit_kernel_size=clip_limit_kernel_size,
            contrast_stretch_percentiles=parsed_percentiles,
            gamma=gamma,
            color_preservation=color_preservation,
            color_preservation_strength=color_preservation_strength,
            calculate_advanced_metrics=calculate_advanced_metrics,
            use_tiling=use_tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        background_tasks.add_task(
            _process_image_background,
            job_id,
            img,
            processing_params,
        )
        return job_manager.get_job(job_id)

    @app.get("/api/v1/jobs/{job_id}", tags=["Jobs"], response_model=JobInfo)
    async def get_job_status(
        job_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/v1/jobs", tags=["Jobs"], response_model=APIResponse)
    async def list_jobs(api_key: str = Depends(verify_api_key)):
        counts = job_manager.job_counts()
        return APIResponse(
            success=True,
            message=f"{sum(counts.values())} jobs tracked",
            data={"counts": counts},
        )

    @app.post("/api/v1/jobs/cleanup", tags=["Jobs"], response_model=APIResponse)
    async def cleanup_jobs(api_key: str = Depends(verify_api_key)):
        removed = job_manager.cleanup_expired_jobs()
        return APIResponse(
            success=True,
            message=f"Removed {removed} expired jobs",
        )

    @app.post("/api/v1/keys", tags=["Authentication"], response_model=APIResponse)
    async def create_api_key_endpoint(
        name: str = Form("New API Key"),
        rate_limit: int = Form(100),
        api_key: str = Depends(verify_api_key_or_bootstrap),
    ):
        existing_keys = api_key_manager.list_keys()
        if existing_keys and not api_key_manager.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="An API key is required to create additional keys.",
            )
        if not existing_keys and not _bootstrap_is_authorized(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bootstrap not authorized. Set CHIAROSCURO_API_KEY and CHIAROSCURO_ALLOW_BOOTSTRAP=true, or provide a valid API key.",
            )
        key = api_key_manager.create_key(name=name, rate_limit=rate_limit)
        return APIResponse(
            success=True,
            message="API key created",
            data={"api_key": key, "rate_limit": rate_limit},
        )

    @app.get("/api/v1/keys", tags=["Authentication"], response_model=APIResponse)
    async def list_api_keys(api_key: str = Depends(verify_api_key)):
        keys = api_key_manager.list_keys()
        return APIResponse(
            success=True,
            message=f"{len(keys)} keys registered",
            data={"keys": keys},
        )

    def _process_image_background(
        job_id: str,
        img: Image.Image,
        processing_params: Optional[ProcessingParams],
    ):
        try:
            job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=0.1)
            img_array = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            job_manager.update_job(job_id, progress=0.3)

            input_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    input_path = tf.name
                skio.imsave(input_path, _ubyte(img_array))

                config = _build_processing_config_from_params(processing_params)
                processed, metrics = _process(input_path, config=config)
                job_manager.update_job(job_id, progress=0.8)

                height, width = processed.shape[:2]
                job_manager.update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    progress=1.0,
                    result={
                        "width": width,
                        "height": height,
                        "metrics": metrics,
                    },
                )
            finally:
                try:
                    os.unlink(input_path)
                except OSError:
                    pass

        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc)
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=type(exc).__name__,
            )

else:
    app = None
    logger.warning(
        "FastAPI not installed. API functionality unavailable. "
        "Install with: pip install 'fastapi[all]' uvicorn"
    )


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
):
    """Run the FastAPI server.

    Parameters
    ----------
    host :
        Host to bind to.
    port :
        Port to listen on.
    reload :
        Enable auto-reload for development.
    ssl_keyfile :
        Path to TLS private key file (PEM). Required for HTTPS.
    ssl_certfile :
        Path to TLS certificate file (PEM). Required for HTTPS.

    Raises
    ------
    ImportError
        If FastAPI is not installed.

    Example
    -------
    >>> from chiaroscuro_forge.api import run_server
    >>> run_server(port=8000)
    >>> # Production with TLS:
    >>> # run_server(port=443, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not installed. " "Install with: pip install 'fastapi[all]' uvicorn"
        )
    if (ssl_keyfile is None) != (ssl_certfile is None):
        raise ValueError(
            "ssl_keyfile and ssl_certfile must be provided together or omitted together"
        )
    uvicorn.run(
        "chiaroscuro_forge.api:app",
        host=host,
        port=port,
        reload=reload,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
