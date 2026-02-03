"""Tests for the REST API module."""

import pytest
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import numpy as np

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient
    from chiaroscuro_forge.api import (
        app, api_key_manager, job_manager,
        JobStatus, JobInfo, APIResponse,
        APIKeyManager, JobManager, ProcessingParams
    )
    FASTAPI_AVAILABLE = app is not None
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed"
)


# ===== Fixtures =====

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """Create a valid API key for testing."""
    return "dev-key-12345"  # Default development key


@pytest.fixture
def test_image():
    """Create a test image."""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture(autouse=True)
def reset_managers():
    """Reset managers before each test."""
    # Reset API key manager
    api_key_manager._keys.clear()
    api_key_manager.create_key("dev-key-12345", name="Development Key")
    
    # Reset job manager
    job_manager._jobs.clear()
    job_manager._job_counter = 0
    
    yield


# ===== API Key Manager Tests =====

class TestAPIKeyManager:
    """Tests for APIKeyManager."""
    
    def test_create_key_default(self):
        """Test creating a key with default parameters."""
        manager = APIKeyManager()
        key = manager.create_key()
        
        assert key.startswith('sk_')
        assert manager.validate_key(key)
    
    def test_create_key_specific(self):
        """Test creating a key with specific value."""
        manager = APIKeyManager()
        key = manager.create_key(key="test-key-123", name="Test Key")
        
        assert key == "test-key-123"
        assert manager.validate_key(key)
    
    def test_validate_key_invalid(self):
        """Test validating an invalid key."""
        manager = APIKeyManager()
        assert not manager.validate_key("invalid-key")
    
    def test_rate_limit_within_limit(self):
        """Test rate limiting within the limit."""
        manager = APIKeyManager()
        key = manager.create_key(rate_limit=10)
        
        # Should allow up to 10 requests
        for _ in range(10):
            assert manager.check_rate_limit(key)
    
    def test_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded."""
        manager = APIKeyManager()
        key = manager.create_key(rate_limit=5)
        
        # Use up the limit
        for _ in range(5):
            manager.check_rate_limit(key)
        
        # Next request should fail
        assert not manager.check_rate_limit(key)
    
    def test_rate_limit_time_window(self):
        """Test that rate limit resets after time window."""
        manager = APIKeyManager()
        key = manager.create_key(rate_limit=2)
        
        # Use up the limit
        manager.check_rate_limit(key)
        manager.check_rate_limit(key)
        
        # Manually expire old requests
        key_data = manager._keys[key]
        old_time = datetime.now() - timedelta(hours=2)
        key_data['requests'] = [old_time, old_time]
        
        # Should allow new request
        assert manager.check_rate_limit(key)


# ===== Job Manager Tests =====

class TestJobManager:
    """Tests for JobManager."""
    
    def test_create_job(self):
        """Test creating a new job."""
        manager = JobManager()
        job_id = manager.create_job()
        
        assert job_id.startswith('job_')
        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.PENDING
    
    def test_get_nonexistent_job(self):
        """Test getting a non-existent job."""
        manager = JobManager()
        job = manager.get_job("nonexistent")
        
        assert job is None
    
    def test_update_job_status(self):
        """Test updating job status."""
        manager = JobManager()
        job_id = manager.create_job()
        
        manager.update_job(job_id, status=JobStatus.PROCESSING, progress=0.5)
        job = manager.get_job(job_id)
        
        assert job.status == JobStatus.PROCESSING
        assert job.progress == 0.5
    
    def test_update_job_completion(self):
        """Test updating job to completed status."""
        manager = JobManager()
        job_id = manager.create_job()
        
        manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result={"test": "data"}
        )
        job = manager.get_job(job_id)
        
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1.0
        assert job.completed_at is not None
    
    def test_update_job_failure(self):
        """Test updating job to failed status."""
        manager = JobManager()
        job_id = manager.create_job()
        
        manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error="Test error"
        )
        job = manager.get_job(job_id)
        
        assert job.status == JobStatus.FAILED
        assert job.error == "Test error"
        assert job.completed_at is not None


# ===== API Endpoint Tests =====

class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "Chiaroscuro Forge" in data["name"]
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_process_image_no_auth(self, client, test_image):
        """Test processing without authentication."""
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")}
        )
        
        assert response.status_code == 401
    
    def test_process_image_invalid_key(self, client, test_image):
        """Test processing with invalid API key."""
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            headers={"X-API-Key": "invalid-key"}
        )
        
        assert response.status_code == 401
    
    def test_process_image_success(self, client, test_image, valid_api_key):
        """Test successful image processing."""
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            data={"gamma": "1.2"},
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
    
    def test_process_image_with_params(self, client, test_image, valid_api_key):
        """Test image processing with parameters."""
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            data={
                "gamma": "1.5",
                "scale_factor": "2.0",
                "application_type": "photography"
            },
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
    
    def test_process_invalid_image(self, client, valid_api_key):
        """Test processing with invalid image data."""
        invalid_data = BytesIO(b"not an image")
        
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.txt", invalid_data, "text/plain")},
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 400
    
    def test_get_job_status_not_found(self, client, valid_api_key):
        """Test getting status of non-existent job."""
        response = client.get(
            "/api/v1/jobs/nonexistent",
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 404
    
    def test_get_job_status_success(self, client, valid_api_key):
        """Test getting status of existing job."""
        # Create a job
        job_id = job_manager.create_job()
        
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
    
    def test_create_api_key(self, client):
        """Test creating a new API key."""
        response = client.post(
            "/api/v1/keys",
            data={"name": "Test Key", "rate_limit": "50"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "api_key" in data["data"]
    
    def test_rate_limit_enforcement(self, client, test_image):
        """Test that rate limiting is enforced."""
        # Create a key with very low rate limit
        limited_key = api_key_manager.create_key(rate_limit=2)
        
        # First two requests should succeed
        for _ in range(2):
            response = client.post(
                "/api/v1/process",
                files={"image": ("test.png", test_image, "image/png")},
                headers={"X-API-Key": limited_key}
            )
            assert response.status_code == 200
        
        # Third request should be rate limited
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            headers={"X-API-Key": limited_key}
        )
        assert response.status_code == 429


# ===== Model Tests =====

class TestModels:
    """Tests for Pydantic models."""
    
    def test_processing_params_valid(self):
        """Test valid processing parameters."""
        params = ProcessingParams(
            gamma=1.2,
            scale_factor=2.0,
            application_type="photography"
        )
        
        assert params.gamma == 1.2
        assert params.scale_factor == 2.0
        assert params.application_type == "photography"
    
    def test_processing_params_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            ProcessingParams(gamma=5.0)  # Exceeds max
        
        with pytest.raises(ValueError):
            ProcessingParams(scale_factor=-1.0)  # Negative not allowed
    
    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
    
    def test_job_info_model(self):
        """Test JobInfo model."""
        now = datetime.now()
        job = JobInfo(
            job_id="test-job",
            status=JobStatus.COMPLETED,
            created_at=now,
            completed_at=now,
            progress=1.0
        )
        
        assert job.job_id == "test-job"
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1.0
    
    def test_api_response_model(self):
        """Test APIResponse model."""
        response = APIResponse(
            success=True,
            message="Operation successful",
            data={"key": "value"}
        )
        
        assert response.success
        assert response.message == "Operation successful"
        assert response.data["key"] == "value"


# ===== Integration Tests =====

class TestAPIIntegration:
    """Integration tests for API workflows."""
    
    def test_complete_processing_workflow(self, client, test_image, valid_api_key):
        """Test complete image processing workflow."""
        # 1. Submit processing job
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            data={"gamma": "1.2"},
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        # 2. Check job status
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-API-Key": valid_api_key}
        )
        
        assert response.status_code == 200
        job_data = response.json()
        assert job_data["job_id"] == job_id
        assert "status" in job_data
    
    def test_multiple_api_keys(self, client, test_image):
        """Test using multiple different API keys."""
        # Create two keys
        key1 = api_key_manager.create_key(name="Key 1")
        key2 = api_key_manager.create_key(name="Key 2")
        
        # Both should work independently
        for key in [key1, key2]:
            response = client.post(
                "/api/v1/process",
                files={"image": ("test.png", test_image, "image/png")},
                headers={"X-API-Key": key}
            )
            assert response.status_code == 200
    
    def test_concurrent_requests(self, client, test_image, valid_api_key):
        """Test handling multiple concurrent requests."""
        responses = []
        
        # Submit multiple jobs
        for _ in range(5):
            response = client.post(
                "/api/v1/process",
                files={"image": ("test.png", test_image, "image/png")},
                headers={"X-API-Key": valid_api_key}
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
        
        # All job IDs should be unique
        job_ids = [r.json()["job_id"] for r in responses]
        assert len(job_ids) == len(set(job_ids))
