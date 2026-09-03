"""Tests for the REST API module."""

from datetime import datetime, timedelta
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from chiaroscuro_forge.api import _build_processing_config_from_params
from chiaroscuro_forge.constants import MAX_FILE_SIZE_MB

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient

    from chiaroscuro_forge.api import (
        APIKeyManager,
        APIResponse,
        JobInfo,
        JobManager,
        JobStatus,
        ProcessingParams,
        api_key_manager,
        app,
        job_manager,
    )

    FASTAPI_AVAILABLE = app is not None
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")

TEST_API_KEY = "test-key-42"


# ===== Fixtures =====


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """Create a valid API key for testing."""
    return TEST_API_KEY


@pytest.fixture
def test_image_factory():
    """Create a fresh test image stream for each request."""

    def _make_image():
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return img_bytes

    return _make_image


@pytest.fixture
def test_image(test_image_factory):
    """Create a test image."""
    return test_image_factory()


@pytest.fixture(autouse=True)
def reset_managers():
    """Reset managers before each test."""
    # Reset API key manager
    api_key_manager._keys.clear()
    api_key_manager.create_key(TEST_API_KEY, name="Test Key")

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

        assert key.startswith("sk_")
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
        key_data["requests"] = [old_time, old_time]

        # Should allow new request
        assert manager.check_rate_limit(key)

    def test_revoke_key_removes_and_lists_active_keys(self):
        """Revoked keys should be removed from validation and no longer listed."""
        manager = APIKeyManager()
        first_key = manager.create_key(key="first-key", name="First")
        second_key = manager.create_key(key="second-key", name="Second")

        assert manager.revoke_key(first_key)
        assert not manager.validate_key(first_key)
        assert manager.validate_key(second_key)

        listed = manager.list_keys()
        names = [entry["name"] for entry in listed]
        assert names == ["Second"]


# ===== Job Manager Tests =====


class TestJobManager:
    """Tests for JobManager."""

    def test_create_job(self):
        """Test creating a new job."""
        manager = JobManager()
        job_id = manager.create_job()

        assert job_id.startswith("job_")
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
            job_id, status=JobStatus.COMPLETED, progress=1.0, result={"test": "data"}
        )
        job = manager.get_job(job_id)

        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1.0
        assert job.completed_at is not None

    def test_update_job_failure(self):
        """Test updating job to failed status."""
        manager = JobManager()
        job_id = manager.create_job()

        manager.update_job(job_id, status=JobStatus.FAILED, error="Test error")
        job = manager.get_job(job_id)

        assert job.status == JobStatus.FAILED
        assert job.error == "Test error"
        assert job.completed_at is not None

    def test_cleanup_expired_jobs_removes_only_old_completed_or_failed_jobs(self):
        """Cleanup should remove only stale completed/failed jobs."""
        manager = JobManager()
        old_completed = manager.create_job()
        recent_completed = manager.create_job()
        pending_job = manager.create_job()
        old_failed = manager.create_job()
        old_pending = manager.create_job()

        manager._jobs[old_completed]["status"] = JobStatus.COMPLETED
        manager._jobs[old_completed]["completed_at"] = datetime.now() - timedelta(hours=25)
        manager._jobs[recent_completed]["status"] = JobStatus.COMPLETED
        manager._jobs[recent_completed]["completed_at"] = datetime.now() - timedelta(hours=1)
        manager._jobs[old_failed]["completed_at"] = datetime.now() - timedelta(hours=25)
        manager._jobs[old_failed]["status"] = JobStatus.FAILED
        manager._jobs[pending_job]["status"] = JobStatus.PENDING
        manager._jobs[old_pending]["status"] = JobStatus.PENDING
        manager._jobs[old_pending]["completed_at"] = datetime.now() - timedelta(hours=25)

        removed = manager.cleanup_expired_jobs()

        assert removed == 2
        assert old_completed not in manager._jobs
        assert old_failed not in manager._jobs
        assert recent_completed in manager._jobs
        assert pending_job in manager._jobs
        assert old_pending in manager._jobs

    def test_job_counts_reports_each_status(self):
        """Job counts should report the current status distribution."""
        manager = JobManager()
        manager.create_job()
        processing_job = manager.create_job()
        completed_job = manager.create_job()
        failed_job = manager.create_job()

        manager.update_job(processing_job, status=JobStatus.PROCESSING)
        manager.update_job(completed_job, status=JobStatus.COMPLETED)
        manager.update_job(failed_job, status=JobStatus.FAILED)

        counts = manager.job_counts()

        assert counts["pending"] == 1
        assert counts["processing"] == 1
        assert counts["completed"] == 1
        assert counts["failed"] == 1

    def test_uptime_seconds_returns_nonnegative_elapsed_value(self):
        """Uptime should be a non-negative elapsed time value."""
        manager = JobManager()
        elapsed = manager.uptime_seconds

        assert elapsed >= 0.0


# ===== API Endpoint Tests =====


class TestProcessingConfigMapping:
    """Regression tests for converting API request params into ProcessingConfig."""

    def test_build_processing_config_maps_params(self):
        class DummyParams:
            def dict(self, exclude_none=True):
                return {
                    "application_type": "photography",
                    "gamma": 1.2,
                    "scale_factor": 2.0,
                    "denoise_type": "bilateral",
                    "equalize_method": "clahe",
                    "clip_limit": 0.02,
                    "color_preservation": "lab",
                    "use_tiling": True,
                    "tile_size": 256,
                    "tile_overlap": 32,
                }

        config = _build_processing_config_from_params(DummyParams())

        assert config.application_type == "photography"
        assert config.gamma_correction == 1.2
        assert config.scale_factor == 2.0
        assert config.denoise_type == "bilateral"
        assert config.equalize_method == "clahe"
        assert config.clip_limit == 0.02
        assert config.color_preservation == "lab"
        assert config.use_tiling is True
        assert config.tile_size == 256
        assert config.tile_overlap == 32


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
            "/api/v1/process", files={"image": ("test.png", test_image, "image/png")}
        )

        assert response.status_code == 401

    def test_process_image_invalid_key(self, client, test_image):
        """Test processing with invalid API key."""
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            headers={"X-API-Key": "invalid-key"},
        )

        assert response.status_code == 401

    def test_process_image_success(self, client, test_image, valid_api_key):
        """Test successful image processing."""
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", test_image, "image/png")},
            data={"gamma": "1.2"},
            headers={"X-API-Key": valid_api_key},
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
            data={"gamma": "1.5", "scale_factor": "2.0", "application_type": "photography"},
            headers={"X-API-Key": valid_api_key},
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
            headers={"X-API-Key": valid_api_key},
        )

        assert response.status_code == 400

    def test_get_job_status_not_found(self, client, valid_api_key):
        """Test getting status of non-existent job."""
        response = client.get("/api/v1/jobs/nonexistent", headers={"X-API-Key": valid_api_key})

        assert response.status_code == 404

    def test_get_job_status_success(self, client, valid_api_key):
        """Test getting status of existing job."""
        # Create a job
        job_id = job_manager.create_job()

        response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": valid_api_key})

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data

    def test_create_api_key(self, client, valid_api_key):
        """Test creating a new API key."""
        response = client.post(
            "/api/v1/keys",
            headers={"X-API-Key": valid_api_key},
            data={"name": "Test Key", "rate_limit": "50"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "api_key" in data["data"]

    def test_create_first_key_bootstrap(self, client, monkeypatch):
        """Test creating the first key when bootstrap is explicitly authorized."""
        api_key_manager._keys.clear()
        monkeypatch.setenv("CHIAROSCURO_API_KEY", "bootstrap-key")
        monkeypatch.setenv("CHIAROSCURO_ALLOW_BOOTSTRAP", "true")
        response = client.post(
            "/api/v1/keys",
            headers={"X-API-Key": "bootstrap-key"},
            data={"name": "Bootstrap Key", "rate_limit": "100"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert len(api_key_manager.list_keys()) == 1

    def test_create_first_key_with_bootstrap_flag(self, client, monkeypatch):
        """Test that the bootstrap flag without the environment key is rejected."""
        api_key_manager._keys.clear()
        monkeypatch.delenv("CHIAROSCURO_API_KEY", raising=False)
        monkeypatch.setenv("CHIAROSCURO_ALLOW_BOOTSTRAP", "true")
        response = client.post(
            "/api/v1/keys",
            headers={"X-API-Key": "bootstrap-flag-key"},
            data={"name": "Bootstrap Flag Key", "rate_limit": "100"},
        )
        assert response.status_code == 401
        assert len(api_key_manager.list_keys()) == 0

    def test_create_first_key_env_key_without_bootstrap_flag(self, client, monkeypatch):
        """Test that the environment key without the bootstrap flag is rejected."""
        api_key_manager._keys.clear()
        monkeypatch.setenv("CHIAROSCURO_API_KEY", "bootstrap-key")
        monkeypatch.delenv("CHIAROSCURO_ALLOW_BOOTSTRAP", raising=False)
        response = client.post(
            "/api/v1/keys",
            headers={"X-API-Key": "bootstrap-key"},
            data={"name": "Key Only", "rate_limit": "100"},
        )
        assert response.status_code == 401
        assert len(api_key_manager.list_keys()) == 0

    def test_create_first_key_requires_explicit_bootstrap_control(self, client, monkeypatch):
        """Test that bootstrap without explicit control is rejected."""
        api_key_manager._keys.clear()
        monkeypatch.delenv("CHIAROSCURO_API_KEY", raising=False)
        monkeypatch.delenv("CHIAROSCURO_ALLOW_BOOTSTRAP", raising=False)
        response = client.post(
            "/api/v1/keys",
            data={"name": "Unauthorized Bootstrap Key", "rate_limit": "100"},
        )
        assert response.status_code == 401

    def test_startup_provisions_environment_key_when_no_keys_exist(self, monkeypatch):
        """Test that startup provisions the configured environment key when no keys exist."""
        api_key_manager._keys.clear()
        monkeypatch.setenv("CHIAROSCURO_API_KEY", "startup-bootstrap-key")
        with TestClient(app):
            pass
        assert len(api_key_manager.list_keys()) == 1
        assert api_key_manager.validate_key("startup-bootstrap-key")

    def test_create_additional_key_requires_auth(self, client):
        """Test that creating a key requires auth when keys already exist."""
        response = client.post(
            "/api/v1/keys",
            data={"name": "Unauthorized Key", "rate_limit": "50"},
        )
        assert response.status_code == 401

    def test_create_key_with_auth(self, client, valid_api_key):
        """Test creating a key with valid authentication."""
        response = client.post(
            "/api/v1/keys",
            headers={"X-API-Key": valid_api_key},
            data={"name": "Authenticated Key", "rate_limit": "50"},
        )
        assert response.status_code == 200

    def test_rate_limit_enforcement(self, client, test_image_factory):
        limited_key = api_key_manager.create_key(rate_limit=2)

        # First two requests should succeed
        for _ in range(2):
            image = test_image_factory()
            response = client.post(
                "/api/v1/process",
                files={"image": ("test.png", image, "image/png")},
                headers={"X-API-Key": limited_key},
            )
            assert response.status_code == 200

        # Third request should be rate limited
        image = test_image_factory()
        response = client.post(
            "/api/v1/process",
            files={"image": ("test.png", image, "image/png")},
            headers={"X-API-Key": limited_key},
        )
        assert response.status_code == 429

    def test_process_rejects_oversized_upload(self, client, valid_api_key):
        """Test that oversized uploads are rejected before decoding."""
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        oversized_bytes = b"x" * (max_bytes + 1024)
        response = client.post(
            "/api/v1/process",
            files={"image": ("oversized.png", oversized_bytes, "image/png")},
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 400

    def test_process_rejects_excessive_pixels(self, client, valid_api_key):
        """Test that decoded images exceeding the pixel threshold are rejected."""
        img = Image.new("RGB", (3, 3), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        with patch("chiaroscuro_forge.api.MAX_IMAGE_PIXELS", 4):
            response = client.post(
                "/api/v1/process",
                files={"image": ("huge.png", img_bytes.read(), "image/png")},
                headers={"X-API-Key": valid_api_key},
            )

        assert response.status_code == 400


# ===== Model Tests =====


class TestModels:
    """Tests for Pydantic models."""

    def test_processing_params_valid(self):
        """Test valid processing parameters."""
        params = ProcessingParams(gamma=1.2, scale_factor=2.0, application_type="photography")

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
            progress=1.0,
        )

        assert job.job_id == "test-job"
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1.0

    def test_api_response_model(self):
        """Test APIResponse model."""
        response = APIResponse(success=True, message="Operation successful", data={"key": "value"})

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
            headers={"X-API-Key": valid_api_key},
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # 2. Check job status
        response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": valid_api_key})

        assert response.status_code == 200
        job_data = response.json()
        assert job_data["job_id"] == job_id
        assert job_data["status"] == "completed"
        assert job_data["progress"] == 1.0

    def test_multiple_api_keys(self, client, test_image_factory):
        """Test using multiple different API keys."""
        # Create two keys
        key1 = api_key_manager.create_key(name="Key 1")
        key2 = api_key_manager.create_key(name="Key 2")

        # Both should work independently
        for key in [key1, key2]:
            image = test_image_factory()
            response = client.post(
                "/api/v1/process",
                files={"image": ("test.png", image, "image/png")},
                headers={"X-API-Key": key},
            )
            assert response.status_code == 200

    def test_concurrent_requests(self, client, test_image_factory, valid_api_key):
        """Test handling multiple concurrent requests."""
        responses = []

        # Submit multiple jobs
        for _ in range(5):
            image = test_image_factory()
            response = client.post(
                "/api/v1/process",
                files={"image": ("test.png", image, "image/png")},
                headers={"X-API-Key": valid_api_key},
            )
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        # All job IDs should be unique
        job_ids = [r.json()["job_id"] for r in responses]
        assert len(job_ids) == len(set(job_ids))
