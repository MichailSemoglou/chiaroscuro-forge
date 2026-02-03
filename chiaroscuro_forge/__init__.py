"""
Chiaroscuro Forge - Intelligent Image Enhancement Tool

This package provides comprehensive image enhancement capabilities inspired by
Renaissance art techniques, featuring automatic parameter detection, advanced
color preservation, and quality metrics.

Main Features:
    - Intelligent enhancement with automatic parameter selection
    - Multiple color preservation methods (LAB, RGB, ratio-based)
    - Quality metrics (SSIM, PSNR, MS-SSIM, perceptual metrics)
    - Batch processing with parallel execution
    - Preset system for reusable configurations

Example:
    Basic usage for image enhancement::

        from chiaroscuro_forge import process_image
        
        processed, metrics = process_image(
            "input.jpg",
            output_path="enhanced.jpg",
            application_type="photography"
        )
        print(f"Quality Score: {metrics['quality_score']:.4f}")

For more examples, see the documentation at:
https://github.com/MichailSemoglou/chiaroscuro-forge
"""

__version__ = "1.0.0"
__author__ = "Michail Semoglou"
__email__ = "m.semoglou@tongji.edu.cn"
__license__ = "MIT"

# Import all functionality from modular structure
from chiaroscuro_forge.processing import process_image
from chiaroscuro_forge.analysis import (
    analyze_image_characteristics,
    get_image_statistics,
)
from chiaroscuro_forge.batch import batch_process_images, analyze_batch
from chiaroscuro_forge.comparison import compare_processing_methods
from chiaroscuro_forge.presets import save_preset, load_preset, list_presets
from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.cache import (
    get_cache_manager,
    invalidate_preset_cache,
    invalidate_stats_cache,
)
from chiaroscuro_forge.tiling import (
    process_image_tiled,
    should_use_tiling,
)
from chiaroscuro_forge.di import (
    ServiceContainer,
    get_container,
    inject,
    setup_default_services,
)
from chiaroscuro_forge.gpu import (
    GPUContext,
    gpu_available,
    get_gpu_info,
    get_gpu_backend,
    GPUBackend,
    GPUInfo,
    benchmark_operation,
)
from chiaroscuro_forge.distributed import (
    TaskQueue,
    LocalQueue,
    DistributedBatchProcessor,
    TaskStatus,
    TaskResult,
    create_queue,
)

# Optional API module - only available if fastapi is installed
try:
    from chiaroscuro_forge.api import (
        app,
        api_key_manager,
        job_manager,
        run_server,
        JobStatus as APIJobStatus,
        JobInfo,
        APIResponse,
        ProcessingParams,
    )
    _api_available = True
except ImportError:
    _api_available = False
    # Provide placeholder None values for optional API components
    app = None
    api_key_manager = None
    job_manager = None
    run_server = None
    APIJobStatus = None
    JobInfo = None
    APIResponse = None
    ProcessingParams = None

__all__ = [
    "process_image",
    "analyze_image_characteristics",
    "get_image_statistics",
    "batch_process_images",
    "analyze_batch",
    "compare_processing_methods",
    "save_preset",
    "load_preset",
    "list_presets",
    "ImageProcessingError",
    # Phase 3.1: Cache management
    "get_cache_manager",
    "invalidate_preset_cache",
    "invalidate_stats_cache",
    # Phase 3.2: Tile-based processing
    "process_image_tiled",
    "should_use_tiling",
    # Phase 3.3: Dependency injection
    "ServiceContainer",
    "get_container",
    "inject",
    "setup_default_services",
    # Phase 4.1: GPU acceleration
    "GPUContext",
    "gpu_available",
    "get_gpu_info",
    "get_gpu_backend",
    "GPUBackend",
    "GPUInfo",
    "benchmark_operation",
    # Phase 4.2: Distributed processing
    "TaskQueue",
    "LocalQueue",
    "DistributedBatchProcessor",
    "TaskStatus",
    "TaskResult",
    "create_queue",
    # Phase 4.3: REST API
    "app",
    "api_key_manager",
    "job_manager",
    "run_server",
    "APIJobStatus",
    "JobInfo",
    "APIResponse",
    "ProcessingParams",
]
