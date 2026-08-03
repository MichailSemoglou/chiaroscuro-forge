"""
GPU acceleration module for compute-intensive image processing operations.

This module provides GPU-accelerated versions of key operations with automatic
fallback to CPU when GPU is unavailable. Supports CUDA (NVIDIA), OpenCL (cross-platform),
and Metal (Apple Silicon).

Architecture:
- Automatic GPU capability detection (lazy, on first use)
- Transparent fallback to CPU
- Async GPU processing for non-blocking operations
- Memory-efficient data transfer
- Benchmarking utilities

Example:
    >>> from chiaroscuro_forge.gpu import GPUContext, gpu_available
    >>>
    >>> if gpu_available():
    ...     with GPUContext() as gpu:
    ...         result = gpu.gaussian_blur(image, sigma=2.0)
    ... else:
    ...     result = cpu_gaussian_blur(image, sigma=2.0)
"""

import functools
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .optional import optional_import

logger = logging.getLogger(__name__)

MAX_GPU_PIXELS = 50_000_000
"""Maximum pixel count to transfer to GPU in a single operation. Larger
images are processed on CPU to avoid out-of-memory errors."""


class GPUBackend(Enum):
    """Supported GPU acceleration backends."""

    NONE = "none"
    CUDA = "cuda"
    OPENCL = "opencl"
    METAL = "metal"


@dataclass
class GPUInfo:
    """GPU device information."""

    backend: GPUBackend
    device_name: str
    compute_capability: Optional[str] = None
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None  # bytes
    supports_fp16: bool = False
    supports_async: bool = False


class GPUCapabilities:
    """Singleton class for GPU capability detection and management.

    Detection is lazy -- no import-side-effect probing. The first access
    to ``info``, ``backend``, or ``is_available()`` triggers detection.
    """

    _instance: Optional["GPUCapabilities"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._detected = False
                    cls._instance._gpu_info: Optional[GPUInfo] = None
                    cls._instance._backend_module: Any = None
        return cls._instance

    def _ensure_detected(self) -> None:
        if self._detected:
            return
        with self._lock:
            if self._detected:
                return
            self._detect_gpu()
            self._detected = True

    def _detect_gpu(self) -> None:
        if self._try_cuda():
            return
        if self._try_metal():
            return
        if self._try_opencl():
            return
        logger.info("No GPU backend available, using CPU")
        self._gpu_info = GPUInfo(backend=GPUBackend.NONE, device_name="CPU")

    def _try_cuda(self) -> bool:
        cp = optional_import("cupy", "CUDA GPU acceleration", "pip install cupy")
        if cp is None:
            return False
        try:
            if cp.cuda.is_available():
                device = cp.cuda.Device()
                cc = device.compute_capability
                mem_info = device.mem_info
                self._gpu_info = GPUInfo(
                    backend=GPUBackend.CUDA,
                    device_name=f"CUDA Device {device.id}",
                    compute_capability=f"{cc[0]}.{cc[1]}",
                    memory_total=mem_info[1],
                    memory_available=mem_info[0],
                    supports_fp16=True,
                    supports_async=True,
                )
                self._backend_module = cp
                logger.info("CUDA GPU detected: %s", self._gpu_info.device_name)
                return True
        except Exception as exc:
            logger.debug("CUDA initialization failed: %s", exc)
        return False

    def _try_metal(self) -> bool:
        metal = optional_import("metal", "Metal GPU (Apple Silicon)", "pip install metal")
        if metal is None:
            return False
        try:
            devices = metal.MTL.MTLCopyAllDevices()
            if devices:
                d = devices[0]
                self._gpu_info = GPUInfo(
                    backend=GPUBackend.METAL,
                    device_name=d.name(),
                    supports_fp16=True,
                    supports_async=True,
                )
                self._backend_module = metal
                logger.info("Metal GPU detected: %s", self._gpu_info.device_name)
                return True
        except Exception as exc:
            logger.debug("Metal initialization failed: %s", exc)
        return False

    def _try_opencl(self) -> bool:
        pycl = optional_import("pyopencl", "OpenCL acceleration", "pip install pyopencl")
        if pycl is None:
            return False
        try:
            platforms = pycl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(device_type=pycl.device_type.GPU)
                if devices:
                    d = devices[0]
                    ext = getattr(d, "extensions", "") or ""
                    fp16 = ext.find("cl_khr_fp16") >= 0
                    self._gpu_info = GPUInfo(
                        backend=GPUBackend.OPENCL,
                        device_name=d.name,
                        memory_total=d.global_mem_size,
                        supports_fp16=fp16,
                        supports_async=True,
                    )
                    self._backend_module = pycl
                    logger.info("OpenCL GPU detected: %s", self._gpu_info.device_name)
                    return True
        except Exception as exc:
            logger.debug("OpenCL initialization failed: %s", exc)
        return False

    @property
    def info(self) -> GPUInfo:
        self._ensure_detected()
        return self._gpu_info

    @property
    def backend(self) -> GPUBackend:
        self._ensure_detected()
        return self._gpu_info.backend

    @property
    def backend_module(self) -> Optional[Any]:
        self._ensure_detected()
        return self._backend_module

    def is_available(self) -> bool:
        self._ensure_detected()
        return self._gpu_info.backend != GPUBackend.NONE


def _get_capabilities() -> GPUCapabilities:
    """Return the singleton GPU capability instance."""
    return GPUCapabilities()


def gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns
    -------
    bool
        ``True`` if GPU is available, ``False`` otherwise.
    """
    return _get_capabilities().is_available()


def get_gpu_info() -> GPUInfo:
    """Get information about the available GPU.

    Returns
    -------
    GPUInfo
        GPU device information.
    """
    return _get_capabilities().info


def get_gpu_backend() -> GPUBackend:
    """Get the active GPU backend.

    Returns
    -------
    GPUBackend
        Active backend (CUDA, OpenCL, Metal, or NONE).
    """
    return _get_capabilities().backend


def _image_too_large_for_gpu(image: np.ndarray) -> bool:
    """Check whether an image exceeds the GPU memory safety threshold."""
    pixel_count = image.shape[0] * image.shape[1] if image.ndim >= 2 else image.size
    return pixel_count > MAX_GPU_PIXELS


def safe_gpu(
    gpu_func: Callable[..., Any],
    cpu_func: Callable[..., Any],
) -> Callable[..., Any]:
    """Create a GPU-safe wrapper factory for two implementations.

    This function must be called with both a GPU implementation and a CPU
    fallback implementation, such as ``safe_gpu(gpu_impl, cpu_impl)``. The
    returned wrapper runs the GPU implementation when available and falls back
    to the CPU implementation when the GPU is unavailable, the input image is
    too large for GPU memory, or the GPU operation fails.

    Both functions receive the same arguments.

    Parameters
    ----------
    gpu_func :
        Function that uses GPU-accelerated primitives.
    cpu_func :
        Equivalent CPU fallback.

    Returns
    -------
    Callable
        A wrapper that dispatches to the GPU implementation or CPU fallback.
    """

    @functools.wraps(gpu_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not gpu_available():
            logger.debug("GPU unavailable, falling back to CPU")
            return cpu_func(*args, **kwargs)

        if args and isinstance(args[0], np.ndarray):
            image = args[0]
            if _image_too_large_for_gpu(image):
                pixel_count = image.shape[0] * image.shape[1] if image.ndim >= 2 else image.size
                logger.debug(
                    "Image exceeds GPU memory threshold (%d pixels), falling back to CPU",
                    pixel_count,
                )
                return cpu_func(*args, **kwargs)

        try:
            result = gpu_func(*args, **kwargs)
            logger.debug("GPU operation succeeded: %s", gpu_func.__name__)
            return result
        except Exception as exc:
            logger.warning(
                "GPU operation %s failed: %s. Falling back to CPU.",
                gpu_func.__name__,
                exc,
            )
            return cpu_func(*args, **kwargs)

    return wrapper


class GPUContext:
    """Context manager for GPU operations with automatic memory management.

    Handles data transfer to/from GPU and ensures proper cleanup.

    Example
    -------
    >>> with GPUContext() as gpu:
    ...     result = gpu.gaussian_blur(image, sigma=2.0)
    """

    def __init__(self):
        caps = _get_capabilities()
        self.backend = caps.backend
        self.backend_module = caps.backend_module
        self._opencl_context: Any = None
        self._opencl_queue: Any = None
        self._active = False

    def __enter__(self):
        self._active = True
        if self.backend == GPUBackend.OPENCL:
            import pyopencl as cl

            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    self._opencl_context = cl.Context([devices[0]])
                    self._opencl_queue = cl.CommandQueue(self._opencl_context)
                    logger.debug("OpenCL context created: %s", devices[0].name)
                    break
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._opencl_queue is not None:
            try:
                self._opencl_queue.finish()
            except Exception as exc:
                logger.warning("Failed to flush OpenCL queue: %s", exc)
        if self.backend == GPUBackend.CUDA and self.backend_module:
            try:
                self.backend_module.get_default_memory_pool().free_all_blocks()
            except Exception as exc:
                logger.warning("Failed to free CUDA memory pool: %s", exc)
        self._active = False

    def to_gpu(self, array: np.ndarray) -> Any:
        if self.backend == GPUBackend.CUDA:
            if _image_too_large_for_gpu(array):
                raise MemoryError(f"Array too large for GPU transfer ({array.size} pixels)")
            return self.backend_module.asarray(array)
        elif self.backend == GPUBackend.OPENCL:
            import pyopencl as cl

            mf = cl.mem_flags
            buf = cl.Buffer(
                self._opencl_context,
                mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=array,
            )
            buf._host_shape = array.shape
            buf._host_dtype = array.dtype
            return buf
        return array

    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        if self.backend == GPUBackend.CUDA:
            result = self.backend_module.asnumpy(gpu_array)
            logger.debug("CUDA→CPU transfer: %s", result.shape)
            return result
        elif self.backend == GPUBackend.OPENCL:
            import pyopencl as cl

            shape = getattr(gpu_array, "_host_shape", None)
            dtype = getattr(gpu_array, "_host_dtype", None)
            if shape is None or dtype is None:
                raise ValueError(
                    "OpenCL buffer metadata is missing: expected _host_shape and _host_dtype"
                )
            result = np.empty(shape, dtype=dtype)
            cl.enqueue_copy(self._opencl_queue, result, gpu_array)
            return result
        return gpu_array if isinstance(gpu_array, np.ndarray) else np.asarray(gpu_array)

    def gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        if self.backend == GPUBackend.CUDA:
            from cupyx.scipy.ndimage import gaussian_filter

            gpu_img = self.to_gpu(image)
            gpu_result = gaussian_filter(gpu_img, sigma=sigma)
            return self.to_cpu(gpu_result)
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(image, sigma=sigma)

    def sobel_filter(self, image: np.ndarray) -> np.ndarray:
        if self.backend == GPUBackend.CUDA:
            from cupyx.scipy.ndimage import sobel

            gpu_img = self.to_gpu(image)
            sx = sobel(gpu_img, axis=0)
            sy = sobel(gpu_img, axis=1)
            gpu_result = self.backend_module.hypot(sx, sy)
            return self.to_cpu(gpu_result)
        from scipy.ndimage import sobel

        sx = sobel(image, axis=0)
        sy = sobel(image, axis=1)
        return np.hypot(sx, sy)


@dataclass
class GPUBenchmark:
    """Results from GPU benchmarking."""

    operation: str
    backend: GPUBackend
    image_shape: Tuple[int, ...]
    gpu_time: Optional[float] = None
    cpu_time: Optional[float] = None
    speedup: Optional[float] = None
    memory_transferred: Optional[int] = None
    gpu_error: Optional[str] = None
    cpu_error: Optional[str] = None

    @property
    def summary(self) -> str:
        parts = [f"{self.operation} [{self.backend.value}]"]
        if self.speedup is not None:
            parts.append(f"{self.speedup:.1f}x speedup")
        if self.gpu_error:
            parts.append(f"GPU error: {self.gpu_error}")
        if self.cpu_error:
            parts.append(f"CPU error: {self.cpu_error}")
        return " | ".join(parts)


def benchmark_operation(
    operation_name: str,
    gpu_func: Callable[..., Any],
    cpu_func: Callable[..., Any],
    test_image: np.ndarray,
    *args: Any,
    warmup_iterations: int = 1,
    **kwargs: Any,
) -> GPUBenchmark:
    """Benchmark GPU vs CPU performance for an operation.

    Parameters
    ----------
    operation_name :
        Name of the operation.
    gpu_func :
        GPU function to benchmark.
    cpu_func :
        CPU function to benchmark.
    test_image :
        Test image array.
    warmup_iterations :
        Number of warm-up iterations (default 1). These are not timed.
    *args, **kwargs :
        Additional arguments passed to both functions.

    Returns
    -------
    GPUBenchmark
        Benchmark results with ``summary`` property.
    """
    result = GPUBenchmark(
        operation=operation_name,
        backend=get_gpu_backend(),
        image_shape=test_image.shape,
    )

    logger.info("Benchmarking %s on image %s", operation_name, test_image.shape)

    for _ in range(warmup_iterations):
        try:
            cpu_func(test_image.copy(), *args, **kwargs)
        except Exception as exc:
            logger.debug("CPU warmup failed: %s", exc, exc_info=True)
        if gpu_available():
            try:
                gpu_func(test_image.copy(), *args, **kwargs)
            except Exception as exc:
                logger.debug("GPU warmup failed: %s", exc, exc_info=True)

    try:
        start = time.perf_counter()
        cpu_func(test_image, *args, **kwargs)
        result.cpu_time = time.perf_counter() - start
        logger.debug("CPU benchmark: %.4fs", result.cpu_time)
    except Exception as exc:
        result.cpu_error = str(exc)
        logger.warning("CPU benchmark failed: %s", exc)

    if gpu_available():
        try:
            start = time.perf_counter()
            gpu_func(test_image, *args, **kwargs)
            result.gpu_time = time.perf_counter() - start
            logger.debug("GPU benchmark: %.4fs", result.gpu_time)
            result.memory_transferred = test_image.nbytes * 2
        except Exception as exc:
            result.gpu_error = str(exc)
            logger.warning("GPU benchmark failed: %s", exc)

        if result.cpu_time is not None and result.gpu_time is not None and result.gpu_time > 0:
            result.speedup = result.cpu_time / result.gpu_time

    logger.info("Benchmark result: %s", result.summary)
    return result
