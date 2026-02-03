"""
GPU acceleration module for compute-intensive image processing operations.

This module provides GPU-accelerated versions of key operations with automatic
fallback to CPU when GPU is unavailable. Supports CUDA (NVIDIA), OpenCL (cross-platform),
and Metal (Apple Silicon).

Architecture:
- Automatic GPU capability detection
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

import logging
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Tuple, Any, Dict, List
import numpy as np
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


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
    """Singleton class for GPU capability detection and management."""
    
    _instance = None
    _lock = threading.Lock()
    _gpu_info: Optional[GPUInfo] = None
    _backend_module: Optional[Any] = None
    
    def __new__(cls):
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._detect_gpu()
        return cls._instance
    
    def _detect_gpu(self) -> None:
        """Detect available GPU backends and select the best one."""
        # Try CUDA (NVIDIA GPUs)
        if self._try_cuda():
            return
        
        # Try Metal (Apple Silicon)
        if self._try_metal():
            return
        
        # Try OpenCL (cross-platform fallback)
        if self._try_opencl():
            return
        
        # No GPU available - fallback to CPU
        logger.info("No GPU backend available, using CPU")
        self._gpu_info = GPUInfo(
            backend=GPUBackend.NONE,
            device_name="CPU"
        )
    
    def _try_cuda(self) -> bool:
        """Try to initialize CUDA backend."""
        try:
            import cupy as cp
            
            # Check if CUDA is available
            if cp.cuda.is_available():
                device = cp.cuda.Device()
                device_name = device.compute_capability
                
                # Get memory info
                mem_info = device.mem_info
                mem_available = mem_info[0]
                mem_total = mem_info[1]
                
                self._gpu_info = GPUInfo(
                    backend=GPUBackend.CUDA,
                    device_name=f"CUDA Device {device.id}",
                    compute_capability=f"{device_name[0]}.{device_name[1]}",
                    memory_total=mem_total,
                    memory_available=mem_available,
                    supports_fp16=True,
                    supports_async=True
                )
                self._backend_module = cp
                logger.info(f"CUDA GPU detected: {self._gpu_info.device_name}")
                return True
        except ImportError:
            logger.debug("CuPy not available, skipping CUDA")
        except Exception as e:
            logger.debug(f"CUDA initialization failed: {e}")
        
        return False
    
    def _try_metal(self) -> bool:
        """Try to initialize Metal backend (Apple Silicon)."""
        try:
            import metal
            
            # Metal is only available on macOS with Apple Silicon
            devices = metal.MTL.MTLCopyAllDevices()
            if devices and len(devices) > 0:
                device = devices[0]
                
                self._gpu_info = GPUInfo(
                    backend=GPUBackend.METAL,
                    device_name=device.name(),
                    supports_fp16=True,
                    supports_async=True
                )
                self._backend_module = metal
                logger.info(f"Metal GPU detected: {self._gpu_info.device_name}")
                return True
        except ImportError:
            logger.debug("Metal not available")
        except Exception as e:
            logger.debug(f"Metal initialization failed: {e}")
        
        return False
    
    def _try_opencl(self) -> bool:
        """Try to initialize OpenCL backend."""
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            if platforms:
                # Get first GPU device
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        device = devices[0]
                        
                        self._gpu_info = GPUInfo(
                            backend=GPUBackend.OPENCL,
                            device_name=device.name,
                            memory_total=device.global_mem_size,
                            supports_fp16=device.extensions.find('cl_khr_fp16') >= 0,
                            supports_async=True
                        )
                        self._backend_module = cl
                        logger.info(f"OpenCL GPU detected: {self._gpu_info.device_name}")
                        return True
        except ImportError:
            logger.debug("PyOpenCL not available")
        except Exception as e:
            logger.debug(f"OpenCL initialization failed: {e}")
        
        return False
    
    @property
    def info(self) -> GPUInfo:
        """Get GPU information."""
        return self._gpu_info
    
    @property
    def backend(self) -> GPUBackend:
        """Get active GPU backend."""
        return self._gpu_info.backend
    
    @property
    def backend_module(self) -> Optional[Any]:
        """Get backend module (cupy, metal, pyopencl)."""
        return self._backend_module
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_info.backend != GPUBackend.NONE


# Global singleton instance
_gpu_capabilities = GPUCapabilities()


def gpu_available() -> bool:
    """Check if GPU acceleration is available.
    
    Returns:
        bool: True if GPU is available, False otherwise
        
    Example:
        >>> if gpu_available():
        ...     print("GPU acceleration enabled")
    """
    return _gpu_capabilities.is_available()


def get_gpu_info() -> GPUInfo:
    """Get information about the available GPU.
    
    Returns:
        GPUInfo: GPU device information
        
    Example:
        >>> info = get_gpu_info()
        >>> print(f"Using {info.backend.value}: {info.device_name}")
    """
    return _gpu_capabilities.info


def get_gpu_backend() -> GPUBackend:
    """Get the active GPU backend.
    
    Returns:
        GPUBackend: Active backend (CUDA, OpenCL, Metal, or NONE)
    """
    return _gpu_capabilities.backend


class GPUContext:
    """Context manager for GPU operations with automatic memory management.
    
    Handles data transfer to/from GPU and ensures proper cleanup.
    
    Example:
        >>> with GPUContext() as gpu:
        ...     result = gpu.gaussian_blur(image, sigma=2.0)
    """
    
    def __init__(self):
        """Initialize GPU context."""
        self.backend = _gpu_capabilities.backend
        self.backend_module = _gpu_capabilities.backend_module
        self._context = None
        self._queue = None
    
    def __enter__(self):
        """Enter GPU context."""
        if self.backend == GPUBackend.OPENCL:
            # Create OpenCL context and command queue
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        self._context = cl.Context([devices[0]])
                        self._queue = cl.CommandQueue(self._context)
                        break
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit GPU context and cleanup resources."""
        if self._queue is not None:
            self._queue.finish()
        
        if self.backend == GPUBackend.CUDA:
            # Force CUDA memory cleanup
            if self.backend_module:
                self.backend_module.get_default_memory_pool().free_all_blocks()
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Transfer array to GPU memory.
        
        Args:
            array: NumPy array to transfer
            
        Returns:
            GPU array object
        """
        if self.backend == GPUBackend.CUDA:
            return self.backend_module.asarray(array)
        elif self.backend == GPUBackend.OPENCL:
            import pyopencl as cl
            mf = cl.mem_flags
            return cl.Buffer(self._context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
        else:
            # Fallback - return original array
            return array
    
    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        """Transfer array from GPU to CPU memory.
        
        Args:
            gpu_array: GPU array to transfer
            
        Returns:
            NumPy array
        """
        if self.backend == GPUBackend.CUDA:
            return self.backend_module.asnumpy(gpu_array)
        elif self.backend == GPUBackend.OPENCL:
            # OpenCL buffer needs explicit read
            result = np.empty_like(gpu_array)
            import pyopencl as cl
            cl.enqueue_copy(self._queue, result, gpu_array)
            return result
        else:
            # Already on CPU
            return gpu_array if isinstance(gpu_array, np.ndarray) else np.asarray(gpu_array)
    
    def gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur using GPU acceleration.
        
        Args:
            image: Input image array
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred image
        """
        if self.backend == GPUBackend.CUDA:
            # Use CuPy's Gaussian filter
            from cupyx.scipy.ndimage import gaussian_filter
            gpu_img = self.to_gpu(image)
            gpu_result = gaussian_filter(gpu_img, sigma=sigma)
            return self.to_cpu(gpu_result)
        else:
            # Fallback to CPU
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image, sigma=sigma)
    
    def sobel_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection using GPU acceleration.
        
        Args:
            image: Input image array (grayscale)
            
        Returns:
            Edge-detected image
        """
        if self.backend == GPUBackend.CUDA:
            # Use CuPy's Sobel filter
            from cupyx.scipy.ndimage import sobel
            gpu_img = self.to_gpu(image)
            
            # Compute gradients in both directions
            sx = sobel(gpu_img, axis=0)
            sy = sobel(gpu_img, axis=1)
            
            # Compute magnitude
            gpu_result = self.backend_module.hypot(sx, sy)
            return self.to_cpu(gpu_result)
        else:
            # Fallback to CPU
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
    gpu_time: Optional[float] = None  # seconds
    cpu_time: Optional[float] = None  # seconds
    speedup: Optional[float] = None
    memory_transferred: Optional[int] = None  # bytes


def benchmark_operation(
    operation_name: str,
    gpu_func: Any,
    cpu_func: Any,
    test_image: np.ndarray,
    *args,
    **kwargs
) -> GPUBenchmark:
    """Benchmark GPU vs CPU performance for an operation.
    
    Args:
        operation_name: Name of the operation
        gpu_func: GPU function to benchmark
        cpu_func: CPU function to benchmark
        test_image: Test image array
        *args: Additional positional arguments for functions
        **kwargs: Additional keyword arguments for functions
        
    Returns:
        GPUBenchmark: Benchmark results
    """
    result = GPUBenchmark(
        operation=operation_name,
        backend=get_gpu_backend(),
        image_shape=test_image.shape
    )
    
    # Benchmark CPU
    try:
        start = time.perf_counter()
        cpu_result = cpu_func(test_image, *args, **kwargs)
        result.cpu_time = time.perf_counter() - start
    except Exception as e:
        logger.warning(f"CPU benchmark failed: {e}")
    
    # Benchmark GPU
    if gpu_available():
        try:
            start = time.perf_counter()
            gpu_result = gpu_func(test_image, *args, **kwargs)
            result.gpu_time = time.perf_counter() - start
            
            # Calculate speedup
            if result.cpu_time and result.gpu_time:
                result.speedup = result.cpu_time / result.gpu_time
            
            # Estimate memory transferred (to GPU + from GPU)
            result.memory_transferred = test_image.nbytes * 2
        except Exception as e:
            logger.warning(f"GPU benchmark failed: {e}")
    
    return result
