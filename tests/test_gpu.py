"""Tests for GPU acceleration module."""

import unittest
from unittest.mock import patch

import numpy as np

from chiaroscuro_forge.gpu import (
    GPUBackend,
    GPUBenchmark,
    GPUCapabilities,
    GPUContext,
    GPUInfo,
    _image_too_large_for_gpu,
    benchmark_operation,
    get_gpu_backend,
    get_gpu_info,
    gpu_available,
)


class TestGPUInfo(unittest.TestCase):
    """Tests for GPUInfo dataclass."""

    def test_create_gpu_info(self):
        """Test creating GPUInfo with all fields."""
        info = GPUInfo(
            backend=GPUBackend.CUDA,
            device_name="NVIDIA RTX 3080",
            compute_capability="8.6",
            memory_total=10 * 1024**3,  # 10GB
            memory_available=8 * 1024**3,  # 8GB
            supports_fp16=True,
            supports_async=True,
        )

        self.assertEqual(info.backend, GPUBackend.CUDA)
        self.assertEqual(info.device_name, "NVIDIA RTX 3080")
        self.assertEqual(info.compute_capability, "8.6")
        self.assertTrue(info.supports_fp16)
        self.assertTrue(info.supports_async)

    def test_create_minimal_gpu_info(self):
        """Test creating GPUInfo with minimal fields."""
        info = GPUInfo(backend=GPUBackend.NONE, device_name="CPU")

        self.assertEqual(info.backend, GPUBackend.NONE)
        self.assertEqual(info.device_name, "CPU")
        self.assertIsNone(info.compute_capability)
        self.assertFalse(info.supports_fp16)


class TestGPUBackend(unittest.TestCase):
    """Tests for GPUBackend enum."""

    def test_backend_values(self):
        """Test GPU backend enum values."""
        self.assertEqual(GPUBackend.NONE.value, "none")
        self.assertEqual(GPUBackend.CUDA.value, "cuda")
        self.assertEqual(GPUBackend.OPENCL.value, "opencl")
        self.assertEqual(GPUBackend.METAL.value, "metal")

    def test_backend_comparison(self):
        """Test comparing GPU backends."""
        self.assertNotEqual(GPUBackend.CUDA, GPUBackend.OPENCL)
        self.assertEqual(GPUBackend.CUDA, GPUBackend.CUDA)


class TestGPUCapabilities(unittest.TestCase):
    """Tests for GPUCapabilities singleton."""

    def test_singleton_pattern(self):
        """Test that GPUCapabilities is a singleton."""
        cap1 = GPUCapabilities()
        cap2 = GPUCapabilities()

        self.assertIs(cap1, cap2)

    def test_gpu_info_property(self):
        """Test GPU info property."""
        cap = GPUCapabilities()
        info = cap.info

        self.assertIsInstance(info, GPUInfo)
        self.assertIsInstance(info.backend, GPUBackend)

    def test_backend_property(self):
        """Test backend property."""
        cap = GPUCapabilities()
        backend = cap.backend

        self.assertIsInstance(backend, GPUBackend)

    def test_is_available(self):
        """Test GPU availability check."""
        cap = GPUCapabilities()
        available = cap.is_available()

        self.assertIsInstance(available, bool)
        # If GPU is available, backend should not be NONE
        if available:
            self.assertNotEqual(cap.backend, GPUBackend.NONE)
        else:
            self.assertEqual(cap.backend, GPUBackend.NONE)


class TestGPUCapabilitiesDetection(unittest.TestCase):
    """Tests for GPU backend detection."""

    @patch("chiaroscuro_forge.gpu.GPUCapabilities._try_cuda")
    @patch("chiaroscuro_forge.gpu.GPUCapabilities._try_metal")
    @patch("chiaroscuro_forge.gpu.GPUCapabilities._try_opencl")
    def test_detection_order(self, mock_opencl, mock_metal, mock_cuda):
        """Test that GPU detection tries backends in correct order."""
        mock_cuda.return_value = False
        mock_metal.return_value = False
        mock_opencl.return_value = False

        instance = GPUCapabilities()
        instance._detected = False
        instance._gpu_info = None
        instance._backend_module = None

        instance._ensure_detected()

        mock_cuda.assert_called_once_with()
        mock_metal.assert_called_once_with()
        mock_opencl.assert_called_once_with()

    @patch("chiaroscuro_forge.gpu.GPUCapabilities._try_cuda")
    @patch("chiaroscuro_forge.gpu.GPUCapabilities._try_metal")
    @patch("chiaroscuro_forge.gpu.GPUCapabilities._try_opencl")
    def test_cuda_detection_success(self, mock_opencl, mock_metal, mock_cuda):
        """Test successful CUDA detection stops further checks."""
        mock_cuda.return_value = True
        mock_metal.return_value = True
        mock_opencl.return_value = True

        instance = GPUCapabilities()
        instance._detected = False
        instance._gpu_info = None
        instance._backend_module = None

        try:
            instance._ensure_detected()

            mock_cuda.assert_called_once_with()
            mock_metal.assert_not_called()
            mock_opencl.assert_not_called()
        finally:
            instance._detected = False
            instance._gpu_info = None
            instance._backend_module = None


class TestGPUFunctions(unittest.TestCase):
    """Tests for module-level GPU functions."""

    def test_image_too_large_for_gpu_uses_pixel_count(self):
        """Thresholding should use height x width pixels rather than channel count."""
        with patch("chiaroscuro_forge.gpu.MAX_GPU_PIXELS", 10_000):
            image = np.ones((100, 100, 100), dtype=np.float32)
            self.assertFalse(_image_too_large_for_gpu(image))

    def test_gpu_available(self):
        """Test gpu_available function."""
        available = gpu_available()
        self.assertIsInstance(available, bool)

    def test_get_gpu_info(self):
        """Test get_gpu_info function."""
        info = get_gpu_info()

        self.assertIsInstance(info, GPUInfo)
        self.assertIsInstance(info.backend, GPUBackend)
        self.assertIsInstance(info.device_name, str)

    def test_get_gpu_backend(self):
        """Test get_gpu_backend function."""
        backend = get_gpu_backend()

        self.assertIsInstance(backend, GPUBackend)


class TestGPUContext(unittest.TestCase):
    """Tests for GPUContext context manager."""

    def test_context_manager(self):
        """Test GPUContext as context manager."""
        with GPUContext() as gpu:
            self.assertIsNotNone(gpu)
            self.assertIsInstance(gpu.backend, GPUBackend)

    def test_to_gpu_cpu_fallback(self):
        """Test to_gpu with CPU fallback."""
        test_array = np.random.rand(10, 10).astype(np.float32)

        with GPUContext() as gpu:
            # Should work even without GPU (fallback to CPU)
            gpu_array = gpu.to_gpu(test_array)
            self.assertIsNotNone(gpu_array)

    def test_to_cpu(self):
        """Test to_cpu conversion."""
        test_array = np.random.rand(10, 10).astype(np.float32)

        with GPUContext() as gpu:
            gpu_array = gpu.to_gpu(test_array)
            cpu_array = gpu.to_cpu(gpu_array)

            self.assertIsInstance(cpu_array, np.ndarray)
            np.testing.assert_array_almost_equal(cpu_array, test_array)

    def test_gaussian_blur(self):
        """Test GPU Gaussian blur."""
        test_image = np.random.rand(100, 100).astype(np.float32)

        with GPUContext() as gpu:
            result = gpu.gaussian_blur(test_image, sigma=2.0)

            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, test_image.shape)
            # Blurred image should have lower variance
            self.assertLess(np.var(result), np.var(test_image))

    def test_gaussian_blur_color(self):
        """Test GPU Gaussian blur on color image."""
        test_image = np.random.rand(100, 100, 3).astype(np.float32)

        with GPUContext() as gpu:
            result = gpu.gaussian_blur(test_image, sigma=2.0)

            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, test_image.shape)

    def test_sobel_filter(self):
        """Test GPU Sobel filter."""
        # Create test image with a vertical edge
        test_image = np.zeros((100, 100), dtype=np.float32)
        test_image[:, :50] = 1.0

        with GPUContext() as gpu:
            result = gpu.sobel_filter(test_image)

            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, test_image.shape)
            # Edge detection should produce non-zero values at the edge
            self.assertGreater(np.max(result), 0)

    def test_sobel_filter_random(self):
        """Test Sobel filter on random image."""
        test_image = np.random.rand(100, 100).astype(np.float32)

        with GPUContext() as gpu:
            result = gpu.sobel_filter(test_image)

            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, test_image.shape)


class TestGPUContextMemoryManagement(unittest.TestCase):
    """Tests for GPU memory management."""

    def test_context_cleanup(self):
        """Test that context manager cleans up resources."""
        test_array = np.random.rand(1000, 1000).astype(np.float32)

        # Create and exit context multiple times
        for _ in range(3):
            with GPUContext() as gpu:
                gpu_array = gpu.to_gpu(test_array)
                _ = gpu.to_cpu(gpu_array)

        # Should not raise memory errors
        self.assertTrue(True)

    def test_multiple_contexts(self):
        """Test creating multiple GPU contexts."""
        test_array = np.random.rand(100, 100).astype(np.float32)

        with GPUContext() as gpu1:
            result1 = gpu1.gaussian_blur(test_array, sigma=1.0)

        with GPUContext() as gpu2:
            result2 = gpu2.gaussian_blur(test_array, sigma=2.0)

        # Both should work independently
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        # Different sigma should give different results
        self.assertFalse(np.array_equal(result1, result2))


class TestGPUBenchmark(unittest.TestCase):
    """Tests for GPU benchmarking."""

    def test_create_benchmark(self):
        """Test creating GPUBenchmark."""
        benchmark = GPUBenchmark(
            operation="gaussian_blur",
            backend=GPUBackend.CUDA,
            image_shape=(1024, 1024, 3),
            gpu_time=0.05,
            cpu_time=0.5,
            speedup=10.0,
            memory_transferred=1024 * 1024 * 3 * 4 * 2,
        )

        self.assertEqual(benchmark.operation, "gaussian_blur")
        self.assertEqual(benchmark.backend, GPUBackend.CUDA)
        self.assertEqual(benchmark.speedup, 10.0)

    def test_benchmark_operation(self):
        """Test benchmarking GPU vs CPU operation."""
        test_image = np.random.rand(100, 100).astype(np.float32)

        def cpu_blur(img, sigma):
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(img, sigma=sigma)

        def gpu_blur(img, sigma):
            with GPUContext() as gpu:
                return gpu.gaussian_blur(img, sigma=sigma)

        result = benchmark_operation("gaussian_blur", gpu_blur, cpu_blur, test_image, sigma=2.0)

        self.assertIsInstance(result, GPUBenchmark)
        self.assertEqual(result.operation, "gaussian_blur")
        self.assertEqual(result.image_shape, test_image.shape)
        self.assertIsNotNone(result.cpu_time)
        self.assertGreater(result.cpu_time, 0)

    def test_benchmark_with_gpu_speedup(self):
        """Test benchmark calculates speedup when GPU is available."""
        test_image = np.random.rand(100, 100).astype(np.float32)

        def cpu_op(img):
            return img * 2

        def gpu_op(img):
            with GPUContext() as gpu:
                gpu_img = gpu.to_gpu(img)
                return gpu.to_cpu(gpu_img * 2)

        result = benchmark_operation("multiplication", gpu_op, cpu_op, test_image)

        if gpu_available():
            self.assertIsNotNone(result.gpu_time)
            if result.speedup:
                self.assertGreater(result.speedup, 0)


class TestGPUEdgeCases(unittest.TestCase):
    """Tests for GPU edge cases and error handling."""

    def test_empty_image(self):
        """Test GPU operations on empty image."""
        test_image = np.array([[]], dtype=np.float32)

        with GPUContext() as gpu:
            result = gpu.gaussian_blur(test_image, sigma=1.0)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 0))

    def test_single_pixel(self):
        """Test GPU operations on single pixel."""
        test_image = np.array([[1.0]], dtype=np.float32)

        with GPUContext() as gpu:
            result = gpu.gaussian_blur(test_image, sigma=0.5)
            self.assertEqual(result.shape, (1, 1))

    def test_large_sigma(self):
        """Test Gaussian blur with very large sigma."""
        test_image = np.random.rand(100, 100).astype(np.float32)

        with GPUContext() as gpu:
            result = gpu.gaussian_blur(test_image, sigma=50.0)

            # Large sigma should heavily blur the image
            self.assertLess(np.var(result), np.var(test_image) * 0.1)

    def test_zero_sigma(self):
        """Test Gaussian blur with zero sigma."""
        test_image = np.random.rand(50, 50).astype(np.float32)

        with GPUContext() as gpu:
            result = gpu.gaussian_blur(test_image, sigma=0.0)

            # Zero sigma should return nearly identical image
            np.testing.assert_array_almost_equal(result, test_image, decimal=5)


if __name__ == "__main__":
    unittest.main()
