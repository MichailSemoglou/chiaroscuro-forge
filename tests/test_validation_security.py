"""
Unit tests for validation.py security features - Phase 2.1

Tests comprehensive security controls including:
- Path traversal prevention
- File size limits
- Dimension limits
- Extension whitelist
- Magic number verification

Target: 13% → 80% coverage for validation.py
"""

import unittest
import tempfile
import os
from pathlib import Path
import numpy as np

from chiaroscuro_forge.validation import (
    validate_array,
    validate_image_path,
    validate_processing_params,
)
from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.constants import (
    MAX_FILE_SIZE_MB,
    MAX_IMAGE_PIXELS,
    MAX_DIMENSION,
    MAX_PATH_LENGTH,
)


class TestValidateArray(unittest.TestCase):
    """Test validate_array() with security limits."""

    def test_valid_2d_grayscale(self):
        """Test valid 2D grayscale array."""
        arr = np.random.rand(100, 100)
        validate_array(arr, "grayscale")  # Should not raise

    def test_valid_3d_color(self):
        """Test valid 3D color array."""
        arr = np.random.rand(100, 100, 3)
        validate_array(arr, "color")  # Should not raise

    def test_not_numpy_array(self):
        """Test that non-numpy arrays are rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_array([[1, 2], [3, 4]], "list")
        self.assertIn("must be a numpy array", str(ctx.exception))

    def test_invalid_dimensions(self):
        """Test that 1D and 4D arrays are rejected."""
        arr_1d = np.array([1, 2, 3])
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_array(arr_1d, "1d")
        self.assertIn("must be 2D (grayscale) or 3D (color)", str(ctx.exception))

        arr_4d = np.random.rand(10, 10, 3, 2)
        with self.assertRaises(ImageProcessingError):
            validate_array(arr_4d, "4d")

    def test_dimension_limit(self):
        """Test MAX_DIMENSION security limit."""
        # Create array with one dimension exceeding limit
        huge_dim = MAX_DIMENSION + 1
        arr = np.zeros((huge_dim, 10))

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_array(arr, "huge")
        self.assertIn("dimension 0 too large", str(ctx.exception))
        # Check for formatted or unformatted number
        self.assertTrue(
            str(MAX_DIMENSION) in str(ctx.exception) or f"{MAX_DIMENSION:,}" in str(ctx.exception)
        )

    def test_pixel_count_limit(self):
        """Test MAX_IMAGE_PIXELS security limit."""
        # Create array exceeding pixel count
        # Use sqrt to create manageable dimensions
        side = int(np.sqrt(MAX_IMAGE_PIXELS)) + 100
        arr = np.zeros((side, side))

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_array(arr, "too_many_pixels")
        self.assertIn("too large", str(ctx.exception))
        self.assertIn("pixels", str(ctx.exception))
        self.assertIn("memory bomb", str(ctx.exception).lower())

    def test_nan_values(self):
        """Test that NaN values are detected."""
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]])

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_array(arr, "nan_arr")
        self.assertIn("NaN", str(ctx.exception))

    def test_inf_values(self):
        """Test that Inf values are detected."""
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_array(arr, "inf_arr")
        self.assertIn("Inf", str(ctx.exception))

    def test_custom_limits(self):
        """Test custom max_pixels and max_dimension parameters."""
        arr = np.zeros((1000, 1000))  # 1M pixels

        # Should pass with default limits
        validate_array(arr, "normal")

        # Should fail with custom low limit
        with self.assertRaises(ImageProcessingError):
            validate_array(arr, "limited", max_pixels=500_000)

        # Test custom dimension limit
        with self.assertRaises(ImageProcessingError):
            validate_array(arr, "limited", max_dimension=500)


class TestValidateImagePathSecurity(unittest.TestCase):
    """Test _validate_image_path() security features."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create valid test image (PNG)
        self.valid_png = os.path.join(self.temp_dir, "test.png")
        with open(self.valid_png, "wb") as f:
            # PNG magic number + minimal header
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)

        # Create valid JPEG
        self.valid_jpg = os.path.join(self.temp_dir, "test.jpg")
        with open(self.valid_jpg, "wb") as f:
            # JPEG magic number
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 28)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_valid_png(self):
        """Test valid PNG file passes validation."""
        validate_image_path(self.valid_png)  # Should not raise

    def test_valid_jpeg(self):
        """Test valid JPEG file passes validation."""
        validate_image_path(self.valid_jpg)  # Should not raise

    def test_empty_path(self):
        """Test empty path is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_path_too_long(self):
        """Test MAX_PATH_LENGTH limit."""
        long_path = "a" * (MAX_PATH_LENGTH + 1)

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path(long_path)
        self.assertIn("Path too long", str(ctx.exception))
        self.assertIn(str(MAX_PATH_LENGTH), str(ctx.exception))

    def test_path_traversal_dotdot(self):
        """Test path traversal with ../ is blocked."""
        traversal_paths = [
            "../etc/passwd",
            "../../secret.jpg",
            "subdir/../../../etc/hosts",
            "../../../../../etc/passwd",
        ]

        for path in traversal_paths:
            with self.assertRaises(ImageProcessingError) as ctx:
                validate_image_path(path)
            self.assertIn("denied pattern", str(ctx.exception).lower())

    def test_path_traversal_tilde(self):
        """Test path traversal with ~ is blocked."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path("~/sensitive.jpg")
        self.assertIn("denied pattern", str(ctx.exception).lower())

    def test_path_traversal_null_byte(self):
        """Test null byte injection is blocked."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path("test.jpg\x00.png")
        self.assertIn("denied pattern", str(ctx.exception).lower())

    def test_file_not_found(self):
        """Test non-existent file is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path("/nonexistent/image.jpg")
        self.assertIn("not found", str(ctx.exception))

    def test_directory_not_file(self):
        """Test directory path is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path(self.temp_dir)
        self.assertIn("not a regular file", str(ctx.exception))

    def test_file_size_limit(self):
        """Test MAX_FILE_SIZE_MB limit."""
        # Create file larger than limit
        large_file = os.path.join(self.temp_dir, "huge.jpg")
        size_bytes = int((MAX_FILE_SIZE_MB + 1) * 1024 * 1024)

        with open(large_file, "wb") as f:
            # Write JPEG magic number first
            f.write(b"\xff\xd8\xff\xe0")
            # Then fill with zeros to reach size
            f.write(b"\x00" * (size_bytes - 4))

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path(large_file)
        self.assertIn("too large", str(ctx.exception))
        self.assertIn("MB", str(ctx.exception))
        self.assertIn("DoS", str(ctx.exception))

    def test_extension_whitelist(self):
        """Test extension whitelist validation."""
        # Create file with disallowed extension
        bad_ext = os.path.join(self.temp_dir, "test.exe")
        with open(bad_ext, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path(bad_ext)
        self.assertIn("not allowed", str(ctx.exception))
        self.assertIn(".exe", str(ctx.exception))

    def test_allowed_extensions(self):
        """Test all allowed extensions pass."""
        allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"]

        for ext in allowed_exts:
            filename = f"test{ext}"
            filepath = os.path.join(self.temp_dir, filename)

            # Create file with appropriate magic number
            with open(filepath, "wb") as f:
                if ext in [".jpg", ".jpeg"]:
                    f.write(b"\xff\xd8\xff\xe0")
                elif ext == ".png":
                    f.write(b"\x89PNG\r\n\x1a\n")
                elif ext in [".tif", ".tiff"]:
                    f.write(b"II*\x00")  # TIFF little-endian
                elif ext == ".bmp":
                    f.write(b"BM")
                elif ext == ".gif":
                    f.write(b"GIF89a")
                elif ext == ".webp":
                    f.write(b"RIFF\x00\x00\x00\x00WEBP")

                # Pad to minimum size
                f.write(b"\x00" * 24)

            # Should not raise
            try:
                validate_image_path(filepath)
            except ImageProcessingError as e:
                self.fail(f"Valid {ext} file failed validation: {e}")

    def test_magic_number_jpeg(self):
        """Test JPEG magic number verification."""
        fake_jpeg = os.path.join(self.temp_dir, "fake.jpg")
        with open(fake_jpeg, "wb") as f:
            f.write(b"FAKE IMAGE DATA")

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path(fake_jpeg)
        self.assertIn("magic number", str(ctx.exception))

    def test_magic_number_png(self):
        """Test PNG magic number verification."""
        fake_png = os.path.join(self.temp_dir, "fake.png")
        with open(fake_png, "wb") as f:
            f.write(b"NOT A PNG FILE")

        with self.assertRaises(ImageProcessingError) as ctx:
            validate_image_path(fake_png)
        self.assertIn("magic number", str(ctx.exception))

    def test_public_api(self):
        """Test public validate_image_path() wrapper."""
        # Should work same as _validate_image_path
        validate_image_path(self.valid_png)  # Should not raise

        with self.assertRaises(ImageProcessingError):
            validate_image_path("../etc/passwd")


class TestValidateProcessingParams(unittest.TestCase):
    """Test validate_processing_params() parameter validation."""

    def test_valid_params(self):
        """Test validation with all valid parameters."""
        params = {
            "scale_factor": 1.0,
            "order": 1,
            "order_rescale": 1,
            "order_rotate": 1,
            "denoise_type": "gaussian",
            "denoise_sigma": 1.0,
            "sharpen": False,
            "sharpen_amount": 1.0,
            "equalize": False,
            "equalize_method": "stretch",
            "clip_limit": 0.03,
            "clip_limit_kernel_size": 8,
            "contrast_stretch_percentiles": (2, 98),
            "gamma_correction": 1.0,
            "color_preservation": "none",
            "color_preservation_strength": 0.7,
            "application_type": "general",
        }

        # Should not raise
        validate_processing_params(**params)

    def test_negative_scale_factor(self):
        """Test negative scale factor is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_processing_params(
                scale_factor=-1.0,
                order=1,
                order_rescale=1,
                order_rotate=1,
                denoise_type="gaussian",
                denoise_sigma=1.0,
                sharpen=False,
                sharpen_amount=1.0,
                equalize=False,
                equalize_method="stretch",
                clip_limit=0.03,
                clip_limit_kernel_size=8,
                contrast_stretch_percentiles=(2, 98),
                gamma_correction=1.0,
                color_preservation="none",
                color_preservation_strength=0.7,
                application_type="general",
            )
        self.assertIn("positive", str(ctx.exception).lower())

    def test_invalid_denoise_type(self):
        """Test invalid denoise type is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_processing_params(
                scale_factor=1.0,
                order=1,
                order_rescale=1,
                order_rotate=1,
                denoise_type="invalid_method",
                denoise_sigma=1.0,
                sharpen=False,
                sharpen_amount=1.0,
                equalize=False,
                equalize_method="stretch",
                clip_limit=0.03,
                clip_limit_kernel_size=8,
                contrast_stretch_percentiles=(2, 98),
                gamma_correction=1.0,
                color_preservation="none",
                color_preservation_strength=0.7,
                application_type="general",
            )
        self.assertIn("denoise", str(ctx.exception).lower())

    def test_invalid_equalize_method(self):
        """Test invalid equalize method is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_processing_params(
                scale_factor=1.0,
                order=1,
                order_rescale=1,
                order_rotate=1,
                denoise_type="gaussian",
                denoise_sigma=1.0,
                sharpen=False,
                sharpen_amount=1.0,
                equalize=True,
                equalize_method="invalid",
                clip_limit=0.03,
                clip_limit_kernel_size=8,
                contrast_stretch_percentiles=(2, 98),
                gamma_correction=1.0,
                color_preservation="none",
                color_preservation_strength=0.7,
                application_type="general",
            )
        self.assertIn("equalize", str(ctx.exception).lower())

    def test_invalid_color_preservation(self):
        """Test invalid color preservation is rejected."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_processing_params(
                scale_factor=1.0,
                order=1,
                order_rescale=1,
                order_rotate=1,
                denoise_type="gaussian",
                denoise_sigma=1.0,
                sharpen=False,
                sharpen_amount=1.0,
                equalize=False,
                equalize_method="stretch",
                clip_limit=0.03,
                clip_limit_kernel_size=8,
                contrast_stretch_percentiles=(2, 98),
                gamma_correction=1.0,
                color_preservation="invalid_method",
                color_preservation_strength=0.7,
                application_type="general",
            )
        self.assertIn("color preservation", str(ctx.exception).lower())

    def test_invalid_percentiles(self):
        """Test invalid contrast stretch percentiles."""
        with self.assertRaises(ImageProcessingError) as ctx:
            validate_processing_params(
                scale_factor=1.0,
                order=1,
                order_rescale=1,
                order_rotate=1,
                denoise_type="gaussian",
                denoise_sigma=1.0,
                sharpen=False,
                sharpen_amount=1.0,
                equalize=False,
                equalize_method="stretch",
                clip_limit=0.03,
                clip_limit_kernel_size=8,
                contrast_stretch_percentiles=(98, 2),  # Reversed!
                gamma_correction=1.0,
                color_preservation="none",
                color_preservation_strength=0.7,
                application_type="general",
            )
        self.assertIn("percentiles", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
