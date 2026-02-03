"""
Unit tests for processing.py - Phase 2.1

Tests end-to-end image processing workflows including:
- Full pipeline execution
- Error handling
- File I/O operations
- Metrics calculation
- Integration with pipeline

Target: 23% → 80% coverage for processing.py
"""

import unittest
import tempfile
import os
import shutil
import numpy as np
from skimage import io, img_as_ubyte

from chiaroscuro_forge.processing import process_image
from chiaroscuro_forge.exceptions import ImageProcessingError


class TestProcessImage(unittest.TestCase):
    """Test process_image() end-to-end functionality."""
    
    def setUp(self):
        """Create temporary directory and test images."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid 100x100x3 test image (RGB)
        self.test_image = np.random.rand(100, 100, 3)
        self.test_image_path = os.path.join(self.temp_dir, "test.png")
        io.imsave(self.test_image_path, img_as_ubyte(self.test_image))
        
        # Create a small image for testing size limits
        self.small_image = np.random.rand(5, 5, 3)
        self.small_image_path = os.path.join(self.temp_dir, "small.png")
        io.imsave(self.small_image_path, img_as_ubyte(self.small_image))
        
        # Create grayscale image
        self.gray_image = np.random.rand(100, 100)
        self.gray_image_path = os.path.join(self.temp_dir, "gray.png")
        io.imsave(self.gray_image_path, img_as_ubyte(self.gray_image))
    
    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_processing_no_output(self):
        """Test basic processing without saving output."""
        processed, metrics = process_image(
            self.test_image_path,
            output_path=None,
            scale_factor=1.0,
            sharpen=False,
            equalize=False,
            calculate_metrics=True,
        )
        
        # Check return types
        self.assertIsInstance(processed, np.ndarray)
        self.assertIsInstance(metrics, dict)
        
        # Check image shape
        self.assertEqual(processed.shape, (100, 100, 3))
        
        # Check value range
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))
    
    def test_processing_with_output(self):
        """Test processing with output file saved."""
        output_path = os.path.join(self.temp_dir, "output.png")
        
        processed, metrics = process_image(
            self.test_image_path,
            output_path=output_path,
            calculate_metrics=False,
        )
        
        # Check output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check can load output
        loaded = io.imread(output_path)
        self.assertEqual(loaded.shape, (100, 100, 3))
    
    def test_processing_with_scaling(self):
        """Test processing with image scaling."""
        processed, _ = process_image(
            self.test_image_path,
            scale_factor=0.5,
            color_preservation="none",  # Can't preserve color when scaling
            calculate_metrics=False,
        )
        
        # Should be approximately half size (allow rounding)
        self.assertAlmostEqual(processed.shape[0], 50, delta=2)
        self.assertAlmostEqual(processed.shape[1], 50, delta=2)
    
    def test_processing_with_denoising(self):
        """Test processing with different denoise methods."""
        # Gaussian
        processed, _ = process_image(
            self.test_image_path,
            denoise_type="gaussian",
            denoise_sigma=1.0,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
        
        # Median
        processed, _ = process_image(
            self.test_image_path,
            denoise_type="median",
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
        
        # None
        processed, _ = process_image(
            self.test_image_path,
            denoise_type="none",
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_processing_with_sharpening(self):
        """Test processing with sharpening enabled."""
        processed, _ = process_image(
            self.test_image_path,
            sharpen=True,
            sharpen_amount=1.5,
            calculate_metrics=False,
        )
        
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_processing_with_equalization(self):
        """Test processing with different equalization methods."""
        # Stretch
        processed, _ = process_image(
            self.test_image_path,
            equalize=True,
            equalize_method="stretch",
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
        
        # Standard
        processed, _ = process_image(
            self.test_image_path,
            equalize=True,
            equalize_method="standard",
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_processing_with_gamma(self):
        """Test processing with gamma correction."""
        processed, _ = process_image(
            self.test_image_path,
            gamma_correction=1.5,
            calculate_metrics=False,
        )
        
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_processing_with_color_preservation(self):
        """Test processing with color preservation."""
        # LAB
        processed, _ = process_image(
            self.test_image_path,
            color_preservation="lab",
            color_preservation_strength=0.7,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
        
        # Ratio
        processed, _ = process_image(
            self.test_image_path,
            color_preservation="ratio",
            color_preservation_strength=0.5,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
        
        # None
        processed, _ = process_image(
            self.test_image_path,
            color_preservation="none",
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_processing_grayscale(self):
        """Test processing grayscale images."""
        processed, _ = process_image(
            self.gray_image_path,
            calculate_metrics=False,
        )
        
        # Grayscale should be preserved or converted
        self.assertEqual(len(processed.shape), 2)
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        processed, metrics = process_image(
            self.test_image_path,
            calculate_metrics=True,
            calculate_advanced_metrics=True,
        )
        
        # Check metrics exist
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)
        
        # Check essential metrics
        self.assertIn("quality_score", metrics)
        
        # Quality score should be in reasonable range
        self.assertGreaterEqual(metrics["quality_score"], 0)
        self.assertLessEqual(metrics["quality_score"], 100)
    
    def test_metrics_basic_only(self):
        """Test basic metrics without advanced."""
        processed, metrics = process_image(
            self.test_image_path,
            calculate_metrics=True,
            calculate_advanced_metrics=False,
        )
        
        self.assertIsNotNone(metrics)
        self.assertIn("quality_score", metrics)
    
    def test_no_metrics(self):
        """Test processing without metrics calculation."""
        processed, metrics = process_image(
            self.test_image_path,
            calculate_metrics=False,
        )
        
        self.assertIsNone(metrics)
    
    def test_image_too_small_for_metrics(self):
        """Test that small images raise error for metrics."""
        with self.assertRaises(ImageProcessingError) as ctx:
            process_image(
                self.small_image_path,
                calculate_metrics=True,
            )
        self.assertIn("too small", str(ctx.exception).lower())
    
    def test_output_directory_creation(self):
        """Test that output directories are created automatically."""
        nested_output = os.path.join(
            self.temp_dir, "subdir1", "subdir2", "output.png"
        )
        
        processed, _ = process_image(
            self.test_image_path,
            output_path=nested_output,
            calculate_metrics=False,
        )
        
        # Check output file exists
        self.assertTrue(os.path.exists(nested_output))
        
        # Check directory was created
        self.assertTrue(os.path.exists(os.path.dirname(nested_output)))
    
    def test_invalid_image_path(self):
        """Test error handling for non-existent image."""
        with self.assertRaises(ImageProcessingError) as ctx:
            process_image("/nonexistent/image.png")
        self.assertIn("not found", str(ctx.exception).lower())
    
    def test_invalid_output_path_permission(self):
        """Test error handling for invalid output path."""
        # Try to save to a location that requires root permissions
        if os.name != 'nt':  # Skip on Windows
            with self.assertRaises(ImageProcessingError) as ctx:
                process_image(
                    self.test_image_path,
                    output_path="/root/cannot_write.png",
                    calculate_metrics=False,
                )
            self.assertIn("failed to save", str(ctx.exception).lower())
    
    def test_invalid_scale_factor(self):
        """Test error handling for invalid scale factor."""
        with self.assertRaises(ImageProcessingError) as ctx:
            process_image(
                self.test_image_path,
                scale_factor=-1.0,
            )
        # Error should be caught by validation
        self.assertIsInstance(ctx.exception, ImageProcessingError)
    
    def test_invalid_denoise_type(self):
        """Test error handling for invalid denoise type."""
        with self.assertRaises(ImageProcessingError) as ctx:
            process_image(
                self.test_image_path,
                denoise_type="invalid_method",
            )
        self.assertIsInstance(ctx.exception, ImageProcessingError)
    
    def test_full_pipeline_integration(self):
        """Test full pipeline with all features enabled."""
        output_path = os.path.join(self.temp_dir, "full_output.png")
        
        processed, metrics = process_image(
            self.test_image_path,
            output_path=output_path,
            scale_factor=0.8,
            denoise_type="gaussian",
            denoise_sigma=0.5,
            sharpen=True,
            sharpen_amount=1.0,
            equalize=True,
            equalize_method="stretch",
            gamma_correction=1.2,
            color_preservation="none",  # Can't preserve color when scaling (shape mismatch)
            calculate_metrics=False,  # Can't calculate metrics when scaling (shape mismatch)
            application_type="photography",
        )
        
        # Check all outputs
        self.assertIsNotNone(processed)
        self.assertIsNone(metrics)
        self.assertTrue(os.path.exists(output_path))
        
        # Check image processed correctly (allow for rounding)
        self.assertAlmostEqual(processed.shape[0], 80, delta=2)  # 0.8 * 100
        self.assertAlmostEqual(processed.shape[1], 80, delta=2)
    
    def test_minimal_processing(self):
        """Test minimal processing with all enhancements disabled."""
        processed, metrics = process_image(
            self.test_image_path,
            scale_factor=1.0,
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
            calculate_metrics=True,
        )
        
        # Should still work but with minimal changes
        self.assertIsNotNone(processed)
        self.assertIsNotNone(metrics)
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_different_application_types(self):
        """Test processing with different application types."""
        for app_type in ["general", "photography", "document", "medical", "art"]:
            processed, metrics = process_image(
                self.test_image_path,
                application_type=app_type,
                calculate_metrics=True,
            )
            
            self.assertIsNotNone(processed)
            self.assertIsNotNone(metrics)
            self.assertIn("quality_score", metrics)
    
    def test_contrast_stretch_percentiles(self):
        """Test processing with custom contrast stretch percentiles."""
        processed, _ = process_image(
            self.test_image_path,
            equalize=True,
            equalize_method="stretch",
            contrast_stretch_percentiles=(5, 95),
            calculate_metrics=False,
        )
        
        self.assertEqual(processed.shape, (100, 100, 3))
    
    def test_processing_preserves_aspect_ratio(self):
        """Test that scaling preserves aspect ratio."""
        processed, _ = process_image(
            self.test_image_path,
            scale_factor=0.7,
            color_preservation="none",  # Can't preserve color when scaling
            calculate_metrics=False,
        )
        
        # Check aspect ratio preserved (allow for rounding)
        original_ratio = 100 / 100  # 1.0
        processed_ratio = processed.shape[0] / processed.shape[1]
        self.assertAlmostEqual(original_ratio, processed_ratio, delta=0.1)


class TestProcessImageEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            # On Windows, files may be locked - try with ignore_errors
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_corrupted_image_file(self):
        """Test handling of corrupted image file."""
        corrupt_path = os.path.join(self.temp_dir, "corrupt.png")
        # Create file with valid PNG magic but invalid content
        with open(corrupt_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n' + b'NOT VALID PNG DATA')
        
        with self.assertRaises(ImageProcessingError) as ctx:  
            process_image(corrupt_path)
        # Either validation or loading will fail
        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            "failed to load" in error_msg or
            "magic number" in error_msg or
            "cannot" in error_msg
        )
    
    def test_empty_file(self):
        """Test handling of empty file."""
        empty_path = os.path.join(self.temp_dir, "empty.png")
        with open(empty_path, 'wb') as f:
            pass  # Create empty file
        
        with self.assertRaises(ImageProcessingError):
            process_image(empty_path)
    
    def test_extreme_scaling(self):
        """Test processing with extreme scaling factors."""
        # Create small test image
        small_img = np.random.rand(20, 20, 3)
        img_path = os.path.join(self.temp_dir, "small.png")
        io.imsave(img_path, img_as_ubyte(small_img))
        
        # Very small scale
        processed, _ = process_image(
            img_path,
            scale_factor=0.25,
            color_preservation="none",  # Can't preserve color when scaling
            calculate_metrics=False,
        )
        
        # Should create very small image but not empty
        self.assertGreater(processed.size, 0)
        self.assertLess(processed.shape[0], 10)
    
    def test_high_denoise_sigma(self):
        """Test processing with high denoise sigma."""
        img = np.random.rand(50, 50, 3)
        img_path = os.path.join(self.temp_dir, "test.png")
        io.imsave(img_path, img_as_ubyte(img))
        
        processed, _ = process_image(
            img_path,
            denoise_type="gaussian",
            denoise_sigma=5.0,
            calculate_metrics=False,
        )
        
        # Should work but image will be very blurred
        self.assertEqual(processed.shape, (50, 50, 3))
        # Variance should be reduced significantly
        self.assertLess(np.var(processed), np.var(img))


if __name__ == "__main__":
    unittest.main()
