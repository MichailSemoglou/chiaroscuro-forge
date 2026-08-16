"""
Comprehensive tests for comparison module.

Tests cover:
- compare_processing_methods: multiple method comparison
- suggest_optimal_params: parameter recommendations
- Error handling and edge cases
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

from chiaroscuro_forge.comparison import compare_processing_methods, suggest_optimal_params
from chiaroscuro_forge.exceptions import ImageProcessingError


class TestCompareProcessingMethods(unittest.TestCase):
    """Test comparing different processing methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        test_img = np.random.rand(100, 100, 3)
        self.test_image_path = os.path.join(self.temp_dir, "test.png")
        io.imsave(self.test_image_path, img_as_ubyte(test_img))

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_compare_methods_basic(self):
        """Test basic method comparison without output."""
        results = compare_processing_methods(
            self.test_image_path, application_type="general", calculate_advanced_metrics=False
        )

        # Should have results for all methods
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 5)  # Multiple methods tested

        # Should identify best method
        self.assertIn("best_method", results)
        self.assertIn("name", results["best_method"])
        self.assertIn("score", results["best_method"])

    def test_compare_methods_with_output_dir(self):
        """Test comparison with output directory."""
        output_dir = os.path.join(self.temp_dir, "output")

        results = compare_processing_methods(
            self.test_image_path,
            output_dir=output_dir,
            application_type="photography",
            calculate_advanced_metrics=True,
        )

        # Output directory should be created
        self.assertTrue(os.path.exists(output_dir))

        # Should have output files
        output_files = os.listdir(output_dir)
        self.assertGreater(len(output_files), 0)

        # Check that results contain output paths
        for method_name, result in results.items():
            if method_name != "best_method" and "error" not in result:
                self.assertIn("output_path", result)
                self.assertIsNotNone(result["output_path"])

    def test_compare_methods_different_app_types(self):
        """Test comparison with different application types."""
        for app_type in ["general", "photography", "medical", "document", "art"]:
            results = compare_processing_methods(
                self.test_image_path, application_type=app_type, calculate_advanced_metrics=False
            )

            self.assertIn("best_method", results)

    def test_compare_methods_invalid_app_type(self):
        """Test error handling for invalid application type."""
        with self.assertRaises(ImageProcessingError) as ctx:
            compare_processing_methods(self.test_image_path, application_type="invalid_type")

        self.assertIn("Application type must be one of", str(ctx.exception))

    def test_compare_methods_invalid_image_path(self):
        """Test error handling for invalid image path."""
        with self.assertRaises(ImageProcessingError):
            compare_processing_methods("/nonexistent/image.jpg", application_type="general")

    def test_compare_methods_output_dir_creation_fails(self):
        """Test error when output directory creation fails."""
        # Use invalid path that cannot be created - works on both Unix and Windows
        if os.name == "nt":  # Windows
            # Use a path with invalid characters or a reserved path
            invalid_path = "CON:\\invalid_dir"  # CON is a reserved device name on Windows
        else:
            # On Unix, use a path without permission
            invalid_path = "/root/cannot_create_this_dir_without_permission"

        with self.assertRaises(ImageProcessingError) as ctx:
            compare_processing_methods(
                self.test_image_path, output_dir=invalid_path, application_type="general"
            )

        self.assertIn("Failed to create output directory", str(ctx.exception))

    def test_compare_methods_all_methods_processed(self):
        """Test that all comparison methods are attempted."""
        results = compare_processing_methods(
            self.test_image_path, application_type="general", calculate_advanced_metrics=True
        )

        # Should have multiple methods + best_method
        self.assertGreaterEqual(len(results), 8)  # At least 7 methods + best

        # Common method names
        expected_methods = [
            "Standard Equalization (HSV)",
            "LAB Color Preservation",
            "Gentle Enhancement",
            "Color Ratio Preservation",
        ]

        for method in expected_methods:
            # Method should be in results (unless error)
            if method in results:
                result = results[method]
                # Either has processed image or error
                self.assertTrue("processed" in result or "error" in result)

    def test_compare_methods_metrics_calculation(self):
        """Test that metrics are calculated for each method."""
        results = compare_processing_methods(
            self.test_image_path, application_type="photography", calculate_advanced_metrics=True
        )

        # Check that successful methods have metrics
        for method_name, result in results.items():
            if method_name != "best_method" and "error" not in result:
                self.assertIn("metrics", result)
                metrics = result["metrics"]
                if metrics:  # Some might be None if metrics calculation disabled
                    self.assertIn("quality_score", metrics)

    def test_compare_methods_skips_nonnumeric_quality_scores(self):
        """Nonnumeric scores should not turn successful processing into errors."""
        call_count = 0

        def process_with_mixed_scores(*args, **kwargs):
            nonlocal call_count
            quality_score = "invalid" if call_count == 0 else "0.75"
            call_count += 1
            return np.zeros((2, 2, 3)), {"quality_score": quality_score}

        with patch("chiaroscuro_forge.comparison.process_image", process_with_mixed_scores):
            results = compare_processing_methods(self.test_image_path)

        self.assertNotIn("error", results["Standard Equalization (HSV)"])
        self.assertEqual(results["best_method"]["score"], 0.75)


class TestSuggestOptimalParams(unittest.TestCase):
    """Test optimal parameter suggestions."""

    def test_suggest_params_bright_images(self):
        """Test parameter suggestions for bright images."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.8, "min": 0.7, "max": 0.9},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 8,
                "grayscale_images": 2,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        self.assertIn("params", suggestions)
        self.assertIn("application_type", suggestions)
        self.assertIn("batch_summary", suggestions)

        # Bright images should have higher gamma
        self.assertGreater(suggestions["params"]["gamma_correction"], 1.0)

    def test_suggest_params_dark_images(self):
        """Test parameter suggestions for dark images."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 8,
                "grayscale_images": 2,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Dark images should have lower gamma
        self.assertLess(suggestions["params"]["gamma_correction"], 1.0)

    def test_suggest_params_low_contrast(self):
        """Test parameter suggestions for low contrast images."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.1, "min": 0.05, "max": 0.15},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 8,
                "grayscale_images": 2,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Low contrast should use CLAHE
        self.assertEqual(suggestions["params"]["equalize_method"], "clahe")
        self.assertIn("clip_limit", suggestions["params"])

    def test_suggest_params_high_noise(self):
        """Test parameter suggestions for noisy images."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.08, "min": 0.06, "max": 0.1},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 8,
                "grayscale_images": 2,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # High noise should have higher denoise sigma
        self.assertGreater(suggestions["params"]["denoise_sigma"], 1.0)

    def test_suggest_params_grayscale_batch(self):
        """Test parameter suggestions for mostly grayscale images."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 2,
                "grayscale_images": 8,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Grayscale batch should not preserve color
        self.assertEqual(suggestions["params"]["color_preservation"], "none")
        self.assertEqual(suggestions["params"]["color_preservation_strength"], 0.0)

    def test_suggest_params_color_batch(self):
        """Test parameter suggestions for color images."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 9,
                "grayscale_images": 1,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Color batch should preserve color
        self.assertEqual(suggestions["params"]["color_preservation"], "lab")
        self.assertGreater(suggestions["params"]["color_preservation_strength"], 0.0)

    def test_suggest_params_high_edge_density(self):
        """Test parameter suggestions for high edge density (documents)."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.12, "min": 0.1, "max": 0.15},
                "color_images": 5,
                "grayscale_images": 5,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # High edge density should reduce sharpening
        self.assertLessEqual(suggestions["params"]["sharpen_amount"], 1.0)

        # Should detect as document
        self.assertEqual(suggestions["application_type"], "document")

    def test_suggest_params_low_edge_density(self):
        """Test parameter suggestions for low edge density."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.3, "min": 0.2, "max": 0.4},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.01, "min": 0.005, "max": 0.015},
                "color_images": 8,
                "grayscale_images": 2,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Low edge density should increase sharpening
        self.assertGreaterEqual(suggestions["params"]["sharpen_amount"], 1.5)

    def test_suggest_params_photography_detection(self):
        """Test detection of photography application type."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.25, "min": 0.2, "max": 0.3},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 9,
                "grayscale_images": 1,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Color images with good contrast should be photography
        self.assertEqual(suggestions["application_type"], "photography")

    def test_suggest_params_general_detection(self):
        """Test detection of general application type."""
        analysis_results = {
            "total_images": 10,
            "summary": {
                "brightness": {"avg": 0.5, "min": 0.4, "max": 0.6},
                "contrast": {"avg": 0.15, "min": 0.1, "max": 0.2},
                "noise_level": {"avg": 0.02, "min": 0.01, "max": 0.03},
                "edge_density": {"avg": 0.05, "min": 0.03, "max": 0.07},
                "color_images": 5,
                "grayscale_images": 5,
            },
        }

        suggestions = suggest_optimal_params(analysis_results)

        # Mixed characteristics should be general
        self.assertEqual(suggestions["application_type"], "general")


if __name__ == "__main__":
    unittest.main()
