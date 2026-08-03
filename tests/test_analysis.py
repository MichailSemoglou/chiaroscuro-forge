"""
Tests for analysis module - image characteristics and statistics.

Tests cover:
- analyze_image_characteristics: characteristic detection and parameter suggestions
- get_image_statistics: comprehensive image statistics
- Error handling
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

from chiaroscuro_forge.analysis import analyze_image_characteristics, get_image_statistics
from chiaroscuro_forge.exceptions import ImageProcessingError


class TestAnalyzeImageCharacteristics(unittest.TestCase):
    """Test image characteristic analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_analyze_bright_color_image(self):
        """Test analysis of bright color image."""
        # Create bright color image
        img = np.ones((100, 100, 3)) * 0.8
        path = os.path.join(self.temp_dir, "bright.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        self.assertIn("characteristics", result)
        self.assertIn("suggested_params", result)
        self.assertIn("suggested_application", result)

        # Bright image should have higher gamma
        self.assertGreater(result["suggested_params"]["gamma_correction"], 1.0)

    def test_analyze_dark_image(self):
        """Test analysis of dark image."""
        # Create dark image
        img = np.ones((100, 100, 3)) * 0.3
        path = os.path.join(self.temp_dir, "dark.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        # Dark image should have lower gamma
        self.assertLess(result["suggested_params"]["gamma_correction"], 1.0)

    def test_analyze_low_contrast_image(self):
        """Test analysis of low contrast image."""
        # Create low contrast image
        img = np.random.rand(100, 100, 3) * 0.2 + 0.4
        path = os.path.join(self.temp_dir, "low_contrast.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        # Low contrast should use clahe or stretch
        self.assertIn(result["suggested_params"]["equalize_method"], ["clahe", "stretch"])

    def test_analyze_high_noise_image(self):
        """Test analysis of noisy image."""
        # Create noisy image
        img = np.random.rand(100, 100, 3) * 0.3 + 0.4
        path = os.path.join(self.temp_dir, "noisy.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        # High noise should have denoise settings
        self.assertIn("denoise_type", result["suggested_params"])
        self.assertIn("denoise_sigma", result["suggested_params"])

    def test_analyze_grayscale_image(self):
        """Test analysis of grayscale image."""
        # Create grayscale image
        img = np.random.rand(100, 100) * 0.5 + 0.3
        path = os.path.join(self.temp_dir, "gray.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        # Grayscale should not preserve color
        self.assertEqual(result["characteristics"]["is_color"], False)
        self.assertEqual(result["suggested_params"]["color_preservation"], "none")

    def test_analyze_high_saturation_image(self):
        """Test analysis of high saturation color image."""
        # Create saturated color image
        img = np.zeros((100, 100, 3))
        img[:, :, 0] = 0.8  # High red
        img[:, :, 1] = 0.2  # Low green
        img[:, :, 2] = 0.1  # Low blue
        path = os.path.join(self.temp_dir, "saturated.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        # High saturation should preserve color strongly
        self.assertEqual(result["suggested_params"]["color_preservation"], "lab")
        self.assertGreater(result["suggested_params"]["color_preservation_strength"], 0.8)

    def test_analyze_invalid_path(self):
        """Test error handling for invalid path."""
        with self.assertRaises(ImageProcessingError):
            analyze_image_characteristics("/nonexistent/image.jpg")

    def test_suggested_application_detection(self):
        """Test application type suggestions."""
        # Create simple test image
        img = np.random.rand(100, 100, 3) * 0.5 + 0.3
        path = os.path.join(self.temp_dir, "test.png")
        io.imsave(path, img_as_ubyte(img))

        result = analyze_image_characteristics(path)

        # Should suggest one of the valid application types
        valid_apps = ["general", "photography", "document", "art", "medical"]
        self.assertIn(result["suggested_application"], valid_apps)


class TestGetImageStatistics(unittest.TestCase):
    """Test image statistics calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_statistics_from_path(self):
        """Test statistics from image path."""
        # Create test image
        img = np.random.rand(100, 100, 3)
        path = os.path.join(self.temp_dir, "test.png")
        io.imsave(path, img_as_ubyte(img))

        stats = get_image_statistics(path)

        # Should have basic statistics in nested structure
        self.assertIn("intensity", stats)
        self.assertIn("mean", stats["intensity"])
        self.assertIn("std", stats["intensity"])
        self.assertIn("min", stats["intensity"])
        self.assertIn("max", stats["intensity"])

    def test_statistics_from_array(self):
        """Test statistics from numpy array."""
        img = np.random.rand(100, 100, 3)

        stats = get_image_statistics(img)

        self.assertIn("intensity", stats)
        self.assertIn("mean", stats["intensity"])
        self.assertIn("std", stats["intensity"])
        self.assertIn("min", stats["intensity"])
        self.assertIn("max", stats["intensity"])

    def test_statistics_invalid_input(self):
        """Test error handling for invalid input."""
        with self.assertRaises(ImageProcessingError):
            get_image_statistics("/nonexistent/image.jpg")


if __name__ == "__main__":
    unittest.main()
