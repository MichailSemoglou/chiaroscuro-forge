"""Tests for the batch processing module."""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
from PIL import Image

from chiaroscuro_forge.batch import (
    _process_single_image,
    _process_single_image_wrapper,
    analyze_batch,
    batch_process_images,
    setup_logger,
)
from chiaroscuro_forge.exceptions import ImageProcessingError


class TestSetupLogger(unittest.TestCase):
    """Tests for logger setup."""

    def test_logger_without_file(self):
        """Test logger setup without log file."""
        logger = setup_logger()
        self.assertEqual(logger.name, "batch_processor")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)  # Only console handler
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_with_file(self):
        """Test logger setup with log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logger(log_file=log_file, log_level=logging.DEBUG)

            self.assertEqual(logger.level, logging.DEBUG)
            self.assertEqual(len(logger.handlers), 2)  # Console + file handlers

            # Close handlers to release file lock on Windows
            for handler in logger.handlers:
                handler.close()
            logger.handlers = []

    def test_logger_clears_existing_handlers(self):
        """Test that logger clears existing handlers."""
        logger1 = setup_logger()
        logger2 = setup_logger()
        # Should only have one console handler after second setup
        self.assertEqual(len(logger2.handlers), 1)


class TestProcessSingleImage(unittest.TestCase):
    """Tests for single image processing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "test_input.jpg")
        self.output_path = os.path.join(self.temp_dir.name, "test_output.jpg")

        # Create a test image
        image = Image.new("RGB", (100, 100), color="white")
        image.save(self.input_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_process_single_image_success(self):
        """Test successful single image processing."""
        result = _process_single_image(self.input_path, self.output_path, {}, "general")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output_path"], self.output_path)
        self.assertIn("metrics", result)
        self.assertTrue(os.path.exists(self.output_path))

    def test_process_single_image_with_logger(self):
        """Test single image processing with logger."""
        logger = setup_logger()
        with patch.object(logger, "info") as mock_info:
            result = _process_single_image(
                self.input_path, self.output_path, {}, "general", logger=logger
            )

            self.assertEqual(result["status"], "success")
            mock_info.assert_called_once()

    def test_process_single_image_with_params(self):
        """Test single image processing with custom parameters."""
        params = {"denoise_sigma": 2.0, "sharpen_amount": 2.0}
        result = _process_single_image(self.input_path, self.output_path, params, "photography")

        self.assertEqual(result["status"], "success")

    def test_process_single_image_error_handling(self):
        """Test error handling in single image processing."""
        with self.assertRaises(ImageProcessingError):
            _process_single_image("nonexistent.jpg", self.output_path, {}, "general")

    def test_process_single_image_wrapper_success(self):
        """Test wrapper function with successful processing."""
        result = _process_single_image_wrapper(self.input_path, self.output_path, {}, "general")

        self.assertEqual(result["status"], "success")
        self.assertNotIn("error", result)

    def test_process_single_image_wrapper_error(self):
        """Test wrapper function with error."""
        result = _process_single_image_wrapper("nonexistent.jpg", self.output_path, {}, "general")

        self.assertIn("error", result)
        self.assertIsInstance(result["error"], str)


class TestBatchProcessImages(unittest.TestCase):
    """Tests for batch processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "input")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.input_dir)

        # Create test images
        for i in range(3):
            image = Image.new("RGB", (100, 100), color="white")
            image.save(os.path.join(self.input_dir, f"test_{i}.jpg"))

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_batch_process_sequential(self):
        """Test sequential batch processing."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        results = batch_process_images(
            pattern,
            self.output_dir,
            n_workers=1,
            generate_report=False,
        )

        self.assertEqual(results["total"], 3)
        self.assertEqual(results["successful"], 3)
        self.assertEqual(results["failed"], 0)
        self.assertEqual(results["skipped"], 0)
        self.assertGreater(results["processing_time"], 0)

    def test_batch_process_parallel(self):
        """Test parallel batch processing."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        results = batch_process_images(
            pattern,
            self.output_dir,
            n_workers=2,
            generate_report=False,
        )

        self.assertEqual(results["total"], 3)
        self.assertEqual(results["successful"], 3)
        self.assertEqual(results["failed"], 0)

    def test_batch_process_with_params(self):
        """Test batch processing with custom parameters."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        params = {"denoise_sigma": 2.0, "sharpen_amount": 2.0}
        results = batch_process_images(
            pattern,
            self.output_dir,
            params=params,
            n_workers=1,
            generate_report=False,
        )

        self.assertEqual(results["successful"], 3)

    def test_batch_process_with_preset(self):
        """Test batch processing with preset."""
        # Create a preset
        from chiaroscuro_forge.presets import save_preset

        preset_params = {"denoise_sigma": 2.0, "sharpen_amount": 2.0}
        save_preset("test_batch_preset", preset_params, "Test preset")

        pattern = os.path.join(self.input_dir, "*.jpg")
        results = batch_process_images(
            pattern,
            self.output_dir,
            preset_name="test_batch_preset",
            n_workers=1,
            generate_report=False,
        )

        self.assertEqual(results["successful"], 3)

    def test_batch_process_skip_existing(self):
        """Test skipping existing files in batch processing."""
        pattern = os.path.join(self.input_dir, "*.jpg")

        # First run
        results1 = batch_process_images(
            pattern,
            self.output_dir,
            n_workers=1,
            generate_report=False,
        )

        # Second run with skip_existing
        results2 = batch_process_images(
            pattern,
            self.output_dir,
            n_workers=1,
            skip_existing=True,
            generate_report=False,
        )

        self.assertEqual(results1["successful"], 3)
        self.assertEqual(results2["skipped"], 3)
        self.assertEqual(results2["successful"], 0)

    def test_batch_process_generate_report(self):
        """Test report generation in batch processing."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        results = batch_process_images(
            pattern,
            self.output_dir,
            n_workers=1,
            generate_report=True,
        )

        report_path = os.path.join(self.output_dir, "batch_processing_report.json")
        self.assertTrue(os.path.exists(report_path))

        with open(report_path) as f:
            report_data = json.load(f)

        self.assertEqual(report_data["total"], 3)
        self.assertEqual(report_data["successful"], 3)

    def test_batch_process_with_log_file(self):
        """Test batch processing with log file."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        log_file = os.path.join(self.temp_dir.name, "batch.log")

        results = batch_process_images(
            pattern,
            self.output_dir,
            n_workers=1,
            generate_report=False,
            log_file=log_file,
        )

        self.assertEqual(results["successful"], 3)
        self.assertTrue(os.path.exists(log_file))

        # Close log handlers to release file lock on Windows
        import logging

        logger = logging.getLogger("batch_processor")
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

    def test_batch_process_no_files_found(self):
        """Test error when no files match pattern."""
        pattern = os.path.join(self.input_dir, "*.nonexistent")

        with self.assertRaises(ImageProcessingError) as cm:
            batch_process_images(
                pattern,
                self.output_dir,
                n_workers=1,
                generate_report=False,
            )

        self.assertIn("No files found", str(cm.exception))

    def test_batch_process_invalid_preset(self):
        """Test error with invalid preset."""
        pattern = os.path.join(self.input_dir, "*.jpg")

        with self.assertRaises(ImageProcessingError):
            batch_process_images(
                pattern,
                self.output_dir,
                preset_name="nonexistent_preset",
                n_workers=1,
                generate_report=False,
            )

    def test_batch_process_creates_output_dir(self):
        """Test that output directory is created if it doesn't exist."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        new_output_dir = os.path.join(self.temp_dir.name, "new_output")

        results = batch_process_images(
            pattern,
            new_output_dir,
            n_workers=1,
            generate_report=False,
        )

        self.assertTrue(os.path.exists(new_output_dir))
        self.assertEqual(results["successful"], 3)

    def test_batch_process_application_types(self):
        """Test batch processing with different application types."""
        pattern = os.path.join(self.input_dir, "*.jpg")

        for app_type in ["general", "photography", "medical"]:
            output_dir = os.path.join(self.temp_dir.name, f"output_{app_type}")
            results = batch_process_images(
                pattern,
                output_dir,
                application_type=app_type,
                n_workers=1,
                generate_report=False,
            )

            self.assertEqual(results["successful"], 3)


class TestAnalyzeBatch(unittest.TestCase):
    """Tests for batch analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "input")
        os.makedirs(self.input_dir)

        # Create test images with different characteristics
        for i, brightness in enumerate([50, 128, 200]):
            image = Image.new("RGB", (100, 100), color=(brightness, brightness, brightness))
            image.save(os.path.join(self.input_dir, f"test_{i}.jpg"))

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_analyze_batch_basic(self):
        """Test basic batch analysis."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        results = analyze_batch(pattern)

        self.assertEqual(results["total_images"], 3)
        self.assertEqual(len(results["analyses"]), 3)
        self.assertIn("summary", results)

    def test_analyze_batch_summary_statistics(self):
        """Test summary statistics in batch analysis."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        results = analyze_batch(pattern)

        summary = results["summary"]
        self.assertIn("brightness", summary)
        self.assertIn("contrast", summary)
        self.assertIn("noise_level", summary)
        self.assertIn("edge_density", summary)

        # Check that averages are calculated
        for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
            self.assertIn("avg", summary[metric])
            self.assertIn("min", summary[metric])
            self.assertIn("max", summary[metric])

    def test_analyze_batch_with_output_file(self):
        """Test batch analysis with output file."""
        pattern = os.path.join(self.input_dir, "*.jpg")
        output_file = os.path.join(self.temp_dir.name, "analysis.json")

        results = analyze_batch(pattern, output_file)

        self.assertTrue(os.path.exists(output_file))

        with open(output_file) as f:
            saved_results = json.load(f)

        self.assertEqual(saved_results["total_images"], results["total_images"])

    def test_analyze_batch_no_files_found(self):
        """Test error when no files match pattern."""
        pattern = os.path.join(self.input_dir, "*.nonexistent")

        with self.assertRaises(ImageProcessingError) as cm:
            analyze_batch(pattern)

        self.assertIn("No files found", str(cm.exception))

    def test_analyze_batch_color_counting(self):
        """Test color image counting in batch analysis."""
        # Add a grayscale image
        gray_image = Image.new("L", (100, 100), color=128)
        gray_image.save(os.path.join(self.input_dir, "gray.jpg"))

        pattern = os.path.join(self.input_dir, "*.jpg")
        results = analyze_batch(pattern)

        # Should have 3 color images and 1 grayscale
        self.assertEqual(results["total_images"], 4)
        # All images converted to RGB by PIL, so all counted as color
        self.assertGreaterEqual(results["summary"]["color_images"], 3)

    def test_analyze_batch_error_handling(self):
        """Test error handling for invalid files in batch analysis."""
        # Create an invalid file
        invalid_file = os.path.join(self.input_dir, "invalid.jpg")
        with open(invalid_file, "w") as f:
            f.write("not an image")

        pattern = os.path.join(self.input_dir, "*.jpg")
        results = analyze_batch(pattern)

        # Should still process valid images
        self.assertGreater(results["total_images"], 0)

        # Check that error is recorded
        error_found = any("error" in analysis for analysis in results["analyses"].values())
        self.assertTrue(error_found)


if __name__ == "__main__":
    unittest.main()
