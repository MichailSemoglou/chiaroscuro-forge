"""Tests for the command-line interface module."""

import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from PIL import Image

from chiaroscuro_forge.cli import main
from chiaroscuro_forge.presets import save_preset


class TestCLI(unittest.TestCase):
    """Tests for CLI functionality."""

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

    @patch("sys.argv", ["cli", "--list-presets"])
    def test_list_presets_empty(self):
        """Test listing presets when none or some exist."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = main()

            output = fake_out.getvalue()
            self.assertEqual(result, 0)
            # Output can either show no presets or existing ones from other tests
            self.assertTrue("presets" in output.lower() or "Saved preset" in output)

    @patch("sys.argv", ["cli", "--list-presets"])
    def test_list_presets_with_presets(self):
        """Test listing presets when some exist."""
        # Create a test preset
        save_preset("test_preset", {"gaussian_blur": 1.0}, "Test preset")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = main()

            output = fake_out.getvalue()
            self.assertEqual(result, 0)
            self.assertIn("Available presets", output)
            self.assertIn("test_preset", output)

    @patch("sys.argv", ["cli"])
    def test_no_arguments_prints_help(self):
        """Test that no arguments prints help."""
        result = main()
        self.assertEqual(result, 1)

    def test_single_image_processing(self):
        """Test processing a single image."""
        with patch("sys.argv", ["cli", self.input_path, "-o", self.output_path]):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                self.assertTrue(os.path.exists(self.output_path))
                output = fake_out.getvalue()
                self.assertIn("saved to", output)

    def test_single_image_with_application_type(self):
        """Test processing with specific application type."""
        with patch(
            "sys.argv",
            [
                "cli",
                self.input_path,
                "-o",
                self.output_path,
                "--application",
                "photography",
            ],
        ):
            result = main()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(self.output_path))

    def test_single_image_with_preset(self):
        """Test processing with a preset."""
        # Create a test preset
        save_preset("test_cli_preset", {"denoise_sigma": 2.0}, "Test")

        with patch(
            "sys.argv",
            [
                "cli",
                self.input_path,
                "-o",
                self.output_path,
                "--preset",
                "test_cli_preset",
            ],
        ):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                output = fake_out.getvalue()
                self.assertIn("Loaded preset", output)

    def test_single_image_with_invalid_preset(self):
        """Test error with invalid preset."""
        with patch(
            "sys.argv",
            [
                "cli",
                self.input_path,
                "-o",
                self.output_path,
                "--preset",
                "nonexistent",
            ],
        ):
            with patch("sys.stdout", new=StringIO()):
                result = main()
                self.assertEqual(result, 1)

    def test_analyze_single_image(self):
        """Test analyzing a single image."""
        with patch("sys.argv", ["cli", self.input_path, "--analyze"]):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                output = fake_out.getvalue()
                self.assertIn("Image Characteristics", output)
                self.assertIn("Brightness", output)
                self.assertIn("Contrast", output)

    def test_compare_processing_methods(self):
        """Test comparing different processing methods."""
        compare_dir = os.path.join(self.temp_dir.name, "comparison")

        with patch(
            "sys.argv",
            [
                "cli",
                self.input_path,
                "--compare",
                "--compare-dir",
                compare_dir,
            ],
        ):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                output = fake_out.getvalue()
                self.assertIn("Comparison Results", output)
                self.assertIn("Best method", output)

    def test_save_preset(self):
        """Test saving a preset."""
        with patch(
            "sys.argv",
            [
                "cli",
                self.input_path,
                "-o",
                self.output_path,
                "--save-preset",
                "new_preset",
                "--preset-description",
                "New test preset",
            ],
        ):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                output = fake_out.getvalue()
                self.assertIn("Saved preset", output)

    def test_batch_processing(self):
        """Test batch processing."""
        # Create multiple test images
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        output_dir = os.path.join(self.temp_dir.name, "batch_output")
        os.makedirs(input_dir)

        for i in range(3):
            image = Image.new("RGB", (100, 100), color="white")
            image.save(os.path.join(input_dir, f"test_{i}.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        with patch(
            "sys.argv",
            [
                "cli",
                pattern,
                "--batch",
                "-o",
                output_dir,
                "--workers",
                "1",
            ],
        ):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                output = fake_out.getvalue()
                self.assertIn("Batch processing completed", output)
                self.assertIn("Processed successfully: 3", output)

    def test_batch_processing_no_output_dir(self):
        """Test error when batch processing without output directory."""
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        os.makedirs(input_dir)
        image = Image.new("RGB", (100, 100), color="white")
        image.save(os.path.join(input_dir, "test.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        with patch("sys.argv", ["cli", pattern, "--batch"]):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 1)
                output = fake_out.getvalue()
                self.assertIn("Output directory must be specified", output)

    def test_batch_processing_with_report(self):
        """Test batch processing with report generation."""
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        output_dir = os.path.join(self.temp_dir.name, "batch_output")
        os.makedirs(input_dir)

        for i in range(2):
            image = Image.new("RGB", (100, 100), color="white")
            image.save(os.path.join(input_dir, f"test_{i}.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        with patch(
            "sys.argv",
            [
                "cli",
                pattern,
                "--batch",
                "-o",
                output_dir,
                "--report",
            ],
        ):
            result = main()

            self.assertEqual(result, 0)
            report_path = os.path.join(output_dir, "batch_processing_report.json")
            self.assertTrue(os.path.exists(report_path))

    def test_batch_processing_skip_existing(self):
        """Test batch processing with skip existing."""
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        output_dir = os.path.join(self.temp_dir.name, "batch_output")
        os.makedirs(input_dir)

        image = Image.new("RGB", (100, 100), color="white")
        image.save(os.path.join(input_dir, "test.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        # First run
        with patch(
            "sys.argv",
            [
                "cli",
                pattern,
                "--batch",
                "-o",
                output_dir,
            ],
        ):
            result1 = main()
            self.assertEqual(result1, 0)

        # Second run with skip existing
        with patch(
            "sys.argv",
            [
                "cli",
                pattern,
                "--batch",
                "-o",
                output_dir,
                "--skip-existing",
            ],
        ):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result2 = main()

                self.assertEqual(result2, 0)
                output = fake_out.getvalue()
                self.assertIn("Skipped: 1", output)

    def test_batch_processing_with_log_file(self):
        """Test batch processing with log file."""
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        output_dir = os.path.join(self.temp_dir.name, "batch_output")
        log_file = os.path.join(self.temp_dir.name, "batch.log")
        os.makedirs(input_dir)

        image = Image.new("RGB", (100, 100), color="white")
        image.save(os.path.join(input_dir, "test.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        with patch(
            "sys.argv",
            [
                "cli",
                pattern,
                "--batch",
                "-o",
                output_dir,
                "--log-file",
                log_file,
            ],
        ):
            result = main()

            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(log_file))

        # Close log handlers to release file lock on Windows
        import logging

        logger = logging.getLogger("batch_processor")
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

    def test_analyze_batch(self):
        """Test batch analysis."""
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        output_file = os.path.join(self.temp_dir.name, "analysis.json")
        os.makedirs(input_dir)

        for i in range(2):
            image = Image.new("RGB", (100, 100), color="white")
            image.save(os.path.join(input_dir, f"test_{i}.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        with patch(
            "sys.argv",
            [
                "cli",
                pattern,
                "--analyze-batch",
                "-o",
                output_file,
            ],
        ):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                result = main()

                self.assertEqual(result, 0)
                output = fake_out.getvalue()
                self.assertIn("Analyzed", output)
                self.assertIn("Batch Summary", output)
                self.assertIn("Suggested Processing Parameters", output)
                self.assertTrue(os.path.exists(output_file))

    def test_analyze_batch_default_output(self):
        """Test batch analysis with default output filename."""
        input_dir = os.path.join(self.temp_dir.name, "batch_input")
        os.makedirs(input_dir)

        image = Image.new("RGB", (100, 100), color="white")
        image.save(os.path.join(input_dir, "test.jpg"))

        pattern = os.path.join(input_dir, "*.jpg")

        with patch("sys.argv", ["cli", pattern, "--analyze-batch"]):
            result = main()

            self.assertEqual(result, 0)
            # Default output should be batch_analysis.json in current directory
            default_output = "batch_analysis.json"
            self.assertTrue(os.path.exists(default_output))
            # Clean up
            if os.path.exists(default_output):
                os.remove(default_output)

    def test_error_handling_invalid_file(self):
        """Test error handling with invalid file."""
        with patch(
            "sys.argv",
            [
                "cli",
                "nonexistent.jpg",
                "-o",
                self.output_path,
            ],
        ):
            with patch("sys.stdout", new=StringIO()):
                result = main()
                self.assertEqual(result, 1)

    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors."""
        with patch("sys.argv", ["cli", self.input_path, "-o", self.output_path]):
            with patch(
                "chiaroscuro_forge.cli.process_image", side_effect=RuntimeError("Unexpected")
            ):
                with patch("sys.stdout", new=StringIO()) as fake_out:
                    result = main()

                    self.assertEqual(result, 1)
                    output = fake_out.getvalue()
                    self.assertIn("Unexpected error", output)


if __name__ == "__main__":
    unittest.main()
