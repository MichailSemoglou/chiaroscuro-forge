"""
Integration and regression tests for the core image processing workflow.

Covers end-to-end paths: load → process → metrics → save, including
batch, tiling, preset-driven runs, config-object integration, and
expanded CLI smoke tests.
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
from unittest.mock import patch, MagicMock
from io import StringIO

import numpy as np
from PIL import Image
from skimage import io, img_as_ubyte

from chiaroscuro_forge.processing import process_image
from chiaroscuro_forge.config import ProcessingConfig
from chiaroscuro_forge import presets as preset_module
from chiaroscuro_forge.presets import save_preset, load_preset, list_presets
from chiaroscuro_forge.batch import batch_process_images
from chiaroscuro_forge.exceptions import ImageProcessingError


class TestProcessingConfigIntegration(unittest.TestCase):
    """Test process_image via the ProcessingConfig codepath."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image = np.random.rand(80, 80, 3)
        self.image_path = os.path.join(self.temp_dir, "test.png")
        io.imsave(self.image_path, img_as_ubyte(self.image))
        self.output_path = os.path.join(self.temp_dir, "out.png")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_config_object_basic(self):
        config = ProcessingConfig(
            denoise_type="median",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
        )
        processed, metrics = process_image(self.image_path, config=config)
        self.assertEqual(processed.shape, self.image.shape)
        self.assertIsNotNone(metrics)

    def test_config_merge_overrides(self):
        config = ProcessingConfig(denoise_type="median")
        config = config.merge({"denoise_type": "gaussian", "denoise_sigma": 0.8})
        processed, _ = process_image(self.image_path, config=config)
        self.assertEqual(processed.shape, self.image.shape)

    def test_config_merge(self):
        base = ProcessingConfig(denoise_type="gaussian", denoise_sigma=0.5)
        merged = base.merge({"denoise_sigma": 2.0, "sharpen": False})
        self.assertEqual(merged.denoise_sigma, 2.0)
        self.assertFalse(merged.sharpen)
        self.assertEqual(merged.denoise_type, "gaussian")

    def test_config_preset_photography(self):
        config = ProcessingConfig.preset("photography")
        self.assertEqual(config.color_preservation, "lab")
        self.assertEqual(config.color_preservation_strength, 0.9)

    def test_config_preset_document(self):
        config = ProcessingConfig.preset("document")
        self.assertEqual(config.equalize_method, "clahe")
        self.assertEqual(config.clip_limit, 0.02)

    def test_config_to_context_keys(self):
        config = ProcessingConfig()
        ctx = config.to_context()
        expected_keys = [
            "scale_factor",
            "order",
            "order_rescale",
            "order_rotate",
            "denoise_type",
            "denoise_sigma",
            "sharpen",
            "sharpen_amount",
            "equalize",
            "equalize_method",
            "clip_limit",
            "clip_limit_kernel_size",
            "contrast_stretch_percentiles",
            "gamma_correction",
            "color_preservation",
            "color_preservation_strength",
        ]
        for k in expected_keys:
            self.assertIn(k, ctx)

    def test_config_to_context_preserves_interpolation_orders(self):
        config = ProcessingConfig(order=2, order_rescale=3, order_rotate=4)
        ctx = config.to_context()
        self.assertEqual(ctx["order"], 2)
        self.assertEqual(ctx["order_rescale"], 3)
        self.assertEqual(ctx["order_rotate"], 4)

    def test_config_roundtrip_via_dict(self):
        config = ProcessingConfig(
            application_type="art",
            denoise_type="bilateral",
            denoise_sigma=0.4,
            contrast_stretch_percentiles=(5, 95),
        )
        d = config.to_dict()
        restored = ProcessingConfig.from_dict(d)
        self.assertEqual(restored.application_type, "art")
        self.assertEqual(restored.denoise_type, "bilateral")
        self.assertEqual(restored.contrast_stretch_percentiles, (5, 95))

    def test_config_object_preserves_output(self):
        config = ProcessingConfig(
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
            calculate_metrics=False,
        )
        processed, metrics = process_image(
            self.image_path, output_path=self.output_path, config=config
        )
        self.assertIsNone(metrics)
        self.assertTrue(os.path.exists(self.output_path))
        reloaded = io.imread(self.output_path)
        self.assertEqual(reloaded.shape[:2], (80, 80))


class TestEndToEndWorkflows(unittest.TestCase):
    """Realistic processing workflows with common parameter combinations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.random_path = os.path.join(self.temp_dir, "random.png")
        random_image = np.random.rand(100, 100, 3)
        io.imsave(self.random_path, img_as_ubyte(random_image))

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_photography_stretch_lab(self):
        processed, metrics = process_image(
            self.random_path,
            application_type="photography",
            equalize_method="stretch",
            contrast_stretch_percentiles=(2, 98),
            color_preservation="lab",
            color_preservation_strength=0.9,
            denoise_type="bilateral",
            denoise_sigma=0.5,
            sharpen=True,
            sharpen_amount=1.2,
        )
        self.assertEqual(processed.shape, (100, 100, 3))
        self.assertIsNotNone(metrics)
        self.assertIn("quality_score", metrics)

    def test_document_stretch_ratio(self):
        processed, _ = process_image(
            self.random_path,
            application_type="document",
            equalize_method="stretch",
            contrast_stretch_percentiles=(2, 98),
            color_preservation="ratio",
            color_preservation_strength=0.5,
            denoise_type="median",
            sharpen=True,
            sharpen_amount=1.8,
            gamma_correction=1.1,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_medical_standard_no_color(self):
        processed, _ = process_image(
            self.random_path,
            application_type="medical",
            equalize_method="standard",
            color_preservation="none",
            denoise_type="bilateral",
            denoise_sigma=0.4,
            sharpen_amount=0.5,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_art_ratio_gamma(self):
        processed, _ = process_image(
            self.random_path,
            application_type="art",
            color_preservation="ratio",
            color_preservation_strength=0.5,
            gamma_correction=1.15,
            sharpen=False,
            denoise_type="gaussian",
            denoise_sigma=0.8,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_clahe_grayscale_no_color(self):
        gray = np.random.rand(100, 100)
        gray_path = os.path.join(self.temp_dir, "gray_clahe.png")
        io.imsave(gray_path, img_as_ubyte(gray))
        processed, _ = process_image(
            gray_path,
            equalize_method="clahe",
            clip_limit=0.03,
            clip_limit_kernel_size=8,
            color_preservation="none",
            denoise_type="none",
            sharpen=False,
            equalize=True,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100))

    def test_clahe_grayscale_kernel_16(self):
        gray = np.random.rand(100, 100)
        gray_path = os.path.join(self.temp_dir, "gray_clahe_16.png")
        io.imsave(gray_path, img_as_ubyte(gray))
        processed, _ = process_image(
            gray_path,
            equalize_method="clahe",
            clip_limit=0.05,
            clip_limit_kernel_size=16,
            color_preservation="none",
            denoise_type="none",
            sharpen=False,
            equalize=True,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100))

    def test_color_preservation_strength_zero(self):
        processed, _ = process_image(
            self.random_path,
            color_preservation="lab",
            color_preservation_strength=0.0,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_color_preservation_strength_one(self):
        processed, _ = process_image(
            self.random_path,
            color_preservation="lab",
            color_preservation_strength=1.0,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_all_features_disabled_pass_through(self):
        processed, _ = process_image(
            self.random_path,
            scale_factor=1.0,
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_grayscale_with_equalization(self):
        gray = np.random.rand(100, 100)
        gray_path = os.path.join(self.temp_dir, "gray.png")
        io.imsave(gray_path, img_as_ubyte(gray))
        processed, _ = process_image(
            gray_path,
            equalize_method="stretch",
            color_preservation="none",
            denoise_type="gaussian",
            calculate_metrics=False,
        )
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.shape, (100, 100))

    def test_metrics_advanced_with_stretch(self):
        processed, metrics = process_image(
            self.random_path,
            equalize_method="stretch",
            color_preservation="lab",
            calculate_metrics=True,
            calculate_advanced_metrics=True,
        )
        self.assertIsNotNone(metrics)
        self.assertIn("quality_score", metrics)
        self.assertIn("ssim", metrics)

    def test_processing_config_photo_pipeline(self):
        config = ProcessingConfig(
            application_type="photography",
            equalize_method="stretch",
            color_preservation="lab",
            color_preservation_strength=0.9,
            denoise_type="bilateral",
            denoise_sigma=0.5,
            sharpen=True,
            sharpen_amount=1.2,
        )
        processed, metrics = process_image(self.random_path, config=config)
        self.assertEqual(processed.shape, (100, 100, 3))
        self.assertIsNotNone(metrics)

    def test_processing_config_document_pipeline(self):
        config = ProcessingConfig(
            application_type="document",
            equalize_method="stretch",
            contrast_stretch_percentiles=(2, 98),
            sharpen=True,
            sharpen_amount=1.8,
            gamma_correction=1.1,
            calculate_metrics=False,
        )
        processed, _ = process_image(self.random_path, config=config)
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_processing_config_merges_kwargs(self):
        config = ProcessingConfig(
            denoise_type="gaussian",
            denoise_sigma=0.5,
            color_preservation="none",
            calculate_metrics=False,
        )
        config = config.merge({"denoise_type": "bilateral"})
        processed, _ = process_image(
            self.random_path,
            config=config,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_contrast_stretch_percentiles_zero_to_hundred(self):
        processed, _ = process_image(
            self.random_path,
            equalize=True,
            equalize_method="stretch",
            contrast_stretch_percentiles=(0, 100),
            color_preservation="none",
            denoise_type="none",
            sharpen=False,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_equalize_method_adaptive_gamma(self):
        processed, _ = process_image(
            self.random_path,
            equalize=True,
            equalize_method="adaptive_gamma",
            color_preservation="none",
            denoise_type="none",
            sharpen=False,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_application_type_art_pipeline(self):
        config = ProcessingConfig.preset("art")
        config = config.merge({"calculate_metrics": False})
        processed, _ = process_image(self.random_path, config=config)
        self.assertEqual(processed.shape, (100, 100, 3))

    def test_application_type_medical_pipeline(self):
        config = ProcessingConfig.preset("medical")
        config.equalize_method = "standard"
        config = config.merge({"calculate_metrics": False})
        processed, _ = process_image(self.random_path, config=config)
        self.assertEqual(processed.shape, (100, 100, 3))


class TestTilingWorkflows(unittest.TestCase):
    """End-to-end tests with explicit tiling control."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_image(self, size):
        path = os.path.join(self.temp_dir, f"img_{size}.png")
        image = np.random.rand(size, size, 3)
        io.imsave(path, img_as_ubyte(image))
        return path

    def test_tiling_explicitly_disabled(self):
        path = self._create_image(256)
        processed, _ = process_image(
            path,
            use_tiling=False,
            color_preservation="none",
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape[:2], (256, 256))

    def test_tiling_explicitly_enabled_small_image(self):
        path = self._create_image(64)
        processed, _ = process_image(
            path,
            use_tiling=True,
            tile_size=32,
            tile_overlap=8,
            color_preservation="none",
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape[:2], (64, 64))

    def test_tiling_with_scaled_lab_color_preservation(self):
        path = self._create_image(128)
        processed, _ = process_image(
            path,
            use_tiling=True,
            tile_size=32,
            tile_overlap=8,
            scale_factor=0.5,
            color_preservation="lab",
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            calculate_metrics=False,
        )
        self.assertEqual(processed.shape[:2], (64, 64))

    def test_tiling_with_denoise_and_metrics(self):
        path = self._create_image(192)
        processed, metrics = process_image(
            path,
            use_tiling=True,
            tile_size=64,
            tile_overlap=16,
            denoise_type="gaussian",
            denoise_sigma=0.5,
            color_preservation="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            calculate_metrics=True,
        )
        self.assertEqual(processed.shape[:2], (192, 192))
        self.assertIsNotNone(metrics)

    def test_tiling_disabled_with_config(self):
        path = self._create_image(128)
        config = ProcessingConfig(
            use_tiling=False,
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
            denoise_type="none",
            calculate_metrics=False,
        )
        processed, _ = process_image(path, config=config)
        self.assertEqual(processed.shape[:2], (128, 128))


class TestBatchWithConfig(unittest.TestCase):
    """Batch processing with ProcessingConfig objects."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir)
        for i in range(3):
            img = np.random.rand(50, 50, 3)
            io.imsave(os.path.join(self.input_dir, f"img_{i}.png"), img_as_ubyte(img))

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_batch_with_config_object(self):
        config = ProcessingConfig(
            denoise_type="median",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
            calculate_metrics=False,
        )
        pattern = os.path.join(self.input_dir, "*.png")
        results = batch_process_images(pattern, self.output_dir, config=config)
        self.assertEqual(results["total"], 3)
        self.assertEqual(results["successful"], 3)
        self.assertEqual(results["failed"], 0)

    def test_batch_config_skips_overridden_params(self):
        config = ProcessingConfig(denoise_type="gaussian")
        pattern = os.path.join(self.input_dir, "*.png")
        results = batch_process_images(pattern, self.output_dir, config=config)
        self.assertEqual(results["successful"], 3)

    def test_batch_config_with_skip_existing(self):
        config = ProcessingConfig(
            denoise_type="none",
            sharpen=False,
            equalize=False,
            gamma_correction=1.0,
            color_preservation="none",
            calculate_metrics=False,
        )
        pattern = os.path.join(self.input_dir, "*.png")
        batch_process_images(pattern, self.output_dir, config=config)
        results = batch_process_images(pattern, self.output_dir, config=config, skip_existing=True)
        self.assertEqual(results["skipped"], 3)
        self.assertEqual(results["successful"], 0)


class TestPresetRegression(unittest.TestCase):
    """End-to-end preset save/load/process cycles."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.preset_dir = os.path.join(self.temp_dir, "presets")
        os.makedirs(self.preset_dir, exist_ok=True)
        self._preset_module_path = os.path.join(self.temp_dir, "presets_module.py")
        self._preset_patch = patch.object(preset_module, "__file__", self._preset_module_path)
        self._preset_patch.start()
        self._preset_suffix = os.path.basename(self.temp_dir).replace("-", "_")
        preset_module.invalidate_preset_cache()
        self.image = np.random.rand(80, 80, 3)
        self.image_path = os.path.join(self.temp_dir, "test.png")
        io.imsave(self.image_path, img_as_ubyte(self.image))

    def tearDown(self):
        self._preset_patch.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _preset_name(self, base_name):
        return f"{base_name}_{self._preset_suffix}"

    def test_save_load_process_cycle(self):
        config = ProcessingConfig(
            application_type="photography",
            equalize_method="stretch",
            contrast_stretch_percentiles=(2, 98),
            color_preservation="lab",
            color_preservation_strength=0.85,
            denoise_type="bilateral",
            denoise_sigma=0.4,
            sharpen=True,
            sharpen_amount=1.3,
        )
        preset_name = self._preset_name("regression_test")
        save_preset(preset_name, config.to_dict(), description="Integration regression")
        loaded = load_preset(preset_name)
        restored = ProcessingConfig.from_dict(loaded)
        processed, metrics = process_image(self.image_path, config=restored)
        self.assertEqual(processed.shape, self.image.shape)
        self.assertIsNotNone(metrics)

    def test_preset_overrides_defaults_only(self):
        minimal_config = ProcessingConfig(
            denoise_sigma=0.2,
            color_preservation_strength=0.3,
        )
        preset_name = self._preset_name("minimal_regression")
        save_preset(preset_name, minimal_config.to_dict(), description="Minimal override preset")
        loaded = load_preset(preset_name)
        restored = ProcessingConfig.from_dict(loaded)
        self.assertEqual(restored.denoise_sigma, 0.2)
        self.assertEqual(restored.denoise_type, "gaussian")

    def test_preset_applied_via_config_merge(self):
        preset_name = self._preset_name("merge_test")
        save_preset(preset_name, {"denoise_type": "bilateral", "denoise_sigma": 0.6})
        loaded = load_preset(preset_name)
        config = ProcessingConfig()
        config = config.merge(loaded)
        self.assertEqual(config.denoise_type, "bilateral")
        self.assertEqual(config.denoise_sigma, 0.6)

    def test_preset_loaded_via_cli_style_processing(self):
        config = ProcessingConfig(
            application_type="medical",
            equalize_method="standard",
            color_preservation="none",
            denoise_type="bilateral",
            denoise_sigma=0.4,
        )
        preset_name = self._preset_name("cli_style")
        save_preset(preset_name, config.to_dict())
        loaded = load_preset(preset_name)
        cli_config = ProcessingConfig.from_dict(loaded)
        processed, _ = process_image(self.image_path, config=cli_config, calculate_metrics=False)
        self.assertEqual(processed.shape, self.image.shape)


class TestCLISmoke(unittest.TestCase):
    """Expanded smoke tests for CLI entry points."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "input.jpg")
        self.output_path = os.path.join(self.temp_dir.name, "output.jpg")
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        image.save(self.input_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_cli(self, *args):
        from chiaroscuro_forge.cli import main

        with patch("sys.argv", ["cli"] + list(args)):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                exit_code = main()
                return exit_code, fake_out.getvalue()

    def test_batch_with_workers_flag(self):
        for i in range(2):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(self.temp_dir.name, f"batch_{i}.jpg"))
        output_dir = os.path.join(self.temp_dir.name, "out")
        pattern = os.path.join(self.temp_dir.name, "batch_*.jpg")
        exit_code, output = self._run_cli(
            pattern, "--batch", "-o", output_dir, "--workers", "2", "--report"
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("Batch processing completed", output)
        self.assertIn("successfully: 2", output)

    def test_single_image_with_application_type_flag(self):
        exit_code, output = self._run_cli(
            self.input_path, "-o", self.output_path, "--application", "photography"
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("saved to", output)

    def test_analyze_single_with_output(self):
        exit_code, output = self._run_cli(self.input_path, "--analyze", "-o", self.output_path)
        self.assertEqual(exit_code, 0)
        self.assertIn("Image Characteristics", output)

    def test_compare_with_output_dir(self):
        compare_dir = os.path.join(self.temp_dir.name, "comparison")
        exit_code, output = self._run_cli(
            self.input_path, "--compare", "--compare-dir", compare_dir
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("Comparison Results", output)

    def test_preset_save_and_list_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_dir = os.path.join(temp_dir, "presets")
            os.makedirs(preset_dir, exist_ok=True)
            preset_module_path = os.path.join(temp_dir, "presets_module.py")
            with patch.object(preset_module, "__file__", preset_module_path):
                preset_module.invalidate_preset_cache()
                exit_code, _ = self._run_cli(
                    self.input_path,
                    "--save-preset",
                    "smoke_preset",
                    "--preset-description",
                    "CLI smoke test",
                )
                self.assertEqual(exit_code, 0)
                exit_code, output = self._run_cli("--list-presets")
                self.assertEqual(exit_code, 0)
                self.assertIn("smoke_preset", output)

                preset_path = os.path.join(preset_dir, "smoke_preset.json")
                self.assertTrue(os.path.exists(preset_path))
                os.remove(preset_path)
                self.assertFalse(os.path.exists(preset_path))

    def test_batch_skip_existing_combined(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(self.temp_dir.name, "bs_0.jpg"))
        output_dir = os.path.join(self.temp_dir.name, "bs_out")
        pattern = os.path.join(self.temp_dir.name, "bs_*.jpg")
        exit_code, output = self._run_cli(
            pattern, "--batch", "-o", output_dir, "--skip-existing", "--report"
        )
        self.assertEqual(exit_code, 0)
        exit_code, output = self._run_cli(
            pattern, "--batch", "-o", output_dir, "--skip-existing", "--report"
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("Skipped: 1", output)
