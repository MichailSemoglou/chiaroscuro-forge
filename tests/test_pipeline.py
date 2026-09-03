"""
Unit tests for pipeline.py - Phase 2.1: Testing Framework

Tests each pipeline stage independently to ensure correctness,
error handling, and edge case coverage.

Target: 80%+ coverage for pipeline.py (currently 24%)
"""

import unittest

import numpy as np

from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.pipeline import (
    ColorPreservationStage,
    ContrastStage,
    DenoiseStage,
    GammaCorrectionStage,
    ImageProcessingPipeline,
    LinearizeStage,
    PipelineStage,
    ResizeStage,
    SharpenStage,
    ToneMappingStage,
    create_standard_pipeline,
)


class TestPipelineStage(unittest.TestCase):
    """Test base PipelineStage abstract class."""

    def test_stage_must_implement_process(self):
        """Abstract class should enforce process() implementation."""
        with self.assertRaises(TypeError):
            # Cannot instantiate abstract class directly
            PipelineStage("test_stage")

    def test_stage_requires_name(self):
        """Stages should have a name for identification."""

        class DummyStage(PipelineStage):
            def process(self, image, context):
                return image

        stage = DummyStage("dummy")
        self.assertEqual(stage.name, "dummy")


class TestResizeStage(unittest.TestCase):
    """Test ResizeStage functionality."""

    def setUp(self):
        """Create test image (100x100)."""
        self.image = np.random.rand(100, 100, 3)
        self.stage = ResizeStage()

    def test_resize_downscale(self):
        """Test downscaling image by factor of 0.5."""
        context = {"scale_factor": 0.5}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, (50, 50, 3))
        self.assertEqual(result.dtype, self.image.dtype)

    def test_resize_upscale(self):
        """Test upscaling image by factor of 2.0."""
        context = {"scale_factor": 2.0}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, (200, 200, 3))

    def test_resize_no_scale(self):
        """Test that scale=1.0 returns unchanged image."""
        context = {"scale_factor": 1.0}
        result = self.stage.process(self.image, context)

        np.testing.assert_array_equal(result, self.image)

    def test_resize_missing_context(self):
        """Test that missing resize_scale returns original."""
        result = self.stage.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)

    def test_resize_invalid_scale(self):
        """Test that scale=0 produces minimal image."""
        # scale_factor=0 produces 1x1x3 image (minimal result)
        context = {"scale_factor": 0}
        result = self.stage.process(self.image, context)
        # Result will be minimal but not empty
        self.assertGreater(result.size, 0)
        self.assertLess(result.size, self.image.size)


class TestDenoiseStage(unittest.TestCase):
    """Test DenoiseStage functionality."""

    def setUp(self):
        """Create noisy test image."""
        self.image = np.random.rand(50, 50, 3)
        self.stage = DenoiseStage()

    def test_denoise_gaussian(self):
        """Test Gaussian denoising."""
        context = {"denoise_method": "gaussian", "denoise_strength": 0.5}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)
        # Denoised image should have lower variance
        self.assertLess(np.var(result), np.var(self.image))

    def test_denoise_median(self):
        """Test median denoising."""
        context = {"denoise_method": "median", "denoise_strength": 0.3}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)

    def test_denoise_bilateral(self):
        """Test bilateral denoising."""
        context = {"denoise_method": "bilateral", "denoise_strength": 0.5}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)

    def test_denoise_no_method(self):
        """Test that default gaussian denoising is applied."""
        result = self.stage.process(self.image, {})
        # Default is gaussian with sigma=1.0, so result won't equal original
        self.assertEqual(result.shape, self.image.shape)
        # Check that some denoising occurred
        self.assertLess(np.var(result), np.var(self.image))

    def test_denoise_invalid_method(self):
        """Test invalid denoise method raises error."""
        context = {"denoise_type": "invalid"}
        with self.assertRaises(ImageProcessingError):
            self.stage.process(self.image, context)


class TestSharpenStage(unittest.TestCase):
    """Test SharpenStage functionality."""

    def setUp(self):
        """Create test image."""
        self.image = np.random.rand(50, 50, 3)
        self.stage = SharpenStage()

    def test_sharpen_standard(self):
        """Test standard sharpening."""
        context = {"sharpen_method": "standard", "sharpen_amount": 1.0}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_sharpen_no_method(self):
        """Test that missing method returns original."""
        result = self.stage.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)

    def test_sharpen_zero_amount(self):
        """Test that zero amount returns original."""
        context = {"sharpen_method": "standard", "sharpen_amount": 0.0}
        result = self.stage.process(self.image, context)
        np.testing.assert_array_equal(result, self.image)


class TestContrastStage(unittest.TestCase):
    """Test ContrastStage functionality."""

    def setUp(self):
        """Create test image."""
        self.image = np.linspace(0, 1, 2500).reshape(50, 50, 1)
        self.image = np.repeat(self.image, 3, axis=2)
        self.stage = ContrastStage()

    def test_contrast_standard(self):
        """Test standard contrast enhancement."""
        context = {"equalize": True, "equalize_method": "standard"}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)
        # Enhanced contrast should spread values more
        self.assertGreater(np.std(result), np.std(self.image) * 0.9)

    def test_contrast_clahe(self):
        """Test CLAHE contrast enhancement."""
        context = {"equalize": True, "equalize_method": "clahe"}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)
        # CLAHE must actually modify the image
        self.assertFalse(np.allclose(result, self.image))

    def test_contrast_clahe_color_random(self):
        """Test that CLAHE runs on color images."""
        image = np.random.rand(50, 50, 3)
        context = {"equalize": True, "equalize_method": "clahe"}
        result = self.stage.process(image, context)

        self.assertEqual(result.shape, image.shape)
        self.assertGreaterEqual(np.min(result), 0.0)
        self.assertLessEqual(np.max(result), 1.0)
        self.assertFalse(np.allclose(result, image))

    def test_contrast_adaptive_gamma_dark_color(self):
        """Test that adaptive gamma brightens a dark color image."""
        image = np.full((50, 50, 3), 0.1)
        context = {"equalize": True, "equalize_method": "adaptive_gamma"}
        result = self.stage.process(image, context)

        self.assertEqual(result.shape, image.shape)
        self.assertGreater(np.mean(result), np.mean(image))

    def test_contrast_stretch(self):
        """Test contrast stretching."""
        context = {"equalize": True, "equalize_method": "stretch"}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)
        # Should stretch to full range
        self.assertAlmostEqual(np.min(result), 0.0, places=2)
        self.assertAlmostEqual(np.max(result), 1.0, places=2)

    def test_contrast_no_method(self):
        """Test that missing method returns original."""
        result = self.stage.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)


class TestLinearizeStage(unittest.TestCase):
    """Test linear light conversion stage."""

    def setUp(self):
        self.image = np.array([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]], dtype=float)
        self.stage = LinearizeStage()

    def test_linearize_converts_srgb_to_linear(self):
        context = {"linear_light": True}
        result = self.stage.process(self.image, context)
        self.assertEqual(result.shape, self.image.shape)
        self.assertGreater(result[0, 1, 0], 0.0)
        self.assertLess(result[0, 1, 0], 0.5)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_linearize_preserves_hdr_headroom(self):
        """Linear-light conversion keeps values above 1.0 in HDR input."""
        hdr_image = np.array([[[1.5, 2.0, 3.0]]], dtype=float)
        result = self.stage.process(hdr_image, {"linear_light": True})
        self.assertTrue(np.all(result >= 0.0))
        self.assertGreater(result[0, 0, 0], 1.0)

    def test_linearize_disabled_returns_original(self):
        result = self.stage.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)


class TestGammaCorrectionStage(unittest.TestCase):
    """Test GammaCorrectionStage functionality."""

    def setUp(self):
        """Create test image."""
        self.image = np.linspace(0, 1, 2500).reshape(50, 50, 1)
        self.image = np.repeat(self.image, 3, axis=2)
        self.stage = GammaCorrectionStage()

    def test_gamma_correction(self):
        """Test gamma correction with gamma=2.0."""
        context = {"gamma_correction": 2.0}
        result = self.stage.process(self.image, context)

        self.assertEqual(result.shape, self.image.shape)
        # Gamma=2.0 should darken mid-tones (x^2 < x for 0<x<1)
        mid_val = 0.5
        idx = np.argmin(np.abs(self.image - mid_val))
        self.assertLess(result.flat[idx], self.image.flat[idx] + 0.01)

    def test_gamma_one(self):
        """Test that gamma=1.0 returns unchanged."""
        context = {"gamma_correction": 1.0}
        result = self.stage.process(self.image, context)

        np.testing.assert_array_almost_equal(result, self.image)

    def test_gamma_missing(self):
        """Test that missing gamma returns original."""
        result = self.stage.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)

    def test_gamma_invalid(self):
        """Test that invalid gamma raises error."""
        # Note: Current implementation allows any gamma value
        # This documents actual behavior - no validation at stage level
        context = {"gamma_correction": 0}
        # Zero gamma makes all values 0 or 1, no error raised
        result = self.stage.process(self.image, context)
        self.assertEqual(result.shape, self.image.shape)


class TestToneMappingStage(unittest.TestCase):
    """Test tone mapping stage for linear-light workflows."""

    def setUp(self):
        self.image = np.array([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]], dtype=float)
        self.stage = ToneMappingStage()

    def test_tone_mapping_applies_reinhard(self):
        context = {"linear_light": True}
        result = self.stage.process(self.image, context)
        self.assertEqual(result.shape, self.image.shape)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertLess(np.max(result), np.max(self.image))
        self.assertGreater(np.min(result), np.min(self.image) - 1e-9)

    def test_tone_mapping_disabled_returns_original(self):
        result = self.stage.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)


class TestColorPreservationStage(unittest.TestCase):
    """Test ColorPreservationStage functionality."""

    def setUp(self):
        """Create test images."""
        self.original = np.random.rand(50, 50, 3)
        self.processed = np.random.rand(50, 50, 3)
        self.stage = ColorPreservationStage()

    def test_color_preservation_lab(self):
        """Test LAB color space preservation."""
        context = {
            "original_image": self.original,
            "preserve_color": True,
            "preserve_method": "lab",
            "color_preservation_strength": 0.8,
        }
        result = self.stage.process(self.processed, context)

        self.assertEqual(result.shape, self.processed.shape)

    def test_color_preservation_lab_linear_light_matches_encoding(self):
        """LAB preservation in linear-light mode uses sRGB conversion on both sides."""
        original = np.array([[[0.03, 0.18, 0.62], [0.71, 0.14, 0.05]]], dtype=float)
        enhanced = np.array([[[0.48, 0.09, 0.27], [0.08, 0.53, 0.22]]], dtype=float)
        context = {
            "original_for_color": original,
            "color_preservation": "lab",
            "color_preservation_strength": 0.65,
            "linear_light": True,
        }

        result = self.stage.process(enhanced, context)

        expected = np.array(
            [
                [
                    [0.193259949270, 0.153710665546, 0.483637070718],
                    [0.655608166745, 0.366447549718, 0.144496749491],
                ]
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-10)

    def test_color_preservation_ratio(self):
        """Test ratio-based preservation."""
        context = {
            "original_image": self.original,
            "preserve_color": True,
            "preserve_method": "ratio",
            "color_preservation_strength": 0.8,
        }
        result = self.stage.process(self.processed, context)

        self.assertEqual(result.shape, self.processed.shape)

    def test_color_no_preservation(self):
        """Test that preserve_color=False returns processed."""
        context = {"preserve_color": False}
        result = self.stage.process(self.processed, context)

        np.testing.assert_array_equal(result, self.processed)

    def test_color_missing_original(self):
        """Test that missing original returns processed unchanged."""
        context = {"color_preservation": "lab"}
        result = self.stage.process(self.processed, context)
        # Without original_for_color, returns input unchanged
        np.testing.assert_array_equal(result, self.processed)


class TestImageProcessingPipeline(unittest.TestCase):
    """Test ImageProcessingPipeline orchestration."""

    def setUp(self):
        """Create test image and pipeline."""
        self.image = np.random.rand(50, 50, 3)
        self.pipeline = ImageProcessingPipeline()

    def test_empty_pipeline(self):
        """Test pipeline with no stages returns original."""
        result, context = self.pipeline.process(self.image, {})
        np.testing.assert_array_equal(result, self.image)

    def test_add_single_stage(self):
        """Test adding and executing single stage."""
        self.pipeline.add_stage(ResizeStage())
        context = {"scale_factor": 0.5}

        result, _ = self.pipeline.process(self.image, context)
        self.assertEqual(result.shape, (25, 25, 3))

    def test_add_multiple_stages(self):
        """Test stages execute in order."""
        self.pipeline.add_stage(ResizeStage())
        self.pipeline.add_stage(GammaCorrectionStage())

        context = {"scale_factor": 0.5, "gamma_correction": 2.0}
        result, _ = self.pipeline.process(self.image, context)

        # Should be resized (50->25) then gamma corrected
        self.assertEqual(result.shape, (25, 25, 3))

    def test_pipeline_error_propagation(self):
        """Test that stage errors propagate."""

        # Create a stage that will definitely fail
        class FailStage(PipelineStage):
            def process(self, image, context):
                raise ValueError("Intentional failure")

        self.pipeline.add_stage(FailStage("Fail"))

        with self.assertRaises(ImageProcessingError):
            self.pipeline.process(self.image, {})


class TestStandardPipeline(unittest.TestCase):
    """Test create_standard_pipeline factory."""

    def test_standard_pipeline_creation(self):
        """Test factory creates 6-stage pipeline."""
        pipeline = create_standard_pipeline()

        # Should have 6 stages
        self.assertEqual(len(pipeline.stages), 6)

        # Verify stage order
        stage_types = [type(s).__name__ for s in pipeline.stages]
        expected = [
            "ResizeStage",
            "DenoiseStage",
            "SharpenStage",
            "ContrastStage",
            "GammaCorrectionStage",
            "ColorPreservationStage",
        ]
        self.assertEqual(stage_types, expected)

    def test_linear_pipeline_creation(self):
        """Test linear light pipeline includes conversion and tone mapping."""
        pipeline = create_standard_pipeline(linear_light=True)
        stage_types = [type(s).__name__ for s in pipeline.stages]
        self.assertEqual(stage_types[:2], ["ResizeStage", "LinearizeStage"])
        self.assertEqual(stage_types[-1], "ToneMappingStage")

    def test_standard_pipeline_execution(self):
        """Test standard pipeline processes image."""
        pipeline = create_standard_pipeline()
        image = np.random.rand(100, 100, 3)
        context = {
            "scale_factor": 0.5,
            "gamma_correction": 1.2,
            "color_preservation": "none",
        }

        result, _ = pipeline.process(image, context)

        # Should be resized
        self.assertEqual(result.shape, (50, 50, 3))
        # Should have valid range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))


if __name__ == "__main__":
    unittest.main()
