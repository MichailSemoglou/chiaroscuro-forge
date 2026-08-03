"""Property-based tests using Hypothesis framework.

This module contains property-based tests that verify invariants and
mathematical properties of the image processing functions.

Note: These tests require the hypothesis package to be installed.
They will be skipped if hypothesis is not available.
"""

import numpy as np
import pytest

# Check if Hypothesis is available
try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HYPOTHESIS_AVAILABLE = True

    # Define strategies when Hypothesis is available
    @st.composite
    def valid_images(draw, min_size=16, max_size=128):
        """Generate valid image arrays for testing."""
        height = draw(st.integers(min_value=min_size, max_value=max_size))
        width = draw(st.integers(min_value=min_size, max_value=max_size))
        channels = draw(st.sampled_from([3]))

        shape = (height, width, channels)
        img = draw(
            arrays(
                dtype=np.float64,
                shape=shape,
                elements=st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        return img

    @st.composite
    def valid_gamma_values(draw):
        """Generate valid gamma correction values."""
        return draw(st.floats(min_value=0.5, max_value=2.5, allow_nan=False, allow_infinity=False))

    @st.composite
    def valid_scale_factors(draw):
        """Generate valid scale factors."""
        return draw(st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False))

except ImportError:
    HYPOTHESIS_AVAILABLE = False

    def given(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def settings(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def assume(*args):
        return None

    def valid_images(*args, **kwargs):
        return None

    def valid_gamma_values(*args, **kwargs):
        return None

    def valid_scale_factors(*args, **kwargs):
        return None


from chiaroscuro_forge.metrics import (
    ms_ssim,
    ssim,
)
from chiaroscuro_forge.pipeline import (
    ContrastStage,
    DenoiseStage,
    ImageProcessingPipeline,
    ResizeStage,
)

pytestmark = pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")


# ===== Property Tests for Pipeline Stages =====


class TestResizeProperties:
    """Property-based tests for image resizing."""

    @given(image=valid_images(), scale=valid_scale_factors())
    @settings(deadline=None, max_examples=30)
    def test_resize_dimensions(self, image, scale):
        """Resizing should change dimensions proportionally."""
        stage = ResizeStage()
        context = {"scale_factor": scale}

        result = stage.process(image, context)

        expected_height = int(image.shape[0] * scale)
        expected_width = int(image.shape[1] * scale)

        # Allow some tolerance due to resampling
        assert abs(result.shape[0] - expected_height) <= 2
        assert abs(result.shape[1] - expected_width) <= 2
        assert result.shape[2] == image.shape[2]  # Channels unchanged

    @given(image=valid_images())
    @settings(deadline=None, max_examples=30)
    def test_resize_identity(self, image):
        """Scale factor of 1.0 should be identity."""
        stage = ResizeStage()
        context = {"scale_factor": 1.0}

        result = stage.process(image, context)

        # Should return the same image
        assert result.shape == image.shape
        np.testing.assert_array_equal(result, image)

    @given(image=valid_images(), scale=valid_scale_factors())
    @settings(deadline=None, max_examples=30)
    def test_resize_preserves_range(self, image, scale):
        """Resizing should preserve value range."""
        stage = ResizeStage()
        context = {"scale_factor": scale}

        result = stage.process(image, context)

        assert np.all(result >= -0.01)
        assert np.all(result <= 1.01)


class TestDenoiseProperties:
    """Property-based tests for denoising."""

    @given(image=valid_images(min_size=32))
    @settings(deadline=None, max_examples=20)
    def test_denoise_preserves_shape(self, image):
        """Denoising should preserve image shape."""
        stage = DenoiseStage()
        context = {"denoise_type": "gaussian", "denoise_sigma": 1.0}

        result = stage.process(image, context)
        assert result.shape == image.shape

    @given(image=valid_images(min_size=32))
    @settings(deadline=None, max_examples=20)
    def test_denoise_preserves_range(self, image):
        """Denoising should keep values in [0, 1]."""
        stage = DenoiseStage()
        context = {"denoise_type": "gaussian", "denoise_sigma": 1.0}

        result = stage.process(image, context)
        assert np.all(result >= -0.01)
        assert np.all(result <= 1.01)


class TestContrastProperties:
    """Property-based tests for contrast enhancement."""

    @given(image=valid_images(min_size=32))
    @settings(deadline=None, max_examples=20)
    def test_contrast_preserves_shape(self, image):
        """Contrast enhancement should preserve image shape."""
        stage = ContrastStage()
        context = {"equalize": True, "equalize_method": "stretch"}

        result = stage.process(image, context)
        assert result.shape == image.shape

    @given(image=valid_images(min_size=32))
    @settings(deadline=None, max_examples=20)
    def test_contrast_preserves_range(self, image):
        """Contrast enhancement should keep values in [0, 1]."""
        stage = ContrastStage()
        context = {"equalize": True, "equalize_method": "stretch"}

        result = stage.process(image, context)
        assert np.all(result >= -0.01)
        assert np.all(result <= 1.01)


# ===== Property Tests for Metrics =====


class TestMetricProperties:
    """Property-based tests for image quality metrics."""

    @given(image=valid_images(min_size=16))
    @settings(deadline=None, max_examples=30)
    def test_ssim_self_is_one(self, image):
        """SSIM of an image with itself should be 1.0."""
        ssim_value = ssim(image, image)
        assert 0.99 <= ssim_value <= 1.01

    @given(image=valid_images(min_size=16))
    @settings(deadline=None, max_examples=30)
    def test_ssim_range(self, image):
        """SSIM should be in [-1, 1] range."""
        noise = np.random.normal(0, 0.1, image.shape)
        noisy = np.clip(image + noise, 0, 1)

        ssim_value = ssim(image, noisy)
        assert -1.0 <= ssim_value <= 1.0

    @given(image=valid_images(min_size=16))
    @settings(deadline=None, max_examples=30)
    def test_ssim_symmetric(self, image):
        """SSIM should be symmetric: SSIM(A, B) = SSIM(B, A)."""
        noise = np.random.normal(0, 0.05, image.shape)
        noisy = np.clip(image + noise, 0, 1)

        ssim1 = ssim(image, noisy)
        ssim2 = ssim(noisy, image)

        np.testing.assert_allclose(ssim1, ssim2, rtol=1e-5, atol=1e-5)

    @given(image=valid_images(min_size=48))
    @settings(deadline=None, max_examples=20)
    def test_ms_ssim_self_is_one(self, image):
        """MS-SSIM of an image with itself should be close to 1.0."""
        ms_ssim_value = ms_ssim(image, image)
        assert 0.95 <= ms_ssim_value <= 1.05

    @given(image=valid_images(min_size=48))
    @settings(deadline=None, max_examples=20)
    def test_ms_ssim_range(self, image):
        """MS-SSIM should be in [0, 1] range."""
        noise = np.random.normal(0, 0.1, image.shape)
        noisy = np.clip(image + noise, 0, 1)

        ms_ssim_value = ms_ssim(image, noisy)
        assert 0.0 <= ms_ssim_value <= 1.0


# ===== Property Tests for Pipeline =====


class TestPipelineProperties:
    """Property-based tests for full pipeline."""

    @given(image=valid_images(min_size=64))
    @settings(deadline=None, max_examples=15)
    def test_pipeline_preserves_channels(self, image):
        """Pipeline should preserve number of channels."""
        pipeline = ImageProcessingPipeline()
        pipeline.add_stage(ResizeStage())

        context = {"scale_factor": 1.0}
        result, _ = pipeline.process(image, context)

        assert result.shape[2] == image.shape[2]

    @given(image=valid_images(min_size=64))
    @settings(deadline=None, max_examples=15)
    def test_pipeline_deterministic(self, image):
        """Pipeline with same parameters should be deterministic."""
        pipeline = ImageProcessingPipeline()
        pipeline.add_stage(ResizeStage())

        context = {"scale_factor": 1.0}

        result1, _ = pipeline.process(image.copy(), context.copy())
        result2, _ = pipeline.process(image.copy(), context.copy())

        np.testing.assert_array_equal(result1, result2)


# ===== Property Tests for Invariants =====


class TestInvariantProperties:
    """Test mathematical invariants across transformations."""

    @given(image=valid_images())
    @settings(deadline=None, max_examples=30)
    def test_resize_compose(self, image):
        """Resizing twice should equal single resize with combined scale."""
        stage = ResizeStage()

        # Resize twice with scale 0.8 each time
        context1 = {"scale_factor": 0.8}
        intermediate = stage.process(image, context1)

        context2 = {"scale_factor": 0.8}
        result = stage.process(intermediate, context2)

        # Should be similar to single resize with scale 0.64
        context_combined = {"scale_factor": 0.64}
        expected = stage.process(image, context_combined)

        # Shapes should match
        assert abs(result.shape[0] - expected.shape[0]) <= 2
        assert abs(result.shape[1] - expected.shape[1]) <= 2

    @given(image=valid_images(min_size=32))
    @settings(deadline=None, max_examples=20)
    def test_ssim_transitivity(self, image):
        """If A is similar to B and B to C, A should be somewhat similar to C."""
        # Create slightly noisy versions
        noise1 = np.random.normal(0, 0.02, image.shape)
        image_b = np.clip(image + noise1, 0, 1)

        noise2 = np.random.normal(0, 0.02, image.shape)
        image_c = np.clip(image_b + noise2, 0, 1)

        ssim_ab = ssim(image, image_b)
        ssim_bc = ssim(image_b, image_c)
        ssim_ac = ssim(image, image_c)

        # If A~B and B~C, then A~C (with some degradation)
        if ssim_ab > 0.9 and ssim_bc > 0.9:
            assert ssim_ac > 0.7  # Allowing for accumulated error


# ===== Extreme Value Tests =====


class TestExtremeValues:
    """Test behavior with edge cases."""

    @given(scale=valid_scale_factors())
    @settings(deadline=None, max_examples=30)
    def test_black_image_processing(self, scale):
        """Black image should remain black after most operations."""
        black = np.zeros((64, 64, 3))

        stage = ResizeStage()
        context = {"scale_factor": scale}
        result = stage.process(black, context)

        assert np.allclose(result, 0, atol=1e-6)

    @given(scale=valid_scale_factors())
    @settings(deadline=None, max_examples=30)
    def test_white_image_processing(self, scale):
        """White image should remain white after most operations."""
        white = np.ones((64, 64, 3))

        stage = ResizeStage()
        context = {"scale_factor": scale}
        result = stage.process(white, context)

        assert np.allclose(result, 1, atol=1e-3)
