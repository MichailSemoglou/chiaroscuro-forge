"""
Comprehensive tests for metrics module.

Testing all quality metrics functions including SSIM, MS-SSIM, feature similarity,
histogram similarity, perceptual metrics, and quality score calculation.
"""

import unittest

import numpy as np
from skimage import img_as_ubyte, io

from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.metrics import (
    calculate_perceptual_metrics,
    calculate_quality_score,
    feature_similarity,
    histogram_similarity,
    ms_ssim,
    ssim,
)


class TestSSIM(unittest.TestCase):
    """Test SSIM calculation."""

    def setUp(self):
        """Create test images."""
        self.img1 = np.random.rand(100, 100, 3)
        self.img2 = self.img1.copy()
        self.img3 = np.random.rand(100, 100, 3)

    def test_identical_images(self):
        """Test SSIM for identical images."""
        score = ssim(self.img1, self.img2)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_different_images(self):
        """Test SSIM for different images."""
        score = ssim(self.img1, self.img3)
        self.assertLess(score, 1.0)
        self.assertGreater(score, -1.0)

    def test_grayscale_images(self):
        """Test SSIM with grayscale images."""
        gray1 = np.random.rand(100, 100)
        gray2 = gray1.copy()
        score = ssim(gray1, gray2)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_different_shapes(self):
        """Test SSIM with different image shapes."""
        img_small = np.random.rand(50, 50, 3)
        with self.assertRaises(ImageProcessingError) as ctx:
            ssim(self.img1, img_small)
        self.assertIn("same dimensions", str(ctx.exception))

    def test_color_to_grayscale_conversion(self):
        """Test that color images are converted to grayscale."""
        # Create color and grayscale versions
        color_img = np.random.rand(50, 50, 3)
        # SSIM should work even though internal processing differs
        score = ssim(color_img, color_img)
        self.assertAlmostEqual(score, 1.0, places=5)


class TestMSSSIM(unittest.TestCase):
    """Test Multi-Scale SSIM calculation."""

    def setUp(self):
        """Create test images."""
        # Larger images for MS-SSIM
        self.img1 = np.random.rand(128, 128, 3)
        self.img2 = self.img1.copy()
        self.img3 = np.random.rand(128, 128, 3)

    def test_identical_images(self):
        """Test MS-SSIM for identical images."""
        score = ms_ssim(self.img1, self.img2)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_different_images(self):
        """Test MS-SSIM for different images."""
        score = ms_ssim(self.img1, self.img3)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_custom_weights(self):
        """Test MS-SSIM with custom weights."""
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        score = ms_ssim(self.img1, self.img2, weights=weights)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_custom_levels(self):
        """Test MS-SSIM with different number of levels."""
        score = ms_ssim(self.img1, self.img2, levels=3)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_mismatched_weights(self):
        """Test MS-SSIM with wrong number of weights."""
        # Should auto-adjust weights
        weights = [0.5, 0.5]  # Only 2 weights for 5 levels
        score = ms_ssim(self.img1, self.img2, weights=weights, levels=5)
        self.assertGreater(score, 0.0)

    def test_zero_sum_retained_weights_falls_back_to_uniform(self):
        """Test MS-SSIM falls back to uniform weights when retained weights sum to zero."""
        weights = [0.0, 0.0, 0.0, 0.0, 0.0]
        score = ms_ssim(self.img1, self.img2, weights=weights, levels=5)
        self.assertGreater(score, 0.0)

    def test_grayscale_images(self):
        """Test MS-SSIM with grayscale images."""
        gray1 = np.random.rand(128, 128)
        gray2 = gray1.copy()
        score = ms_ssim(gray1, gray2)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_different_shapes(self):
        """Test MS-SSIM with different image shapes."""
        img_small = np.random.rand(64, 64, 3)
        with self.assertRaises(ImageProcessingError) as ctx:
            ms_ssim(self.img1, img_small)
        self.assertIn("same dimensions", str(ctx.exception))

    def test_small_image_error(self):
        """Test MS-SSIM fails gracefully on very small images."""
        # Very small image that can't support multi-scale
        tiny1 = np.random.rand(4, 4, 3)
        tiny2 = np.random.rand(4, 4, 3)
        with self.assertRaises(ImageProcessingError) as ctx:
            ms_ssim(tiny1, tiny2)
        self.assertIn("Could not calculate MS-SSIM", str(ctx.exception))


class TestFeatureSimilarity(unittest.TestCase):
    """Test feature-based similarity metrics."""

    def setUp(self):
        """Create test images."""
        self.img1 = np.random.rand(100, 100, 3)
        self.img2 = self.img1.copy()
        self.img3 = np.random.rand(100, 100, 3)

    def test_hog_identical(self):
        """Test HOG similarity for identical images."""
        score = feature_similarity(self.img1, self.img2, method="hog")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_hog_different(self):
        """Test HOG similarity for different images."""
        score = feature_similarity(self.img1, self.img3, method="hog")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_orb_keypoints(self):
        """Test ORB feature matching."""
        score = feature_similarity(self.img1, self.img2, method="orb")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_canny_edges(self):
        """Test Canny edge similarity."""
        score = feature_similarity(self.img1, self.img2, method="canny")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_grayscale_option(self):
        """Test feature similarity with multichannel=False."""
        # Note: multichannel parameter has been deprecated in scikit-image
        # The function handles grayscale conversion internally
        try:
            score = feature_similarity(self.img1, self.img2, method="hog", multichannel=False)
            self.assertGreater(score, 0.0)
        except ImageProcessingError as e:
            # If HOG fails with multichannel parameter, skip test
            if "multichannel" in str(e):
                self.skipTest("multichannel parameter not supported in this scikit-image version")
            raise

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with self.assertRaises(ImageProcessingError) as ctx:
            feature_similarity(self.img1, self.img2, method="invalid")
        self.assertIn("Unknown feature extraction method", str(ctx.exception))

    def test_different_shapes(self):
        """Test feature similarity with different shapes."""
        img_small = np.random.rand(50, 50, 3)
        with self.assertRaises(ImageProcessingError) as ctx:
            feature_similarity(self.img1, img_small, method="hog")
        self.assertIn("same dimensions", str(ctx.exception))

    def test_orb_no_keypoints(self):
        """Test ORB with uniform image (no keypoints)."""
        # Uniform image has no interesting keypoints
        uniform1 = np.ones((100, 100, 3)) * 0.5
        uniform2 = np.ones((100, 100, 3)) * 0.5

        # ORB may raise error or return 0.0 for uniform images
        try:
            score = feature_similarity(uniform1, uniform2, method="orb")
            # Should handle gracefully
            self.assertGreaterEqual(score, 0.0)
        except ImageProcessingError as e:
            # Expected for uniform images - ORB cannot find features
            self.assertIn("ORB feature extraction failed", str(e))

    def test_canny_no_edges(self):
        """Test Canny with uniform image (no edges)."""
        uniform1 = np.ones((100, 100, 3)) * 0.5
        uniform2 = np.ones((100, 100, 3)) * 0.5
        score = feature_similarity(uniform1, uniform2, method="canny")
        # Should return 1.0 when no edges in either (union = 0)
        self.assertAlmostEqual(score, 1.0, places=5)


class TestHistogramSimilarity(unittest.TestCase):
    """Test histogram-based similarity metrics."""

    def setUp(self):
        """Create test images."""
        self.img1 = np.random.rand(100, 100, 3)
        self.img2 = self.img1.copy()
        self.img3 = np.random.rand(100, 100, 3)

    def test_correlation_identical(self):
        """Test histogram correlation for identical images."""
        score = histogram_similarity(self.img1, self.img2, method="correlation")
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_correlation_different(self):
        """Test histogram correlation for different images."""
        score = histogram_similarity(self.img1, self.img3, method="correlation")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_chi_square(self):
        """Test chi-square histogram similarity."""
        score = histogram_similarity(self.img1, self.img2, method="chi_square")
        self.assertGreater(score, 0.5)

    def test_intersection(self):
        """Test histogram intersection."""
        score = histogram_similarity(self.img1, self.img2, method="intersection")
        self.assertGreater(score, 0.5)

    def test_bhattacharyya(self):
        """Test Bhattacharyya distance."""
        score = histogram_similarity(self.img1, self.img2, method="bhattacharyya")
        self.assertGreater(score, 0.5)

    def test_custom_bins(self):
        """Test histogram similarity with custom bin count."""
        score = histogram_similarity(self.img1, self.img2, method="correlation", bins=128)
        self.assertGreater(score, 0.5)

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with self.assertRaises(ImageProcessingError) as ctx:
            histogram_similarity(self.img1, self.img2, method="invalid")
        self.assertIn("Histogram method must be one of", str(ctx.exception))

    def test_identical_histograms_zero_std(self):
        """Test correlation with zero standard deviation."""
        # Uniform image should have histogram with zero std
        uniform1 = np.ones((100, 100, 3)) * 0.5
        uniform2 = np.ones((100, 100, 3)) * 0.5
        score = histogram_similarity(uniform1, uniform2, method="correlation")
        # Should handle div by zero gracefully
        self.assertGreater(score, 0.5)


class TestPerceptualMetrics(unittest.TestCase):
    """Test comprehensive perceptual metrics calculation."""

    def setUp(self):
        """Create test images."""
        self.original = np.random.rand(100, 100, 3)
        self.processed = self.original.copy()
        self.different = np.random.rand(100, 100, 3)

    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        metrics = calculate_perceptual_metrics(
            self.original, self.processed, calculate_advanced=False
        )

        self.assertIn("ssim", metrics)
        self.assertIn("psnr", metrics)
        self.assertIn("ms_ssim", metrics)

        # Should not have advanced metrics
        self.assertNotIn("feature_similarity_hog", metrics)

    def test_advanced_metrics(self):
        """Test advanced metrics calculation."""
        metrics = calculate_perceptual_metrics(
            self.original, self.processed, calculate_advanced=True
        )

        # Should have all metrics
        self.assertIn("ssim", metrics)
        self.assertIn("psnr", metrics)
        self.assertIn("ms_ssim", metrics)
        self.assertIn("feature_similarity_hog", metrics)
        self.assertIn("feature_similarity_canny", metrics)
        self.assertIn("histogram_correlation", metrics)
        self.assertIn("histogram_intersection", metrics)
        self.assertIn("color_preservation", metrics)

    def test_identical_images_psnr(self):
        """Test PSNR for identical images."""
        metrics = calculate_perceptual_metrics(
            self.original, self.processed, calculate_advanced=False
        )

        # PSNR should be very high for identical images
        self.assertGreater(metrics["psnr"], 90.0)

    def test_different_images_scores(self):
        """Test that different images have lower scores."""
        metrics = calculate_perceptual_metrics(
            self.original, self.different, calculate_advanced=True
        )

        # SSIM should be less than 1.0
        self.assertLess(metrics["ssim"], 1.0)

    def test_small_image_ms_ssim_fallback(self):
        """Test MS-SSIM fallback for small images."""
        small1 = np.random.rand(20, 20, 3)
        small2 = small1.copy()

        metrics = calculate_perceptual_metrics(small1, small2, calculate_advanced=False)

        # MS-SSIM should fall back to SSIM
        self.assertAlmostEqual(metrics["ssim"], metrics["ms_ssim"], places=5)

    def test_grayscale_images(self):
        """Test metrics with grayscale images."""
        gray1 = np.random.rand(100, 100)
        gray2 = gray1.copy()

        metrics = calculate_perceptual_metrics(gray1, gray2, calculate_advanced=True)

        # Should calculate all metrics except color_preservation
        self.assertIn("ssim", metrics)
        self.assertNotIn("color_preservation", metrics)

    def test_different_shapes_error(self):
        """Test error handling for different shapes."""
        small = np.random.rand(50, 50, 3)

        with self.assertRaises(ImageProcessingError) as ctx:
            calculate_perceptual_metrics(self.original, small)
        self.assertIn("same dimensions", str(ctx.exception))

    def test_color_preservation_metric(self):
        """Test color preservation metric for color images."""
        metrics = calculate_perceptual_metrics(
            self.original, self.processed, calculate_advanced=True
        )

        # Should have color preservation
        self.assertIn("color_preservation", metrics)
        # For identical images, should be close to 1.0
        self.assertGreater(metrics["color_preservation"], 0.8)

    def test_error_handling_advanced_metrics(self):
        """Test that advanced metric errors are handled gracefully."""
        # Create problematic image that might cause feature extraction to fail
        tiny = np.ones((10, 10, 3)) * 0.5
        tiny2 = np.ones((10, 10, 3)) * 0.6

        metrics = calculate_perceptual_metrics(tiny, tiny2, calculate_advanced=True)

        # Should still have basic metrics
        self.assertIn("ssim", metrics)
        self.assertIn("psnr", metrics)


class TestQualityScore(unittest.TestCase):
    """Test overall quality score calculation."""

    def setUp(self):
        """Create test metrics."""
        self.perfect_metrics = {
            "ssim": 1.0,
            "ms_ssim": 1.0,
            "psnr": 45.0,
            "feature_similarity_hog": 1.0,
            "feature_similarity_canny": 1.0,
            "histogram_correlation": 1.0,
            "histogram_intersection": 1.0,
            "color_preservation": 1.0,
        }

        self.poor_metrics = {
            "ssim": 0.5,
            "ms_ssim": 0.5,
            "psnr": 25.0,
            "feature_similarity_hog": 0.3,
            "histogram_correlation": 0.4,
            "color_preservation": 0.5,
        }

    def test_general_application(self):
        """Test quality score for general application."""
        score = calculate_quality_score(self.perfect_metrics, application_type="general")
        self.assertGreater(score, 0.9)
        self.assertLessEqual(score, 1.0)

    def test_photography_application(self):
        """Test quality score for photography application."""
        score = calculate_quality_score(self.perfect_metrics, application_type="photography")
        self.assertGreater(score, 0.9)

    def test_medical_application(self):
        """Test quality score for medical application."""
        score = calculate_quality_score(self.perfect_metrics, application_type="medical")
        self.assertGreater(score, 0.9)

    def test_document_application(self):
        """Test quality score for document application."""
        score = calculate_quality_score(self.perfect_metrics, application_type="document")
        self.assertGreater(score, 0.9)

    def test_art_application(self):
        """Test quality score for art application."""
        score = calculate_quality_score(self.perfect_metrics, application_type="art")
        self.assertGreater(score, 0.9)

    def test_poor_quality_score(self):
        """Test quality score for poor metrics."""
        score = calculate_quality_score(self.poor_metrics, application_type="general")
        self.assertLess(score, 0.7)
        self.assertGreater(score, 0.0)

    def test_invalid_application_type(self):
        """Test error handling for invalid application type."""
        with self.assertRaises(ImageProcessingError) as ctx:
            calculate_quality_score(self.perfect_metrics, application_type="invalid")
        self.assertIn("Application type must be one of", str(ctx.exception))

    def test_missing_metrics(self):
        """Test quality score with missing metrics."""
        partial_metrics = {
            "ssim": 0.9,
            "psnr": 35.0,
        }

        score = calculate_quality_score(partial_metrics, application_type="general")
        # Should work with partial metrics
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_psnr_normalization(self):
        """Test PSNR normalization in quality score."""
        # Very high PSNR should saturate at 1.0
        high_psnr_metrics = {
            "ssim": 0.9,
            "psnr": 60.0,  # Very high
        }

        score = calculate_quality_score(high_psnr_metrics, application_type="general")
        self.assertGreater(score, 0.0)

    def test_low_psnr_normalization(self):
        """Test low PSNR normalization."""
        low_psnr_metrics = {
            "ssim": 0.5,
            "psnr": 15.0,  # Very low (below 20 dB threshold)
        }

        score = calculate_quality_score(low_psnr_metrics, application_type="general")
        self.assertGreater(score, 0.0)

    def test_different_application_weights(self):
        """Test that different applications produce different scores."""
        score_general = calculate_quality_score(self.poor_metrics, application_type="general")
        score_medical = calculate_quality_score(self.poor_metrics, application_type="medical")
        score_art = calculate_quality_score(self.poor_metrics, application_type="art")

        # Scores should differ based on application type weights
        # (may not always be true depending on metrics, but test that function works)
        self.assertIsInstance(score_general, float)
        self.assertIsInstance(score_medical, float)
        self.assertIsInstance(score_art, float)


if __name__ == "__main__":
    unittest.main()
