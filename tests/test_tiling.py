"""
Tests for Tile-Based Image Processing

Comprehensive tests for memory-efficient large image processing using tiles.
"""

import unittest

import numpy as np

from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.tiling import (
    blend_overlap,
    calculate_tile_grid,
    extract_tile,
    process_image_tiled,
    should_use_tiling,
    stitch_tiles,
)


class TestCalculateTileGrid(unittest.TestCase):
    """Test tile grid calculation."""

    def test_single_tile(self):
        """Test when image fits in single tile."""
        coords, grid_shape = calculate_tile_grid((256, 256), tile_size=512, overlap=64)

        self.assertEqual(len(coords), 1)
        self.assertEqual(grid_shape, (1, 1))
        self.assertEqual(coords[0], (0, 256, 0, 256))

    def test_multiple_tiles_no_overlap(self):
        """Test grid with no overlap."""
        coords, grid_shape = calculate_tile_grid((1024, 1024), tile_size=512, overlap=0)

        self.assertEqual(grid_shape, (2, 2))
        self.assertEqual(len(coords), 4)

        # Check first tile
        self.assertEqual(coords[0], (0, 512, 0, 512))
        # Check last tile
        self.assertEqual(coords[-1], (512, 1024, 512, 1024))

    def test_multiple_tiles_with_overlap(self):
        """Test grid with overlap."""
        coords, grid_shape = calculate_tile_grid((1024, 1024), tile_size=512, overlap=64)

        # With overlap=64, stride=448
        # Tiles needed: ceil((1024-64)/448) = ceil(960/448) = 3 per dimension
        self.assertEqual(grid_shape, (3, 3))
        self.assertEqual(len(coords), 9)

    def test_rectangular_image(self):
        """Test with non-square image."""
        coords, grid_shape = calculate_tile_grid((512, 1024), tile_size=512, overlap=64)

        # Height: 1 tile (512 fits in 512)
        # Width: 1024 with 512 tile and 64 overlap -> stride=448
        # Tiles needed: ceil((1024-64)/448) = ceil(960/448) = 3
        self.assertEqual(grid_shape, (1, 3))
        self.assertEqual(len(coords), 3)

    def test_invalid_tile_size(self):
        """Test with invalid tile size."""
        with self.assertRaises(ImageProcessingError):
            calculate_tile_grid((1024, 1024), tile_size=-512, overlap=64)

    def test_invalid_overlap(self):
        """Test with invalid overlap."""
        with self.assertRaises(ImageProcessingError):
            calculate_tile_grid((1024, 1024), tile_size=512, overlap=-10)

    def test_overlap_too_large(self):
        """Test when overlap >= tile_size."""
        with self.assertRaises(ImageProcessingError):
            calculate_tile_grid((1024, 1024), tile_size=512, overlap=512)

    def test_edge_adjustment(self):
        """Test that edge tiles are properly adjusted."""
        # 1000x1000 image with 512 tiles, overlap 64
        coords, grid_shape = calculate_tile_grid((1000, 1000), tile_size=512, overlap=64)

        # Check that no tile goes beyond image bounds
        for y_start, y_end, x_start, x_end in coords:
            self.assertGreaterEqual(y_start, 0)
            self.assertGreaterEqual(x_start, 0)
            self.assertLessEqual(y_end, 1000)
            self.assertLessEqual(x_end, 1000)


class TestExtractTile(unittest.TestCase):
    """Test tile extraction."""

    def setUp(self):
        """Create test image."""
        self.image = np.random.rand(512, 512, 3).astype(np.float64)

    def test_extract_full_tile(self):
        """Test extracting a full tile."""
        tile = extract_tile(self.image, (0, 256, 0, 256))

        self.assertEqual(tile.shape, (256, 256, 3))
        np.testing.assert_array_equal(tile, self.image[0:256, 0:256])

    def test_extract_edge_tile(self):
        """Test extracting an edge tile."""
        tile = extract_tile(self.image, (256, 512, 256, 512))

        self.assertEqual(tile.shape, (256, 256, 3))
        np.testing.assert_array_equal(tile, self.image[256:512, 256:512])

    def test_extract_grayscale(self):
        """Test extracting from grayscale image."""
        gray_image = np.random.rand(512, 512).astype(np.float64)
        tile = extract_tile(gray_image, (0, 256, 0, 256))

        self.assertEqual(tile.shape, (256, 256))

    def test_invalid_coordinates(self):
        """Test with invalid coordinates."""
        with self.assertRaises(ImageProcessingError):
            extract_tile(self.image, (-10, 256, 0, 256))

    def test_coordinates_exceed_bounds(self):
        """Test when coordinates exceed image bounds."""
        with self.assertRaises(ImageProcessingError):
            extract_tile(self.image, (0, 600, 0, 256))

    def test_inverted_coordinates(self):
        """Test with start > end."""
        with self.assertRaises(ImageProcessingError):
            extract_tile(self.image, (256, 0, 0, 256))


class TestBlendOverlap(unittest.TestCase):
    """Test overlap blending."""

    def test_vertical_blend(self):
        """Test vertical blending (top-bottom)."""
        tile1 = np.ones((256, 256, 3)) * 0.2
        tile2 = np.ones((256, 256, 3)) * 0.8

        blended = blend_overlap(tile1, tile2, overlap=64, axis=0)

        # Check shape preserved
        self.assertEqual(blended.shape, (256, 256, 3))

        # Check blending: bottom of tile1 should be between 0.2 and 0.8
        bottom_values = blended[-32, 128, 0]  # Middle of blend region
        self.assertGreater(bottom_values, 0.2)
        self.assertLess(bottom_values, 0.8)

    def test_horizontal_blend(self):
        """Test horizontal blending (left-right)."""
        tile1 = np.ones((256, 256, 3)) * 0.2
        tile2 = np.ones((256, 256, 3)) * 0.8

        blended = blend_overlap(tile1, tile2, overlap=64, axis=1)

        # Check shape preserved
        self.assertEqual(blended.shape, (256, 256, 3))

        # Check blending: right of tile1 should be between 0.2 and 0.8
        right_values = blended[128, -32, 0]  # Middle of blend region
        self.assertGreater(right_values, 0.2)
        self.assertLess(right_values, 0.8)

    def test_grayscale_blend(self):
        """Test blending grayscale tiles."""
        tile1 = np.ones((256, 256)) * 0.3
        tile2 = np.ones((256, 256)) * 0.7

        blended = blend_overlap(tile1, tile2, overlap=32, axis=0)

        self.assertEqual(blended.shape, (256, 256))

    def test_zero_overlap(self):
        """Test with zero overlap."""
        tile1 = np.ones((256, 256, 3)) * 0.5
        tile2 = np.ones((256, 256, 3)) * 0.8

        blended = blend_overlap(tile1, tile2, overlap=0, axis=0)

        # Should return tile1 unchanged
        np.testing.assert_array_equal(blended, tile1)

    def test_mismatched_shapes(self):
        """Test with different tile shapes."""
        tile1 = np.ones((256, 256, 3))
        tile2 = np.ones((128, 128, 3))

        with self.assertRaises(ImageProcessingError):
            blend_overlap(tile1, tile2, overlap=64, axis=0)

    def test_invalid_axis(self):
        """Test with invalid axis."""
        tile1 = np.ones((256, 256, 3))
        tile2 = np.ones((256, 256, 3))

        with self.assertRaises(ImageProcessingError):
            blend_overlap(tile1, tile2, overlap=64, axis=2)


class TestStitchTiles(unittest.TestCase):
    """Test tile stitching."""

    def test_stitch_simple_grid(self):
        """Test stitching a 2x2 grid."""
        # Create 4 tiles with different values
        tiles = [
            np.ones((256, 256, 3)) * 0.2,  # Top-left
            np.ones((256, 256, 3)) * 0.4,  # Top-right
            np.ones((256, 256, 3)) * 0.6,  # Bottom-left
            np.ones((256, 256, 3)) * 0.8,  # Bottom-right
        ]

        coords = [
            (0, 256, 0, 256),
            (0, 256, 256, 512),
            (256, 512, 0, 256),
            (256, 512, 256, 512),
        ]

        result = stitch_tiles(tiles, coords, (512, 512), overlap=0)

        self.assertEqual(result.shape, (512, 512, 3))

        # Check corners
        np.testing.assert_allclose(result[0, 0, 0], 0.2, rtol=0.01)
        np.testing.assert_allclose(result[0, 500, 0], 0.4, rtol=0.01)
        np.testing.assert_allclose(result[500, 0, 0], 0.6, rtol=0.01)
        np.testing.assert_allclose(result[500, 500, 0], 0.8, rtol=0.01)

    def test_stitch_with_overlap(self):
        """Test stitching with overlap blending."""
        # Create 2 tiles with overlap
        tiles = [
            np.ones((320, 256, 3)) * 0.3,
            np.ones((320, 256, 3)) * 0.7,
        ]

        coords = [
            (0, 320, 0, 256),
            (256, 576, 0, 256),  # 64 pixel overlap
        ]

        result = stitch_tiles(tiles, coords, (576, 256), overlap=64)

        self.assertEqual(result.shape, (576, 256, 3))

        # Check overlap region is blended (should be between 0.3 and 0.7)
        overlap_value = result[288, 128, 0]  # Middle of overlap
        self.assertGreater(overlap_value, 0.3)
        self.assertLess(overlap_value, 0.7)

    def test_stitch_grayscale(self):
        """Test stitching grayscale tiles."""
        tiles = [
            np.ones((256, 256)) * 0.3,
            np.ones((256, 256)) * 0.6,
        ]

        coords = [
            (0, 256, 0, 256),
            (0, 256, 256, 512),
        ]

        result = stitch_tiles(tiles, coords, (256, 512), overlap=0)

        self.assertEqual(result.shape, (256, 512))

    def test_empty_tiles_list(self):
        """Test with empty tiles list."""
        with self.assertRaises(ImageProcessingError):
            stitch_tiles([], [], (512, 512), overlap=0)

    def test_mismatched_counts(self):
        """Test when tile count doesn't match coord count."""
        tiles = [np.ones((256, 256, 3))]
        coords = [(0, 256, 0, 256), (0, 256, 256, 512)]

        with self.assertRaises(ImageProcessingError):
            stitch_tiles(tiles, coords, (256, 512), overlap=0)


class TestProcessImageTiled(unittest.TestCase):
    """Test full tiled processing pipeline."""

    def test_process_small_image(self):
        """Test processing an image that fits in one tile."""
        image = np.random.rand(256, 256, 3).astype(np.float64)

        def simple_enhance(tile, **kwargs):
            return tile * 1.2

        result = process_image_tiled(image, simple_enhance, tile_size=512, overlap=64)

        self.assertEqual(result.shape, image.shape)
        np.testing.assert_allclose(result, image * 1.2, rtol=0.01)

    def test_process_large_image(self):
        """Test processing a large image with multiple tiles."""
        image = np.random.rand(1024, 1024, 3).astype(np.float64)

        def simple_enhance(tile, **kwargs):
            return tile * 0.8

        result = process_image_tiled(image, simple_enhance, tile_size=512, overlap=64)

        self.assertEqual(result.shape, image.shape)
        # Result should be close to image * 0.8 everywhere
        np.testing.assert_allclose(result, image * 0.8, rtol=0.05)

    def test_process_with_kwargs(self):
        """Test passing kwargs to processing function."""
        image = np.random.rand(512, 512, 3).astype(np.float64)

        def enhance_with_factor(tile, factor=1.0, **kwargs):
            return tile * factor

        result = process_image_tiled(
            image, enhance_with_factor, tile_size=256, overlap=32, process_fn_kwargs={"factor": 1.5}
        )

        self.assertEqual(result.shape, image.shape)
        np.testing.assert_allclose(result, image * 1.5, rtol=0.05)

    def test_process_grayscale(self):
        """Test processing grayscale image."""
        image = np.random.rand(512, 512).astype(np.float64)

        def invert(tile, **kwargs):
            return 1.0 - tile

        result = process_image_tiled(image, invert, tile_size=256, overlap=32)

        self.assertEqual(result.shape, image.shape)
        np.testing.assert_allclose(result, 1.0 - image, rtol=0.05)

    def test_process_function_error(self):
        """Test when processing function raises error."""
        image = np.random.rand(256, 256, 3).astype(np.float64)

        def failing_function(tile, **kwargs):
            raise ValueError("Processing failed")

        with self.assertRaises(ImageProcessingError):
            process_image_tiled(image, failing_function)

    def test_seamless_stitching(self):
        """Test that tiles are stitched seamlessly."""
        # Create a gradient image
        x = np.linspace(0, 1, 1024)
        y = np.linspace(0, 1, 1024)
        xx, yy = np.meshgrid(x, y)
        image = (xx + yy) / 2.0
        image = np.stack([image, image, image], axis=2)

        def identity(tile, **kwargs):
            return tile

        result = process_image_tiled(image, identity, tile_size=512, overlap=64)

        # Result should be very close to original (within blending tolerance)
        np.testing.assert_allclose(result, image, atol=0.01)

    def test_edge_handling(self):
        """Test proper handling of image edges."""
        # Create image with distinct edges
        image = np.zeros((600, 800, 3))
        image[:300, :, :] = 1.0  # Top half white

        def identity(tile, **kwargs):
            return tile

        result = process_image_tiled(image, identity, tile_size=400, overlap=50)

        # Check edge is preserved (with some tolerance for blending)
        self.assertGreater(np.mean(result[:100, :, :]), 0.8)  # Top should be white
        self.assertLess(np.mean(result[-100:, :, :]), 0.2)  # Bottom should be black


class TestShouldUseTiling(unittest.TestCase):
    """Test tiling decision logic."""

    def test_small_image_no_tiling(self):
        """Test that small images don't use tiling."""
        # 512x512 RGB = 0.75 MB < 100 MB threshold
        result = should_use_tiling((512, 512), memory_threshold_mb=100.0, bytes_per_pixel=3)
        self.assertFalse(result)

    def test_large_image_uses_tiling(self):
        """Test that large images use tiling."""
        # 8192x8192 RGB = 192 MB > 100 MB threshold
        result = should_use_tiling((8192, 8192), memory_threshold_mb=100.0, bytes_per_pixel=3)
        self.assertTrue(result)

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        # Calculate size for exactly 100 MB with RGB
        # 100 MB = 100 * 1024 * 1024 bytes / 3 bytes_per_pixel = 34,952,533 pixels
        # sqrt(34952533) ≈ 5912
        result_below = should_use_tiling((5900, 5900), memory_threshold_mb=100.0, bytes_per_pixel=3)
        result_above = should_use_tiling((5920, 5920), memory_threshold_mb=100.0, bytes_per_pixel=3)

        self.assertFalse(result_below)
        self.assertTrue(result_above)

    def test_grayscale_larger_threshold(self):
        """Test that grayscale images have effectively larger threshold."""
        size = (7000, 7000)
        # Same size, but grayscale uses 1/3 the memory
        rgb_result = should_use_tiling(size, memory_threshold_mb=100.0, bytes_per_pixel=3)
        gray_result = should_use_tiling(size, memory_threshold_mb=100.0, bytes_per_pixel=1)

        # RGB should trigger tiling, grayscale might not
        self.assertTrue(rgb_result)
        # Grayscale 7000x7000 = 49MB < 100MB
        self.assertFalse(gray_result)

    def test_custom_threshold(self):
        """Test with custom memory threshold."""
        size = (4000, 4000)
        # 4000x4000 RGB = 48 MB

        result_high = should_use_tiling(size, memory_threshold_mb=100.0, bytes_per_pixel=3)
        result_low = should_use_tiling(size, memory_threshold_mb=10.0, bytes_per_pixel=3)

        self.assertFalse(result_high)  # 48 MB < 100 MB
        self.assertTrue(result_low)  # 48 MB > 10 MB


if __name__ == "__main__":
    unittest.main()
