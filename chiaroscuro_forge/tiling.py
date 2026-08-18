"""
Tile-Based Image Processing Module

Provides memory-efficient processing for large images through tiling.
Supports configurable tile sizes with overlap for seamless stitching.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .exceptions import ImageProcessingError
from .validation import validate_array


def calculate_tile_grid(
    image_shape: Tuple[int, int], tile_size: int, overlap: int
) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Calculate tile positions for processing a large image.

    Args:
        image_shape: (height, width) of the image
        tile_size: Size of each tile (square tiles)
        overlap: Overlap between adjacent tiles in pixels

    Returns:
        Tuple of (tile_coords, grid_shape) where:
        - tile_coords: List of (y_start, y_end, x_start, x_end) for each tile
        - grid_shape: (rows, cols) number of tiles in each dimension

    Raises:
        ImageProcessingError: If parameters are invalid
    """
    if tile_size <= 0:
        raise ImageProcessingError("Tile size must be positive")
    if overlap < 0:
        raise ImageProcessingError("Overlap cannot be negative")
    if overlap >= tile_size:
        raise ImageProcessingError("Overlap must be less than tile size")

    height, width = image_shape[:2]

    if height <= 0 or width <= 0:
        raise ImageProcessingError("Image dimensions must be positive")

    # Calculate stride (step between tiles)
    stride = tile_size - overlap

    # Calculate number of tiles needed in each dimension
    rows = max(1, (height - overlap) // stride + (1 if (height - overlap) % stride > 0 else 0))
    cols = max(1, (width - overlap) // stride + (1 if (width - overlap) % stride > 0 else 0))

    tile_coords = []

    for row in range(rows):
        y_start = row * stride
        y_end = min(y_start + tile_size, height)

        # Adjust start if we're at the end and tile would be too small
        if y_end == height and (y_end - y_start) < tile_size // 2:
            y_start = max(0, height - tile_size)

        for col in range(cols):
            x_start = col * stride
            x_end = min(x_start + tile_size, width)

            # Adjust start if we're at the end and tile would be too small
            if x_end == width and (x_end - x_start) < tile_size // 2:
                x_start = max(0, width - tile_size)

            tile_coords.append((y_start, y_end, x_start, x_end))

    return tile_coords, (rows, cols)


def extract_tile(image: np.ndarray, tile_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract a tile from an image.

    Args:
        image: Input image array
        tile_coords: (y_start, y_end, x_start, x_end)

    Returns:
        Extracted tile as numpy array

    Raises:
        ImageProcessingError: If extraction fails
    """
    validate_array(image)

    y_start, y_end, x_start, x_end = tile_coords

    if y_start < 0 or x_start < 0:
        raise ImageProcessingError("Tile coordinates cannot be negative")
    if y_end > image.shape[0] or x_end > image.shape[1]:
        raise ImageProcessingError("Tile coordinates exceed image dimensions")
    if y_start >= y_end or x_start >= x_end:
        raise ImageProcessingError("Invalid tile coordinates")

    return image[y_start:y_end, x_start:x_end].copy()


def blend_overlap(tile1: np.ndarray, tile2: np.ndarray, overlap: int, axis: int) -> np.ndarray:
    """
    Blend the overlapping region between two adjacent tiles.

    Uses linear blending to create seamless transitions.

    Args:
        tile1: First tile
        tile2: Second tile
        overlap: Number of pixels to blend
        axis: 0 for vertical blend (top-bottom), 1 for horizontal (left-right)

    Returns:
        Blended tile

    Raises:
        ImageProcessingError: If blending fails
    """
    if overlap <= 0:
        return tile1

    validate_array(tile1)
    validate_array(tile2)

    if tile1.shape != tile2.shape:
        raise ImageProcessingError("Tiles must have the same shape for blending")

    if axis not in (0, 1):
        raise ImageProcessingError("Axis must be 0 (vertical) or 1 (horizontal)")

    # Create blending weights
    weights: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(0, 1, overlap)

    # Reshape weights for broadcasting
    if axis == 0:  # Vertical blend
        if overlap > tile1.shape[0]:
            overlap = tile1.shape[0]
            weights = np.linspace(0, 1, overlap)
        weights = weights.reshape(-1, 1)
        if tile1.ndim == 3:
            weights = weights.reshape(-1, 1, 1)

        # Blend the overlap region at the bottom of tile1 and top of tile2
        blended = tile1.copy()
        blended[-overlap:] = tile1[-overlap:] * (1 - weights) + tile2[:overlap] * weights
    else:  # Horizontal blend
        if overlap > tile1.shape[1]:
            overlap = tile1.shape[1]
            weights = np.linspace(0, 1, overlap)
        weights = weights.reshape(1, -1)
        if tile1.ndim == 3:
            weights = weights.reshape(1, -1, 1)

        # Blend the overlap region at the right of tile1 and left of tile2
        blended = tile1.copy()
        blended[:, -overlap:] = tile1[:, -overlap:] * (1 - weights) + tile2[:, :overlap] * weights

    return blended


def stitch_tiles(
    tiles: List[np.ndarray],
    tile_coords: List[Tuple[int, int, int, int]],
    output_shape: Tuple[int, int],
    overlap: int,
) -> np.ndarray:
    """
    Stitch processed tiles back into a single image.

    Uses weighted blending in overlap regions for seamless results.

    Args:
        tiles: List of processed tiles
        tile_coords: List of (y_start, y_end, x_start, x_end) for each tile
        output_shape: (height, width) or (height, width, channels) of output
        overlap: Overlap used during tiling

    Returns:
        Stitched image

    Raises:
        ImageProcessingError: If stitching fails
    """
    if not tiles:
        raise ImageProcessingError("No tiles to stitch")
    if len(tiles) != len(tile_coords):
        raise ImageProcessingError("Number of tiles must match number of coordinates")

    # Determine output dtype and channels
    sample_tile = tiles[0]
    validate_array(sample_tile)

    if sample_tile.ndim == 3:
        output = np.zeros((*output_shape, sample_tile.shape[2]), dtype=np.float32)
        weights = np.zeros((*output_shape, sample_tile.shape[2]), dtype=np.float32)
    else:
        output = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)

    # Place each tile with weighted blending
    for tile, (y_start, y_end, x_start, x_end) in zip(tiles, tile_coords):
        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Create weight map for this tile
        weight_map = np.ones((tile_h, tile_w), dtype=np.float32)

        # Apply distance-based weighting in overlap regions
        if overlap > 0:
            # Vertical edges (top and bottom)
            if y_start > 0:  # Not the first row
                fade_region = min(overlap, tile_h)
                weight_map[:fade_region, :] *= np.linspace(0, 1, fade_region).reshape(-1, 1)
            if y_end < output_shape[0]:  # Not the last row
                fade_region = min(overlap, tile_h)
                weight_map[-fade_region:, :] *= np.linspace(1, 0, fade_region).reshape(-1, 1)

            # Horizontal edges (left and right)
            if x_start > 0:  # Not the first column
                fade_region = min(overlap, tile_w)
                weight_map[:, :fade_region] *= np.linspace(0, 1, fade_region).reshape(1, -1)
            if x_end < output_shape[1]:  # Not the last column
                fade_region = min(overlap, tile_w)
                weight_map[:, -fade_region:] *= np.linspace(1, 0, fade_region).reshape(1, -1)

        # Expand weight map for color channels if needed
        if sample_tile.ndim == 3:
            weight_map = weight_map[:, :, np.newaxis]

        # Accumulate weighted tile and weights
        output[y_start:y_end, x_start:x_end] += tile * weight_map
        weights[y_start:y_end, x_start:x_end] += weight_map

    # Normalize by total weights to get final image
    # Avoid division by zero
    weights = np.maximum(weights, 1e-8)
    output = output / weights

    return output.astype(sample_tile.dtype)


def process_image_tiled(
    image: np.ndarray,
    process_fn: Callable[..., np.ndarray],
    tile_size: int = 512,
    overlap: int = 64,
    process_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Process a large image using tile-based processing.

    Automatically splits the image into tiles, processes each independently,
    and stitches them back together with seamless blending.

    Args:
        image: Input image to process
        process_fn: Function to apply to each tile
        tile_size: Size of square tiles (default: 512)
        overlap: Overlap between tiles for seamless stitching (default: 64)
        process_fn_kwargs: Additional keyword arguments for process_fn

    Returns:
        Processed image

    Raises:
        ImageProcessingError: If processing fails

    Example:
        >>> def enhance_tile(tile, **kwargs):
        ...     return some_enhancement(tile)
        >>> result = process_image_tiled(large_image, enhance_tile)
    """
    validate_array(image)

    if process_fn_kwargs is None:
        process_fn_kwargs = {}

    # Get tile grid
    tile_coords, grid_shape = calculate_tile_grid(image.shape[:2], tile_size, overlap)

    # Process each tile
    processed_tiles = []
    for coords in tile_coords:
        tile = extract_tile(image, coords)

        try:
            processed_tile = process_fn(tile, **process_fn_kwargs)
            validate_array(processed_tile)
            processed_tiles.append(processed_tile)
        except Exception as e:
            raise ImageProcessingError(f"Failed to process tile at {coords}: {e}")

    # Stitch tiles back together
    result = stitch_tiles(processed_tiles, tile_coords, image.shape[:2], overlap)

    return result


def should_use_tiling(
    image_shape: Tuple[int, int], memory_threshold_mb: float = 100.0, bytes_per_pixel: int = 3
) -> bool:
    """
    Determine if an image should use tile-based processing.

    Args:
        image_shape: (height, width) of the image
        memory_threshold_mb: Memory threshold in megabytes (default: 100 MB)
        bytes_per_pixel: Bytes per pixel (3 for RGB, 1 for grayscale, 4 for RGBA)

    Returns:
        True if tiling should be used, False otherwise
    """
    height, width = image_shape[:2]

    # Calculate memory usage in MB
    memory_mb = (height * width * bytes_per_pixel) / (1024 * 1024)

    return memory_mb > memory_threshold_mb
