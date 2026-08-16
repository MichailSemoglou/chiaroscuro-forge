"""
Image processing functions using clean pipeline pattern.

This module contains the refactored main image processing function
that uses a pipeline pattern for better maintainability.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
from skimage import io, transform
from skimage.util import img_as_float, img_as_ubyte

from .config import ProcessingConfig
from .constants import (
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    TILING_MEMORY_THRESHOLD_MB,
)
from .exceptions import ImageProcessingError
from .metrics import calculate_perceptual_metrics, calculate_quality_score
from .pipeline import create_standard_pipeline
from .tiling import process_image_tiled, should_use_tiling
from .validation import _validate_image_path, _validate_output_path, validate_config


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    config: Optional[ProcessingConfig] = None,
    # -- individual parameter overrides (backward compatible) ------------
    order: int = 1,
    order_rescale: int = 1,
    order_rotate: int = 1,
    scale_factor: float = 1.0,
    denoise_type: str = "gaussian",
    denoise_sigma: float = 1.0,
    sharpen: bool = True,
    sharpen_amount: float = 1.5,
    equalize: bool = True,
    equalize_method: str = "stretch",
    clip_limit: float = 0.03,
    clip_limit_kernel_size: int = 8,
    contrast_stretch_percentiles: Tuple[float, float] = (2.0, 98.0),
    gamma_correction: float = 1.0,
    color_preservation: str = "lab",
    color_preservation_strength: float = 0.7,
    calculate_metrics: bool = True,
    calculate_advanced_metrics: bool = True,
    application_type: str = "general",
    use_tiling: Optional[bool] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    rotation_angle: float = 0.0,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    Process an image using a clean pipeline architecture.

    Parameters may be supplied individually (backward compatible) or
    through a single ``ProcessingConfig`` object. When ``config`` is
    provided, individual keyword arguments are ignored; use
    ``config.merge()`` to combine overrides before calling.

    Parameters
    ----------
    image_path : str
        Path to input image file.
    output_path : str, optional
        Path to save processed image.
    config : ProcessingConfig, optional
        Unified configuration object.

    Returns
    -------
    tuple
        (processed_image, metrics_dict) where metrics_dict is None if
        ``calculate_metrics=False``.

    Raises
    ------
    ImageProcessingError
        If validation fails, processing fails, or file operations fail.
    """
    try:
        # 1. Build unified config
        if config is None:
            cfg = ProcessingConfig(
                application_type=application_type,
                scale_factor=scale_factor,
                order=order,
                order_rescale=order_rescale,
                order_rotate=order_rotate,
                rotation_angle=rotation_angle,
                denoise_type=denoise_type,
                denoise_sigma=denoise_sigma,
                sharpen=sharpen,
                sharpen_amount=sharpen_amount,
                equalize=equalize,
                equalize_method=equalize_method,
                clip_limit=clip_limit,
                clip_limit_kernel_size=clip_limit_kernel_size,
                contrast_stretch_percentiles=contrast_stretch_percentiles,
                gamma_correction=gamma_correction,
                color_preservation=color_preservation,
                color_preservation_strength=color_preservation_strength,
                calculate_metrics=calculate_metrics,
                calculate_advanced_metrics=calculate_advanced_metrics,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
        else:
            cfg = config

        # 2. Validate config
        validate_config(cfg)

        # 3. Validate image path
        _validate_image_path(image_path)

        # 4. Load image
        try:
            original_image = io.imread(image_path)
            image = img_as_float(original_image)
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

        # 5. Build processing context
        context = cfg.to_context()
        context["original_for_color"] = image.copy()

        # 6. Determine if tiling should be used
        actual_use_tiling = cfg.use_tiling
        if actual_use_tiling is None:
            bytes_per_pixel = image.shape[2] if image.ndim == 3 else 1
            actual_use_tiling = should_use_tiling(
                image.shape[:2],
                memory_threshold_mb=TILING_MEMORY_THRESHOLD_MB,
                bytes_per_pixel=bytes_per_pixel,
            )

        # Tiling is incompatible with resize operations that change each tile's
        # spatial dimensions, so fall back to the full-image pipeline in that case.
        if actual_use_tiling and cfg.scale_factor != 1.0:
            actual_use_tiling = False

        # 7. Process through pipeline
        pipeline = create_standard_pipeline()

        if actual_use_tiling:

            def process_tile(tile: np.ndarray, **kwargs) -> np.ndarray:
                tile_context = context.copy()
                tile_context["original_for_color"] = tile.copy()
                processed_tile, _ = pipeline.process(tile, tile_context)
                return processed_tile

            processed_image = process_image_tiled(
                image,
                process_tile,
                tile_size=cfg.tile_size,
                overlap=cfg.tile_overlap,
            )
        else:
            processed_image, context = pipeline.process(image, context)

        # 8. Calculate metrics
        metrics_dict = None
        if cfg.calculate_metrics:
            if min(original_image.shape[:2]) < 7:
                raise ImageProcessingError("Image too small for perceptual metrics calculation")

            original_for_metrics = original_image

            if cfg.scale_factor != 1.0:
                original_for_metrics = transform.resize(
                    original_for_metrics,
                    processed_image.shape[:2],
                    anti_aliasing=True,
                    preserve_range=True,
                )

            try:
                metrics_dict = calculate_perceptual_metrics(
                    img_as_float(original_for_metrics),
                    processed_image,
                    calculate_advanced=cfg.calculate_advanced_metrics,
                )
                metrics_dict["quality_score"] = calculate_quality_score(
                    metrics_dict,
                    application_type=cfg.application_type,
                )
            except Exception as e:
                raise ImageProcessingError(f"Could not calculate metrics: {str(e)}")

        # 9. Save output
        if output_path:
            safe_output = _validate_output_path(output_path)
            try:
                output_dir = os.path.dirname(safe_output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                io.imsave(safe_output, img_as_ubyte(processed_image))
            except Exception as e:
                raise ImageProcessingError(f"Failed to save output image: {str(e)}")

        return processed_image, metrics_dict

    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(f"Unexpected error in image processing: {str(e)}")
