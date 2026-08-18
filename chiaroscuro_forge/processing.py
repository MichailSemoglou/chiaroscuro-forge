"""High-level image processing entry points.

This module contains the main public processing workflow for Chiaroscuro Forge.
It builds a unified ``ProcessingConfig`` from either explicit arguments or a
single configuration object, validates the requested workflow, performs image
loading and output handling, and returns processed pixels together with the
computed quality metrics.

The API intentionally keeps the historical keyword-based calling pattern for
backward compatibility while encouraging the more structured
``ProcessingConfig`` flow for new integrations.
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
from .pipeline import create_standard_pipeline, srgb_to_linear
from .tiling import process_image_tiled, should_use_tiling
from .validation import _validate_image_path, _validate_output_path, validate_config


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    config: Optional[ProcessingConfig] = None,
    # Legacy keyword arguments remain accepted for backward compatibility.
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
    linear_light: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """Process an image and return the enhanced result with optional metrics.

    This is the library's main public entry point. The caller may provide a
    complete ``ProcessingConfig`` object or use the legacy keyword-based form,
    which is preserved for compatibility with older integrations.

    Parameters
    ----------
    image_path : str
        Path to the source image file.
    output_path : str, optional
        Destination path for the processed image. If omitted, the image is only
        returned in memory.
    config : ProcessingConfig, optional
        Unified configuration object. When supplied, it takes precedence over
        legacy keyword arguments.

    Returns
    -------
    tuple[np.ndarray, dict | None]
        A tuple of ``(processed_image, metrics_dict)``. The metrics dictionary is
        ``None`` when metric calculation is disabled.

    Raises
    ------
    ImageProcessingError
        Raised when validation, file loading, processing, output writing, or
        metric computation fails.

    Notes
    -----
    The ``linear_light`` option is opt-in and is intended for experimental
    workflows that require sRGB-to-linear conversion before enhancement and a
    tone-mapping step before final output.
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
                linear_light=linear_light,
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
        context["original_for_color"] = (
            srgb_to_linear(image.copy()) if cfg.linear_light else image.copy()
        )

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
        pipeline = create_standard_pipeline(linear_light=cfg.linear_light)

        if actual_use_tiling:

            def process_tile(tile: np.ndarray, **kwargs) -> np.ndarray:
                tile_context = context.copy()
                tile_context["original_for_color"] = (
                    srgb_to_linear(tile.copy()) if cfg.linear_light else tile.copy()
                )
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
