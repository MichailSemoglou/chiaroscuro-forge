"""
Image processing functions using clean pipeline pattern.

This module contains the refactored main image processing function
that uses a pipeline pattern for better maintainability.
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple
from skimage import io
from skimage import img_as_float, img_as_ubyte

from .exceptions import ImageProcessingError
from .validation import _validate_image_path, _validate_processing_params
from .metrics import calculate_perceptual_metrics, calculate_quality_score
from .pipeline import create_standard_pipeline
from .tiling import should_use_tiling, process_image_tiled
from .constants import (
    DEFAULT_TILE_SIZE,
    DEFAULT_TILE_OVERLAP,
    TILING_MEMORY_THRESHOLD_MB
)


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
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
    contrast_stretch_percentiles: Tuple[int, int] = (2, 98),
    gamma_correction: float = 1.0,
    color_preservation: str = "lab",
    color_preservation_strength: float = 0.7,
    calculate_metrics: bool = True,
    calculate_advanced_metrics: bool = True,
    application_type: str = "general",
    use_tiling: Optional[bool] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    Process an image using a clean pipeline architecture.
    
    This function validates inputs, loads the image, processes it through
    a series of pipeline stages, calculates metrics, and saves the output.
    
    Refactored to use pipeline pattern - complexity reduced from ~250 lines
    to <100 lines in this function, with stage logic isolated in pipeline.py.
    
    For large images, tile-based processing is automatically used to reduce
    memory consumption.

    Parameters
    ----------
    image_path : str
        Path to input image file.
    output_path : str, optional
        Path to save processed image.
    scale_factor : float, default=1.0
        Scaling factor for resizing.
    denoise_type : str, default="gaussian"
        Denoising method: "gaussian", "median", "bilateral", or "none".
    denoise_sigma : float, default=1.0
        Sigma parameter for denoising.
    sharpen : bool, default=True
        Whether to apply sharpening.
    sharpen_amount : float, default=1.5
        Amount of sharpening to apply.
    equalize : bool, default=True
        Whether to apply histogram equalization.
    equalize_method : str, default="stretch"
        Equalization method: "standard", "clahe", "stretch", "adaptive_gamma".
    clip_limit : float, default=0.03
        CLAHE clip limit.
    clip_limit_kernel_size : int, default=8
        CLAHE kernel size.
    contrast_stretch_percentiles : tuple, default=(2, 98)
        Percentiles for contrast stretching.
    gamma_correction : float, default=1.0
        Gamma correction value.
    color_preservation : str, default="lab"
        Color preservation method: "none", "lab", "rgb", "ratio".
    color_preservation_strength : float, default=0.7
        Strength of color preservation (0-1).
    calculate_metrics : bool, default=True
        Whether to calculate quality metrics.
    calculate_advanced_metrics : bool, default=True
        Whether to calculate advanced metrics.
    application_type : str, default="general"
        Application type for optimization.
    use_tiling : bool, optional
        Force tiling on/off. If None, automatically determined based on image size.
    tile_size : int, default=512
        Size of tiles for large image processing.
    tile_overlap : int, default=64
        Overlap between tiles for seamless stitching.

    Returns
    -------
    tuple
        (processed_image, metrics_dict) where metrics_dict is None if 
        calculate_metrics=False.
        
    Raises
    ------
    ImageProcessingError
        If validation fails, processing fails, or file operations fail.
    """
    try:
        # 1. Validation
        _validate_image_path(image_path)
        _validate_processing_params(
            scale_factor, order, order_rescale, order_rotate,
            denoise_type, denoise_sigma, sharpen, sharpen_amount,
            equalize, equalize_method, clip_limit, clip_limit_kernel_size,
            contrast_stretch_percentiles, gamma_correction,
            color_preservation, color_preservation_strength, application_type,
        )

        # 2. Load image
        try:
            original_image = io.imread(image_path)
            image = img_as_float(original_image)
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

        # 3. Build processing context
        context = {
            # Resizing
            'scale_factor': scale_factor,
            'order_rescale': order_rescale,
            
            # Denoising
            'denoise_type': denoise_type,
            'denoise_sigma': denoise_sigma,
            
            # Sharpening
            'sharpen': sharpen,
            'sharpen_amount': sharpen_amount,
            
            # Contrast
            'equalize': equalize,
            'equalize_method': equalize_method,
            'clip_limit': clip_limit,
            'clip_limit_kernel_size': clip_limit_kernel_size,
            'contrast_stretch_percentiles': contrast_stretch_percentiles,
            
            # Gamma
            'gamma_correction': gamma_correction,
            
            # Color preservation
            'color_preservation': color_preservation,
            'color_preservation_strength': color_preservation_strength,
            'original_for_color': image.copy(),
        }

        # 4. Determine if tiling should be used
        if use_tiling is None:
            # Auto-detect based on image size
            bytes_per_pixel = 3 if image.ndim == 3 else 1
            use_tiling = should_use_tiling(
                image.shape[:2],
                memory_threshold_mb=TILING_MEMORY_THRESHOLD_MB,
                bytes_per_pixel=bytes_per_pixel
            )
        
        # 5. Process through pipeline (with or without tiling)
        pipeline = create_standard_pipeline()
        
        if use_tiling:
            # Define a wrapper function for pipeline processing
            def process_tile(tile: np.ndarray, **kwargs) -> np.ndarray:
                # Update context with tile-specific original for color preservation
                tile_context = context.copy()
                tile_context['original_for_color'] = tile.copy()
                processed_tile, _ = pipeline.process(tile, tile_context)
                return processed_tile
            
            processed_image = process_image_tiled(
                image,
                process_tile,
                tile_size=tile_size,
                overlap=tile_overlap
            )
        else:
            processed_image, context = pipeline.process(image, context)
        
        # 6. Calculate metrics
        metrics_dict = None
        if calculate_metrics:
            if min(original_image.shape[:2]) < 7:
                raise ImageProcessingError(
                    "Image too small for perceptual metrics calculation"
                )
            
            try:
                metrics_dict = calculate_perceptual_metrics(
                    original_image,
                    processed_image,
                    calculate_advanced=calculate_advanced_metrics
                )
                metrics_dict["quality_score"] = calculate_quality_score(
                    metrics_dict,
                    application_type=application_type
                )
            except Exception as e:
                raise ImageProcessingError(f"Could not calculate metrics: {str(e)}")

        # 7. Save output
        if output_path:
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                io.imsave(output_path, img_as_ubyte(processed_image))
            except Exception as e:
                raise ImageProcessingError(f"Failed to save output image: {str(e)}")

        return processed_image, metrics_dict

    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(f"Unexpected error in image processing: {str(e)}")
