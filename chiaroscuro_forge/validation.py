"""
Validation Utilities for Chiaroscuro Forge

This module provides validation functions for image arrays, file paths,
and processing parameters.
"""

import os
from typing import Tuple
import numpy as np

from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.constants import (
    VALID_APP_TYPES,
    VALID_DENOISE_TYPES,
    VALID_EQUALIZE_METHODS,
    VALID_COLOR_METHODS,
)


def validate_array(arr: np.ndarray, name: str = "array") -> None:
    """
    Validate that an array is suitable for image processing.
    
    Checks that the array is a valid numpy array with correct dimensions
    and no invalid values (NaN or Inf).
    
    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str, optional
        Name of the array for error messages, by default "array".
        
    Raises
    ------
    ImageProcessingError
        If the array is invalid.
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100, 3)
    >>> validate_array(img, "test_image")  # No error
    >>> invalid = np.array([1, 2, 3])
    >>> validate_array(invalid)  # Raises ImageProcessingError
    """
    if not isinstance(arr, np.ndarray):
        raise ImageProcessingError(f"{name} must be a numpy array")

    if arr.ndim not in [2, 3]:
        raise ImageProcessingError(
            f"{name} must be 2D (grayscale) or 3D (color) array"
        )

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ImageProcessingError(f"{name} contains NaN or Inf values")


def validate_image_path(image_path: str) -> None:
    """
    Validate that an image file path exists and is accessible.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
        
    Raises
    ------
    ImageProcessingError
        If the path is invalid or inaccessible.
        
    Examples
    --------
    >>> validate_image_path("photo.jpg")  # Raises if file doesn't exist
    """
    if not image_path:
        raise ImageProcessingError("Image path cannot be empty")

    if not os.path.exists(image_path):
        raise ImageProcessingError(f"Image file not found: {image_path}")

    if not os.path.isfile(image_path):
        raise ImageProcessingError(f"The path is not a file: {image_path}")


def validate_processing_params(
    scale_factor: float,
    order: int,
    order_rescale: int,
    order_rotate: int,
    denoise_type: str,
    denoise_sigma: float,
    sharpen: bool,
    sharpen_amount: float,
    equalize: bool,
    equalize_method: str,
    clip_limit: float,
    clip_limit_kernel_size: int,
    contrast_stretch_percentiles: Tuple[float, float],
    gamma_correction: float,
    color_preservation: str,
    color_preservation_strength: float,
    application_type: str,
) -> None:
    """
    Validate all processing parameters.
    
    Performs comprehensive validation of all image processing parameters
    to ensure they are within valid ranges and of correct types.
    
    Parameters
    ----------
    scale_factor : float
        Scaling factor for image resizing.
    order : int
        Interpolation order (0-5).
    order_rescale : int
        Interpolation order for rescaling (0-5).
    order_rotate : int
        Interpolation order for rotation (0-5).
    denoise_type : str
        Type of denoising to apply.
    denoise_sigma : float
        Sigma value for denoising.
    sharpen : bool
        Whether to apply sharpening.
    sharpen_amount : float
        Amount of sharpening to apply.
    equalize : bool
        Whether to apply histogram equalization.
    equalize_method : str
        Method of equalization.
    clip_limit : float
        CLAHE clip limit.
    clip_limit_kernel_size : int
        CLAHE kernel size.
    contrast_stretch_percentiles : Tuple[float, float]
        Percentiles for contrast stretching.
    gamma_correction : float
        Gamma correction value.
    color_preservation : str
        Color preservation method.
    color_preservation_strength : float
        Strength of color preservation (0.0-1.0).
    application_type : str
        Application type for optimization.
        
    Raises
    ------
    ImageProcessingError
        If any parameter is invalid.
    """
    if scale_factor <= 0:
        raise ImageProcessingError("Scale factor must be positive")

    if denoise_type not in VALID_DENOISE_TYPES:
        raise ImageProcessingError(
            f"Denoise type must be one of {VALID_DENOISE_TYPES}"
        )

    if denoise_sigma < 0:
        raise ImageProcessingError("Denoise sigma must be non-negative")

    if sharpen_amount <= 0:
        raise ImageProcessingError("Sharpen amount must be positive")

    if equalize_method not in VALID_EQUALIZE_METHODS:
        raise ImageProcessingError(
            f"Equalize method must be one of {VALID_EQUALIZE_METHODS}"
        )

    if clip_limit <= 0:
        raise ImageProcessingError("CLAHE clip limit must be positive")

    if clip_limit_kernel_size <= 0:
        raise ImageProcessingError("CLAHE kernel size must be positive")

    if not (
        0 <= contrast_stretch_percentiles[0] < contrast_stretch_percentiles[1] <= 100
    ):
        raise ImageProcessingError(
            "Contrast stretch percentiles must be in range [0,100] and low < high"
        )

    if gamma_correction <= 0:
        raise ImageProcessingError("Gamma correction must be positive")

    if color_preservation not in VALID_COLOR_METHODS:
        raise ImageProcessingError(
            f"Color preservation method must be one of {VALID_COLOR_METHODS}"
        )

    if not (0.0 <= color_preservation_strength <= 1.0):
        raise ImageProcessingError(
            "Color preservation strength must be between 0.0 and 1.0"
        )

    if application_type not in VALID_APP_TYPES:
        raise ImageProcessingError(
            f"Application type must be one of {VALID_APP_TYPES}"
        )

    if not all(
        0 <= order_val <= 5 for order_val in [order, order_rescale, order_rotate]
    ):
        raise ImageProcessingError("Interpolation orders must be between 0 and 5")


# Backward compatibility aliases with leading underscore
_validate_image_path = validate_image_path
_validate_processing_params = validate_processing_params
