"""
Validation Utilities for Chiaroscuro Forge

This module provides validation functions for image arrays, file paths,
and processing parameters.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from chiaroscuro_forge.constants import (
    ALLOWED_EXTENSIONS,
    DENIED_PATH_PATTERNS,
    MAX_DIMENSION,
    MAX_FILE_SIZE_MB,
    MAX_IMAGE_PIXELS,
    MAX_PATH_LENGTH,
    VALID_APP_TYPES,
    VALID_COLOR_METHODS,
    VALID_DENOISE_TYPES,
    VALID_EQUALIZE_METHODS,
)
from chiaroscuro_forge.exceptions import ImageProcessingError

if TYPE_CHECKING:
    from chiaroscuro_forge.config import ProcessingConfig


def validate_array(
    arr: np.ndarray,
    name: str = "array",
    max_pixels: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> None:
    """
    Validate that an array is suitable for image processing with security limits.

    Checks that the array is a valid numpy array with correct dimensions,
    no invalid values (NaN or Inf), and within safe size limits to prevent
    memory exhaustion attacks.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str, optional
        Name of the array for error messages, by default "array".
    max_pixels : int, optional
        Maximum total pixels allowed. Defaults to MAX_IMAGE_PIXELS.
    max_dimension : int, optional
        Maximum single dimension allowed. Defaults to MAX_DIMENSION.

    Raises
    ------
    ImageProcessingError
        If the array is invalid, too large, or contains invalid values.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100, 3)
    >>> validate_array(img, "test_image")  # No error
    >>> huge = np.zeros((50000, 50000))  # Too large
    >>> validate_array(huge)  # Raises ImageProcessingError
    """
    if not isinstance(arr, np.ndarray):
        raise ImageProcessingError(f"{name} must be a numpy array")

    if arr.ndim not in [2, 3]:
        raise ImageProcessingError(f"{name} must be 2D (grayscale) or 3D (color) array")

    # Security: Check dimension limits (memory bomb prevention)
    max_pix = max_pixels or MAX_IMAGE_PIXELS
    max_dim = max_dimension or MAX_DIMENSION

    total_pixels = arr.shape[0] * arr.shape[1] if arr.ndim >= 2 else arr.size
    if total_pixels > max_pix or arr.size > max_pix * 4:
        raise ImageProcessingError(
            f"{name} too large: {total_pixels:,} pixels "
            f"(max: {max_pix:,} pixels). Potential memory bomb attack."
        )

    # Check individual dimensions
    for i, dim_size in enumerate(arr.shape[:2]):
        if dim_size > max_dim:
            raise ImageProcessingError(
                f"{name} dimension {i} too large: {dim_size:,} pixels " f"(max: {max_dim:,})"
            )

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ImageProcessingError(f"{name} contains NaN or Inf values")


def _is_path_traversal_tilde(path: str) -> bool:
    """
    Check if ~ in path indicates home directory traversal (not Windows short name).

    Windows uses ~ in 8.3 short filenames (e.g., RUNNER~1), which are legitimate.
    ~ at the start of a path or after a separator indicates home directory expansion.

    Parameters
    ----------
    path : str
        Path to check.

    Returns
    -------
    bool
        True if ~ indicates path traversal, False if it's a Windows short name.
    """
    # Normalize path separators for consistent checking
    normalized = path.replace("\\", "/")

    # Check for ~ at start or after path separator (indicates home directory)
    # Pattern matches: ~/path, /~/path, //~/path, etc.
    if normalized.startswith("~/"):
        return True
    if "/~/" in normalized:
        return True
    if normalized.endswith("/~"):
        return True

    # Check for standalone ~ (entire path is just ~)
    if normalized == "~":
        return True

    # ~ embedded in a path component (like RUNNER~1) is OK - Windows short name
    return False


def _validate_image_path(image_path: str) -> None:
    """
    Comprehensive security validation for image file paths.

    Performs multiple security checks including:
    - Path traversal attack prevention (../, ~, null bytes)
    - Path length limits
    - File size limits
    - Extension whitelist validation
    - File type verification (imghdr)

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Raises
    ------
    ImageProcessingError
        If the path is invalid, insecure, too large, or wrong type.

    Examples
    --------
    >>> _validate_image_path("photo.jpg")  # OK
    >>> _validate_image_path("../etc/passwd")  # Raises - path traversal
    >>> _validate_image_path("huge_file.jpg")  # Raises if >100MB
    """
    # Basic validation
    if not image_path:
        raise ImageProcessingError("Image path cannot be empty")

    # Security: Path length limit
    if len(image_path) > MAX_PATH_LENGTH:
        raise ImageProcessingError(
            f"Path too long: {len(image_path)} characters " f"(max: {MAX_PATH_LENGTH})"
        )

    # Security: Path traversal attack prevention
    path_str = str(image_path)
    for pattern in DENIED_PATH_PATTERNS:
        if pattern == "~":
            # Special handling for ~ to allow Windows 8.3 short filenames
            if _is_path_traversal_tilde(path_str):
                raise ImageProcessingError(
                    f"Path contains denied pattern '{pattern}'. " "Potential path traversal attack."
                )
        elif pattern in path_str:
            raise ImageProcessingError(
                f"Path contains denied pattern '{pattern}'. " "Potential path traversal attack."
            )

    # Security: Resolve and check canonicalization
    try:
        resolved_path = Path(image_path).resolve(strict=False)
        if ".." in resolved_path.parts:
            raise ImageProcessingError(
                "Path contains parent directory references. " "Potential path traversal attack."
            )
    except Exception as e:
        raise ImageProcessingError(f"Invalid path: {e}")

    # Check file exists
    if not os.path.exists(image_path):
        raise ImageProcessingError(f"Image file not found: {image_path}")

    # Check it's a file, not a directory or special file
    if not os.path.isfile(image_path):
        raise ImageProcessingError(f"Path is not a regular file: {image_path}")

    # Security: File size limit (DoS prevention)
    try:
        file_size_bytes = os.path.getsize(image_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ImageProcessingError(
                f"File too large: {file_size_mb:.2f} MB "
                f"(max: {MAX_FILE_SIZE_MB} MB). Potential DoS attack."
            )
    except OSError as e:
        raise ImageProcessingError(f"Cannot access file size: {e}")

    # Security: Extension whitelist
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ImageProcessingError(
            f"File extension '{file_ext}' not allowed. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Security: Verify file type by content (not just extension)
    # Check magic numbers for common image formats
    try:
        with open(image_path, "rb") as f:
            header = f.read(32)

        # Check magic numbers for common image formats
        is_valid_image = (
            header[:2] == b"\xff\xd8"  # JPEG
            or header[:8] == b"\x89PNG\r\n\x1a\n"  # PNG
            or header[:2] in (b"II", b"MM")  # TIFF
            or header[:2] == b"BM"  # BMP
            or (header[:4] == b"RIFF" and header[8:12] == b"WEBP")  # WebP
            or header[:3] == b"GIF"  # GIF
        )

        if not is_valid_image:
            raise ImageProcessingError(
                "File does not appear to be a valid image " "(magic number check failed)"
            )
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(f"Cannot verify file type: {e}")


def validate_image_path(image_path: str) -> None:
    """
    Validate that an image file path is secure and accessible.

    This is the public API that wraps _validate_image_path with
    comprehensive security checks.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Raises
    ------
    ImageProcessingError
        If the path is invalid, insecure, or inaccessible.

    Examples
    --------
    >>> validate_image_path("photo.jpg")  # Raises if file doesn't exist
    >>> validate_image_path("../sensitive.jpg")  # Raises - security issue
    """
    _validate_image_path(image_path)


def _validate_output_path(output_path: str) -> str:

    if not output_path:
        raise ImageProcessingError("Output path cannot be empty")

    path_str = str(output_path)
    if len(path_str) > MAX_PATH_LENGTH:
        raise ImageProcessingError(
            f"Path too long: {len(path_str)} characters " f"(max: {MAX_PATH_LENGTH})"
        )

    for pattern in DENIED_PATH_PATTERNS:
        if pattern == "~":
            if _is_path_traversal_tilde(path_str):
                raise ImageProcessingError(
                    f"Path contains denied pattern '{pattern}'. " "Potential path traversal attack."
                )
        elif pattern in path_str:
            raise ImageProcessingError(
                f"Path contains denied pattern '{pattern}'. " "Potential path traversal attack."
            )

    try:
        resolved_path = Path(output_path).resolve(strict=False)
        if ".." in resolved_path.parts:
            raise ImageProcessingError(
                "Path contains parent directory references after "
                "resolution. Potential path traversal attack."
            )
        return str(resolved_path)
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(f"Invalid output path: {type(e).__name__}")


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
    Validate all processing parameters (delegates to ``validate_config``).

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
    from chiaroscuro_forge.config import ProcessingConfig

    config = ProcessingConfig(
        scale_factor=scale_factor,
        order=order,
        order_rescale=order_rescale,
        order_rotate=order_rotate,
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
        application_type=application_type,
    )
    validate_config(config)


# Note: _validate_image_path is the actual implementation above.
# The public API validate_image_path() calls it.
# Do NOT create backward compat aliases here - they cause infinite recursion.
# For backward compatibility with code expecting underscore prefix:
_validate_processing_params = validate_processing_params


def validate_config(config: "ProcessingConfig") -> None:
    """Validate a ``ProcessingConfig`` instance.

    Parameters
    ----------
    config : ProcessingConfig

    Raises
    ------
    ImageProcessingError
        If any field value is invalid.
    """
    if config.scale_factor <= 0:
        raise ImageProcessingError("Scale factor must be positive")

    if config.denoise_type not in VALID_DENOISE_TYPES:
        raise ImageProcessingError(f"Denoise type must be one of {VALID_DENOISE_TYPES}")

    if config.denoise_sigma < 0:
        raise ImageProcessingError("Denoise sigma must be non-negative")

    if config.sharpen_amount <= 0:
        raise ImageProcessingError("Sharpen amount must be positive")

    if config.equalize_method not in VALID_EQUALIZE_METHODS:
        raise ImageProcessingError(f"Equalize method must be one of {VALID_EQUALIZE_METHODS}")

    if config.clip_limit <= 0:
        raise ImageProcessingError("CLAHE clip limit must be positive")

    if config.clip_limit_kernel_size <= 0:
        raise ImageProcessingError("CLAHE kernel size must be positive")

    p_low, p_high = config.contrast_stretch_percentiles
    if not (0 <= p_low < p_high <= 100):
        raise ImageProcessingError(
            "Contrast stretch percentiles must be in range [0,100] and low < high"
        )

    if config.gamma_correction <= 0:
        raise ImageProcessingError("Gamma correction must be positive")

    if config.color_preservation not in VALID_COLOR_METHODS:
        raise ImageProcessingError(
            f"Color preservation method must be one of {VALID_COLOR_METHODS}"
        )

    if not (0.0 <= config.color_preservation_strength <= 1.0):
        raise ImageProcessingError("Color preservation strength must be between 0.0 and 1.0")

    if config.application_type not in VALID_APP_TYPES:
        raise ImageProcessingError(f"Application type must be one of {VALID_APP_TYPES}")

    for name, val in [
        ("order", config.order),
        ("order_rescale", config.order_rescale),
        ("order_rotate", config.order_rotate),
    ]:
        if not (0 <= val <= 5):
            raise ImageProcessingError(f"Interpolation {name} must be between 0 and 5")
