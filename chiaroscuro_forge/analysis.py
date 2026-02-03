"""
Image analysis and statistics functions.

This module provides tools for analyzing image characteristics and calculating
comprehensive statistics to help determine optimal processing parameters.
"""

import numpy as np
from typing import Dict, Any, Union
from skimage import io, color, filters, feature
from skimage import img_as_float, img_as_ubyte
from .exceptions import ImageProcessingError
from .validation import _validate_image_path, validate_array
from .cache import cached_image_stats


@cached_image_stats(ttl=3600)  # Cache for 1 hour
def analyze_image_characteristics(image_path: str) -> Dict[str, Any]:
    """
    Analyze image characteristics and suggest optimal processing parameters.

    Examines color saturation, brightness, contrast, noise level, edge density,
    and texture to automatically determine appropriate enhancement settings.
    
    Results are cached for 1 hour to improve performance on repeated analyses.

    Parameters
    ----------
    image_path : str
        Path to the image file to analyze

    Returns
    -------
    dict
        Dictionary containing:
        - characteristics: Dict of measured image properties
        - suggested_params: Dict of recommended processing parameters
        - suggested_application: Recommended application type

    Raises
    ------
    ImageProcessingError
        If image cannot be loaded or analyzed

    Examples
    --------
    >>> analysis = analyze_image_characteristics("photo.jpg")
    >>> print(f"Brightness: {analysis['characteristics']['brightness']:.2f}")
    >>> process_image("photo.jpg", **analysis['suggested_params'])
    """
    try:
        _validate_image_path(image_path)

        image = img_as_float(io.imread(image_path))

        is_color = image.ndim == 3 and image.shape[2] >= 3

        if is_color:
            lab_image = color.rgb2lab(image)

            luminance = lab_image[:, :, 0]
            min_luminance = np.min(luminance)
            max_luminance = np.max(luminance)
            mean_luminance = np.mean(luminance)
            std_luminance = np.std(luminance)

            contrast = std_luminance / 100.0

            color_a = lab_image[:, :, 1]
            color_b = lab_image[:, :, 2]

            color_saturation = np.sqrt(np.mean(color_a**2 + color_b**2)) / 128.0
            color_variance = (np.std(color_a) + np.std(color_b)) / 128.0

        else:
            min_luminance = np.min(image) * 100
            max_luminance = np.max(image) * 100
            mean_luminance = np.mean(image) * 100
            std_luminance = np.std(image) * 100

            contrast = std_luminance / 100.0
            color_saturation = 0
            color_variance = 0

        try:
            if is_color:
                gray_image = color.rgb2gray(image)
            else:
                gray_image = image

            noise_estimator = filters.rank.windowed_variance(
                img_as_ubyte(gray_image), filters.rank.square(5)
            )
            noise_level = np.mean(noise_estimator) / 255.0
        except Exception:
            if is_color:
                blue_channel = image[:, :, 2]
                noise_level = np.std(
                    blue_channel - filters.gaussian(blue_channel, sigma=1)
                )
            else:
                noise_level = np.std(image - filters.gaussian(image, sigma=1))

        edges = feature.canny(color.rgb2gray(image) if is_color else image, sigma=1.0)
        edge_density = np.mean(edges)

        if is_color:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image

        gradient_x = filters.sobel_h(gray_image)
        gradient_y = filters.sobel_v(gray_image)
        gradient_magnitude = np.hypot(gradient_x, gradient_y)
        texture_level = np.mean(gradient_magnitude)

        characteristics = {
            "is_color": is_color,
            "brightness": mean_luminance / 100.0,
            "contrast": contrast,
            "dynamic_range": (max_luminance - min_luminance) / 100.0,
            "color_saturation": color_saturation,
            "color_variance": color_variance,
            "noise_level": noise_level,
            "edge_density": edge_density,
            "texture_level": texture_level,
        }

        suggested_params = {}

        if noise_level > 0.05:
            suggested_params["denoise_type"] = (
                "bilateral" if edge_density > 0.05 else "gaussian"
            )
            suggested_params["denoise_sigma"] = min(1.5, noise_level * 15)
        else:
            suggested_params["denoise_type"] = "gaussian"
            suggested_params["denoise_sigma"] = 0.5

        if contrast < 0.15:
            if texture_level > 0.1:
                suggested_params["equalize_method"] = "clahe"
                suggested_params["clip_limit"] = min(0.03, 0.01 + contrast)
                suggested_params["clip_limit_kernel_size"] = 8
            else:
                suggested_params["equalize_method"] = "stretch"
                suggested_params["contrast_stretch_percentiles"] = (
                    max(1, 10 - int(contrast * 100)),
                    min(99, 90 + int(contrast * 100)),
                )
        else:
            suggested_params["equalize_method"] = "stretch"
            suggested_params["contrast_stretch_percentiles"] = (5, 95)

        if mean_luminance / 100.0 < 0.4:
            suggested_params["gamma_correction"] = 0.8
        elif mean_luminance / 100.0 > 0.7:
            suggested_params["gamma_correction"] = 1.2
        else:
            suggested_params["gamma_correction"] = 1.0

        if edge_density < 0.02:
            suggested_params["sharpen"] = True
            suggested_params["sharpen_amount"] = 1.5
        elif edge_density > 0.1:
            suggested_params["sharpen"] = True
            suggested_params["sharpen_amount"] = 0.8
        else:
            suggested_params["sharpen"] = True
            suggested_params["sharpen_amount"] = 1.2

        if is_color:
            if color_saturation > 0.3 or color_variance > 0.2:
                suggested_params["color_preservation"] = "lab"
                suggested_params["color_preservation_strength"] = 0.9
            else:
                suggested_params["color_preservation"] = "lab"
                suggested_params["color_preservation_strength"] = 0.7
        else:
            suggested_params["color_preservation"] = "none"

        if edge_density > 0.1 and texture_level < 0.05:
            suggested_application = "document"
        elif color_saturation > 0.3 and color_variance > 0.2:
            suggested_application = "art"
        elif edge_density > 0.05 and texture_level > 0.1:
            suggested_application = "photography"
        else:
            suggested_application = "general"

        return {
            "characteristics": characteristics,
            "suggested_params": suggested_params,
            "suggested_application": suggested_application,
        }

    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(f"Error analyzing image: {str(e)}")


def get_image_statistics(image: Union[str, np.ndarray]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for an image.

    Provides a quick overview of image properties including intensity statistics,
    histogram information, and basic quality indicators.

    Parameters
    ----------
    image : str or np.ndarray
        Either a file path to an image or a numpy array containing image data

    Returns
    -------
    dict
        Dictionary containing:
        - dimensions: (height, width, channels)
        - dtype: Data type of array
        - intensity: Min, max, mean, std, median values
        - histogram: Percentile values (p1, p5, p25, p50, p75, p95, p99)
        - dynamic_range: Effective dynamic range
        - is_color: Boolean indicating color vs grayscale
        - contrast_ratio: Ratio between brightest and darkest regions
        - brightness: Overall brightness (0-1 scale)

    Raises
    ------
    ImageProcessingError
        If image cannot be loaded or is invalid

    Examples
    --------
    >>> stats = get_image_statistics("photo.jpg")
    >>> print(f"Image brightness: {stats['brightness']:.2f}")
    >>> print(f"Dynamic range: {stats['dynamic_range']:.2f}")
    """
    # Load image if path is provided
    if isinstance(image, str):
        _validate_image_path(image)
        try:
            img = io.imread(image)
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {e}")
    else:
        img = image

    validate_array(img, "image")

    # Convert to float for consistent calculations
    img_float = img_as_float(img)

    # Determine if color or grayscale
    is_color = img_float.ndim == 3 and img_float.shape[2] >= 3

    # Calculate dimensions
    if is_color:
        height, width, channels = img_float.shape
    else:
        height, width = img_float.shape
        channels = 1

    # Flatten for statistics (use luminance for color images)
    if is_color:
        luminance = 0.2126 * img_float[:, :, 0] + 0.7152 * img_float[:, :, 1] + 0.0722 * img_float[:, :, 2]
        flat = luminance.flatten()
    else:
        flat = img_float.flatten()

    # Calculate intensity statistics
    intensity_stats = {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
    }

    # Calculate percentiles for histogram analysis
    percentiles = {
        "p1": float(np.percentile(flat, 1)),
        "p5": float(np.percentile(flat, 5)),
        "p25": float(np.percentile(flat, 25)),
        "p50": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
    }

    # Calculate dynamic range (using 1st and 99th percentile to avoid outliers)
    dynamic_range = percentiles["p99"] - percentiles["p1"]

    # Calculate contrast ratio
    dark_region = percentiles["p5"]
    bright_region = percentiles["p95"]
    contrast_ratio = bright_region / max(dark_region, 0.001)

    # Overall brightness (mean normalized to 0-1)
    brightness = intensity_stats["mean"]

    return {
        "dimensions": (height, width, channels),
        "dtype": str(img.dtype),
        "intensity": intensity_stats,
        "histogram": percentiles,
        "dynamic_range": float(dynamic_range),
        "is_color": is_color,
        "contrast_ratio": float(contrast_ratio),
        "brightness": float(brightness),
    }


__all__ = ["analyze_image_characteristics", "get_image_statistics"]
