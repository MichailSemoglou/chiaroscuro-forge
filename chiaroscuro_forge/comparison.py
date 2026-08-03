"""
Comparison and optimization functions for processing methods.

This module provides functionality for comparing different processing methods
and suggesting optimal parameters based on image or batch analysis.
"""

import os
from typing import Any, Dict, Optional

from .config import ProcessingConfig
from .exceptions import ImageProcessingError
from .processing import process_image
from .validation import _validate_image_path, _validate_output_path


def compare_processing_methods(
    image_path: str,
    output_dir: Optional[str] = None,
    application_type: str = "general",
    calculate_advanced_metrics: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple processing methods on a single image.

    Args:
        image_path: Path to input image
        output_dir: Optional directory to save processed images
        application_type: Application type for processing optimization
        calculate_advanced_metrics: Whether to calculate advanced quality metrics

    Returns:
        Dictionary with results for each method and the best method
    """
    _validate_image_path(image_path)

    valid_app_types = ["general", "photography", "medical", "document", "art"]
    if application_type not in valid_app_types:
        raise ImageProcessingError(f"Application type must be one of {valid_app_types}")

    if output_dir:
        output_dir = _validate_output_path(output_dir)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                raise ImageProcessingError(
                    f"Failed to create output directory: {type(e).__name__}"
                ) from e

    methods = [
        {
            "name": "Standard Equalization (HSV)",
            "config": ProcessingConfig(
                equalize_method="standard",
                color_preservation="none",
                denoise_type="gaussian",
                denoise_sigma=0.8,
                sharpen=True,
                sharpen_amount=1.2,
                gamma_correction=1.05,
            ),
        },
        {
            "name": "LAB Color Preservation",
            "config": ProcessingConfig(
                equalize_method="stretch",
                color_preservation="lab",
                color_preservation_strength=0.9,
                denoise_type="gaussian",
                denoise_sigma=0.8,
                sharpen=True,
                sharpen_amount=1.2,
                gamma_correction=1.05,
            ),
        },
        {
            "name": "Gentle Enhancement",
            "config": ProcessingConfig(
                equalize_method="stretch",
                contrast_stretch_percentiles=(5, 95),
                color_preservation="lab",
                color_preservation_strength=0.9,
                denoise_type="gaussian",
                denoise_sigma=0.5,
                sharpen=True,
                sharpen_amount=1.0,
                gamma_correction=1.0,
            ),
        },
        {
            "name": "Color Ratio Preservation",
            "config": ProcessingConfig(
                equalize_method="stretch",
                color_preservation="ratio",
                contrast_stretch_percentiles=(2, 98),
                denoise_type="gaussian",
                denoise_sigma=0.8,
                sharpen=True,
                sharpen_amount=1.2,
                gamma_correction=1.05,
            ),
        },
        {
            "name": "Detail Preservation",
            "config": ProcessingConfig(
                equalize_method="clahe",
                clip_limit=0.02,
                clip_limit_kernel_size=16,
                color_preservation="lab",
                color_preservation_strength=0.8,
                denoise_type="bilateral",
                denoise_sigma=0.6,
                sharpen=True,
                sharpen_amount=1.4,
                gamma_correction=1.0,
            ),
        },
        {
            "name": "High Dynamic Range (HDR)",
            "config": ProcessingConfig(
                equalize_method="clahe",
                clip_limit=0.03,
                clip_limit_kernel_size=16,
                color_preservation="lab",
                color_preservation_strength=0.85,
                denoise_type="bilateral",
                denoise_sigma=0.6,
                sharpen=True,
                sharpen_amount=1.6,
                gamma_correction=0.9,
            ),
        },
        {
            "name": "Vintage Look",
            "config": ProcessingConfig(
                equalize_method="standard",
                color_preservation="lab",
                color_preservation_strength=0.5,
                denoise_type="median",
                denoise_sigma=1.0,
                sharpen=False,
                gamma_correction=1.2,
            ),
        },
        {
            "name": "Black & White Contrast",
            "config": ProcessingConfig(
                equalize_method="stretch",
                contrast_stretch_percentiles=(2, 98),
                color_preservation="none",
                denoise_type="gaussian",
                denoise_sigma=0.7,
                sharpen=True,
                sharpen_amount=1.3,
                gamma_correction=1.1,
            ),
        },
        {
            "name": "Focus Enhancement",
            "config": ProcessingConfig(
                equalize_method="stretch",
                contrast_stretch_percentiles=(5, 95),
                color_preservation="lab",
                color_preservation_strength=0.95,
                denoise_type="bilateral",
                denoise_sigma=0.4,
                sharpen=True,
                sharpen_amount=2.0,
                gamma_correction=1.0,
            ),
        },
    ]

    results = {}
    best_method = None
    best_score = -1.0

    for method in methods:
        name = method["name"]
        m_config = method["config"].merge({"application_type": application_type})

        if output_dir:
            output_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.jpg")
        else:
            output_path = None

        try:
            m_config.calculate_advanced_metrics = calculate_advanced_metrics
            processed, metrics = process_image(
                image_path,
                output_path=output_path,
                config=m_config,
            )

            results[name] = {
                "processed": processed,
                "metrics": metrics,
                "output_path": output_path,
            }

            if metrics and "quality_score" in metrics and metrics["quality_score"] > best_score:
                best_score = metrics["quality_score"]
                best_method = name

        except Exception as e:
            results[name] = {"error": str(e)}

    if best_method:
        results["best_method"] = {"name": best_method, "score": best_score}

    return results


def suggest_optimal_params(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest optimal processing parameters based on batch analysis results.

    Args:
        analysis_results: Results from analyze_batch function

    Returns:
        Dictionary with suggested parameters
    """
    # Extract summary statistics
    summary = analysis_results.get("summary", {})
    total_images = analysis_results.get("total_images", 0)

    avg_brightness = (
        summary.get("brightness", {}).get("avg", 0.5) if isinstance(summary, dict) else 0.5
    )
    avg_contrast = summary.get("contrast", {}).get("avg", 0.3) if isinstance(summary, dict) else 0.3
    avg_noise = (
        summary.get("noise_level", {}).get("avg", 0.03) if isinstance(summary, dict) else 0.03
    )
    avg_edge_density = (
        summary.get("edge_density", {}).get("avg", 0.05) if isinstance(summary, dict) else 0.05
    )

    color_images = summary.get("color_images", 0) if isinstance(summary, dict) else 0
    is_mostly_color = color_images > (total_images / 2) if total_images else False

    # Start with default parameters
    params = {
        "equalize_method": "stretch",
        "contrast_stretch_percentiles": (5, 95),
        "denoise_type": "gaussian",
        "denoise_sigma": 0.8,
        "sharpen": True,
        "sharpen_amount": 1.2,
        "gamma_correction": 1.0,
        "color_preservation": "lab" if is_mostly_color else "none",
        "color_preservation_strength": 0.8 if is_mostly_color else 0.0,
    }

    # Adjust based on average characteristics

    # Brightness adjustment
    if avg_brightness < 0.4:
        params["gamma_correction"] = 0.85
    elif avg_brightness > 0.7:
        params["gamma_correction"] = 1.15

    # Contrast adjustment
    if avg_contrast < 0.15:
        params["equalize_method"] = "clahe"
        params["clip_limit"] = 0.03
        params["clip_limit_kernel_size"] = 8
    elif avg_contrast < 0.25:
        params["contrast_stretch_percentiles"] = (2, 98)
    else:
        params["contrast_stretch_percentiles"] = (5, 95)

    # Noise adjustment
    if avg_noise > 0.06:
        params["denoise_type"] = "bilateral" if avg_edge_density > 0.05 else "gaussian"
        params["denoise_sigma"] = min(1.5, avg_noise * 15)
    elif avg_noise > 0.03:
        params["denoise_type"] = "gaussian"
        params["denoise_sigma"] = min(1.0, avg_noise * 12)
    else:
        params["denoise_type"] = "gaussian"
        params["denoise_sigma"] = 0.5

    # Edge/sharpening adjustment
    if avg_edge_density < 0.02:
        params["sharpen"] = True
        params["sharpen_amount"] = 1.8
    elif avg_edge_density > 0.1:
        params["sharpen"] = True
        params["sharpen_amount"] = 0.9
    else:
        params["sharpen"] = True
        params["sharpen_amount"] = 1.2

    # Determine application type
    if avg_edge_density > 0.1 and avg_contrast > 0.25:
        application_type = "document"
    elif is_mostly_color and avg_contrast > 0.2:
        application_type = "photography"
    else:
        application_type = "general"

    return {
        "params": params,
        "application_type": application_type,
        "batch_summary": summary,
    }


__all__ = ["compare_processing_methods", "suggest_optimal_params"]
