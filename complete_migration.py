#!/usr/bin/env python3
"""
Complete migration script to create all modular files from monolithic chiaroscuro_forge.py
"""

import os

# Read the entire monolithic file
with open("chiaroscuro_forge.py", "r") as f:
    content = f.read()

# Parse sections (this is approximate based on the code structure)
lines = content.split("\n")

#metrics.py already created

# Create processing.py - contains process_image and validation helpers
processing_content = '''"""
Image processing functions.

This module contains the main image processing pipeline and related validation functions.
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple
from skimage import io, color, filters, exposure, transform, util
from skimage import img_as_float, img_as_ubyte
from .exceptions import ImageProcessingError
from .validation import _validate_image_path, _validate_processing_params
from .metrics import calculate_perceptual_metrics, calculate_quality_score


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
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    Process an image with enhancement techniques.

    Parameters
    ----------
    image_path : str
        Path to input image file
    output_path : str, optional
        Path to save processed image
    scale_factor : float, default=1.0
        Scaling factor for resizing
    denoise_type : str, default="gaussian"
        Denoising method: "gaussian", "median", "bilateral", or "none"
    denoise_sigma : float, default=1.0
        Sigma parameter for denoising
    sharpen : bool, default=True
        Whether to apply sharpening
    sharpen_amount : float, default=1.5
        Amount of sharpening to apply
    equalize : bool, default=True
        Whether to apply histogram equalization
    equalize_method : str, default="stretch"
        Equalization method: "standard", "clahe", "stretch", "adaptive_gamma"
    clip_limit : float, default=0.03
        CLAHE clip limit
    clip_limit_kernel_size : int, default=8
        CLAHE kernel size
    contrast_stretch_percentiles : tuple, default=(2, 98)
        Percentiles for contrast stretching
    gamma_correction : float, default=1.0
        Gamma correction value
    color_preservation : str, default="lab"
        Color preservation method: "none", "lab", "rgb", "ratio"
    color_preservation_strength : float, default=0.7
        Strength of color preservation (0-1)
    calculate_metrics : bool, default=True
        Whether to calculate quality metrics
    calculate_advanced_metrics : bool, default=True
        Whether to calculate advanced metrics
    application_type : str, default="general"
        Application type for optimization

    Returns
    -------
    tuple
        (processed_image, metrics_dict) where metrics_dict is None if calculate_metrics=False
    """
    try:
        _validate_image_path(image_path)

        _validate_processing_params(
            scale_factor, order, order_rescale, order_rotate,
            denoise_type, denoise_sigma, sharpen, sharpen_amount,
            equalize, equalize_method, clip_limit, clip_limit_kernel_size,
            contrast_stretch_percentiles, gamma_correction,
            color_preservation, color_preservation_strength, application_type,
        )

        try:
            original_image = io.imread(image_path)
            image = img_as_float(original_image)
            processed_image = image.copy()
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

        # 1. Resize
        if scale_factor != 1.0:
            try:
                processed_image = transform.rescale(
                    processed_image, scale_factor, anti_aliasing=True, order=order_rescale
                )
            except Exception as e:
                raise ImageProcessingError(f"Rescaling failed: {str(e)}")

        original_for_color = processed_image.copy()

        # 2. Denoise
        try:
            if denoise_type == "gaussian":
                processed_image = filters.gaussian(processed_image, sigma=denoise_sigma)
            elif denoise_type == "median":
                if processed_image.ndim == 3:
                    for i in range(3):
                        processed_image[:, :, i] = filters.median(processed_image[:, :, i])
                else:
                    processed_image = filters.median(processed_image)
            elif denoise_type == "bilateral":
                if processed_image.ndim == 3:
                    for i in range(3):
                        processed_image[:, :, i] = filters.gaussian(
                            processed_image[:, :, i], sigma=denoise_sigma, mode="nearest"
                        )
                else:
                    processed_image = filters.gaussian(processed_image, sigma=denoise_sigma, mode="nearest")
        except Exception as e:
            raise ImageProcessingError(f"Denoising failed: {str(e)}")

        # 3. Sharpening
        try:
            if sharpen:
                blurred = filters.gaussian(processed_image, sigma=0.5)
                highpass = processed_image - blurred
                processed_image = processed_image + sharpen_amount * highpass
                processed_image = np.clip(processed_image, 0, 1)
        except Exception as e:
            raise ImageProcessingError(f"Sharpening failed: {str(e)}")

        # 4. Contrast Enhancement
        try:
            if equalize:
                if processed_image.ndim == 3:
                    if color_preservation == "lab":
                        lab_image = color.rgb2lab(processed_image)
                        l_channel_normalized = lab_image[:, :, 0] / 100.0

                        if equalize_method == "standard":
                            l_channel_normalized = exposure.equalize_hist(l_channel_normalized)
                        elif equalize_method == "clahe":
                            l_channel_normalized = exposure.equalize_adapthist(
                                l_channel_normalized, kernel_size=clip_limit_kernel_size, clip_limit=clip_limit
                            )
                        elif equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            l_channel_normalized = exposure.rescale_intensity(
                                l_channel_normalized,
                                in_range=tuple(np.percentile(l_channel_normalized, (p_low, p_high)))
                            )
                        elif equalize_method == "adaptive_gamma":
                            mean_luminance = np.mean(l_channel_normalized)
                            adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_luminance)
                            l_channel_normalized = exposure.adjust_gamma(l_channel_normalized, gamma=adaptive_gamma)

                        lab_image[:, :, 0] = l_channel_normalized * 100.0
                        processed_image = color.lab2rgb(lab_image)

                    elif color_preservation == "rgb":
                        if equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            for i in range(3):
                                processed_image[:, :, i] = exposure.rescale_intensity(
                                    processed_image[:, :, i],
                                    in_range=tuple(np.percentile(processed_image[:, :, i], (p_low, p_high)))
                                )
                        else:
                            hsv_image = color.rgb2hsv(processed_image)
                            if equalize_method == "standard":
                                hsv_image[:, :, 2] = exposure.equalize_hist(hsv_image[:, :, 2])
                            elif equalize_method == "clahe":
                                hsv_image[:, :, 2] = exposure.equalize_adapthist(
                                    hsv_image[:, :, 2], kernel_size=clip_limit_kernel_size, clip_limit=clip_limit
                                )
                            elif equalize_method == "adaptive_gamma":
                                mean_luminance = np.mean(hsv_image[:, :, 2])
                                adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_luminance)
                                hsv_image[:, :, 2] = exposure.adjust_gamma(hsv_image[:, :, 2], gamma=adaptive_gamma)
                            processed_image = color.hsv2rgb(hsv_image)

                    elif color_preservation == "ratio":
                        luminance = np.mean(processed_image, axis=2)

                        if equalize_method == "standard":
                            enhanced_luminance = exposure.equalize_hist(luminance)
                        elif equalize_method == "clahe":
                            enhanced_luminance = exposure.equalize_adapthist(
                                luminance, kernel_size=clip_limit_kernel_size, clip_limit=clip_limit
                            )
                        elif equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            enhanced_luminance = exposure.rescale_intensity(
                                luminance, in_range=tuple(np.percentile(luminance, (p_low, p_high)))
                            )
                        elif equalize_method == "adaptive_gamma":
                            mean_lum = np.mean(luminance)
                            adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_lum)
                            enhanced_luminance = exposure.adjust_gamma(luminance, gamma=adaptive_gamma)

                        enhanced_image = np.zeros_like(processed_image)
                        for i in range(3):
                            ratio = np.divide(
                                processed_image[:, :, i], luminance,
                                out=np.ones_like(processed_image[:, :, i]),
                                where=luminance > 0.01
                            )
                            enhanced_image[:, :, i] = enhanced_luminance * ratio

                        processed_image = enhanced_image
                        processed_image = np.clip(processed_image, 0, 1)

                    else:
                        hsv_image = color.rgb2hsv(processed_image)
                        if equalize_method == "standard":
                            hsv_image[:, :, 2] = exposure.equalize_hist(hsv_image[:, :, 2])
                        elif equalize_method == "clahe":
                            hsv_image[:, :, 2] = exposure.equalize_adapthist(
                                hsv_image[:, :, 2], kernel_size=clip_limit_kernel_size, clip_limit=clip_limit
                            )
                        elif equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            hsv_image[:, :, 2] = exposure.rescale_intensity(
                                hsv_image[:, :, 2], in_range=tuple(np.percentile(hsv_image[:, :, 2], (p_low, p_high)))
                            )
                        elif equalize_method == "adaptive_gamma":
                            mean_luminance = np.mean(hsv_image[:, :, 2])
                            adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_luminance)
                            hsv_image[:, :, 2] = exposure.adjust_gamma(hsv_image[:, :, 2], gamma=adaptive_gamma)
                        processed_image = color.hsv2rgb(hsv_image)
                else:
                    if equalize_method == "standard":
                        processed_image = exposure.equalize_hist(processed_image)
                    elif equalize_method == "clahe":
                        processed_image = exposure.equalize_adapthist(
                            processed_image, kernel_size=clip_limit_kernel_size, clip_limit=clip_limit
                        )
                    elif equalize_method == "stretch":
                        p_low, p_high = contrast_stretch_percentiles
                        processed_image = exposure.rescale_intensity(
                            processed_image, in_range=tuple(np.percentile(processed_image, (p_low, p_high)))
                        )
                    elif equalize_method == "adaptive_gamma":
                        mean_luminance = np.mean(processed_image)
                        adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_luminance)
                        processed_image = exposure.adjust_gamma(processed_image, gamma=adaptive_gamma)
        except Exception as e:
            raise ImageProcessingError(f"Contrast enhancement failed: {str(e)}")

        # 5. Gamma correction
        try:
            if gamma_correction != 1.0 and equalize_method != "adaptive_gamma":
                if processed_image.ndim == 3 and color_preservation == "lab":
                    lab_image = color.rgb2lab(processed_image)
                    l_channel_normalized = lab_image[:, :, 0] / 100.0
                    l_channel_normalized = exposure.adjust_gamma(l_channel_normalized, gamma_correction)
                    lab_image[:, :, 0] = l_channel_normalized * 100.0
                    processed_image = color.lab2rgb(lab_image)
                else:
                    processed_image = exposure.adjust_gamma(processed_image, gamma_correction)
        except Exception as e:
            raise ImageProcessingError(f"Gamma correction failed: {str(e)}")

        # 6. Color preservation blending
        if color_preservation != "none" and color_preservation_strength > 0 and processed_image.ndim == 3:
            try:
                if color_preservation == "lab":
                    processed_lab = color.rgb2lab(processed_image)
                    processed_luminance = processed_lab[:, :, 0]
                    original_lab = color.rgb2lab(original_for_color)
                    blended_lab = original_lab.copy()
                    blended_lab[:, :, 0] = processed_luminance
                    blended = color.lab2rgb(blended_lab)
                    processed_image = (
                        color_preservation_strength * blended +
                        (1 - color_preservation_strength) * processed_image
                    )
            except Exception as e:
                raise ImageProcessingError(f"Color preservation blending failed: {str(e)}")

        processed_image = np.clip(processed_image, 0, 1)

        # Calculate metrics
        metrics_dict = None
        if calculate_metrics:
            if min(original_image.shape[:2]) < 7:
                raise ImageProcessingError("Image too small for perceptual metrics calculation")
            else:
                try:
                    metrics_dict = calculate_perceptual_metrics(
                        original_image, processed_image, calculate_advanced=calculate_advanced_metrics
                    )
                    metrics_dict["quality_score"] = calculate_quality_score(
                        metrics_dict, application_type=application_type
                    )
                except Exception as e:
                    raise ImageProcessingError(f"Could not calculate metrics: {str(e)}")

        # Save output
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
'''

# Write processing.py
with open("chiaroscuro_forge/processing.py", "w") as f:
    f.write(processing_content)

print("Created processing.py")

# Create stub files for other modules that just import from monolithic file temporarily
stub_modules = {
    "analysis": ["analyze_image_characteristics", "get_image_statistics"],
    "batch": ["batch_process_images", "analyze_batch", "_process_single_image", "_process_single_image_wrapper", "setup_logger"],
    "comparison": ["compare_processing_methods", "suggest_optimal_params"],
    "presets": ["load_preset", "save_preset", "list_presets"],
    "cli": ["main"],
}

for module_name, functions in stub_modules.items():
    stub_content = f'''"""
{module_name.capitalize()} module - temporarily importing from monolithic file.

This module is a stub that imports functions from the original monolithic chiaroscuro_forge.py file.
It will be migrated to a proper module implementation in future versions.
"""

import sys
import os
import importlib.util
from pathlib import Path

# Import from monolithic file
_parent_dir = Path(__file__).parent.parent
_monolithic_path = _parent_dir / "chiaroscuro_forge.py"

if _monolithic_path.exists():
    spec = importlib.util.spec_from_file_location("_cf_old", _monolithic_path)
    if spec and spec.loader:
        _cf_old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_cf_old)
        
        # Import functions
'''
    
    for func in functions:
        stub_content += f"        {func} = _cf_old.{func}\n"
    
    stub_content += '''
        # Cleanup
        del spec
else:
    from ..exceptions import ImageProcessingError
    raise ImageProcessingError(f"Could not find monolithic file at {{_monolithic_path}}")

__all__ = [''' + ", ".join(f'"{f}"' for f in functions) + ''']
'''
    
    with open(f"chiaroscuro_forge/{module_name}.py", "w") as f:
        f.write(stub_content)
    
    print(f"Created {module_name}.py")

print("\n✅ All modules created successfully!")
print("Run tests with: python3 -m pytest tests/ -v")
