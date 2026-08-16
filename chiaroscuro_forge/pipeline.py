"""
Image Processing Pipeline

This module implements a clean pipeline pattern for image processing,
reducing complexity and improving maintainability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from skimage import color, exposure, filters, transform

from .exceptions import ImageProcessingError


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage performs a specific transformation on an image
    and can be chained together.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Process the image and return the result.

        Parameters
        ----------
        image : np.ndarray
            Input image to process.
        context : dict
            Shared context with configuration and intermediate results.

        Returns
        -------
        np.ndarray
            Processed image.
        """
        pass

    def __call__(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Allow stage to be called like a function."""
        try:
            return self.process(image, context)
        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"{self.name} stage failed: {str(e)}")


class ResizeStage(PipelineStage):
    """Resize image by scale factor."""

    def __init__(self):
        super().__init__("Resize")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        scale_factor = context.get("scale_factor", 1.0)
        if scale_factor == 1.0:
            return image

        order = context.get("order_rescale", context.get("order", 1))
        resized_image = transform.rescale(
            image,
            scale_factor,
            anti_aliasing=True,
            order=order,
            channel_axis=-1 if image.ndim == 3 else None,
        )

        original_for_color = context.get("original_for_color")
        if original_for_color is not None:
            context["original_for_color"] = np.asarray(
                transform.rescale(
                    original_for_color,
                    scale_factor,
                    anti_aliasing=True,
                    order=order,
                    channel_axis=-1 if original_for_color.ndim == 3 else None,
                )
            )

        return np.asarray(resized_image)


class RotateStage(PipelineStage):
    """Apply rotation with the configured interpolation order."""

    def __init__(self):
        super().__init__("Rotate")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        angle = context.get("rotation_angle")
        if angle in (None, 0):
            return image

        order = context.get("order_rotate", context.get("order", 1))
        rotated = transform.rotate(
            image,
            angle,
            order=order,
            resize=False,
            preserve_range=False,
            mode="reflect",
        )

        original_for_color = context.get("original_for_color")
        if original_for_color is not None:
            context["original_for_color"] = np.asarray(
                transform.rotate(
                    original_for_color,
                    angle,
                    order=order,
                    resize=False,
                    preserve_range=False,
                    mode="reflect",
                )
            )

        return np.asarray(rotated)


class DenoiseStage(PipelineStage):
    """Apply denoising to reduce noise."""

    def __init__(self):
        super().__init__("Denoise")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        denoise_type = context.get("denoise_type", "gaussian")
        denoise_sigma = context.get("denoise_sigma", 1.0)

        if denoise_type == "none":
            return image
        elif denoise_type == "gaussian":
            return np.asarray(
                filters.gaussian(
                    image, sigma=denoise_sigma, channel_axis=-1 if image.ndim == 3 else None
                )
            )
        elif denoise_type == "median":
            if image.ndim == 3:
                result = image.copy()
                for i in range(image.shape[2]):
                    result[:, :, i] = filters.median(image[:, :, i])
                return np.asarray(result)
            return np.asarray(filters.median(image))
        elif denoise_type == "bilateral":
            # True edge-preserving bilateral filter.
            # Note: ~10–100× slower than Gaussian. For throughput-sensitive
            # workloads, consider denoise_type="gaussian" instead.
            from skimage.restoration import denoise_bilateral

            if image.ndim == 3:
                return np.asarray(
                    denoise_bilateral(
                        image,
                        sigma_color=denoise_sigma,
                        sigma_spatial=denoise_sigma / 2,
                        channel_axis=-1,
                    )
                )
            return np.asarray(
                denoise_bilateral(
                    image,
                    sigma_color=denoise_sigma,
                    sigma_spatial=denoise_sigma / 2,
                    channel_axis=None,
                )
            )
        else:
            raise ImageProcessingError(f"Unknown denoise type: {denoise_type}")


class SharpenStage(PipelineStage):
    """Apply unsharp masking for sharpening."""

    def __init__(self):
        super().__init__("Sharpen")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        if not context.get("sharpen", False):
            return image

        sharpen_amount = context.get("sharpen_amount", 1.5)
        blurred = filters.gaussian(image, sigma=0.5, channel_axis=-1 if image.ndim == 3 else None)
        highpass = image - blurred
        sharpened = image + sharpen_amount * highpass
        return np.asarray(np.clip(sharpened, 0, 1))


class ContrastStage(PipelineStage):
    """Apply contrast enhancement."""

    def __init__(self):
        super().__init__("Contrast Enhancement")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        if not context.get("equalize", False):
            return image

        equalize_method = context.get("equalize_method", "stretch")

        # Store original for potential color preservation
        context["pre_contrast_image"] = image.copy()

        if image.ndim == 3:
            # Convert to LAB for processing
            lab_image = color.rgb2lab(image)
            l_channel = lab_image[:, :, 0]

            # Apply method to L channel
            l_enhanced = self._apply_method(l_channel, equalize_method, context)

            # Reconstruct
            lab_enhanced = lab_image.copy()
            lab_enhanced[:, :, 0] = l_enhanced
            enhanced = color.lab2rgb(lab_enhanced)
        else:
            # Grayscale
            enhanced = self._apply_method(image, equalize_method, context)

        return np.asarray(np.clip(enhanced, 0, 1))

    def _apply_method(
        self, channel: np.ndarray, method: str, context: Dict[str, Any]
    ) -> np.ndarray:
        """Apply specific contrast enhancement method."""
        if method == "standard":
            return np.asarray(exposure.equalize_hist(channel))
        elif method == "clahe":
            clip_limit = context.get("clip_limit", 0.03)
            kernel_size = context.get("clip_limit_kernel_size", 8)
            return np.asarray(
                exposure.equalize_adapthist(channel, kernel_size=kernel_size, clip_limit=clip_limit)
            )
        elif method == "stretch":
            p_low, p_high = context.get("contrast_stretch_percentiles", (2, 98))
            p_low_val, p_high_val = np.percentile(channel, [p_low, p_high])
            return np.asarray(exposure.rescale_intensity(channel, in_range=(p_low_val, p_high_val)))
        elif method == "adaptive_gamma":
            # Adaptive gamma based on mean luminance
            mean_val = np.mean(channel)
            gamma = 1.0 if mean_val > 0.5 else 0.5 + mean_val
            return np.asarray(exposure.adjust_gamma(channel, gamma))
        else:
            raise ImageProcessingError(f"Unknown equalize method: {method}")


class GammaCorrectionStage(PipelineStage):
    """Apply gamma correction."""

    def __init__(self):
        super().__init__("Gamma Correction")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        gamma = context.get("gamma_correction", 1.0)
        if gamma == 1.0:
            return image
        return np.asarray(exposure.adjust_gamma(image, gamma))


class ColorPreservationStage(PipelineStage):
    """Preserve colors from original image."""

    def __init__(self):
        super().__init__("Color Preservation")

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        method = context.get("color_preservation", "none")
        if method == "none" or image.ndim != 3:
            return image

        strength = context.get("color_preservation_strength", 0.7)
        original = context.get("original_for_color")

        if original is None:
            return image

        if method == "lab":
            return self._preserve_lab(image, original, strength)
        elif method == "ratio":
            return self._preserve_ratio(image, original, strength)
        elif method == "rgb":
            return self._preserve_rgb(image, original, strength)
        else:
            return image

    def _preserve_lab(
        self, enhanced: np.ndarray, original: np.ndarray, strength: float
    ) -> np.ndarray:
        """Preserve colors in LAB space."""
        lab_enhanced = color.rgb2lab(enhanced)
        lab_original = color.rgb2lab(original)

        # Blend a and b channels
        lab_result = lab_enhanced.copy()
        lab_result[:, :, 1] = (
            strength * lab_original[:, :, 1] + (1 - strength) * lab_enhanced[:, :, 1]
        )
        lab_result[:, :, 2] = (
            strength * lab_original[:, :, 2] + (1 - strength) * lab_enhanced[:, :, 2]
        )

        return np.asarray(color.lab2rgb(lab_result))

    def _preserve_ratio(
        self, enhanced: np.ndarray, original: np.ndarray, strength: float
    ) -> np.ndarray:
        """Preserve color ratios."""
        orig_sum = np.sum(original, axis=2, keepdims=True)
        orig_sum = np.where(orig_sum < 1e-6, 1.0, orig_sum)

        orig_ratios = original / orig_sum
        enh_intensity = np.mean(enhanced, axis=2, keepdims=True)

        reconstructed = orig_ratios * enh_intensity
        return np.asarray(np.clip(strength * reconstructed + (1 - strength) * enhanced, 0, 1))

    def _preserve_rgb(
        self, enhanced: np.ndarray, original: np.ndarray, strength: float
    ) -> np.ndarray:
        """Simple RGB blending."""
        return np.asarray(np.clip(strength * original + (1 - strength) * enhanced, 0, 1))


class ImageProcessingPipeline:
    """
    Main processing pipeline that chains stages together.

    This reduces the complexity of process_image() from ~250 lines
    to a clean, modular pipeline.
    """

    def __init__(self):
        self.stages = []

    def add_stage(self, stage: PipelineStage) -> "ImageProcessingPipeline":
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        return self

    def process(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process image through all stages.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        context : dict
            Processing configuration and shared state.

        Returns
        -------
        tuple
            (processed_image, context) with updated context.
        """
        result = image.copy()

        for stage in self.stages:
            result = stage(result, context)

        return result, context


def create_standard_pipeline() -> ImageProcessingPipeline:
    """
    Create the standard image processing pipeline.

    Returns
    -------
    ImageProcessingPipeline
        Configured pipeline with all stages.
    """
    pipeline = ImageProcessingPipeline()
    pipeline.add_stage(ResizeStage())
    pipeline.add_stage(DenoiseStage())
    pipeline.add_stage(SharpenStage())
    pipeline.add_stage(ContrastStage())
    pipeline.add_stage(GammaCorrectionStage())
    pipeline.add_stage(ColorPreservationStage())
    return pipeline
