"""
Quality metrics for image processing assessment.

This module provides various perceptual quality metrics for evaluating
processed images, including SSIM, MS-SSIM, PSNR, CIEDE2000 color
difference, LPIPS (learned perceptual similarity), feature similarity,
and histogram-based comparisons.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage
from skimage import color, feature
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.transform import pyramid_gaussian
from skimage.util import img_as_float

from .constants import VALID_HISTOGRAM_METHODS
from .exceptions import ImageProcessingError
from .validation import validate_array

logger = logging.getLogger(__name__)
_lpips_lock = threading.Lock()
_lpips_model: Optional[Any] = None


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image array (original or reference)
    img2 : np.ndarray
        Second image array (processed or comparison)

    Returns
    -------
    float
        SSIM value between -1 and 1 (1 indicates identical images)

    Raises
    ------
    ImageProcessingError
        If input validation fails
    """
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if img1.shape != img2.shape:
        raise ImageProcessingError("Images must have the same dimensions for SSIM calculation")

    # Convert to grayscale if color
    if img1.ndim == 3:
        img1 = color.rgb2gray(img_as_float(img1))
    if img2.ndim == 3:
        img2 = color.rgb2gray(img_as_float(img2))

    return float(ssim_skimage(img1, img2, data_range=1.0))


def ms_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    weights: Optional[List[float]] = None,
    levels: int = 5,
    method: str = "wang",
) -> float:
    """
    Calculate Multi-Scale Structural Similarity Index (MS-SSIM).

    The default implementation follows the Wang et al. (2003) definition,
    which multiplies the luminance term at the coarsest scale by a weighted
    product of the contrast and structure terms across intermediate scales.
    The historical averaging behavior remains available through
    ``method="mean"`` for backward compatibility.

    Parameters
    ----------
    img1 : np.ndarray
        First image array (original or reference)
    img2 : np.ndarray
        Second image array (processed or comparison)
    weights : list of float, optional
        Weight vector for the multilayer product. If None, uses the published
        Wang et al. (2003) defaults.
    levels : int, default=5
        Number of Gaussian scales to evaluate.
    method : str, default="wang"
        ``"wang"`` for the published formulation or ``"mean"`` for the legacy
        per-scale average.

    Returns
    -------
    float
        MS-SSIM value between 0 and 1 (1 indicates identical images)

    Raises
    ------
    ImageProcessingError
        If input validation fails or images are too small

    Notes
    -----
    This implementation departs from the authors' reference code in four
    ways: the luminance, contrast, and structure maps are clipped to
    [0, 1] per term; the term maps are reduced by full-image means; local
    statistics use a Gaussian filter with sigma 1.5 instead of the
    reference 11 x 11 circular-symmetric Gaussian window; and the weight
    vector is renormalized when fewer levels than requested fit the image.
    The published default weights and the coarsest-scale luminance rule
    follow Wang et al. (2003) exactly.
    """
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if img1.shape != img2.shape:
        raise ImageProcessingError("Images must have the same dimensions for MS-SSIM calculation")

    if not isinstance(levels, (int, np.integer)) or isinstance(levels, bool):
        raise ImageProcessingError("MS-SSIM levels must be an integer")
    if levels < 2:
        raise ImageProcessingError("MS-SSIM requires at least two scales")

    if method not in {"wang", "mean"}:
        raise ImageProcessingError("MS-SSIM method must be 'wang' or 'mean'")

    # Convert to grayscale if color
    if img1.ndim == 3:
        img1 = color.rgb2gray(img_as_float(img1))
    if img2.ndim == 3:
        img2 = color.rgb2gray(img_as_float(img2))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    else:
        weights = list(weights)

    if not weights:
        raise ImageProcessingError("MS-SSIM weights cannot be empty")

    weights_array = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights_array)):
        raise ImageProcessingError("MS-SSIM weights must be finite")
    if np.any(weights_array < 0.0):
        raise ImageProcessingError("MS-SSIM weights must be non-negative")

    # Build pyramids; level 0 is the highest-resolution image and the final
    # level is the coarsest one. This matches the Wang et al. (2003) multi-scale
    # formulation, which evaluates luminance on the coarsest scale and contrast/
    # structure terms on the intermediate scales.
    pyramid1 = list(pyramid_gaussian(img1, max_layer=max(0, levels - 1), downscale=2))
    pyramid2 = list(pyramid_gaussian(img2, max_layer=max(0, levels - 1), downscale=2))
    usable_levels = min(levels, len(pyramid1))

    if usable_levels < 2:
        raise ImageProcessingError("Could not calculate MS-SSIM at any scale")

    if method == "mean":
        scale_values = []
        for level in range(usable_levels):
            try:
                scale_values.append(ssim_skimage(pyramid1[level], pyramid2[level], data_range=1.0))
            except Exception as exc:  # pragma: no cover - defensive path
                raise ImageProcessingError(f"Could not calculate MS-SSIM at scale {level}: {exc}")
        legacy_weights = weights_array[: len(scale_values)]
        legacy_weight_sum = legacy_weights.sum()
        if np.isclose(legacy_weight_sum, 0.0):
            legacy_weights = np.full(len(legacy_weights), 1.0 / len(legacy_weights))
        else:
            legacy_weights = legacy_weights / legacy_weight_sum
        return float(np.dot(legacy_weights, scale_values[: len(legacy_weights)]))

    if len(weights_array) != levels:
        weights_array = np.asarray(weights_array[:levels], dtype=float)
        if len(weights_array) < levels:
            weights_array = np.pad(
                weights_array, (0, levels - len(weights_array)), constant_values=1.0 / levels
            )

    weights_sum = weights_array.sum()
    if np.isclose(weights_sum, 0.0):
        weights_array = np.full(levels, 1.0 / levels, dtype=float)
    else:
        weights_array = weights_array / weights_sum

    def _ssim_terms(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        sigma = 1.5

        mu_a = ndimage.gaussian_filter(a, sigma=sigma, mode="reflect")
        mu_b = ndimage.gaussian_filter(b, sigma=sigma, mode="reflect")
        sigma2_a = ndimage.gaussian_filter(a * a, sigma=sigma, mode="reflect") - mu_a**2
        sigma2_b = ndimage.gaussian_filter(b * b, sigma=sigma, mode="reflect") - mu_b**2
        sigma_ab = ndimage.gaussian_filter(a * b, sigma=sigma, mode="reflect") - mu_a * mu_b

        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        c3 = c2 / 2.0

        luminance = (2.0 * mu_a * mu_b + c1) / (mu_a**2 + mu_b**2 + c1)
        contrast = (2.0 * np.sqrt(np.maximum(sigma2_a * sigma2_b, 0.0)) + c2) / (
            sigma2_a + sigma2_b + c2
        )
        structure = (sigma_ab + c3) / (np.sqrt(np.maximum(sigma2_a * sigma2_b, 0.0)) + c3)
        return (
            np.clip(luminance, 0.0, 1.0),
            np.clip(contrast, 0.0, 1.0),
            np.clip(structure, 0.0, 1.0),
        )

    active_weights = weights_array[:usable_levels]
    active_weight_sum = active_weights.sum()
    if np.isclose(active_weight_sum, 0.0):
        active_weights = np.full(usable_levels, 1.0 / usable_levels, dtype=float)
    else:
        active_weights = active_weights / active_weight_sum

    product = 1.0
    for level in range(usable_levels - 1):
        _, c_term, s_term = _ssim_terms(pyramid1[level], pyramid2[level])
        product *= np.mean(c_term * s_term) ** active_weights[level]

    coarsest_l, coarsest_c, coarsest_s = _ssim_terms(pyramid1[-1], pyramid2[-1])
    product *= np.mean(coarsest_l) ** active_weights[-1]
    product *= np.mean(coarsest_c * coarsest_s) ** active_weights[-1]
    return float(np.clip(product, 0.0, 1.0))


def feature_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "hog",
    multichannel: bool = True,
) -> float:
    """
    Calculate similarity between images based on extracted features.

    Parameters
    ----------
    img1 : np.ndarray
        First image array (original or reference)
    img2 : np.ndarray
        Second image array (processed or comparison)
    method : str, default="hog"
        Feature extraction method: "hog", "orb", or "canny"
    multichannel : bool, default=True
        Whether to process color images as multichannel

    Returns
    -------
    float
        Feature similarity score between 0 and 1

    Raises
    ------
    ImageProcessingError
        If input validation fails or method is invalid
    """
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if img1.shape != img2.shape:
        raise ImageProcessingError("Images must have the same dimensions")

    # Convert to grayscale if needed
    if img1.ndim == 3 and not multichannel:
        img1 = color.rgb2gray(img_as_float(img1))
    if img2.ndim == 3 and not multichannel:
        img2 = color.rgb2gray(img_as_float(img2))

    if method == "hog":
        # HOG feature extraction
        try:
            # Note: multichannel parameter deprecated in scikit-image 0.19+
            # Now inferred from ndim (3 = multichannel)
            hog1 = feature.hog(img1, channel_axis=-1 if img1.ndim == 3 else None)
            hog2 = feature.hog(img2, channel_axis=-1 if img2.ndim == 3 else None)

            norm1 = np.linalg.norm(hog1)
            norm2 = np.linalg.norm(hog2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(hog1, hog2) / (norm1 * norm2)
            return float(max(0.0, min(1.0, (similarity + 1) / 2)))
        except Exception as e:
            raise ImageProcessingError(f"HOG feature extraction failed: {e}")

    elif method == "orb":
        # ORB keypoint matching
        try:
            orb = feature.ORB(n_keypoints=500)

            orb.detect_and_extract(color.rgb2gray(img1) if img1.ndim == 3 else img1)
            keypoints1 = orb.keypoints
            descriptors1 = orb.descriptors

            orb.detect_and_extract(color.rgb2gray(img2) if img2.ndim == 3 else img2)
            keypoints2 = orb.keypoints
            descriptors2 = orb.descriptors

            if descriptors1 is None or descriptors2 is None:
                return 0.0

            # Match descriptors
            matches = feature.match_descriptors(descriptors1, descriptors2, cross_check=True)

            if len(keypoints1) == 0 or len(keypoints2) == 0:
                return 0.0

            similarity = len(matches) / max(len(keypoints1), len(keypoints2))
            return float(min(1.0, similarity))
        except Exception as e:
            raise ImageProcessingError(f"ORB feature extraction failed: {e}")

    elif method == "canny":
        # Edge similarity using Canny
        try:
            edges1 = feature.canny(color.rgb2gray(img1) if img1.ndim == 3 else img1, sigma=1.0)
            edges2 = feature.canny(color.rgb2gray(img2) if img2.ndim == 3 else img2, sigma=1.0)

            # Calculate overlap
            intersection = np.logical_and(edges1, edges2).sum()
            union = np.logical_or(edges1, edges2).sum()

            if union == 0:
                return 1.0

            similarity = intersection / union
            return float(similarity)
        except Exception as e:
            raise ImageProcessingError(f"Canny edge detection failed: {e}")

    else:
        raise ImageProcessingError(f"Unknown feature extraction method: {method}")


def histogram_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "correlation",
    bins: int = 256,
) -> float:
    """
    Calculate histogram-based similarity between images.

    Parameters
    ----------
    img1 : np.ndarray
        First image array (original or reference)
    img2 : np.ndarray
        Second image array (processed or comparison)
    method : str, default="correlation"
        Comparison method: "correlation", "chi_square", "intersection", or "bhattacharyya"
    bins : int, default=256
        Number of histogram bins

    Returns
    -------
    float
        Histogram similarity score (0-1 for most methods)

    Raises
    ------
    ImageProcessingError
        If input validation fails
    """
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if method not in VALID_HISTOGRAM_METHODS:
        raise ImageProcessingError(f"Histogram method must be one of {VALID_HISTOGRAM_METHODS}")

    if method == "bhattacharyya":
        pass

    # Convert to float and flatten
    img1_float = img_as_float(img1)
    img2_float = img_as_float(img2)

    # Calculate histograms
    hist1, _ = np.histogram(img1_float.flatten(), bins=bins, range=(0, 1))
    hist2, _ = np.histogram(img2_float.flatten(), bins=bins, range=(0, 1))

    # Normalize histograms
    hist1 = hist1.astype(float) / hist1.sum()
    hist2 = hist2.astype(float) / hist2.sum()

    if method == "correlation":
        # Correlation coefficient
        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)

        numerator = np.sum((hist1 - mean1) * (hist2 - mean2))
        denominator = np.sqrt(np.sum((hist1 - mean1) ** 2) * np.sum((hist2 - mean2) ** 2))

        if denominator == 0:
            return 1.0 if np.allclose(hist1, hist2) else 0.0

        return float((numerator / denominator + 1) / 2)

    elif method == "chi_square":
        # Chi-square distance (inverted and normalized)
        epsilon = 1e-10
        chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
        return float(1.0 / (1.0 + chi_square))

    elif method == "intersection":
        # Histogram intersection
        return float(np.minimum(hist1, hist2).sum())

    elif method == "bhattacharyya":
        # Bhattacharyya distance (converted to similarity)
        bhattacharyya = -np.log(np.sum(np.sqrt(hist1 * hist2)))
        return float(np.exp(-bhattacharyya))

    raise ImageProcessingError(f"Histogram method must be one of {VALID_HISTOGRAM_METHODS}")


def lpips_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    Computes perceptual distance using deep features from a pretrained
    AlexNet or VGG network. Requires the optional ``lpips`` package.
    Falls back to MS-SSIM via ``ms_ssim`` when not installed.

    Input arrays are scaled from [0, 1] to [-1, 1] before inference,
    matching the calibration of the published networks.

    LPIPS is a distance metric (lower = more similar), correlating with
    human judgments at r ≈ 0.9 versus SSIM's r ≈ 0.7 (Zhang et al.,
    CVPR 2018).

    Parameters
    ----------
    img1 : np.ndarray
        First image array (original or reference).
    img2 : np.ndarray
        Second image array (processed or comparison).

    Returns
    -------
    float
        Perceptual similarity between 0 and 1 (1 = identical).

    Raises
    ------
    ImageProcessingError
        If input validation fails.
    """
    global _lpips_model

    import importlib

    try:
        importlib.import_module("lpips")
    except ImportError:
        logger.debug("lpips not installed, falling back to MS-SSIM")
        return ms_ssim(img1, img2)

    try:
        import lpips
        import torch

        with _lpips_lock:
            if _lpips_model is None:
                _lpips_model = lpips.LPIPS(net="alex", verbose=False)
            loss_fn = _lpips_model

        # The published LPIPS networks are calibrated on inputs in [-1, 1]
        t1 = torch.from_numpy(img_as_float(img1).clip(0, 1) * 2.0 - 1.0).float()
        t2 = torch.from_numpy(img_as_float(img2).clip(0, 1) * 2.0 - 1.0).float()

        if t1.ndim == 2:
            t1 = t1.unsqueeze(0).unsqueeze(0)
            t2 = t2.unsqueeze(0).unsqueeze(0)
        else:
            t1 = t1.permute(2, 0, 1).unsqueeze(0)
            t2 = t2.permute(2, 0, 1).unsqueeze(0)

        if t1.shape[1] == 1:
            t1 = t1.repeat(1, 3, 1, 1)
            t2 = t2.repeat(1, 3, 1, 1)

        with torch.no_grad():
            distance = loss_fn(t1, t2).item()

        return 1.0 - float(np.clip(distance, 0.0, 1.0))
    except Exception as exc:
        logger.debug("LPIPS calculation failed: %s, falling back to MS-SSIM", exc)
        return ms_ssim(img1, img2)


def calculate_perceptual_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    calculate_advanced: bool = True,
) -> Dict[str, float]:
    """
    Calculate comprehensive perceptual quality metrics.

    Parameters
    ----------
    original : np.ndarray
        Original image array
    processed : np.ndarray
        Processed image array
    calculate_advanced : bool, default=True
        Whether to calculate advanced metrics (feature similarity, etc.)

    Returns
    -------
    dict
        Dictionary with metric names as keys and scores as values

    Raises
    ------
    ImageProcessingError
        If input validation fails
    """
    validate_array(original, "original")
    validate_array(processed, "processed")

    if original.shape != processed.shape:
        raise ImageProcessingError("Images must have the same dimensions")

    metrics_dict = {}

    # SSIM
    try:
        metrics_dict["ssim"] = ssim(original, processed)
    except Exception as e:
        raise ImageProcessingError(f"SSIM calculation failed: {e}")

    # PSNR
    try:
        mse = np.mean((img_as_float(original) - img_as_float(processed)) ** 2)
        if mse == 0:
            metrics_dict["psnr"] = 100.0
        else:
            metrics_dict["psnr"] = float(20 * np.log10(1.0 / np.sqrt(mse)))
    except Exception as e:
        raise ImageProcessingError(f"PSNR calculation failed: {e}")

    # MS-SSIM (if image is large enough)
    if min(original.shape[:2]) >= 32:
        try:
            metrics_dict["ms_ssim"] = ms_ssim(original, processed)
        except Exception as exc:
            logger.debug("MS-SSIM calculation failed: %s", exc)
            metrics_dict["ms_ssim"] = metrics_dict["ssim"]
    else:
        metrics_dict["ms_ssim"] = metrics_dict["ssim"]

    if calculate_advanced:
        # Feature similarity
        try:
            metrics_dict["feature_similarity_hog"] = feature_similarity(
                original, processed, method="hog"
            )
        except Exception as exc:
            logger.debug("HOG feature similarity failed: %s", exc)

        try:
            metrics_dict["feature_similarity_canny"] = feature_similarity(
                original, processed, method="canny"
            )
        except Exception as exc:
            logger.debug("Canny feature similarity failed: %s", exc)

        # Histogram similarity
        try:
            metrics_dict["histogram_correlation"] = histogram_similarity(
                original, processed, method="correlation"
            )
        except Exception as exc:
            logger.debug("Histogram correlation failed: %s", exc)

        try:
            metrics_dict["histogram_intersection"] = histogram_similarity(
                original, processed, method="intersection"
            )
        except Exception as exc:
            logger.debug("Histogram intersection failed: %s", exc)

        # LPIPS (learned perceptual similarity)
        try:
            metrics_dict["lpips"] = lpips_similarity(original, processed)
        except Exception as exc:
            logger.debug("LPIPS calculation failed: %s", exc)

        # Color preservation (if color images)
        if original.ndim == 3 and processed.ndim == 3:
            try:
                orig_lab = color.rgb2lab(img_as_float(original))
                proc_lab = color.rgb2lab(img_as_float(processed))

                from skimage.color import deltaE_ciede2000

                deltae = deltaE_ciede2000(orig_lab, proc_lab)
                metrics_dict["color_preservation"] = float(
                    1.0 - np.clip(np.mean(deltae) / 20.0, 0.0, 1.0)
                )
            except Exception as exc:
                logger.debug("Color preservation metric failed: %s", exc)

    return metrics_dict


def calculate_quality_score(
    metrics_dict: Dict[str, float],
    application_type: str = "general",
) -> float:
    """
    Calculate an overall quality score from individual metrics.

    Weights metrics based on the application type to provide a single
    quality score between 0 and 1.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metric values
    application_type : str, default="general"
        Type of application: "general", "photography", "medical", "document", or "art"

    Returns
    -------
    float
        Overall quality score between 0 and 1

    Raises
    ------
    ImageProcessingError
        If application type is invalid
    """
    valid_app_types = ["general", "photography", "medical", "document", "art"]
    if application_type not in valid_app_types:
        raise ImageProcessingError(f"Application type must be one of {valid_app_types}")

    # Define weights for different application types
    weights = {
        "general": {
            "ssim": 0.25,
            "ms_ssim": 0.15,
            "psnr": 0.15,
            "lpips": 0.15,
            "feature_similarity_hog": 0.1,
            "histogram_correlation": 0.1,
            "color_preservation": 0.1,
        },
        "photography": {
            "ssim": 0.2,
            "ms_ssim": 0.1,
            "psnr": 0.05,
            "lpips": 0.2,
            "feature_similarity_hog": 0.1,
            "histogram_correlation": 0.1,
            "color_preservation": 0.25,
        },
        "medical": {
            "ssim": 0.35,
            "ms_ssim": 0.25,
            "psnr": 0.2,
            "feature_similarity_hog": 0.15,
            "histogram_correlation": 0.05,
            "color_preservation": 0.0,
        },
        "document": {
            "ssim": 0.3,
            "ms_ssim": 0.2,
            "psnr": 0.15,
            "feature_similarity_canny": 0.2,
            "feature_similarity_hog": 0.1,
            "histogram_correlation": 0.05,
        },
        "art": {
            "ssim": 0.15,
            "ms_ssim": 0.1,
            "psnr": 0.05,
            "lpips": 0.2,
            "feature_similarity_hog": 0.15,
            "histogram_correlation": 0.1,
            "color_preservation": 0.25,
        },
    }

    app_weights = weights[application_type]

    score = 0.0
    total_weight = 0.0

    for metric, weight in app_weights.items():
        if metric in metrics_dict:
            value = metrics_dict[metric]

            # Normalize PSNR (typically 20-50 dB, saturate at 50)
            if metric == "psnr":
                value = min(1.0, max(0.0, (value - 20.0) / 30.0))

            # Ensure value is in 0-1 range
            value = max(0.0, min(1.0, value))

            score += weight * value
            total_weight += weight

    if total_weight > 0:
        score = score / total_weight

    return float(max(0.0, min(1.0, score)))
