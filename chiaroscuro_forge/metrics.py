"""
Quality metrics for image processing assessment.

This module provides various perceptual quality metrics for evaluating
processed images, including SSIM, MS-SSIM, PSNR, feature similarity,
and histogram-based comparisons.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from skimage import color, img_as_float, feature
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.transform import pyramid_gaussian
from .validation import validate_array
from .exceptions import ImageProcessingError


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
) -> float:
    """
    Calculate Multi-Scale Structural Similarity Index (MS-SSIM).

    MS-SSIM evaluates image quality at multiple resolutions, providing
    a more comprehensive similarity measure than single-scale SSIM.

    Parameters
    ----------
    img1 : np.ndarray
        First image array (original or reference)
    img2 : np.ndarray
        Second image array (processed or comparison)
    weights : list of float, optional
        Weights for each scale. If None, uses default weights.
    levels : int, default=5
        Number of scales to use in the pyramid

    Returns
    -------
    float
        MS-SSIM value between 0 and 1 (1 indicates identical images)

    Raises
    ------
    ImageProcessingError
        If input validation fails or images are too small
    """
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if img1.shape != img2.shape:
        raise ImageProcessingError("Images must have the same dimensions for MS-SSIM calculation")

    # Convert to grayscale if color
    if img1.ndim == 3:
        img1 = color.rgb2gray(img_as_float(img1))
    if img2.ndim == 3:
        img2 = color.rgb2gray(img_as_float(img2))

    # Default weights
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    # Ensure we have the right number of weights
    if len(weights) != levels:
        weights = weights[:levels]
        weights = [w / sum(weights) for w in weights]

    # Build pyramids
    pyramid1 = list(pyramid_gaussian(img1, max_layer=levels - 1, downscale=2))
    pyramid2 = list(pyramid_gaussian(img2, max_layer=levels - 1, downscale=2))

    ms_ssim_values = []

    for level in range(min(levels, len(pyramid1))):
        try:
            ssim_val = ssim_skimage(pyramid1[level], pyramid2[level], data_range=1.0)
            ms_ssim_values.append(ssim_val)
        except Exception:
            # If SSIM calculation fails at this level, stop here
            break

    if not ms_ssim_values:
        raise ImageProcessingError("Could not calculate MS-SSIM at any scale")

    # Weight and combine
    weights_used = weights[:len(ms_ssim_values)]
    weights_normalized = [w / sum(weights_used) for w in weights_used]

    return float(sum(w * s for w, s in zip(weights_normalized, ms_ssim_values)))


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

            # Calculate cosine similarity
            similarity = np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2))
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
        denominator = np.sqrt(np.sum((hist1 - mean1)**2) * np.sum((hist2 - mean2)**2))
        
        if denominator == 0:
            return 1.0 if np.allclose(hist1, hist2) else 0.0
        
        return float((numerator / denominator + 1) / 2)

    elif method == "chi_square":
        # Chi-square distance (inverted and normalized)
        epsilon = 1e-10
        chi_square = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + epsilon))
        return float(1.0 / (1.0 + chi_square))

    elif method == "intersection":
        # Histogram intersection
        return float(np.minimum(hist1, hist2).sum())

    elif method == "bhattacharyya":
        # Bhattacharyya distance (converted to similarity)
        bhattacharyya = -np.log(np.sum(np.sqrt(hist1 * hist2)))
        return float(np.exp(-bhattacharyya))

    else:
        raise ImageProcessingError(f"Unknown histogram comparison method: {method}")


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
        except Exception:
            metrics_dict["ms_ssim"] = metrics_dict["ssim"]
    else:
        metrics_dict["ms_ssim"] = metrics_dict["ssim"]

    if calculate_advanced:
        # Feature similarity
        try:
            metrics_dict["feature_similarity_hog"] = feature_similarity(
                original, processed, method="hog"
            )
        except Exception:
            metrics_dict["feature_similarity_hog"] = 0.0

        try:
            metrics_dict["feature_similarity_canny"] = feature_similarity(
                original, processed, method="canny"
            )
        except Exception:
            metrics_dict["feature_similarity_canny"] = 0.0

        # Histogram similarity
        try:
            metrics_dict["histogram_correlation"] = histogram_similarity(
                original, processed, method="correlation"
            )
        except Exception:
            metrics_dict["histogram_correlation"] = 0.0

        try:
            metrics_dict["histogram_intersection"] = histogram_similarity(
                original, processed, method="intersection"
            )
        except Exception:
            metrics_dict["histogram_intersection"] = 0.0

        # Color preservation (if color images)
        if original.ndim == 3 and processed.ndim == 3:
            try:
                orig_lab = color.rgb2lab(img_as_float(original))
                proc_lab = color.rgb2lab(img_as_float(processed))

                # Color distance in ab plane
                color_dist = np.sqrt(
                    (orig_lab[:, :, 1] - proc_lab[:, :, 1])**2 +
                    (orig_lab[:, :, 2] - proc_lab[:, :, 2])**2
                )
                
                # Normalize to 0-1 (typical ab values range ~-128 to 128)
                metrics_dict["color_preservation"] = float(1.0 - np.mean(color_dist) / 180.0)
            except Exception:
                metrics_dict["color_preservation"] = 1.0

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
            "ssim": 0.3,
            "ms_ssim": 0.2,
            "psnr": 0.15,
            "feature_similarity_hog": 0.15,
            "histogram_correlation": 0.1,
            "color_preservation": 0.1,
        },
        "photography": {
            "ssim": 0.25,
            "ms_ssim": 0.2,
            "psnr": 0.1,
            "feature_similarity_hog": 0.15,
            "histogram_correlation": 0.1,
            "color_preservation": 0.2,
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
            "ssim": 0.2,
            "ms_ssim": 0.15,
            "psnr": 0.05,
            "feature_similarity_hog": 0.2,
            "histogram_correlation": 0.15,
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
