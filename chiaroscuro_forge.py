import os
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import numpy as np
from scipy.signal import convolve2d
from scipy.stats import entropy
import scipy.ndimage as ndi

import skimage.io as io
import skimage.transform as transform
import skimage.filters as filters
import skimage.util as util
import skimage.exposure as exposure
import skimage.metrics as metrics
import skimage.feature as feature
from skimage import color, img_as_float, img_as_ubyte


__version__ = "0.1.0"


class ImageProcessingError(Exception):
    pass


def validate_array(arr: np.ndarray, name: str = "array") -> None:
    if not isinstance(arr, np.ndarray):
        raise ImageProcessingError(f"{name} must be a numpy array")

    if arr.ndim not in [2, 3]:
        raise ImageProcessingError(f"{name} must be 2D (grayscale) or 3D (color) array")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ImageProcessingError(f"{name} contains NaN or Inf values")


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if img1.shape != img2.shape:
        raise ImageProcessingError(
            f"Image shapes don't match: {img1.shape} vs {img2.shape}"
        )

    multichannel = img1.ndim == 3
    min_dim = min(min(img1.shape[:2]), min(img2.shape[:2]))
    win_size = min(7, min_dim - (1 if min_dim % 2 == 0 else 0))
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    try:
        if multichannel:
            return metrics.structural_similarity(
                img1,
                img2,
                win_size=win_size,
                data_range=img2.max() - img2.min(),
                channel_axis=-1,
            )
        else:
            return metrics.structural_similarity(
                img1, img2, win_size=win_size, data_range=img2.max() - img2.min()
            )
    except TypeError:
        return metrics.structural_similarity(
            img1,
            img2,
            win_size=win_size,
            data_range=img2.max() - img2.min(),
            multichannel=multichannel,
        )


def ms_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    weights: Optional[List[float]] = None,
    levels: int = 5,
) -> float:
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    if img1.shape != img2.shape:
        raise ImageProcessingError(
            f"Image shapes don't match: {img1.shape} vs {img2.shape}"
        )

    if levels <= 0:
        raise ImageProcessingError("Levels must be a positive integer")

    if weights is not None:
        if not isinstance(weights, (list, np.ndarray)):
            raise ImageProcessingError("Weights must be a list or numpy array")

        if len(weights) != levels:
            raise ImageProcessingError(f"Expected {levels} weights, got {len(weights)}")

        if any(w < 0 for w in weights):
            raise ImageProcessingError("Weights must be non-negative")

        if sum(weights) == 0:
            raise ImageProcessingError("Sum of weights must be positive")

    if weights is None:
        weights = np.ones(levels) / levels

    weights = np.array(weights) / np.sum(weights)

    min_dim = min(min(img1.shape[:2]), min(img2.shape[:2]))
    max_levels = int(np.log2(min_dim)) - 2
    levels = min(levels, max_levels)

    if levels < 1:
        return ssim(img1, img2)

    multichannel = img1.ndim == 3
    ssim_values = np.zeros(levels)

    img1_float = img_as_float(img1)
    img2_float = img_as_float(img2)

    current_img1 = img1_float.copy()
    current_img2 = img2_float.copy()

    for i in range(levels):
        ssim_values[i] = ssim(current_img1, current_img2)

        if i < levels - 1:
            if multichannel:
                current_img1 = transform.resize(
                    current_img1,
                    (
                        current_img1.shape[0] // 2,
                        current_img1.shape[1] // 2,
                        current_img1.shape[2],
                    ),
                    anti_aliasing=True,
                )
                current_img2 = transform.resize(
                    current_img2,
                    (
                        current_img2.shape[0] // 2,
                        current_img2.shape[1] // 2,
                        current_img2.shape[2],
                    ),
                    anti_aliasing=True,
                )
            else:
                current_img1 = transform.resize(
                    current_img1,
                    (current_img1.shape[0] // 2, current_img1.shape[1] // 2),
                    anti_aliasing=True,
                )
                current_img2 = transform.resize(
                    current_img2,
                    (current_img2.shape[0] // 2, current_img2.shape[1] // 2),
                    anti_aliasing=True,
                )

    return np.prod(ssim_values**weights)


def feature_similarity(
    img1: np.ndarray, img2: np.ndarray, method: str = "hog", multichannel: bool = True
) -> float:
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    valid_methods = ["hog", "orb", "canny"]
    if method not in valid_methods:
        raise ImageProcessingError(
            f"Method must be one of {valid_methods}, got '{method}'"
        )

    if multichannel and img1.ndim == 3:
        img1_gray = color.rgb2gray(img1)
        img2_gray = color.rgb2gray(img2)
    else:
        img1_gray = img1
        img2_gray = img2

    if method == "hog":
        try:
            min_dim = min(min(img1_gray.shape), min(img2_gray.shape))
            cell_size = max(8, min_dim // 32)

            hog1 = feature.hog(
                img1_gray,
                pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(2, 2),
                visualize=False,
                feature_vector=True,
            )

            hog2 = feature.hog(
                img2_gray,
                pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(2, 2),
                visualize=False,
                feature_vector=True,
            )

            similarity = np.dot(hog1, hog2) / (
                np.linalg.norm(hog1) * np.linalg.norm(hog2)
            )
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            raise ImageProcessingError(f"HOG feature extraction failed: {e}")

    elif method == "orb":
        try:
            from skimage.feature import ORB, match_descriptors

            num_keypoints = min(500, img1_gray.shape[0] * img1_gray.shape[1] // 1000)

            orb = ORB(n_keypoints=num_keypoints, fast_threshold=0.05)

            orb.detect_and_extract(img1_gray)
            keypoints1 = orb.keypoints
            descriptors1 = orb.descriptors

            orb.detect_and_extract(img2_gray)
            keypoints2 = orb.keypoints
            descriptors2 = orb.descriptors

            if (
                descriptors1 is None
                or descriptors2 is None
                or len(descriptors1) == 0
                or len(descriptors2) == 0
            ):
                return 0.0

            matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

            similarity = len(matches) / min(len(keypoints1), len(keypoints2))
            return similarity
        except ImportError:
            raise ImageProcessingError(
                "ORB feature matching requires skimage.feature with ORB support"
            )
        except Exception as e:
            raise ImageProcessingError(f"ORB feature matching failed: {e}")

    elif method == "canny":
        try:
            min_dim = min(img1_gray.shape)
            sigma = max(1.0, min_dim / 500)

            edges1 = feature.canny(img1_gray, sigma=sigma)
            edges2 = feature.canny(img2_gray, sigma=sigma)

            intersection = np.logical_and(edges1, edges2).sum()
            union = np.logical_or(edges1, edges2).sum()

            if union == 0:
                return 1.0

            return intersection / union
        except Exception as e:
            raise ImageProcessingError(f"Canny edge detection failed: {e}")

    raise ImageProcessingError(f"Unexpected method: {method}")


def histogram_similarity(
    img1: np.ndarray, img2: np.ndarray, method: str = "correlation", bins: int = 256
) -> float:
    validate_array(img1, "img1")
    validate_array(img2, "img2")

    valid_methods = ["correlation", "chi-square", "intersection", "kl_divergence"]
    if method not in valid_methods:
        raise ImageProcessingError(
            f"Method must be one of {valid_methods}, got '{method}'"
        )

    if bins <= 0:
        raise ImageProcessingError("Bins must be a positive integer")

    multichannel = img1.ndim == 3

    if multichannel:
        similarity = 0
        for i in range(3):
            hist1, _ = np.histogram(
                img1[:, :, i], bins=bins, range=(0, 1), density=True
            )
            hist2, _ = np.histogram(
                img2[:, :, i], bins=bins, range=(0, 1), density=True
            )

            if method == "correlation":
                correlation = np.corrcoef(hist1, hist2)[0, 1]
                similarity += max(0, correlation)

            elif method == "chi-square":
                chi_square = (
                    np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10)) / bins
                )
                channel_similarity = np.exp(-chi_square)
                similarity += channel_similarity

            elif method == "intersection":
                channel_similarity = np.sum(np.minimum(hist1, hist2)) / np.sum(hist2)
                similarity += channel_similarity

            elif method == "kl_divergence":
                epsilon = 1e-10
                kl_div = entropy(hist1 + epsilon, hist2 + epsilon)
                channel_similarity = np.exp(-kl_div)
                similarity += channel_similarity

        return similarity / 3

    else:
        hist1, _ = np.histogram(img1, bins=bins, range=(0, 1), density=True)
        hist2, _ = np.histogram(img2, bins=bins, range=(0, 1), density=True)

        if method == "correlation":
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            return max(0, correlation)

        elif method == "chi-square":
            chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10)) / bins
            return np.exp(-chi_square)

        elif method == "intersection":
            return np.sum(np.minimum(hist1, hist2)) / np.sum(hist2)

        elif method == "kl_divergence":
            epsilon = 1e-10
            kl_div = entropy(hist1 + epsilon, hist2 + epsilon)
            return np.exp(-kl_div)

    raise ImageProcessingError(f"Unexpected method: {method}")


def calculate_perceptual_metrics(
    original: np.ndarray, processed: np.ndarray, calculate_advanced: bool = True
) -> Dict[str, float]:
    validate_array(original, "original")
    validate_array(processed, "processed")

    original_float = img_as_float(original)
    processed_float = img_as_float(processed)

    if original_float.shape != processed_float.shape:
        processed_float = transform.resize(
            processed_float, original_float.shape, anti_aliasing=True
        )

    multichannel = original_float.ndim == 3

    metrics_dict = {
        "ssim": ssim(original_float, processed_float),
        "mse": metrics.mean_squared_error(original_float, processed_float),
        "psnr": metrics.peak_signal_noise_ratio(original_float, processed_float),
        "ncc": np.mean(
            np.corrcoef(original_float.flatten(), processed_float.flatten())[0, 1]
        ),
    }

    if calculate_advanced:
        try:
            metrics_dict["hist_correlation"] = histogram_similarity(
                original_float, processed_float, method="correlation"
            )

            metrics_dict["hist_intersection"] = histogram_similarity(
                original_float, processed_float, method="intersection"
            )

            metrics_dict["edge_similarity"] = feature_similarity(
                original_float,
                processed_float,
                method="canny",
                multichannel=multichannel,
            )

            try:
                metrics_dict["feature_similarity"] = feature_similarity(
                    original_float,
                    processed_float,
                    method="hog",
                    multichannel=multichannel,
                )
            except ImageProcessingError:
                metrics_dict["feature_similarity"] = np.nan

            min_dim = min(original_float.shape[:2])
            if min_dim >= 32:
                metrics_dict["ms_ssim"] = ms_ssim(
                    original_float,
                    processed_float,
                    weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333][
                        : min(5, int(np.log2(min_dim)) - 2)
                    ],
                )
            else:
                metrics_dict["ms_ssim"] = metrics_dict["ssim"]

            try:
                from skimage.filters import scharr

                if multichannel:
                    orig_saliency = np.zeros(original_float.shape[:2])
                    proc_saliency = np.zeros(processed_float.shape[:2])

                    for i in range(3):
                        orig_saliency += np.hypot(
                            scharr(original_float[:, :, i], axis=0),
                            scharr(original_float[:, :, i], axis=1),
                        )
                        proc_saliency += np.hypot(
                            scharr(processed_float[:, :, i], axis=0),
                            scharr(processed_float[:, :, i], axis=1),
                        )

                    orig_saliency /= 3
                    proc_saliency /= 3
                else:
                    orig_saliency = np.hypot(
                        scharr(original_float, axis=0), scharr(original_float, axis=1)
                    )
                    proc_saliency = np.hypot(
                        scharr(processed_float, axis=0), scharr(processed_float, axis=1)
                    )

                orig_saliency = orig_saliency / (np.max(orig_saliency) + 1e-10)
                proc_saliency = proc_saliency / (np.max(proc_saliency) + 1e-10)

                metrics_dict["saliency_similarity"] = np.corrcoef(
                    orig_saliency.flatten(), proc_saliency.flatten()
                )[0, 1]
            except Exception:
                metrics_dict["saliency_similarity"] = np.nan

        except Exception as e:
            raise ImageProcessingError(f"Could not calculate advanced metrics: {e}")

    metrics_dict = {k: v for k, v in metrics_dict.items() if not np.isnan(v)}

    return metrics_dict


def calculate_quality_score(
    metrics: Dict[str, float], application_type: str = "general"
) -> float:
    if not metrics:
        return 0.0

    valid_app_types = ["general", "photography", "medical", "document", "art"]
    if application_type not in valid_app_types:
        raise ImageProcessingError(f"Application type must be one of {valid_app_types}")

    weights = {
        "general": {
            "ssim": 0.30,
            "ms_ssim": 0.15,
            "psnr": 0.15,
            "mse": 0.10,
            "feature_similarity": 0.10,
            "edge_similarity": 0.10,
            "hist_correlation": 0.05,
            "saliency_similarity": 0.05,
        },
        "photography": {
            "ssim": 0.20,
            "ms_ssim": 0.20,
            "psnr": 0.10,
            "mse": 0.05,
            "feature_similarity": 0.15,
            "edge_similarity": 0.10,
            "hist_correlation": 0.10,
            "saliency_similarity": 0.10,
        },
        "medical": {
            "ssim": 0.35,
            "ms_ssim": 0.25,
            "psnr": 0.20,
            "mse": 0.10,
            "feature_similarity": 0.05,
            "edge_similarity": 0.05,
            "hist_correlation": 0.00,
            "saliency_similarity": 0.00,
        },
        "document": {
            "ssim": 0.25,
            "ms_ssim": 0.15,
            "psnr": 0.10,
            "mse": 0.05,
            "feature_similarity": 0.25,
            "edge_similarity": 0.20,
            "hist_correlation": 0.00,
            "saliency_similarity": 0.00,
        },
        "art": {
            "ssim": 0.15,
            "ms_ssim": 0.10,
            "psnr": 0.05,
            "mse": 0.05,
            "feature_similarity": 0.20,
            "edge_similarity": 0.15,
            "hist_correlation": 0.15,
            "saliency_similarity": 0.15,
        },
    }

    app_weights = weights[application_type]

    normalized_metrics = {}

    if "ssim" in metrics:
        normalized_metrics["ssim"] = metrics["ssim"]

    if "ms_ssim" in metrics:
        normalized_metrics["ms_ssim"] = metrics["ms_ssim"]

    if "psnr" in metrics:
        normalized_metrics["psnr"] = min(1.0, max(0.0, (metrics["psnr"] - 20) / 30))

    if "mse" in metrics:
        normalized_metrics["mse"] = min(1.0, max(0.0, 1.0 - metrics["mse"] * 20))

    if "ncc" in metrics:
        normalized_metrics["ncc"] = min(1.0, max(0.0, metrics["ncc"]))

    for metric in [
        "feature_similarity",
        "edge_similarity",
        "hist_correlation",
        "hist_intersection",
        "saliency_similarity",
    ]:
        if metric in metrics:
            normalized_metrics[metric] = min(1.0, max(0.0, metrics[metric]))

    score = 0.0
    total_weight = 0.0

    for metric, weight in app_weights.items():
        if metric in normalized_metrics:
            score += normalized_metrics[metric] * weight
            total_weight += weight

    if total_weight > 0:
        score = score / total_weight
    else:
        if "ssim" in normalized_metrics:
            score = normalized_metrics["ssim"]
        elif "psnr" in normalized_metrics:
            score = normalized_metrics["psnr"]
        else:
            score = 0.5

    return score


def _validate_image_path(image_path: str) -> None:
    if not image_path:
        raise ImageProcessingError("Image path cannot be empty")

    if not os.path.exists(image_path):
        raise ImageProcessingError(f"Image file not found: {image_path}")

    if not os.path.isfile(image_path):
        raise ImageProcessingError(f"The path is not a file: {image_path}")


def _validate_processing_params(
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
    if scale_factor <= 0:
        raise ImageProcessingError("Scale factor must be positive")

    valid_denoise_types = ["gaussian", "median", "bilateral", "none"]
    if denoise_type not in valid_denoise_types:
        raise ImageProcessingError(f"Denoise type must be one of {valid_denoise_types}")

    if denoise_sigma < 0:
        raise ImageProcessingError("Denoise sigma must be non-negative")

    if sharpen_amount <= 0:
        raise ImageProcessingError("Sharpen amount must be positive")

    valid_equalize_methods = ["standard", "clahe", "stretch", "adaptive_gamma"]
    if equalize_method not in valid_equalize_methods:
        raise ImageProcessingError(
            f"Equalize method must be one of {valid_equalize_methods}"
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

    valid_color_methods = ["none", "lab", "rgb", "ratio"]
    if color_preservation not in valid_color_methods:
        raise ImageProcessingError(
            f"Color preservation method must be one of {valid_color_methods}"
        )

    if not (0.0 <= color_preservation_strength <= 1.0):
        raise ImageProcessingError(
            "Color preservation strength must be between 0.0 and 1.0"
        )

    valid_app_types = ["general", "photography", "medical", "document", "art"]
    if application_type not in valid_app_types:
        raise ImageProcessingError(f"Application type must be one of {valid_app_types}")

    if not all(
        0 <= order_val <= 5 for order_val in [order, order_rescale, order_rotate]
    ):
        raise ImageProcessingError("Interpolation orders must be between 0 and 5")


def load_preset(preset_name: str) -> Dict[str, Any]:
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
    preset_path = os.path.join(preset_dir, f"{preset_name}.json")

    if not os.path.exists(preset_path):
        raise ImageProcessingError(f"Preset file not found: {preset_path}")

    try:
        with open(preset_path, "r") as f:
            preset_data = json.load(f)

        if "name" not in preset_data or "params" not in preset_data:
            raise ImageProcessingError(
                "Invalid preset format: missing 'name' or 'params' fields"
            )

        return preset_data["params"]
    except json.JSONDecodeError as e:
        raise ImageProcessingError(f"Failed to parse preset file: {e}")
    except Exception as e:
        raise ImageProcessingError(f"Error loading preset: {e}")


def save_preset(
    preset_name: str, params: Dict[str, Any], description: str = ""
) -> None:
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

    if not os.path.exists(preset_dir):
        try:
            os.makedirs(preset_dir)
        except Exception as e:
            raise ImageProcessingError(f"Failed to create presets directory: {e}")

    preset_path = os.path.join(preset_dir, f"{preset_name}.json")

    preset_data = {"name": preset_name, "description": description, "params": params}

    try:
        with open(preset_path, "w") as f:
            json.dump(preset_data, f, indent=4)
    except Exception as e:
        raise ImageProcessingError(f"Failed to save preset: {e}")


def list_presets() -> List[Dict[str, Any]]:
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

    if not os.path.exists(preset_dir):
        return []

    presets = []

    for filename in os.listdir(preset_dir):
        if filename.endswith(".json"):
            preset_path = os.path.join(preset_dir, filename)
            try:
                with open(preset_path, "r") as f:
                    preset_data = json.load(f)

                presets.append(
                    {
                        "name": preset_data.get("name", filename[:-5]),
                        "description": preset_data.get("description", ""),
                        "filename": filename,
                    }
                )
            except:
                continue

    return presets


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
    try:
        _validate_image_path(image_path)

        _validate_processing_params(
            scale_factor,
            order,
            order_rescale,
            order_rotate,
            denoise_type,
            denoise_sigma,
            sharpen,
            sharpen_amount,
            equalize,
            equalize_method,
            clip_limit,
            clip_limit_kernel_size,
            contrast_stretch_percentiles,
            gamma_correction,
            color_preservation,
            color_preservation_strength,
            application_type,
        )

        try:
            original_image = io.imread(image_path)
            image = img_as_float(original_image)
            processed_image = image.copy()
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

        # 1. Resize (Initial Rescale)
        if scale_factor != 1.0:
            try:
                processed_image = transform.rescale(
                    processed_image,
                    scale_factor,
                    anti_aliasing=True,
                    order=order_rescale,
                )
            except Exception as e:
                raise ImageProcessingError(f"Rescaling failed: {str(e)}")

        # Store a copy of the original image for color preservation
        original_for_color = processed_image.copy()

        # 2. Denoise
        try:
            if denoise_type == "gaussian":
                processed_image = filters.gaussian(processed_image, sigma=denoise_sigma)
            elif denoise_type == "median":
                if processed_image.ndim == 3:  # Handle color images
                    for i in range(3):
                        processed_image[:, :, i] = filters.median(
                            processed_image[:, :, i]
                        )
                else:
                    processed_image = filters.median(processed_image)
            elif denoise_type == "bilateral":
                # Bilateral filter preserves edges better
                if processed_image.ndim == 3:  # Handle color images
                    for i in range(3):
                        processed_image[:, :, i] = filters.gaussian(
                            processed_image[:, :, i],
                            sigma=denoise_sigma,
                            mode="nearest",
                        )
                else:
                    processed_image = filters.gaussian(
                        processed_image, sigma=denoise_sigma, mode="nearest"
                    )
        except Exception as e:
            raise ImageProcessingError(f"Denoising failed: {str(e)}")

        # 3. Sharpening using unsharp mask
        try:
            if sharpen:
                blurred = filters.gaussian(processed_image, sigma=0.5)
                highpass = processed_image - blurred
                processed_image = processed_image + sharpen_amount * highpass
                processed_image = np.clip(processed_image, 0, 1)
        except Exception as e:
            raise ImageProcessingError(f"Sharpening failed: {str(e)}")

        # 4. Contrast Enhancement with Color Preservation
        try:
            if equalize:
                if processed_image.ndim == 3:  # Color image
                    if color_preservation == "lab":
                        lab_image = color.rgb2lab(processed_image)

                        l_channel_normalized = lab_image[:, :, 0] / 100.0

                        if equalize_method == "standard":
                            l_channel_normalized = exposure.equalize_hist(
                                l_channel_normalized
                            )

                        elif equalize_method == "clahe":
                            l_channel_normalized = exposure.equalize_adapthist(
                                l_channel_normalized,
                                kernel_size=clip_limit_kernel_size,
                                clip_limit=clip_limit,
                            )

                        elif equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            l_channel_normalized = exposure.rescale_intensity(
                                l_channel_normalized,
                                in_range=tuple(
                                    np.percentile(l_channel_normalized, (p_low, p_high))
                                ),
                            )

                        elif equalize_method == "adaptive_gamma":
                            mean_luminance = np.mean(l_channel_normalized)
                            adaptive_gamma = gamma_correction * (
                                1.0 - 0.5 * mean_luminance
                            )
                            l_channel_normalized = exposure.adjust_gamma(
                                l_channel_normalized, gamma=adaptive_gamma
                            )

                        lab_image[:, :, 0] = l_channel_normalized * 100.0
                        processed_image = color.lab2rgb(lab_image)

                    elif color_preservation == "rgb":
                        if equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            for i in range(3):
                                processed_image[:, :, i] = exposure.rescale_intensity(
                                    processed_image[:, :, i],
                                    in_range=tuple(
                                        np.percentile(
                                            processed_image[:, :, i], (p_low, p_high)
                                        )
                                    ),
                                )
                        else:
                            hsv_image = color.rgb2hsv(processed_image)
                            if equalize_method == "standard":
                                hsv_image[:, :, 2] = exposure.equalize_hist(
                                    hsv_image[:, :, 2]
                                )
                            elif equalize_method == "clahe":
                                hsv_image[:, :, 2] = exposure.equalize_adapthist(
                                    hsv_image[:, :, 2],
                                    kernel_size=clip_limit_kernel_size,
                                    clip_limit=clip_limit,
                                )
                            elif equalize_method == "adaptive_gamma":
                                mean_luminance = np.mean(hsv_image[:, :, 2])
                                adaptive_gamma = gamma_correction * (
                                    1.0 - 0.5 * mean_luminance
                                )
                                hsv_image[:, :, 2] = exposure.adjust_gamma(
                                    hsv_image[:, :, 2], gamma=adaptive_gamma
                                )
                            processed_image = color.hsv2rgb(hsv_image)

                    elif color_preservation == "ratio":
                        luminance = np.mean(processed_image, axis=2)

                        if equalize_method == "standard":
                            enhanced_luminance = exposure.equalize_hist(luminance)
                        elif equalize_method == "clahe":
                            enhanced_luminance = exposure.equalize_adapthist(
                                luminance,
                                kernel_size=clip_limit_kernel_size,
                                clip_limit=clip_limit,
                            )
                        elif equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            enhanced_luminance = exposure.rescale_intensity(
                                luminance,
                                in_range=tuple(
                                    np.percentile(luminance, (p_low, p_high))
                                ),
                            )
                        elif equalize_method == "adaptive_gamma":
                            mean_lum = np.mean(luminance)
                            adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_lum)
                            enhanced_luminance = exposure.adjust_gamma(
                                luminance, gamma=adaptive_gamma
                            )

                        enhanced_image = np.zeros_like(processed_image)
                        for i in range(3):
                            ratio = np.divide(
                                processed_image[:, :, i],
                                luminance,
                                out=np.ones_like(processed_image[:, :, i]),
                                where=luminance > 0.01,
                            )
                            enhanced_image[:, :, i] = enhanced_luminance * ratio

                        processed_image = enhanced_image
                        processed_image = np.clip(processed_image, 0, 1)

                    else:  # 'none' or default HSV method
                        hsv_image = color.rgb2hsv(processed_image)
                        if equalize_method == "standard":
                            hsv_image[:, :, 2] = exposure.equalize_hist(
                                hsv_image[:, :, 2]
                            )
                        elif equalize_method == "clahe":
                            hsv_image[:, :, 2] = exposure.equalize_adapthist(
                                hsv_image[:, :, 2],
                                kernel_size=clip_limit_kernel_size,
                                clip_limit=clip_limit,
                            )
                        elif equalize_method == "stretch":
                            p_low, p_high = contrast_stretch_percentiles
                            hsv_image[:, :, 2] = exposure.rescale_intensity(
                                hsv_image[:, :, 2],
                                in_range=tuple(
                                    np.percentile(hsv_image[:, :, 2], (p_low, p_high))
                                ),
                            )
                        elif equalize_method == "adaptive_gamma":
                            mean_luminance = np.mean(hsv_image[:, :, 2])
                            adaptive_gamma = gamma_correction * (
                                1.0 - 0.5 * mean_luminance
                            )
                            hsv_image[:, :, 2] = exposure.adjust_gamma(
                                hsv_image[:, :, 2], gamma=adaptive_gamma
                            )
                        processed_image = color.hsv2rgb(hsv_image)
                else:
                    # Grayscale image processing
                    if equalize_method == "standard":
                        processed_image = exposure.equalize_hist(processed_image)
                    elif equalize_method == "clahe":
                        processed_image = exposure.equalize_adapthist(
                            processed_image,
                            kernel_size=clip_limit_kernel_size,
                            clip_limit=clip_limit,
                        )
                    elif equalize_method == "stretch":
                        p_low, p_high = contrast_stretch_percentiles
                        processed_image = exposure.rescale_intensity(
                            processed_image,
                            in_range=tuple(
                                np.percentile(processed_image, (p_low, p_high))
                            ),
                        )
                    elif equalize_method == "adaptive_gamma":
                        mean_luminance = np.mean(processed_image)
                        adaptive_gamma = gamma_correction * (1.0 - 0.5 * mean_luminance)
                        processed_image = exposure.adjust_gamma(
                            processed_image, gamma=adaptive_gamma
                        )
        except Exception as e:
            raise ImageProcessingError(f"Contrast enhancement failed: {str(e)}")

        # 5. Gamma correction (if not using adaptive gamma)
        try:
            if gamma_correction != 1.0 and equalize_method != "adaptive_gamma":
                if processed_image.ndim == 3 and color_preservation == "lab":
                    lab_image = color.rgb2lab(processed_image)

                    l_channel_normalized = lab_image[:, :, 0] / 100.0
                    l_channel_normalized = exposure.adjust_gamma(
                        l_channel_normalized, gamma_correction
                    )
                    lab_image[:, :, 0] = l_channel_normalized * 100.0

                    processed_image = color.lab2rgb(lab_image)
                else:
                    processed_image = exposure.adjust_gamma(
                        processed_image, gamma_correction
                    )
        except Exception as e:
            raise ImageProcessingError(f"Gamma correction failed: {str(e)}")

        # 6. Blend with original if color preservation strength is set
        if (
            color_preservation != "none"
            and color_preservation_strength > 0
            and processed_image.ndim == 3
        ):
            try:
                if color_preservation == "lab":
                    processed_lab = color.rgb2lab(processed_image)
                    processed_luminance = processed_lab[:, :, 0]

                    original_lab = color.rgb2lab(original_for_color)

                    blended_lab = original_lab.copy()
                    blended_lab[:, :, 0] = processed_luminance

                    blended = color.lab2rgb(blended_lab)

                    processed_image = (
                        color_preservation_strength * blended
                        + (1 - color_preservation_strength) * processed_image
                    )
            except Exception as e:
                raise ImageProcessingError(
                    f"Color preservation blending failed: {str(e)}"
                )

        # Ensure final image is in the valid range
        processed_image = np.clip(processed_image, 0, 1)

        # Calculate perceptual metrics if requested
        metrics_dict = None
        if calculate_metrics:
            if min(original_image.shape[:2]) < 7:
                raise ImageProcessingError(
                    "Image too small for perceptual metrics calculation"
                )
            else:
                try:
                    metrics_dict = calculate_perceptual_metrics(
                        original_image,
                        processed_image,
                        calculate_advanced=calculate_advanced_metrics,
                    )

                    metrics_dict["quality_score"] = calculate_quality_score(
                        metrics_dict, application_type=application_type
                    )
                except Exception as e:
                    raise ImageProcessingError(f"Could not calculate metrics: {str(e)}")

        # Save the output if requested
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


def compare_processing_methods(
    image_path: str,
    output_dir: Optional[str] = None,
    application_type: str = "general",
    calculate_advanced_metrics: bool = True,
) -> Dict[str, Dict[str, Any]]:

    _validate_image_path(image_path)

    valid_app_types = ["general", "photography", "medical", "document", "art"]
    if application_type not in valid_app_types:
        raise ImageProcessingError(f"Application type must be one of {valid_app_types}")

    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            raise ImageProcessingError(f"Failed to create output directory: {str(e)}")

    methods = [
        {
            "name": "Standard Equalization (HSV)",
            "params": {
                "equalize_method": "standard",
                "color_preservation": "none",
                "denoise_type": "gaussian",
                "denoise_sigma": 0.8,
                "sharpen": True,
                "sharpen_amount": 1.2,
                "gamma_correction": 1.05,
            },
        },
        {
            "name": "LAB Color Preservation",
            "params": {
                "equalize_method": "stretch",
                "color_preservation": "lab",
                "color_preservation_strength": 0.9,
                "denoise_type": "gaussian",
                "denoise_sigma": 0.8,
                "sharpen": True,
                "sharpen_amount": 1.2,
                "gamma_correction": 1.05,
            },
        },
        {
            "name": "Gentle Enhancement",
            "params": {
                "equalize_method": "stretch",
                "contrast_stretch_percentiles": (5, 95),
                "color_preservation": "lab",
                "color_preservation_strength": 0.9,
                "denoise_type": "gaussian",
                "denoise_sigma": 0.5,
                "sharpen": True,
                "sharpen_amount": 1.0,
                "gamma_correction": 1.0,
            },
        },
        {
            "name": "Color Ratio Preservation",
            "params": {
                "equalize_method": "stretch",
                "color_preservation": "ratio",
                "contrast_stretch_percentiles": (2, 98),
                "denoise_type": "gaussian",
                "denoise_sigma": 0.8,
                "sharpen": True,
                "sharpen_amount": 1.2,
                "gamma_correction": 1.05,
            },
        },
        {
            "name": "Detail Preservation",
            "params": {
                "equalize_method": "clahe",
                "clip_limit": 0.02,
                "clip_limit_kernel_size": 16,
                "color_preservation": "lab",
                "color_preservation_strength": 0.8,
                "denoise_type": "bilateral",
                "denoise_sigma": 0.6,
                "sharpen": True,
                "sharpen_amount": 1.4,
                "gamma_correction": 1.0,
            },
        },
        {
            "name": "High Dynamic Range (HDR)",
            "params": {
                "equalize_method": "clahe",
                "clip_limit": 0.03,
                "clip_limit_kernel_size": 16,
                "color_preservation": "lab",
                "color_preservation_strength": 0.85,
                "denoise_type": "bilateral",
                "denoise_sigma": 0.6,
                "sharpen": True,
                "sharpen_amount": 1.6,
                "gamma_correction": 0.9,
            },
        },
        {
            "name": "Vintage Look",
            "params": {
                "equalize_method": "standard",
                "color_preservation": "lab",
                "color_preservation_strength": 0.5,
                "denoise_type": "median",
                "denoise_sigma": 1.0,
                "sharpen": False,
                "gamma_correction": 1.2,
            },
        },
        {
            "name": "Black & White Contrast",
            "params": {
                "equalize_method": "stretch",
                "contrast_stretch_percentiles": (2, 98),
                "color_preservation": "none",
                "denoise_type": "gaussian",
                "denoise_sigma": 0.7,
                "sharpen": True,
                "sharpen_amount": 1.3,
                "gamma_correction": 1.1,
            },
        },
        {
            "name": "Focus Enhancement",
            "params": {
                "equalize_method": "stretch",
                "contrast_stretch_percentiles": (5, 95),
                "color_preservation": "lab",
                "color_preservation_strength": 0.95,
                "denoise_type": "bilateral",
                "denoise_sigma": 0.4,
                "sharpen": True,
                "sharpen_amount": 2.0,
                "gamma_correction": 1.0,
            },
        },
    ]

    results = {}
    best_method = None
    best_score = -1.0

    for method in methods:
        name = method["name"]
        params = method["params"]

        if output_dir:
            output_path = os.path.join(
                output_dir, f"{name.replace(' ', '_').lower()}.jpg"
            )
        else:
            output_path = None

        try:
            processed, metrics = process_image(
                image_path,
                output_path=output_path,
                application_type=application_type,
                calculate_advanced_metrics=calculate_advanced_metrics,
                **params,
            )

            results[name] = {
                "processed": processed,
                "metrics": metrics,
                "output_path": output_path,
            }

            if (
                metrics
                and "quality_score" in metrics
                and metrics["quality_score"] > best_score
            ):
                best_score = metrics["quality_score"]
                best_method = name

        except Exception as e:
            results[name] = {"error": str(e)}

    if best_method:
        results["best_method"] = {"name": best_method, "score": best_score}

    return results


def _process_single_image_wrapper(input_path, output_path, params, application_type):
    """Wrapper function for ProcessPoolExecutor to handle exceptions."""
    try:
        return _process_single_image(input_path, output_path, params, application_type)
    except Exception as e:
        return {"error": str(e)}


def _process_single_image(
    input_path: str,
    output_path: str,
    params: Dict[str, Any],
    application_type: str,
    logger=None,
) -> Dict[str, Any]:
    """Process a single image and return results."""
    if logger:
        logger.info(f"Processing: {input_path} -> {output_path}")

    try:
        # Process the image
        _, metrics = process_image(
            input_path,
            output_path=output_path,
            application_type=application_type,
            **params,
        )

        # Return success result with metrics
        return {"status": "success", "output_path": output_path, "metrics": metrics}
    except Exception as e:
        if logger:
            logger.error(f"Error processing {input_path}: {e}")
        raise


def analyze_batch(
    input_pattern: str, output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze multiple images to extract characteristics.

    Args:
        input_pattern: Glob pattern to match input images
        output_file: Optional file to save analysis results (JSON)

    Returns:
        Dictionary with analysis results
    """
    # Get list of input files
    input_files = glob.glob(input_pattern)
    if not input_files:
        raise ImageProcessingError(f"No files found matching pattern: {input_pattern}")

    results = {
        "total_images": len(input_files),
        "analyses": {},
        "summary": {
            "brightness": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "contrast": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "noise_level": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "edge_density": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "color_images": 0,
        },
    }

    # Analyze each image
    for input_path in input_files:
        try:
            analysis = analyze_image_characteristics(input_path)
            results["analyses"][input_path] = analysis

            # Update summary statistics
            chars = analysis["characteristics"]
            if chars["is_color"]:
                results["summary"]["color_images"] += 1

            for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
                value = chars[metric]
                results["summary"][metric]["min"] = min(
                    results["summary"][metric]["min"], value
                )
                results["summary"][metric]["max"] = max(
                    results["summary"][metric]["max"], value
                )
                results["summary"][metric]["sum"] += value

        except Exception as e:
            results["analyses"][input_path] = {"error": str(e)}

    # Calculate averages
    successful_analyses = len(results["analyses"]) - sum(
        1 for result in results["analyses"].values() if "error" in result
    )

    if successful_analyses > 0:
        for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
            results["summary"][metric]["avg"] = (
                results["summary"][metric]["sum"] / successful_analyses
            )

    # Save output if requested
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise ImageProcessingError(f"Failed to save analysis results: {e}")

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
    summary = analysis_results["summary"]

    # Determine general characteristics of the batch
    avg_brightness = summary["brightness"].get("avg", 0.5)
    avg_contrast = summary["contrast"].get("avg", 0.3)
    avg_noise = summary["noise_level"].get("avg", 0.03)
    avg_edge_density = summary["edge_density"].get("avg", 0.05)

    is_mostly_color = summary["color_images"] > (analysis_results["total_images"] / 2)

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


# Update main() function to support batch processing
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Image Processing Tool")

    # Input/output arguments
    input_group = parser.add_argument_group("Input/Output")
    input_group.add_argument(
        "image_path",
        nargs="?",
        help="Path to the input image or glob pattern for batch processing",
    )
    input_group.add_argument(
        "--output",
        "-o",
        help="Path for the output image or directory for batch processing",
    )
    input_group.add_argument(
        "--batch", "-b", action="store_true", help="Enable batch processing mode"
    )

    # Processing parameters
    process_group = parser.add_argument_group("Processing Parameters")
    process_group.add_argument(
        "--application",
        "-a",
        choices=["general", "photography", "medical", "document", "art"],
        default="general",
        help="Application type for optimization",
    )
    process_group.add_argument("--preset", help="Name of a preset to use")

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--analyze", action="store_true", help="Analyze image and suggest parameters"
    )
    analysis_group.add_argument(
        "--analyze-batch",
        action="store_true",
        help="Analyze multiple images and suggest optimal parameters",
    )
    analysis_group.add_argument(
        "--compare", action="store_true", help="Compare different processing methods"
    )
    analysis_group.add_argument(
        "--compare-dir",
        default="comparison",
        help="Output directory for comparison results",
    )

    # Preset management
    preset_group = parser.add_argument_group("Preset Management")
    preset_group.add_argument(
        "--save-preset",
        help="Save the current parameters as a preset with the given name",
    )
    preset_group.add_argument(
        "--list-presets", action="store_true", help="List all available presets"
    )
    preset_group.add_argument(
        "--preset-description",
        help="Description for the preset when using --save-preset",
    )

    # Batch processing options
    batch_group = parser.add_argument_group("Batch Processing Options")
    batch_group.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers for batch processing",
    )
    batch_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output file already exists",
    )
    batch_group.add_argument(
        "--report",
        action="store_true",
        help="Generate a JSON report with batch processing results",
    )
    batch_group.add_argument("--log-file", help="Path to log file for batch processing")

    args = parser.parse_args()

    try:
        if args.list_presets:
            presets = list_presets()
            if presets:
                print("\nAvailable presets:")
                for preset in presets:
                    print(f"- {preset['name']}: {preset['description']}")
            else:
                print("\nNo presets found.")
            return 0

        if not args.image_path and not args.list_presets:
            parser.print_help()
            return 1

        app_type = args.application
        params = {}

        if args.preset:
            try:
                preset_params = load_preset(args.preset)
                params.update(preset_params)
                print(f"Loaded preset: {args.preset}")
            except ImageProcessingError as e:
                print(f"Error: {e}")
                return 1

        # Batch analysis mode
        if args.analyze_batch:
            if not args.output:
                output_file = "batch_analysis.json"
            else:
                output_file = args.output

            print(f"Analyzing images matching: {args.image_path}")
            results = analyze_batch(args.image_path, output_file)

            suggestions = suggest_optimal_params(results)

            print(f"\nAnalyzed {results['total_images']} images")
            print("\nBatch Summary:")
            for metric, values in results["summary"].items():
                if metric == "color_images":
                    print(f"Color Images: {values}/{results['total_images']}")
                elif "avg" in values:
                    print(
                        f"{metric.capitalize()}: Avg={values['avg']:.4f}, Min={values['min']:.4f}, Max={values['max']:.4f}"
                    )

            print("\nSuggested Processing Parameters:")
            for param, value in suggestions["params"].items():
                print(f"{param}: {value}")

            print(f"\nSuggested Application Type: {suggestions['application_type']}")
            print(f"\nDetailed analysis saved to: {output_file}")
            return 0

        # Single image analysis mode
        if args.analyze and not args.batch:
            analysis = analyze_image_characteristics(args.image_path)

            print("\nImage Characteristics:")
            print(
                f"Color Image: {'Yes' if analysis['characteristics']['is_color'] else 'No'}"
            )
            print(f"Brightness: {analysis['characteristics']['brightness']:.2f}")
            print(f"Contrast: {analysis['characteristics']['contrast']:.2f}")
            print(f"Noise Level: {analysis['characteristics']['noise_level']:.4f}")
            print(f"Edge Density: {analysis['characteristics']['edge_density']:.4f}")

            print("\nSuggested Processing Parameters:")
            for param, value in analysis["suggested_params"].items():
                print(f"{param}: {value}")

            print(f"\nSuggested Application Type: {analysis['suggested_application']}")

            params.update(analysis["suggested_params"])

            if args.application == "general":
                app_type = analysis["suggested_application"]

        # Comparison mode (single image only)
        if args.compare and not args.batch:
            results = compare_processing_methods(
                args.image_path, output_dir=args.compare_dir, application_type=app_type
            )

            print("\nComparison Results:")
            for name, result in results.items():
                if name == "best_method":
                    continue

                if "metrics" in result and result["metrics"]:
                    metrics = result["metrics"]
                    print(f"\n{name}:")
                    print(f"  SSIM: {metrics['ssim']:.4f}")
                    print(f"  PSNR: {metrics['psnr']:.2f} dB")
                    print(f"  Quality Score: {metrics['quality_score']:.4f}")
                elif "error" in result:
                    print(f"\n{name}: Error - {result['error']}")

            if "best_method" in results:
                print(
                    f"\nBest method: {results['best_method']['name']} "
                    f"(Score: {results['best_method']['score']:.4f})"
                )

                if not args.output:
                    best_name = results["best_method"]["name"]
                    if "output_path" in results[best_name]:
                        print(
                            f"Best result saved to: {results[best_name]['output_path']}"
                        )
                    return 0

        # Batch processing mode
        if args.batch:
            if not args.output:
                print("Error: Output directory must be specified for batch processing")
                return 1

            print(f"Batch processing images matching: {args.image_path}")
            print(f"Output directory: {args.output}")
            print(f"Using {args.workers} workers")

            results = batch_process_images(
                args.image_path,
                args.output,
                params=params,
                preset_name=args.preset if not params else None,
                application_type=app_type,
                n_workers=args.workers,
                skip_existing=args.skip_existing,
                generate_report=args.report,
                log_file=args.log_file,
            )

            print(
                f"\nBatch processing completed in {results['processing_time']:.2f} seconds"
            )
            print(f"Total images: {results['total']}")
            print(f"Processed successfully: {results['successful']}")
            print(f"Failed: {results['failed']}")
            print(f"Skipped: {results['skipped']}")

            if args.report:
                report_path = os.path.join(args.output, "batch_processing_report.json")
                print(f"\nDetailed report saved to: {report_path}")

            return 0

        # Single image processing mode
        if args.output and not args.batch:
            print(f"\nProcessing image with {app_type} application type...")
            processed, metrics = process_image(
                args.image_path,
                output_path=args.output,
                application_type=app_type,
                **params,
            )

            if metrics:
                print("\nQuality Metrics:")
                print(f"SSIM: {metrics['ssim']:.4f}")
                print(f"PSNR: {metrics['psnr']:.2f} dB")
                print(f"Quality Score: {metrics['quality_score']:.4f}")

            print(f"Processed image saved to: {args.output}")

        if args.save_preset:
            try:
                save_preset(
                    args.save_preset, params, description=args.preset_description or ""
                )
                print(f"\nSaved preset '{args.save_preset}'")
            except ImageProcessingError as e:
                print(f"Error saving preset: {e}")
                return 1

    except ImageProcessingError as e:
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

    return 0


def analyze_image_characteristics(image_path: str) -> Dict[str, Any]:
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


# Batch Processing Functions
def setup_logger(log_file=None, log_level=logging.INFO):
    """Setup logging configuration."""
    logger = logging.getLogger("batch_processor")
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def batch_process_images(
    input_pattern: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    preset_name: Optional[str] = None,
    application_type: str = "general",
    n_workers: int = 4,
    skip_existing: bool = False,
    generate_report: bool = True,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process multiple images matching the input pattern.

    Args:
        input_pattern: Glob pattern to match input images (e.g., "input/*.jpg")
        output_dir: Directory to save processed images
        params: Processing parameters to use for all images
        preset_name: Name of a preset to use (alternative to params)
        application_type: Application type for processing optimization
        n_workers: Number of parallel workers for processing
        skip_existing: Skip processing if output file already exists
        generate_report: Generate a JSON report with processing results
        log_file: Path to log file (optional)

    Returns:
        Dictionary with processing results
    """
    logger = setup_logger(log_file)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise ImageProcessingError(f"Failed to create output directory: {e}")

    # Get list of input files
    input_files = glob.glob(input_pattern)
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        raise ImageProcessingError(f"No files found matching pattern: {input_pattern}")

    logger.info(f"Found {len(input_files)} files to process")

    # Load preset if specified
    if preset_name:
        try:
            processing_params = load_preset(preset_name)
            logger.info(f"Loaded preset: {preset_name}")
        except ImageProcessingError as e:
            logger.error(f"Error loading preset: {e}")
            raise
    else:
        processing_params = params or {}

    # Prepare processing tasks
    tasks = []
    for input_path in input_files:
        filename = os.path.basename(input_path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}_processed{ext}")

        if skip_existing and os.path.exists(output_path):
            logger.info(f"Skipping existing file: {output_path}")
            continue

        tasks.append((input_path, output_path))

    logger.info(f"Preparing to process {len(tasks)} images with {n_workers} workers")

    # Initialize results
    results = {
        "successful": 0,
        "failed": 0,
        "skipped": len(input_files) - len(tasks),
        "total": len(input_files),
        "processing_time": 0,
        "files": {},
    }

    # Process images in parallel
    start_time = time.time()

    if n_workers <= 1:
        # Sequential processing
        for input_path, output_path in tasks:
            try:
                results["files"][input_path] = _process_single_image(
                    input_path, output_path, processing_params, application_type, logger
                )
                results["successful"] += 1
            except Exception as e:
                logger.error(f"Error processing {input_path}: {e}")
                results["files"][input_path] = {"error": str(e)}
                results["failed"] += 1
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for input_path, output_path in tasks:
                future = executor.submit(
                    _process_single_image_wrapper,
                    input_path,
                    output_path,
                    processing_params,
                    application_type,
                )
                futures[future] = input_path

            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    result = future.result()
                    results["files"][input_path] = result
                    results["successful"] += 1
                    logger.info(f"Successfully processed: {input_path}")
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")
                    results["files"][input_path] = {"error": str(e)}
                    results["failed"] += 1

    end_time = time.time()
    results["processing_time"] = end_time - start_time

    logger.info(
        f"Batch processing completed in {results['processing_time']:.2f} seconds"
    )
    logger.info(
        f"Successful: {results['successful']}, Failed: {results['failed']}, Skipped: {results['skipped']}"
    )

    # Generate report if requested
    if generate_report:
        report_path = os.path.join(output_dir, "batch_processing_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    return results


if __name__ == "__main__":
    import sys

    sys.exit(main())
