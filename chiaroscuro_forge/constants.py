"""
Configuration Constants for Chiaroscuro Forge

This module contains all configuration constants used throughout the package,
including default values and thresholds.
"""

import warnings

# SSIM Calculation Constants
DEFAULT_WIN_SIZE = 7
"""Default window size for SSIM calculation"""

MIN_WIN_SIZE = 3
"""Minimum window size for SSIM"""

# Feature Detection Constants
MIN_CELL_SIZE = 8
"""Minimum cell size for HOG feature extraction"""

CELL_SIZE_DIVISOR = 32
"""Divisor for calculating dynamic cell size"""

MAX_ORB_KEYPOINTS = 500
"""Maximum number of ORB keypoints to detect"""

ORB_KEYPOINT_DENSITY = 1000
"""Density factor for calculating dynamic keypoint count"""

ORB_FAST_THRESHOLD = 0.05
"""FAST corner detection threshold for ORB"""

# Edge Detection Constants
DEFAULT_EDGE_SIGMA = 1.0
"""Default sigma for Canny edge detection"""

MIN_EDGE_DENSITY_LOW = 0.02
"""Threshold for low edge density (needs more sharpening)"""

MAX_EDGE_DENSITY_HIGH = 0.10
"""Threshold for high edge density (needs less sharpening)"""

# Noise Estimation Constants
NOISE_WINDOW_SIZE = 5
"""Window size for noise estimation"""

HIGH_NOISE_THRESHOLD = 0.06
"""Threshold for high noise level"""

MEDIUM_NOISE_THRESHOLD = 0.03
"""Threshold for medium noise level"""

# Brightness and Contrast Constants
BRIGHTNESS_DARK_THRESHOLD = 0.4
"""Threshold for dark images"""

BRIGHTNESS_BRIGHT_THRESHOLD = 0.7
"""Threshold for bright images"""

CONTRAST_LOW_THRESHOLD = 0.15
"""Threshold for low contrast"""

CONTRAST_MEDIUM_THRESHOLD = 0.25
"""Threshold for medium contrast"""

# Color Characteristics Constants
COLOR_SATURATION_THRESHOLD = 0.3
"""Threshold for high color saturation"""

COLOR_VARIANCE_THRESHOLD = 0.2
"""Threshold for high color variance"""

# Processing Defaults
DEFAULT_DENOISE_SIGMA = 0.8
"""Default denoising sigma"""

DEFAULT_SHARPEN_AMOUNT = 1.2
"""Default sharpening amount"""

DEFAULT_GAMMA = 1.0
"""Default gamma correction value"""

DEFAULT_CLIP_LIMIT = 0.03
"""Default CLAHE clip limit"""

DEFAULT_CLIP_LIMIT_KERNEL_SIZE = 8
"""Default CLAHE kernel size"""

DEFAULT_CONTRAST_PERCENTILES = (2, 98)
"""Default percentiles for contrast stretching"""

# Batch Processing Defaults
DEFAULT_WORKERS = 4
"""Default number of parallel workers"""

# Tile-Based Processing Constants
DEFAULT_TILE_SIZE = 512
"""Default tile size for large image processing (pixels)"""

DEFAULT_TILE_OVERLAP = 64
"""Default overlap between tiles for seamless stitching (pixels)"""

TILING_MEMORY_THRESHOLD_MB = 100.0
"""Memory threshold in MB above which tiling is used"""

MIN_TILE_SIZE = 128
"""Minimum allowed tile size"""

MAX_TILE_SIZE = 2048
"""Maximum allowed tile size"""

# Valid Application Types
VALID_APP_TYPES = ["general", "photography", "medical", "document", "art"]
"""List of valid application types"""

# Valid Processing Methods
VALID_DENOISE_TYPES = ["gaussian", "median", "bilateral", "none"]
"""List of valid denoising methods"""

VALID_EQUALIZE_METHODS = ["standard", "clahe", "stretch", "adaptive_gamma"]
"""List of valid equalization methods"""

VALID_COLOR_METHODS = ["none", "lab", "rgb", "ratio"]
"""List of valid color preservation methods"""

VALID_HISTOGRAM_METHODS = ["correlation", "chi_square", "intersection", "bhattacharyya"]
"""List of valid histogram similarity methods"""

VALID_FEATURE_METHODS = ["hog", "orb", "canny"]
"""List of valid feature similarity methods"""

# Normalization Constants
PSNR_MIN = 20.0
"""Minimum PSNR value for normalization"""

PSNR_RANGE = 30.0
"""PSNR range for normalization (max - min)"""

MSE_SCALE = 20.0
"""Scale factor for MSE normalization"""

# Luminance Coefficients (ITU-R BT.709)
LUMINANCE_R = 0.2126
"""Red channel weight for luminance calculation"""

LUMINANCE_G = 0.7152
"""Green channel weight for luminance calculation"""

LUMINANCE_B = 0.0722
"""Blue channel weight for luminance calculation"""

# Security Constants
MAX_FILE_SIZE_MB = 100
"""Maximum allowed file size in megabytes (default: 100MB)"""

MAX_IMAGE_PIXELS = 100_000_000
"""Maximum allowed image pixels (default: 100 megapixels, ~10000x10000)"""

MAX_DIMENSION = 32768
"""Maximum allowed single dimension in pixels (default: 32768)"""

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
"""Set of allowed image file extensions"""

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/gif",
    "image/webp",
}
"""Set of allowed MIME types for images"""

# Path Security
MAX_PATH_LENGTH = 4096
"""Maximum allowed path length"""

DENIED_PATH_PATTERNS = [
    "..",  # Parent directory traversal
    "~",  # Home directory expansion
    "\x00",  # Null byte injection
]
"""Patterns that indicate potential path traversal attacks"""


# Deprecated compatibility alias, scheduled for removal in 3.0.0
_DEPRECATED_QUALITY_WEIGHTS = {
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


def __getattr__(name):
    """Provide the deprecated QUALITY_WEIGHTS constant with a warning."""
    if name == "QUALITY_WEIGHTS":
        warnings.warn(
            "QUALITY_WEIGHTS is deprecated and will be removed in 3.0.0; "
            "the composite score weights live in calculate_quality_score",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_QUALITY_WEIGHTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
