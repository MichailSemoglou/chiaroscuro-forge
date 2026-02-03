"""
Quality Metrics Module

This module provides comprehensive image quality metrics including SSIM, MS-SSIM,
PSNR, feature similarity, histogram similarity, and perceptual quality scores.

The metrics are based on established research in image quality assessment and
are optimized for various application types (photography, medical imaging, etc.).

Migration Status:
    ⚠️  This module is currently a stub that imports from the monolithic file.
    Functions will be gradually migrated here in upcoming releases.
    
    Expected completion: v0.4.0
"""

from typing import Dict, List, Optional
import numpy as np

# Temporary imports from monolithic file for backward compatibility
import sys
import os

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    # Import from monolithic file
    import chiaroscuro_forge as _cf_old
    
    # Re-export functions (these will be native implementations in future versions)
    ssim = _cf_old.ssim
    ms_ssim = _cf_old.ms_ssim
    feature_similarity = _cf_old.feature_similarity
    histogram_similarity = _cf_old.histogram_similarity
    calculate_perceptual_metrics = _cf_old.calculate_perceptual_metrics
    calculate_quality_score = _cf_old.calculate_quality_score
    
    __all__ = [
        "ssim",
        "ms_ssim",
        "feature_similarity",
        "histogram_similarity",
        "calculate_perceptual_metrics",
        "calculate_quality_score",
    ]

except (ImportError, AttributeError) as e:
    import warnings
    warnings.warn(
        f"Could not import metrics from monolithic file: {e}. "
        "Native implementations not yet available.",
        ImportWarning
    )
    __all__ = []

# Clean up namespace
try:
    del _cf_old
except NameError:
    pass

# TODO: Migrate these functions from chiaroscuro_forge.py:
# - ssim()
# - ms_ssim()
# - feature_similarity()
# - histogram_similarity()
# - calculate_perceptual_metrics()
# - calculate_quality_score()
# - Helper function: calculate_window_size()
