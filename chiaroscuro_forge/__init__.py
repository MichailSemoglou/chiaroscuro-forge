"""
Chiaroscuro Forge - Intelligent Image Enhancement Tool

This package provides comprehensive image enhancement capabilities inspired by
Renaissance art techniques, featuring automatic parameter detection, advanced
color preservation, and quality metrics.

Main Features:
    - Intelligent enhancement with automatic parameter selection
    - Multiple color preservation methods (LAB, RGB, ratio-based)
    - Quality metrics (SSIM, PSNR, MS-SSIM, perceptual metrics)
    - Batch processing with parallel execution
    - Preset system for reusable configurations

Example:
    Basic usage for image enhancement::

        from chiaroscuro_forge import process_image
        
        processed, metrics = process_image(
            "input.jpg",
            output_path="enhanced.jpg",
            application_type="photography"
        )
        print(f"Quality Score: {metrics['quality_score']:.4f}")

For more examples, see the documentation at:
https://github.com/MichailSemoglou/chiaroscuro-forge
"""

__version__ = "0.2.1"
__author__ = "Michail Semoglou"
__email__ = "m.semoglou@tongji.edu.cn"
__license__ = "MIT"

from chiaroscuro_forge.processing import process_image
from chiaroscuro_forge.analysis import (
    analyze_image_characteristics,
    get_image_statistics,
)
from chiaroscuro_forge.batch import batch_process_images, analyze_batch
from chiaroscuro_forge.comparison import compare_processing_methods
from chiaroscuro_forge.presets import save_preset, load_preset, list_presets
from chiaroscuro_forge.exceptions import ImageProcessingError

__all__ = [
    "process_image",
    "analyze_image_characteristics",
    "get_image_statistics",
    "batch_process_images",
    "analyze_batch",
    "compare_processing_methods",
    "save_preset",
    "load_preset",
    "list_presets",
    "ImageProcessingError",
]
