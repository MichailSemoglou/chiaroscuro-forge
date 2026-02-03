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

Migration Note:
    This package is undergoing a gradual migration from a monolithic structure
    to a modular package. Currently, all functionality is imported from the
    original chiaroscuro_forge.py file to ensure 100% backward compatibility.
    
    The modular structure will be built incrementally:
    - v0.3.x: Gradual module implementation (backward compatible)
    - v0.4.x: Deprecation warnings for old imports
    - v0.5.x: Complete migration to modular structure
    
    Your existing code will continue to work without any changes during this
    transition period. See MIGRATION.md for details.
"""

__version__ = "0.3.0"
__author__ = "Michail Semoglou"
__email__ = "m.semoglou@tongji.edu.cn"
__license__ = "MIT"

# Import strategy: Try monolithic first (backward compatibility),
# fall back to modular structure when available

_import_successful = False

# Strategy 1: Import from the original monolithic file
if not _import_successful:
    try:
        import sys
        import importlib.util
        from pathlib import Path
        
        # Get the parent directory (where chiaroscuro_forge.py should be)
        _package_dir = Path(__file__).parent
        _parent_dir = _package_dir.parent
        _monolithic_path = _parent_dir / "chiaroscuro_forge.py"
        
        if _monolithic_path.exists():
            # Import the monolithic file directly using importlib
            spec = importlib.util.spec_from_file_location("_cf_monolithic", _monolithic_path)
            if spec and spec.loader:
                _cf_monolithic = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_cf_monolithic)
                
                # Extract the functions we need
                process_image = _cf_monolithic.process_image
                analyze_image_characteristics = _cf_monolithic.analyze_image_characteristics
                get_image_statistics = _cf_monolithic.get_image_statistics
                batch_process_images = _cf_monolithic.batch_process_images
                analyze_batch = _cf_monolithic.analyze_batch
                compare_processing_methods = _cf_monolithic.compare_processing_methods
                save_preset = _cf_monolithic.save_preset
                load_preset = _cf_monolithic.load_preset
                list_presets = _cf_monolithic.list_presets
                ImageProcessingError = _cf_monolithic.ImageProcessingError
                
                _import_successful = True
                
                # Clean up
                del spec, _cf_monolithic
        
        # Clean up temporary variables
        del sys, importlib, Path, _package_dir, _parent_dir, _monolithic_path
        
    except Exception as e:
        # Store error for debugging but continue to modular imports
        _import_error = str(e)
        pass

# Strategy 2: Import from modular structure (future default)
if not _import_successful:
    try:
        from chiaroscuro_forge.processing import process_image
        from chiaroscuro_forge.analysis import (
            analyze_image_characteristics,
            get_image_statistics,
        )
        from chiaroscuro_forge.batch import batch_process_images, analyze_batch
        from chiaroscuro_forge.comparison import compare_processing_methods
        from chiaroscuro_forge.presets import save_preset, load_preset, list_presets
        from chiaroscuro_forge.exceptions import ImageProcessingError
        from chiaroscuro_forge.cache import (
            get_cache_manager,
            invalidate_preset_cache,
            invalidate_stats_cache,
        )
        
        _import_successful = True
        
    except ImportError as e:
        _import_error_2 = str(e)
        pass

# Error if neither strategy worked
if not _import_successful:
    error_msg = "Could not import Chiaroscuro Forge functions.\n"
    try:
        error_msg += f"Monolithic import error: {_import_error}\n"
    except NameError:
        pass
    try:
        error_msg += f"Modular import error: {_import_error_2}\n"
    except NameError:
        pass
    error_msg += "Please ensure the package is properly installed: pip install -e ."
    raise ImportError(error_msg)

# Clean up
del _import_successful
try:
    del _import_error
except NameError:
    pass
try:
    del _import_error_2
except NameError:
    pass

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
    "get_cache_manager",
    "invalidate_preset_cache",
    "invalidate_stats_cache",
]
