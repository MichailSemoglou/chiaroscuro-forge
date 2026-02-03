"""
Processing Method Comparison Module

This module provides functionality to compare different image enhancement methods
and identify the best approach for a given image.

Migration Status:
    ⚠️  This module is currently a stub that imports from the monolithic file.
    Functions will be gradually migrated here in upcoming releases.
    
    Expected completion: v0.4.0
"""

from typing import Dict, Any, Optional

# Temporary imports from monolithic file for backward compatibility
import sys
import os

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    # Import from monolithic file
    import chiaroscuro_forge as _cf_old
    
    # Re-export comparison function
    compare_processing_methods = _cf_old.compare_processing_methods
    
    __all__ = ["compare_processing_methods"]

except (ImportError, AttributeError) as e:
    import warnings
    warnings.warn(
        f"Could not import comparison functions from monolithic file: {e}. "
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
# - compare_processing_methods()
