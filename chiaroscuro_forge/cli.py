"""
Cli module - temporarily importing from monolithic file.

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
        main = _cf_old.main

        # Cleanup
        del spec
else:
    from ..exceptions import ImageProcessingError
    raise ImageProcessingError(f"Could not find monolithic file at {{_monolithic_path}}")

__all__ = ["main"]
