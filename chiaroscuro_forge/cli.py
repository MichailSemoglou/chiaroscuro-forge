"""
Command-Line Interface Module

This module provides the command-line interface for Chiaroscuro Forge,
handling argument parsing and user interaction.

Migration Status:
    ⚠️  This module is currently a stub that imports from the monolithic file.
    Functions will be gradually migrated here in upcoming releases.
    
    Expected completion: v0.4.0
"""

import sys
import os

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    # Import from monolithic file
    import chiaroscuro_forge as _cf_old
    
    # Re-export main CLI function
    main = _cf_old.main
    
    __all__ = ["main"]

except (ImportError, AttributeError) as e:
    import warnings
    warnings.warn(
        f"Could not import CLI functions from monolithic file: {e}. "
        "Native implementations not yet available.",
        ImportWarning
    )
    
    def main():
        """Fallback main function if import fails."""
        print("Error: Could not load Chiaroscuro Forge CLI.")
        print("Please ensure the package is properly installed: pip install -e .")
        return 1
    
    __all__ = ["main"]

# Clean up namespace
try:
    del _cf_old
except NameError:
    pass

# TODO: Migrate these functions from chiaroscuro_forge.py:
# - main() (CLI entry point ~400 lines)
