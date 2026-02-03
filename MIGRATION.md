# Migration Guide: Version 0.2.x → 0.3.0

## Overview

Chiaroscuro Forge has been refactored to follow professional open-source standards. The monolithic `chiaroscuro_forge.py` file has been split into a modular package structure for better maintainability, testing, and documentation.

## What Changed

### Package Structure

**Before (v0.2.x):**
```
chiaroscuro-forge/
├── chiaroscuro_forge.py  (2,202 lines - everything in one file)
├── setup.py
└── tests/
    └── test_chiaroscuro.py
```

**After (v0.3.0):**
```
chiaroscuro-forge/
├── chiaroscuro_forge/
│   ├── __init__.py         # Package interface
│   ├── constants.py        # Configuration constants
│   ├── exceptions.py       # Custom exceptions
│   ├── validation.py       # Input validation
│   ├── metrics.py          # Quality metrics (SSIM, PSNR, etc.)
│   ├── processing.py       # Main image processing
│   ├── analysis.py         # Image analysis functions
│   ├── batch.py            # Batch processing
│   ├── comparison.py       # Method comparison
│   ├── presets.py          # Preset management
│   └── cli.py              # Command-line interface
├── chiaroscuro_forge.py    # DEPRECATED: Kept for backward compatibility
├── tests/
│   ├── test_metrics.py
│   ├── test_processing.py
│   ├── test_analysis.py
│   ├── test_validation.py
│   └── test_integration.py
├── .github/workflows/
│   └── test.yml            # CI/CD configuration
├── requirements-dev.txt    # Development dependencies
└── setup.cfg               # Tool configuration
```

### API Compatibility

**Good News: The public API remains 100% backward compatible!**

All existing code will continue to work without modifications:

```python
# This still works exactly as before
from chiaroscuro_forge import process_image, analyze_image_characteristics

processed, metrics = process_image("input.jpg", output_path="output.jpg")
analysis = analyze_image_characteristics("photo.jpg")
```

### New Import Options

You can now import from specific modules for better code organization:

```python
# Option 1: Import from main package (recommended, backward compatible)
from chiaroscuro_forge import process_image, get_image_statistics

# Option 2: Import from specific modules (new, more explicit)
from chiaroscuro_forge.processing import process_image
from chiaroscuro_forge.analysis import get_image_statistics
from chiaroscuro_forge.metrics import ssim, ms_ssim
from chiaroscuro_forge.batch import batch_process_images
```

## Deprecation Notice

### Old Monolithic File

The original `chiaroscuro_forge.py` file is **DEPRECATED** but will be maintained for **6 months** (until August 2026) for backward compatibility.

**Timeline:**
- **v0.3.0 (Now)**: Old file deprecated, new package structure introduced
- **v0.4.0 (May 2026)**: Deprecation warnings added
- **v0.5.0 (August 2026)**: Old file removed

### Migration Recommendations

1. **For existing projects**: No immediate action required, but plan to update imports
2. **For new projects**: Use the new package structure imports
3. **Update your code gradually**: Both styles work in v0.3.0

## New Features in v0.3.0

### 1. Comprehensive Documentation

Every function now has detailed docstrings with:
- Parameter descriptions
- Return value documentation
- Usage examples
- References to academic papers (where applicable)

```python
>>> help(process_image)
# Shows comprehensive documentation
```

### 2. Better Type Hints

All functions have complete type annotations for better IDE support:

```python
from chiaroscuro_forge import process_image
from typing import Tuple, Optional, Dict
import numpy as np

# IDE now provides full autocomplete and type checking
result: Tuple[np.ndarray, Optional[Dict[str, float]]] = process_image(...)
```

### 3. Constants Module

Magic numbers replaced with named constants:

```python
from chiaroscuro_forge.constants import (
    DEFAULT_DENOISE_SIGMA,
    DEFAULT_SHARPEN_AMOUNT,
    VALID_APP_TYPES,
    QUALITY_WEIGHTS
)
```

### 4. Improved Error Messages

More informative error messages with context:

```python
# Before
ImageProcessingError: Invalid parameter

# After
ImageProcessingError: Denoise type must be one of ['gaussian', 'median', 'bilateral', 'none'], got 'unknown'
```

### 5. Enhanced Testing

- Test coverage increased from ~20% to >70%
- Comprehensive test suite with edge cases
- CI/CD pipeline with GitHub Actions
- Automated testing on multiple Python versions (3.8-3.11)
- Automated testing on multiple OS (Ubuntu, macOS, Windows)

### 6. Code Quality Tools

Integrated development tools:
- **Black**: Automatic code formatting
- **Flake8**: Style guide enforcement
- **MyPy**: Static type checking
- **Pytest**: Modern testing framework
- **Coverage**: Code coverage reporting

Install development tools:
```bash
pip install -r requirements-dev.txt
```

## Breaking Changes

### None! 🎉

This release maintains 100% backward compatibility with v0.2.x.

However, if you were directly importing internal functions (not part of the public API), you may need to update imports:

```python
# If you were doing this (not recommended):
from chiaroscuro_forge import _process_single_image  # Internal function

# Update to:
from chiaroscuro_forge.batch import _process_single_image
```

## Testing Your Migration

Run this script to verify compatibility:

```python
"""Test script to verify migration compatibility."""
import sys

def test_backward_compatibility():
    """Test that old import style still works."""
    try:
        # Test old-style imports
        from chiaroscuro_forge import (
            process_image,
            analyze_image_characteristics,
            get_image_statistics,
            batch_process_images,
            compare_processing_methods,
            ImageProcessingError
        )
        
        # Test new-style imports
        from chiaroscuro_forge.processing import process_image as process_image_new
        from chiaroscuro_forge.analysis import analyze_image_characteristics as analyze_new
        
        # Verify they're the same
        assert process_image is process_image_new
        assert analyze_image_characteristics is analyze_new
        
        print("✅ All backward compatibility tests passed!")
        return True
        
    except (ImportError, AssertionError) as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_backward_compatibility()
    sys.exit(0 if success else 1)
```

## Getting Help

If you encounter any issues during migration:

1. Check the [documentation](https://github.com/MichailSemoglou/chiaroscuro-forge)
2. Review the [examples](docs/examples/)
3. Open an [issue](https://github.com/MichailSemoglou/chiaroscuro-forge/issues)

## Contributors

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

The new modular structure makes it much easier to:
- Add new features
- Fix bugs
- Write tests
- Improve documentation
- Understand the codebase

## Acknowledgments

Thank you to all users and contributors who made this project successful. This refactoring ensures Chiaroscuro Forge remains maintainable and extensible for years to come.
