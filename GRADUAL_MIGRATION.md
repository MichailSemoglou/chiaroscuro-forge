# Gradual Migration Progress Tracker

## Overview

This document tracks the gradual migration from the monolithic `chiaroscuro_forge.py`
to a modular package structure. The migration follows Option A: maintaining 100%
backward compatibility while incrementally building the new structure.

## Timeline

- **v0.3.0** (Current): Initial modular structure with stub modules
- **v0.3.x**: Gradual function migration to modules
- **v0.4.0**: Add deprecation warnings for monolithic imports
- **v0.5.0**: Complete migration, remove monolithic file

## Module Migration Status

### ✅ Completed

#### Infrastructure (v0.3.0)

- [x] Package directory structure created
- [x] `__init__.py` with backward-compatible imports
- [x] `constants.py` - Configuration constants
- [x] `exceptions.py` - Custom exception classes
- [x] `validation.py` - Input validation functions
- [x] CI/CD workflow (GitHub Actions)
- [x] Development tooling (black, flake8, mypy, pytest)
- [x] Documentation (MIGRATION.md)

#### Stub Modules (v0.3.0)

- [x] `metrics.py` - Stub created, imports from monolithic
- [x] `processing.py` - Stub created, imports from monolithic
- [x] `analysis.py` - Stub created, imports from monolithic
- [x] `batch.py` - Stub created, imports from monolithic
- [x] `comparison.py` - Stub created, imports from monolithic
- [x] `presets.py` - Stub created, imports from monolithic
- [x] `cli.py` - Stub created, imports from monolithic

### 🚧 In Progress

#### Testing (v0.3.1)

- [x] `test_backward_compatibility.py` - Verify imports work
- [ ] `test_validation.py` - Test validation functions
- [ ] `test_constants.py` - Test constant values
- [ ] Achieve 70%+ coverage on completed modules

### 📋 Planned

#### Metrics Module Migration (v0.3.2)

Functions to migrate from `chiaroscuro_forge.py`:

- [ ] `ssim()` - Structural Similarity Index (~40 lines)
- [ ] `ms_ssim()` - Multi-Scale SSIM (~80 lines)
- [ ] `feature_similarity()` - Feature-based similarity (~100 lines)
- [ ] `histogram_similarity()` - Histogram comparison (~70 lines)
- [ ] `calculate_perceptual_metrics()` - Comprehensive metrics (~100 lines)
- [ ] `calculate_quality_score()` - Quality scoring (~80 lines)
- [ ] Helper: `calculate_window_size()` (~10 lines)

**Estimated effort**: 4-6 hours

#### Analysis Module Migration (v0.3.3)

Functions to migrate:

- [ ] `analyze_image_characteristics()` - Image analysis (~150 lines)
- [ ] `get_image_statistics()` - Statistics calculation (~100 lines)

**Estimated effort**: 3-4 hours

#### Presets Module Migration (v0.3.4)

Functions to migrate:

- [ ] `load_preset()` - Load preset from JSON (~30 lines)
- [ ] `save_preset()` - Save preset to JSON (~30 lines)
- [ ] `list_presets()` - List available presets (~30 lines)

**Estimated effort**: 1-2 hours

#### Batch Module Migration (v0.3.5)

Functions to migrate:

- [ ] `batch_process_images()` - Parallel processing (~150 lines)
- [ ] `analyze_batch()` - Batch analysis (~100 lines)
- [ ] `suggest_optimal_params()` - Parameter suggestion (~80 lines)
- [ ] `setup_logger()` - Logging configuration (~30 lines)
- [ ] `_process_single_image()` - Single image processor (~40 lines)
- [ ] `_process_single_image_wrapper()` - Wrapper for parallel (~10 lines)

**Estimated effort**: 5-7 hours

#### Comparison Module Migration (v0.3.6)

Functions to migrate:

- [ ] `compare_processing_methods()` - Method comparison (~200 lines)

**Estimated effort**: 2-3 hours

#### Processing Module Migration (v0.3.7-0.3.9)

Functions to migrate:

- [ ] `process_image()` - Main processing function (~300 lines)
  - This is the largest and most complex function
  - May need to be split into sub-functions
  - Should be done last after all helpers are migrated

**Estimated effort**: 8-12 hours

#### CLI Module Migration (v0.3.10)

Functions to migrate:

- [ ] `main()` - CLI entry point (~400 lines)
  - Argument parsing
  - Workflow orchestration
  - Error handling

**Estimated effort**: 4-6 hours

## Testing Strategy

For each module migration:

1. **Before Migration**
   - Verify backward compatibility tests pass
   - Document current function signatures
   - Identify dependencies

2. **During Migration**
   - Copy function to new module
   - Update imports to use constants/validation modules
   - Add comprehensive docstrings
   - Add type hints if missing
   - Extract magic numbers to constants

3. **After Migration**
   - Write unit tests for the function
   - Verify backward compatibility still works
   - Update stub module to use native function
   - Run full test suite
   - Update documentation

## Deprecation Strategy (v0.4.0)

When ready to deprecate the monolithic file:

```python
import warnings

# In chiaroscuro_forge.py
warnings.warn(
    "Importing from 'chiaroscuro_forge.py' is deprecated and will be "
    "removed in v0.5.0. Please import from 'chiaroscuro_forge' package instead:\n"
    "  from chiaroscuro_forge import process_image\n"
    "See MIGRATION.md for details.",
    DeprecationWarning,
    stacklevel=2
)
```

## Removal Strategy (v0.5.0)

Steps for final removal:

1. **Update **init**.py**
   - Remove monolithic import fallback
   - Use only modular imports

2. **Remove Files**
   - Delete `chiaroscuro_forge.py`
   - Update `.gitignore` if needed

3. **Update Documentation**
   - Mark migration as complete
   - Update examples to use new structure

4. **Version Bump**
   - Increment to v0.5.0 (breaking change)
   - Update CHANGELOG

## Benefits of Gradual Migration

✅ **Zero breaking changes** during transition
✅ **Users can adopt at their own pace**
✅ **Each module can be tested independently**
✅ **Easier to review and merge changes**
✅ **Rollback is simple if issues arise**
✅ **Incremental improvement of code quality**

## Quality Checks for Each Module

Before marking a module as "complete":

- [ ] All functions migrated with docstrings
- [ ] Unit tests with >80% coverage
- [ ] Type hints on all functions
- [ ] No magic numbers (use constants)
- [ ] Passes flake8 linting
- [ ] Passes mypy type checking
- [ ] Formatted with black
- [ ] Backward compatibility verified

## Current Status Summary

**Phase**: Initial Setup Complete ✅
**Next Step**: Begin metrics module migration
**Overall Progress**: 20% (Infrastructure complete, 0/7 modules migrated)
**ETA for Completion**: v0.4.0 (estimated 30-40 hours of work)

---

Last Updated: 2026-02-03
