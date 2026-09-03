# Changelog

All notable changes to Chiaroscuro Forge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.2.0] – 2026-09-03

### Fixed

- Color contrast enhancement now runs the LAB L channel in [0, 1] and rescales afterward; CLAHE raised an error on color images with scikit-image 0.26, and adaptive gamma never triggered on color images
- LPIPS similarity scales inputs to [-1, 1], the calibration of the published networks; reported values shift slightly from previous releases

### Changed

- The MS-SSIM docstring documents four implementation departures from the reference code of Wang, Simoncelli, and Bovik (2003)

### Removed

- `QUALITY_WEIGHTS` constant from `constants.py`; the composite score weights live in `calculate_quality_score`

## [2.1.0] – 2026-08-18

### Added

- Opt-in linear-light processing path with a `linear_light` config flag and `--linear` CLI option
- Tone mapping stage for an experimental linear-light workflow that maps processed values back to sRGB output

### Changed

- The processing pipeline can convert sRGB input to linear-light values before subsequent stages when the feature is enabled
- The linear-light mode remains opt-in to preserve backward compatibility with the default gamma-encoded workflow
- MS-SSIM exposes a compatibility mode for the historical per-scale averaging behavior alongside the current multiscale implementation

## [2.0.1] – 2026-08-16

### Fixed

- Release metadata update for PyPI publishing compatibility
- CI publication metadata support updated for the current GitHub Actions platform

## [2.0.0] – 2026-08-03

### Added

- `ProcessingConfig` dataclass unifying configuration across CLI, API, presets, and batch runner
- `optional.py` module for graceful optional-dependency isolation
- `tests/test_integration.py` with 43 end-to-end and regression tests
- CI pipeline (`.github/workflows/test.yml`) with Python matrix, `black`/`flake8` lint, and `pip-audit`
- Full test suite with coverage gate (`test-full` CI job) on pull requests
- Output-path validation (`_validate_output_path`) applied across all file-write paths

### Changed

- Replaced CIE76 ΔE\*ab color distance metric with CIEDE2000 in quality assessment
- Added LPIPS perceptual similarity metric with MS-SSIM fallback (optional `[perceptual]` extra)
- Replaced "bilateral" denoising approximation with true edge-preserving bilateral filter
- Raised minimum Python version from 3.8 to 3.9
- Updated `numpy>=1.22.0`, `scikit-image>=0.20.0`, `python-multipart>=0.0.7` security floors
- Added `Pillow>=10.0.0` as a declared dependency
- Restricted API CORS to localhost origins; added security headers
- Hardened API authentication bootstrap flow
- CI now runs `black --check` and `flake8` (was syntax-only lint)
- `CONTRIBUTING.md` updated with correct `pytest` commands, `black`/`flake8` steps, and coverage guidance
- API version string reads from package `__version__` instead of hardcoded literal
- 40 Python files reformatted with Black

### Fixed

- Tiling output accumulated with `float32` precision (was `uint8`, causing truncation)
- OpenCL GPU→CPU transfer buffer allocation (was always 1 element)
- TOCTOU race in `JobManager.update_job`
- RGBA crash in `color.rgb2lab` call (alpha channel stripped)
- Inverted contrast percentile logic in `analysis.py`
- Histogram method name reconciliation between `VALID_HISTOGRAM_METHODS` and implementation
- Path-traversal vulnerability in output file paths (now validated)
- Information leakage through exception messages in API responses
- Preset absolute-path leak in error messages
- Duplicate wrapper functions in batch processing

### Added (community infrastructure)

- `SECURITY.md` – vulnerability reporting policy with deployment guidance
- `CODE_OF_CONDUCT.md` – Contributor Covenant 2.0
- `CHANGELOG.md` – Keep a Changelog format
- Issue templates (`bug_report.md`, `feature_request.md`), PR template
- `CODEOWNERS`, `dependabot.yml`

### Security

- Output-path traversal validation
- API error messages sanitized to avoid internal path and exception disclosure
- Bootstrap key window closes after first key; env var provisioning
- CORS restricted to localhost; security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`)
- `pip-audit` runs in CI

## [1.0.1] – 2026-02-03

### Fixed

- Fixed `ImportError` when FastAPI is not installed (module guards)
- Fixed CI test failures across Python 3.8–3.11
- Fixed `setup.py` attempting to read from deleted legacy file
- Fixed GitHub Actions workflow referencing non-existent files

### Changed

- Removed legacy monolithic file (78 KB reduction)
- Simplified package structure
- Made FastAPI truly optional (graceful import fallback)

### Removed

- Deprecated `chiaroscuro_forge.py` legacy entry-point

## [1.0.0] – 2025-11-15

### Added

- Initial public release
- Intelligent image enhancement with automatic parameter detection
- Multiple color preservation methods (LAB, RGB, ratio)
- Quality metrics (SSIM, PSNR, MS-SSIM)
- Batch processing with parallel execution
- Preset system for reusable configurations
- GPU acceleration (CUDA, OpenCL, Metal) with CPU fallback
- FastAPI REST API with authentication and rate limiting
- Distributed processing with local/Redis/Celery backends
- Tiling support for large image processing
