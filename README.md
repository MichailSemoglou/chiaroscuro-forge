# Chiaroscuro Forge

[![PyPI version](https://img.shields.io/pypi/v/chiaroscuro-forge.svg)](https://pypi.org/project/chiaroscuro-forge/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/chiaroscuro-forge?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=MAGENTA&left_text=downloads)](https://pepy.tech/projects/chiaroscuro-forge)
[![Python versions](https://img.shields.io/pypi/pyversions/chiaroscuro-forge.svg)](https://pypi.org/project/chiaroscuro-forge/)
[![License](https://img.shields.io/pypi/l/chiaroscuro-forge.svg)](https://github.com/MichailSemoglou/chiaroscuro-forge/blob/main/LICENSE)
[![CI](https://github.com/MichailSemoglou/chiaroscuro-forge/actions/workflows/test.yml/badge.svg)](https://github.com/MichailSemoglou/chiaroscuro-forge/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18200571.svg)](https://doi.org/10.5281/zenodo.18200571)

Chiaroscuro Forge is a Python library for image enhancement and quality assessment. It applies configurable enhancement stages, color-preservation strategies, and perceptual metrics to improve image appearance while keeping the workflow transparent and scriptable.

## Overview

This project is designed for both practical image processing and research-oriented experimentation. It supports:

- image enhancement workflows with configurable stages
- color-preservation modes for LAB-, RGB-, and ratio-based processing
- perceptual and structural quality metrics
- batch processing and preset-based workflows
- optional GPU acceleration and an API layer

## Features

### Core processing

- automatic parameter analysis for common enhancement tasks
- contrast, denoising, sharpening, and gamma adjustments
- LAB and other color-preservation strategies
- quality metrics including SSIM, PSNR, MS-SSIM, CIEDE2000, and LPIPS
- reusable presets for batch or repeated processing

### Advanced capabilities

- CUDA, OpenCL, and Metal support when available
- FastAPI-based API service for remote or automated workflows
- distributed processing with local, Redis, or Celery-backed options
- tile-based processing for large input images
- comparison tools for evaluating multiple processing settings

## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- scikit-image

### Optional dependencies

- GPU acceleration: `pyopencl`, `cupy`
- REST API: `fastapi`, `uvicorn`, `httpx`, `python-multipart`
- Distributed processing: `redis`, `celery`
- Property-based testing: `hypothesis`

### Install from PyPI

```bash
pip install chiaroscuro-forge
```

### Install from source

```bash
git clone https://github.com/MichailSemoglou/chiaroscuro-forge.git
cd chiaroscuro-forge
pip install -e .
```

## Quick start

### Process a single image from the command line

```bash
chiaroscuro-forge input.jpg --output enhanced.jpg
```

### Use the experimental linear-light workflow

```bash
chiaroscuro-forge input.jpg --output enhanced.jpg --linear
```

This opt-in mode converts sRGB input to linear-light values before enhancement and maps the result back to sRGB output.

### Analyze an image and suggest parameters

```bash
chiaroscuro-forge input.jpg --analyze
```

### Batch process multiple images

```bash
chiaroscuro-forge "images/*.jpg" --output processed/ --batch
```

### Save and reuse a preset

```bash
# Save a preset after analysis
chiaroscuro-forge input.jpg --analyze --save-preset my_preset

# Use the preset later
chiaroscuro-forge input.jpg --output enhanced.jpg --preset my_preset
```

## Common examples

### 1. Basic enhancement

```bash
chiaroscuro-forge photo.jpg --output enhanced.jpg
```

### 2. Apply a specific application profile

```bash
chiaroscuro-forge document.jpg --output enhanced.jpg --application document
```

### 3. Compare processing methods

```bash
chiaroscuro-forge photo.jpg --compare
```

### 4. Batch processing with report output

```bash
chiaroscuro-forge "photos/*.jpg" --output enhanced/ --batch --workers 8 --report
```

### 5. Run a linear-light enhancement pass

```bash
chiaroscuro-forge portrait.jpg --output portrait_linear.jpg --linear --application photography
```

## Python API

### Basic processing

```python
from chiaroscuro_forge import process_image

processed, metrics = process_image(
    "input.jpg",
    output_path="enhanced.jpg",
    application_type="photography",
)

print(metrics)
```

### Inspect image statistics

```python
from chiaroscuro_forge import get_image_statistics

stats = get_image_statistics("photo.jpg")
print(stats)
```

### Analyze image characteristics

```python
from chiaroscuro_forge import analyze_image_characteristics

analysis = analyze_image_characteristics("photo.jpg")
print(analysis)
```

### GPU processing

```python
from chiaroscuro_forge.gpu import GPUContext, gpu_available

if gpu_available():
    with GPUContext() as gpu:
        result = gpu.gaussian_blur(image, sigma=2.0)
```

### API server

Start the local API server:

```bash
uvicorn chiaroscuro_forge.api:app --reload
```

Then interact with it from Python or the browser UI at `http://localhost:8000/docs`.

## CLI reference

### Input and output

- `image_path`: path to an input image or a glob pattern for batch processing
- `--output, -o`: output image path or output directory
- `--batch, -b`: process multiple images

### Processing options

- `--application, -a`: image-use profile such as general, photography, medical, document, or art
- `--preset`: name of a preset to use
- `--linear`: enable the experimental linear-light workflow

### Analysis and comparison

- `--analyze`: analyze an image and suggest parameters
- `--analyze-batch`: analyze multiple images
- `--compare`: compare multiple processing methods
- `--compare-dir`: save comparison output to a directory

### Preset management

- `--save-preset`: save current parameters as a preset
- `--list-presets`: list available presets
- `--preset-description`: set a description for a preset

### Batch execution

- `--workers, -w`: number of parallel workers
- `--skip-existing`: skip files already processed
- `--report`: generate a JSON report
- `--log-file`: path for logs

## Project structure

- `chiaroscuro_forge/processing.py`: main image-processing entry points
- `chiaroscuro_forge/pipeline.py`: stage-based processing pipeline
- `chiaroscuro_forge/analysis.py`: image analysis and parameter estimation
- `chiaroscuro_forge/metrics.py`: quality metrics
- `chiaroscuro_forge/config.py`: shared configuration model
- `chiaroscuro_forge/validation.py`: validation and security checks
- `chiaroscuro_forge/batch.py`: batch processing logic
- `chiaroscuro_forge/presets.py`: preset save/load functions
- `chiaroscuro_forge/tiling.py`: tiled processing for large images
- `chiaroscuro_forge/gpu.py`: GPU support and fallback behavior
- `chiaroscuro_forge/api.py`: REST API implementation
- `chiaroscuro_forge/distributed.py`: distributed task processing

## Development

The project includes automated tests, linting, and CI validation. Contributions are welcome through pull requests on the repository.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. The project uses a standard GitHub contribution flow with a feature-branch workflow.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
