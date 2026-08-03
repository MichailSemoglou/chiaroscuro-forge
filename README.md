# Chiaroscuro Forge

[![PyPI version](https://img.shields.io/pypi/v/chiaroscuro-forge.svg)](https://pypi.org/project/chiaroscuro-forge/)
[![PyPI downloads](https://img.shields.io/pypi/dm/chiaroscuro-forge.svg)](https://pypi.org/project/chiaroscuro-forge/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/chiaroscuro-forge?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/chiaroscuro-forge)
[![Python versions](https://img.shields.io/pypi/pyversions/chiaroscuro-forge.svg)](https://pypi.org/project/chiaroscuro-forge/)
[![License](https://img.shields.io/pypi/l/chiaroscuro-forge.svg)](https://github.com/MichailSemoglou/chiaroscuro-forge/blob/main/LICENSE)
[![CI](https://github.com/MichailSemoglou/chiaroscuro-forge/actions/workflows/test.yml/badge.svg)](https://github.com/MichailSemoglou/chiaroscuro-forge/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18200572.svg)](https://doi.org/10.5281/zenodo.18200572)

An intelligent image enhancement tool inspired by Renaissance techniques. Features automatic parameter detection, advanced color preservation, quality metrics, parallel batch processing, GPU acceleration, REST API, and distributed processing. Perfect for photographers and developers seeking to transform ordinary images with artistic precision.

## Features

### Core Processing

- **Intelligent Enhancement**: Automatically analyzes image characteristics and applies optimal processing parameters
- **Advanced Color Preservation**: Maintains color fidelity while enhancing contrast and details
- **Multiple Enhancement Methods**: LAB, RGB, and ratio-based color processing modes
- **Quality Metrics**: Calculates SSIM, PSNR, MS-SSIM, CIEDE2000 color difference, and LPIPS perceptual similarity
- **Preset System**: Save and reuse customized enhancement settings
- **Application Types**: Specialized processing for photography, documents, medical images, and art

### Advanced Features

- **GPU Acceleration**: CUDA (NVIDIA), OpenCL (cross-platform), and Metal (Apple Silicon) support for compute-intensive operations
- **REST API**: FastAPI-based HTTP API with authentication, rate limiting, and async job processing
- **Distributed Processing**: Task queue system with Redis and Celery backend support (local queue included)
- **Property-Based Testing**: Hypothesis framework integration for comprehensive edge case validation
- **Batch Processing**: Process multiple images in parallel with detailed reporting

## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- scikit-image

### Optional Dependencies

- **GPU Acceleration**: `pyopencl` (OpenCL), `cupy` (NVIDIA CUDA)
- **REST API**: `fastapi`, `uvicorn`, `httpx`, `python-multipart`
- **Distributed Processing**: `redis`, `celery` (for production deployments)
- **Property-Based Testing**: `hypothesis` (for advanced testing)

### Install from PyPI

```bash
pip install chiaroscuro-forge
```

After installing, the CLI entrypoint is available as:

```bash
chiaroscuro-forge --help
```

### Install from source

```bash
git clone https://github.com/MichailSemoglou/chiaroscuro-forge.git
cd chiaroscuro-forge
pip install -e .
```

## Quick Start

### Process a single image

```bash
chiaroscuro-forge input.jpg --output enhanced.jpg
```

### Analyze an image and suggest parameters

```bash
chiaroscuro-forge input.jpg --analyze
```

### Process multiple images in batch mode

```bash
chiaroscuro-forge "images/*.jpg" --output processed/ --batch
```

### Create and use presets

```bash
# Save parameters as preset
chiaroscuro-forge input.jpg --analyze --save-preset my_preset

# Use preset to process images
chiaroscuro-forge input.jpg --output enhanced.jpg --preset my_preset
```

## Command-Line Options

### Input/Output

- `image_path`: Path to input image or glob pattern for batch processing
- `--output, -o`: Path for output image or directory for batch processing
- `--batch, -b`: Enable batch processing mode

### Processing Parameters

- `--application, -a`: Application type (general, photography, medical, document, art)
- `--preset`: Name of a preset to use

### Analysis Options

- `--analyze`: Analyze image and suggest parameters
- `--analyze-batch`: Analyze multiple images and suggest optimal parameters
- `--compare`: Compare different processing methods
- `--compare-dir`: Output directory for comparison results

### Preset Management

- `--save-preset`: Save parameters as a preset
- `--list-presets`: List all available presets
- `--preset-description`: Description for the preset

### Batch Processing Options

- `--workers, -w`: Number of parallel workers (default: 4)
- `--skip-existing`: Skip files that have already been processed
- `--report`: Generate a JSON report with processing results
- `--log-file`: Path to log file for batch processing

## Examples

### Basic Enhancement

```bash
chiaroscuro-forge photo.jpg --output enhanced.jpg
```

### Custom Application Type

```bash
chiaroscuro-forge document.jpg --output enhanced.jpg --application document
```

### Analyze and Process

```bash
chiaroscuro-forge photo.jpg --analyze --output enhanced.jpg
```

### Compare Processing Methods

```bash
chiaroscuro-forge photo.jpg --compare
```

### Batch Processing with Report

```bash
chiaroscuro-forge "photos/*.jpg" --output enhanced/ --batch --workers 8 --report
```

## Python API

You can use Chiaroscuro Forge directly in your Python code:

### Basic Image Processing

```python
from chiaroscuro_forge import process_image

processed, metrics = process_image(
    "input.jpg",
    output_path="enhanced.jpg",
    application_type="photography"
)
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

### Get Image Statistics

```python
from chiaroscuro_forge import get_image_statistics

stats = get_image_statistics("photo.jpg")
print(f"Dimensions: {stats['dimensions']}")
print(f"Brightness: {stats['brightness']:.2f}")
print(f"Dynamic range: {stats['dynamic_range']:.2f}")
print(f"Contrast ratio: {stats['contrast_ratio']:.2f}")
```

### Analyze Image Characteristics

```python
from chiaroscuro_forge import analyze_image_characteristics

analysis = analyze_image_characteristics("photo.jpg")
print(f"Suggested parameters: {analysis['suggested_params']}")
```

### GPU Acceleration

```python
from chiaroscuro_forge.gpu import GPUContext, gpu_available

if gpu_available():
    with GPUContext() as gpu:
        result = gpu.gaussian_blur(image, sigma=2.0)
        edges = gpu.sobel_filter(image)
```

### REST API Usage

Start the API server:

```bash
uvicorn chiaroscuro_forge.api:app --reload
```

Then use it from Python:

```python
import requests

# Create API key
response = requests.post(
    "http://localhost:8000/api/v1/keys",
    data={"name": "my-app", "rate_limit": "100"}
)
api_key = response.json()["data"]["api_key"]

# Process an image
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/process",
        headers={"X-API-Key": api_key},
        files={"image": f},
        data={"gamma": 1.5, "application_type": "photography"}
    )
job_id = response.json()["job_id"]
```

Visit http://localhost:8000/docs for interactive API documentation.

## Development

The project is structured around core image processing functions with a focus on quality and customizability:

### Core Modules

- `processing.py`: Main image processing pipeline with unified `ProcessingConfig`
- `pipeline.py`: Stage-based pipeline pattern (resize, denoise, sharpen, contrast, gamma, color preservation)
- `analysis.py`: Image analysis and automatic parameter detection
- `metrics.py`: Quality metrics (SSIM, PSNR, MS-SSIM, CIEDE2000, LPIPS)
- `config.py`: Shared `ProcessingConfig` dataclass for all entry points
- `validation.py`: Input and output path validation, parameter checks, security guards
- `batch.py`: Parallel batch processing with progress tracking
- `presets.py`: JSON-based preset save/load system
- `tiling.py`: Tile-based processing for large images
- `comparison.py`: Multi-method comparison and optimal parameter suggestion
- `exceptions.py`: Custom exception hierarchy
- `constants.py`: Centralized constants and thresholds

### Advanced Modules

- `gpu.py`: GPU acceleration with CUDA, OpenCL, and Metal support, CPU fallback
- `api.py`: FastAPI REST API with authentication, rate limiting, and job management
- `distributed.py`: Task queue system with local/Redis/Celery backend support
- `optional.py`: Graceful optional-dependency isolation
- `cache.py`: Intelligent caching for performance optimization
- `di.py`: Dependency injection container

### Testing

- 425+ passing tests with coverage enforcement
- Property-based testing with Hypothesis
- GPU and API integration tests
- Comprehensive unit and integration test suites

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
