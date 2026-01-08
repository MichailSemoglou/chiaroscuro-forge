# Chiaroscuro Forge

An intelligent image enhancement tool inspired by Renaissance techniques. Features automatic parameter detection, advanced color preservation, quality metrics, and parallel batch processing. Perfect for photographers and developers seeking to transform ordinary images with artistic precision.

## Features

- **Intelligent Enhancement**: Automatically analyzes image characteristics and applies optimal processing parameters
- **Advanced Color Preservation**: Maintains color fidelity while enhancing contrast and details
- **Multiple Enhancement Methods**: LAB, RGB, and ratio-based color processing modes
- **Quality Metrics**: Calculates SSIM, PSNR, MS-SSIM, and other perceptual quality scores
- **Batch Processing**: Process multiple images in parallel with detailed reporting
- **Preset System**: Save and reuse customized enhancement settings
- **Application Types**: Specialized processing for photography, documents, medical images, and art

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-image

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

## Development

The project is structured around core image processing functions with a focus on quality and customizability:

- `analyze_image_characteristics()`: Extracts characteristics from images
- `process_image()`: Main processing function with numerous customizable parameters
- `compare_processing_methods()`: Compares different enhancement approaches
- `batch_process_images()`: Handles processing of multiple images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
