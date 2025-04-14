# Chiaroscuro Forge 
## Basic Usage Examples

This document provides examples of common usage patterns for ChiaroscuroForge.

## Single Image Processing

The most basic way to use Chiaroscuro Forge is to process a single image with default settings:

```bash
python chiaroscuro_forge.py input.jpg --output enhanced.jpg
```

This applies intelligent enhancement with default parameters optimized for general images.

## Analyzing Image Characteristics

Before processing, you can analyze an image to see its characteristics and get parameter suggestions:

```bash
python chiaroscuro_forge.py input.jpg --analyze
```

This will output information about the image's:
- Brightness
- Contrast
- Noise level
- Edge density
- Color properties
- And more...

It also suggests optimal processing parameters based on this analysis.

## Specific Application Types

Different types of images benefit from different enhancement approaches:

```bash
# For photographs
python chiaroscuro_forge.py photo.jpg --output enhanced.jpg --application photography

# For documents
python chiaroscuro_forge.py scan.jpg --output enhanced.jpg --application document

# For medical images
python chiaroscuro_forge.py xray.jpg --output enhanced.jpg --application medical

# For artwork
python chiaroscuro_forge.py painting.jpg --output enhanced.jpg --application art
```

## Comparing Enhancement Methods

To see which enhancement method works best for a particular image:

```bash
python chiaroscuro_forge.py input.jpg --compare
```

This will:
1. Process the image with multiple enhancement methods
2. Calculate quality metrics for each method
3. Determine the best method based on these metrics
4. Save all processed images to a "comparison" directory

## Creating and Using Presets

Save your favorite settings as presets for future use:

```bash
# Create a preset after analysis
python chiaroscuro_forge.py input.jpg --analyze --save-preset my_preset --preset-description "My custom settings"

# Use a preset
python chiaroscuro_forge.py input.jpg --output enhanced.jpg --preset my_preset

# List available presets
python chiaroscuro_forge.py --list-presets
```

## Batch Processing

Process multiple images at once:

```bash
# Basic batch processing
python chiaroscuro_forge.py "images/*.jpg" --output processed/ --batch

# With parallel processing (8 workers)
python chiaroscuro_forge.py "images/*.jpg" --output processed/ --batch --workers 8

# Skip already processed images
python chiaroscuro_forge.py "images/*.jpg" --output processed/ --batch --skip-existing

# Generate a JSON report
python chiaroscuro_forge.py "images/*.jpg" --output processed/ --batch --report

# With logging
python chiaroscuro_forge.py "images/*.jpg" --output processed/ --batch --log-file batch.log
```

## Analyzing a Batch of Images

To analyze multiple images and get optimal parameters for the entire set:

```bash
python chiaroscuro_forge.py "images/*.jpg" --analyze-batch --output batch_analysis.json
```

This generates a JSON file with:
- Individual image characteristics
- Summary statistics for the entire batch
- Suggested processing parameters optimized for the whole batch
