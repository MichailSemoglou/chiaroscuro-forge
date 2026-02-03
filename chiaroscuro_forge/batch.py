"""
Batch processing functions for handling multiple images.

This module provides functionality for processing multiple images in parallel,
analyzing batches of images, and generating processing reports.
"""

import os
import glob
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

from .exceptions import ImageProcessingError
from .processing import process_image
from .presets import load_preset
from .analysis import analyze_image_characteristics


def setup_logger(log_file=None, log_level=logging.INFO):
    """Setup logging configuration."""
    logger = logging.getLogger("batch_processor")
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _process_single_image_wrapper(input_path, output_path, params, application_type):
    """Wrapper function for ProcessPoolExecutor to handle exceptions."""
    try:
        return _process_single_image(input_path, output_path, params, application_type)
    except Exception as e:
        return {"error": str(e)}


def _process_single_image(
    input_path: str,
    output_path: str,
    params: Dict[str, Any],
    application_type: str,
    logger=None,
) -> Dict[str, Any]:
    """Process a single image and return results."""
    if logger:
        logger.info(f"Processing: {input_path} -> {output_path}")

    try:
        # Process the image
        _, metrics = process_image(
            input_path,
            output_path=output_path,
            application_type=application_type,
            **params,
        )

        # Return success result with metrics
        return {"status": "success", "output_path": output_path, "metrics": metrics}
    except Exception as e:
        if logger:
            logger.error(f"Error processing {input_path}: {e}")
        raise


def batch_process_images(
    input_pattern: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    preset_name: Optional[str] = None,
    application_type: str = "general",
    n_workers: int = 4,
    skip_existing: bool = False,
    generate_report: bool = True,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process multiple images matching the input pattern.

    Args:
        input_pattern: Glob pattern to match input images (e.g., "input/*.jpg")
        output_dir: Directory to save processed images
        params: Processing parameters to use for all images
        preset_name: Name of a preset to use (alternative to params)
        application_type: Application type for processing optimization
        n_workers: Number of parallel workers for processing
        skip_existing: Skip processing if output file already exists
        generate_report: Generate a JSON report with processing results
        log_file: Path to log file (optional)

    Returns:
        Dictionary with processing results
    """
    logger = setup_logger(log_file)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise ImageProcessingError(f"Failed to create output directory: {e}")

    # Get list of input files
    input_files = glob.glob(input_pattern)
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        raise ImageProcessingError(f"No files found matching pattern: {input_pattern}")

    logger.info(f"Found {len(input_files)} files to process")

    # Load preset if specified
    if preset_name:
        try:
            processing_params = load_preset(preset_name)
            logger.info(f"Loaded preset: {preset_name}")
        except ImageProcessingError as e:
            logger.error(f"Error loading preset: {e}")
            raise
    else:
        processing_params = params or {}

    # Prepare processing tasks
    tasks = []
    for input_path in input_files:
        filename = os.path.basename(input_path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}_processed{ext}")

        if skip_existing and os.path.exists(output_path):
            logger.info(f"Skipping existing file: {output_path}")
            continue

        tasks.append((input_path, output_path))

    logger.info(f"Preparing to process {len(tasks)} images with {n_workers} workers")

    # Initialize results
    results = {
        "successful": 0,
        "failed": 0,
        "skipped": len(input_files) - len(tasks),
        "total": len(input_files),
        "processing_time": 0,
        "files": {},
    }

    # Process images in parallel
    start_time = time.time()

    if n_workers <= 1:
        # Sequential processing
        for input_path, output_path in tasks:
            try:
                results["files"][input_path] = _process_single_image(
                    input_path, output_path, processing_params, application_type, logger
                )
                results["successful"] += 1
            except Exception as e:
                logger.error(f"Error processing {input_path}: {e}")
                results["files"][input_path] = {"error": str(e)}
                results["failed"] += 1
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for input_path, output_path in tasks:
                future = executor.submit(
                    _process_single_image_wrapper,
                    input_path,
                    output_path,
                    processing_params,
                    application_type,
                )
                futures[future] = input_path

            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    result = future.result()
                    results["files"][input_path] = result
                    results["successful"] += 1
                    logger.info(f"Successfully processed: {input_path}")
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")
                    results["files"][input_path] = {"error": str(e)}
                    results["failed"] += 1

    end_time = time.time()
    results["processing_time"] = end_time - start_time

    logger.info(
        f"Batch processing completed in {results['processing_time']:.2f} seconds"
    )
    logger.info(
        f"Successful: {results['successful']}, Failed: {results['failed']}, Skipped: {results['skipped']}"
    )

    # Generate report if requested
    if generate_report:
        report_path = os.path.join(output_dir, "batch_processing_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    return results


def analyze_batch(
    input_pattern: str, output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze multiple images to extract characteristics.

    Args:
        input_pattern: Glob pattern to match input images
        output_file: Optional file to save analysis results (JSON)

    Returns:
        Dictionary with analysis results
    """
    # Get list of input files
    input_files = glob.glob(input_pattern)
    if not input_files:
        raise ImageProcessingError(f"No files found matching pattern: {input_pattern}")

    results = {
        "total_images": len(input_files),
        "analyses": {},
        "summary": {
            "brightness": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "contrast": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "noise_level": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "edge_density": {"min": 1.0, "max": 0.0, "sum": 0.0},
            "color_images": 0,
        },
    }

    # Analyze each image
    for input_path in input_files:
        try:
            analysis = analyze_image_characteristics(input_path)
            results["analyses"][input_path] = analysis

            # Update summary statistics
            chars = analysis["characteristics"]
            if chars["is_color"]:
                results["summary"]["color_images"] += 1

            for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
                value = chars[metric]
                results["summary"][metric]["min"] = min(
                    results["summary"][metric]["min"], value
                )
                results["summary"][metric]["max"] = max(
                    results["summary"][metric]["max"], value
                )
                results["summary"][metric]["sum"] += value

        except Exception as e:
            results["analyses"][input_path] = {"error": str(e)}

    # Calculate averages
    successful_analyses = len(results["analyses"]) - sum(
        1 for result in results["analyses"].values() if "error" in result
    )

    if successful_analyses > 0:
        for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
            results["summary"][metric]["avg"] = (
                results["summary"][metric]["sum"] / successful_analyses
            )

    # Save output if requested
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise ImageProcessingError(f"Failed to save analysis results: {e}")

    return results


__all__ = [
    "batch_process_images",
    "analyze_batch",
    "setup_logger",
    "_process_single_image",
    "_process_single_image_wrapper",
]
