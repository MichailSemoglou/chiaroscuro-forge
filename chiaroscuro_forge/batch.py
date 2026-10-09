"""
Batch processing functions for handling multiple images.

This module provides functionality for processing multiple images in parallel,
analyzing batches of images, and generating processing reports.
"""

import glob
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional

from .analysis import analyze_image_characteristics
from .config import ProcessingConfig
from .exceptions import ImageProcessingError
from .presets import load_preset
from .processing import process_image
from .validation import _validate_output_path


def setup_logger(log_file=None, log_level=logging.INFO):
    """Set up logging for batch processing.

    Parameters
    ----------
    log_file : str, optional
        Path to a log file. When given, a file handler is added and its
        parent directory is created if missing.
    log_level : int, optional
        Logging level for the logger (default: logging.INFO).

    Returns
    -------
    logging.Logger
        The configured "batch_processor" logger, with existing handlers
        replaced.
    """
    logger = logging.getLogger("batch_processor")
    logger.setLevel(log_level)

    if logger.handlers:
        logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        # Ensure parent directory exists (needed for Windows)
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _process_single_image_wrapper(
    input_path, output_path, params=None, application_type="general", config=None
):
    """Send one image through the pool with either params or config.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path for the processed image.
    params : dict, optional
        Processing parameters forwarded to process_image.
    application_type : str, optional
        Application type for processing optimization (default: "general").
    config : ProcessingConfig, optional
        Configuration object. Takes precedence over params when given.

    Returns
    -------
    dict
        Result dictionary, or ``{"error": message}`` if processing fails.
    """
    try:
        if config is not None:
            return _process_single_image(
                input_path, output_path, config=config, application_type=application_type
            )
        return _process_single_image(
            input_path, output_path, params=params, application_type=application_type
        )
    except Exception as e:
        return {"error": str(e)}


def _process_single_image(
    input_path: str,
    output_path: str,
    params: Optional[Dict[str, Any]] = None,
    application_type: str = "general",
    config=None,
    logger=None,
) -> Dict[str, Any]:
    """Process a single image and return the result.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path for the processed image.
    params : dict, optional
        Processing parameters forwarded to process_image. Ignored when
        config is given.
    application_type : str, optional
        Application type for processing optimization (default: "general").
    config : ProcessingConfig, optional
        Configuration object. Takes precedence over params when given.
    logger : logging.Logger, optional
        Logger for progress and error messages.

    Returns
    -------
    dict
        Dictionary with status, output_path, and metrics keys.

    Raises
    ------
    Exception
        Re-raises any processing failure after logging it.
    """
    if logger:
        logger.info("Processing: %s", os.path.basename(input_path))

    try:
        if config is not None:
            _, metrics = process_image(
                input_path,
                output_path=output_path,
                config=config,
            )
        else:
            expanded_params = dict(params or {})
            expanded_params["application_type"] = application_type
            _, metrics = process_image(
                input_path,
                output_path=output_path,
                **expanded_params,
            )

        return {"status": "success", "output_path": output_path, "metrics": metrics}
    except Exception as e:
        if logger:
            logger.error("Error processing %s: %s", os.path.basename(input_path), type(e).__name__)
        raise


def batch_process_images(
    input_pattern: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    config: "Optional['ProcessingConfig']" = None,
    preset_name: Optional[str] = None,
    application_type: str = "general",
    n_workers: int = 4,
    skip_existing: bool = False,
    generate_report: bool = True,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Process multiple images matching the input pattern.

    Parameters
    ----------
    input_pattern : str
        Glob pattern to match input images (e.g., "input/*.jpg").
    output_dir : str
        Directory to save processed images.
    params : dict, optional
        Processing parameters to use for all images.
    config : ProcessingConfig, optional
        Configuration object applied to all images. Takes precedence over
        preset_name and params when given.
    preset_name : str, optional
        Name of a preset to use (alternative to params).
    application_type : str, optional
        Application type for processing optimization (default: "general").
    n_workers : int, optional
        Number of parallel workers (default: 4). A value of 1 or less runs
        sequentially in the current process.
    skip_existing : bool, optional
        Skip processing if the output file already exists (default: False).
    generate_report : bool, optional
        Write a JSON report with processing results (default: True).
    log_file : str, optional
        Path to a log file.

    Returns
    -------
    dict
        Counts of successful, failed, and skipped images, the total
        processing time, and a per-file result mapping.

    Raises
    ------
    ImageProcessingError
        If the output directory cannot be created or no files match
        input_pattern.
    """
    logger = setup_logger(log_file)

    safe_output = _validate_output_path(output_dir)
    if not os.path.exists(safe_output):
        try:
            os.makedirs(safe_output)
            logger.info("Created output directory")
        except Exception as e:
            logger.error("Failed to create output directory: %s", type(e).__name__)
            raise ImageProcessingError(f"Failed to create output directory: {e}") from e

    input_files = glob.glob(input_pattern)
    if not input_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        raise ImageProcessingError(f"No files found matching pattern: {input_pattern}")

    logger.info(f"Found {len(input_files)} files to process")

    # A provided config takes precedence over preset_name and params.
    use_config = config is not None

    if use_config:
        processing_params = {}
        logger.info("Using provided ProcessingConfig")
    elif preset_name:
        try:
            processing_params = load_preset(preset_name)
            logger.info(f"Loaded preset: {preset_name}")
        except ImageProcessingError as e:
            logger.error(f"Error loading preset: {e}")
            raise
    else:
        processing_params = params or {}

    tasks = []
    for input_path in input_files:
        filename = os.path.basename(input_path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(safe_output, f"{base_name}_processed{ext}")

        if skip_existing and os.path.exists(output_path):
            logger.info(f"Skipping existing file: {output_path}")
            continue

        tasks.append((input_path, output_path))

    logger.info(f"Preparing to process {len(tasks)} images with {n_workers} workers")

    file_results: Dict[str, Any] = {}
    results: Dict[str, Any] = {
        "successful": 0,
        "failed": 0,
        "skipped": len(input_files) - len(tasks),
        "total": len(input_files),
        "processing_time": 0,
        "files": file_results,
    }

    start_time = time.time()

    if n_workers <= 1:
        # Sequential processing
        for input_path, output_path in tasks:
            try:
                if use_config:
                    results["files"][input_path] = _process_single_image(
                        input_path,
                        output_path,
                        config=config,
                        logger=logger,
                    )
                else:
                    results["files"][input_path] = _process_single_image(
                        input_path,
                        output_path,
                        params=processing_params,
                        application_type=application_type,
                        logger=logger,
                    )
                results["successful"] += 1
            except Exception as e:
                logger.error(
                    "Error processing %s: %s", os.path.basename(input_path), type(e).__name__
                )
                results["files"][input_path] = {"error": str(e)}
                results["failed"] += 1
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures: Dict[Any, str] = {}
            for input_path, output_path in tasks:
                if use_config:
                    future = executor.submit(
                        _process_single_image_wrapper,
                        input_path,
                        output_path,
                        config=config,
                    )
                else:
                    future = executor.submit(
                        _process_single_image_wrapper,
                        input_path,
                        output_path,
                        params=processing_params,
                        application_type=application_type,
                    )
                futures[future] = input_path

            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    result = future.result()
                    results["files"][input_path] = result
                    if isinstance(result, dict) and "error" in result:
                        results["failed"] += 1
                        logger.error(
                            "Error processing %s: %s",
                            os.path.basename(input_path),
                            result["error"],
                        )
                    else:
                        results["successful"] += 1
                        logger.info("Successfully processed: %s", os.path.basename(input_path))
                except Exception as e:
                    logger.error(
                        "Error processing %s: %s", os.path.basename(input_path), type(e).__name__
                    )
                    results["files"][input_path] = {"error": str(e)}
                    results["failed"] += 1

    end_time = time.time()
    results["processing_time"] = end_time - start_time

    logger.info(f"Batch processing completed in {results['processing_time']:.2f} seconds")
    logger.info(
        f"Successful: {results['successful']}, Failed: {results['failed']}, Skipped: {results['skipped']}"
    )

    if generate_report:
        report_path = os.path.join(safe_output, "batch_processing_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Report saved")
        except Exception as e:
            logger.error("Failed to save report: %s", type(e).__name__)

    return results


def analyze_batch(input_pattern: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Analyze multiple images to extract characteristics.

    Parameters
    ----------
    input_pattern : str
        Glob pattern to match input images.
    output_file : str, optional
        Path to save analysis results as JSON.

    Returns
    -------
    dict
        Per-image analyses plus summary statistics (min, max, and average
        for brightness, contrast, noise level, and edge density).

    Raises
    ------
    ImageProcessingError
        If no files match input_pattern or the results cannot be saved.
    """
    input_files = glob.glob(input_pattern)
    if not input_files:
        raise ImageProcessingError(f"No files found matching pattern: {input_pattern}")

    results: Dict[str, Any] = {
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

    for input_path in input_files:
        try:
            analysis = analyze_image_characteristics(input_path)
            results["analyses"][input_path] = analysis

            chars = analysis["characteristics"]
            aggregate_summary: Dict[str, Any] = results["summary"]
            if chars["is_color"]:
                aggregate_summary["color_images"] += 1

            for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
                value = chars[metric]
                metric_stats: Dict[str, Any] = aggregate_summary[metric]
                metric_stats["min"] = min(metric_stats["min"], value)
                metric_stats["max"] = max(metric_stats["max"], value)
                metric_stats["sum"] += value

        except Exception as e:
            results["analyses"][input_path] = {"error": str(e)}

    successful_analyses = len(results["analyses"]) - sum(
        1 for result in results["analyses"].values() if "error" in result
    )

    if successful_analyses > 0:
        summary_stats: Dict[str, Any] = results["summary"]
        for metric in ["brightness", "contrast", "noise_level", "edge_density"]:
            metric_averages: Dict[str, Any] = summary_stats[metric]
            metric_averages["avg"] = metric_averages["sum"] / successful_analyses

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
