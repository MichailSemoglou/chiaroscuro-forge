"""
Command-line interface for chiaroscuro-forge.

This module provides the main CLI entry point with comprehensive argument parsing
for single image processing, batch processing, analysis, and preset management.
"""

import argparse
import os

from .analysis import analyze_image_characteristics
from .batch import analyze_batch, batch_process_images
from .comparison import compare_processing_methods, suggest_optimal_params
from .config import ProcessingConfig
from .exceptions import ImageProcessingError
from .presets import list_presets, load_preset, save_preset
from .processing import process_image


def main():
    """Main CLI entry point for chiaroscuro-forge."""
    parser = argparse.ArgumentParser(description="Enhanced Image Processing Tool")

    # Input/output arguments
    input_group = parser.add_argument_group("Input/Output")
    input_group.add_argument(
        "image_path",
        nargs="?",
        help="Path to the input image or glob pattern for batch processing",
    )
    input_group.add_argument(
        "--output",
        "-o",
        help="Path for the output image or directory for batch processing",
    )
    input_group.add_argument(
        "--batch", "-b", action="store_true", help="Enable batch processing mode"
    )

    # Processing parameters
    process_group = parser.add_argument_group("Processing Parameters")
    process_group.add_argument(
        "--application",
        "-a",
        choices=["general", "photography", "medical", "document", "art"],
        default="general",
        help="Application type for optimization",
    )
    process_group.add_argument("--preset", help="Name of a preset to use")

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--analyze", action="store_true", help="Analyze image and suggest parameters"
    )
    analysis_group.add_argument(
        "--analyze-batch",
        action="store_true",
        help="Analyze multiple images and suggest optimal parameters",
    )
    analysis_group.add_argument(
        "--compare", action="store_true", help="Compare different processing methods"
    )
    analysis_group.add_argument(
        "--compare-dir",
        default="comparison",
        help="Output directory for comparison results",
    )

    # Preset management
    preset_group = parser.add_argument_group("Preset Management")
    preset_group.add_argument(
        "--save-preset",
        help="Save the current parameters as a preset with the given name",
    )
    preset_group.add_argument(
        "--list-presets", action="store_true", help="List all available presets"
    )
    preset_group.add_argument(
        "--preset-description",
        help="Description for the preset when using --save-preset",
    )

    # Batch processing options
    batch_group = parser.add_argument_group("Batch Processing Options")
    batch_group.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers for batch processing",
    )
    batch_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output file already exists",
    )
    batch_group.add_argument(
        "--report",
        action="store_true",
        help="Generate a JSON report with batch processing results",
    )
    batch_group.add_argument("--log-file", help="Path to log file for batch processing")

    args = parser.parse_args()

    try:
        # List presets mode
        if args.list_presets:
            presets = list_presets()
            if presets:
                print("\nAvailable presets:")
                for preset in presets:
                    print(f"- {preset['name']}: {preset['description']}")
            else:
                print("\nNo presets found.")
            return 0

        # Require image_path for all other operations
        if not args.image_path and not args.list_presets:
            parser.print_help()
            return 1

        app_type = args.application
        config = ProcessingConfig(application_type=app_type)

        # Load preset if specified
        if args.preset:
            try:
                preset_params = load_preset(args.preset)
                config = config.merge(preset_params)
                print(f"Loaded preset: {args.preset}")
            except ImageProcessingError as e:
                print(f"Error: {e}")
                return 1

        # Batch analysis mode
        if args.analyze_batch:
            if not args.output:
                output_file = "batch_analysis.json"
            else:
                output_file = args.output

            print(f"Analyzing images matching: {args.image_path}")
            results = analyze_batch(args.image_path, output_file)

            suggestions = suggest_optimal_params(results)

            print(f"\nAnalyzed {results['total_images']} images")
            print("\nBatch Summary:")
            for metric, values in results["summary"].items():
                if metric == "color_images":
                    print(f"Color Images: {values}/{results['total_images']}")
                elif "avg" in values:
                    print(
                        f"{metric.capitalize()}: Avg={values['avg']:.4f}, Min={values['min']:.4f}, Max={values['max']:.4f}"
                    )

            print("\nSuggested Processing Parameters:")
            for param, value in suggestions["params"].items():
                print(f"{param}: {value}")

            print(f"\nSuggested Application Type: {suggestions['application_type']}")
            print(f"\nDetailed analysis saved to: {output_file}")
            return 0

        # Single image analysis mode
        if args.analyze and not args.batch:
            analysis = analyze_image_characteristics(args.image_path)

            print("\nImage Characteristics:")
            print(f"Color Image: {'Yes' if analysis['characteristics']['is_color'] else 'No'}")
            print(f"Brightness: {analysis['characteristics']['brightness']:.2f}")
            print(f"Contrast: {analysis['characteristics']['contrast']:.2f}")
            print(f"Noise Level: {analysis['characteristics']['noise_level']:.4f}")
            print(f"Edge Density: {analysis['characteristics']['edge_density']:.4f}")

            print("\nSuggested Processing Parameters:")
            for param, value in analysis["suggested_params"].items():
                print(f"{param}: {value}")

            print(f"\nSuggested Application Type: {analysis['suggested_application']}")

            config = config.merge(analysis["suggested_params"])

            # Use suggested application type if still using default
            if args.application == "general":
                app_type = analysis["suggested_application"]
                config.application_type = app_type

        # Comparison mode (single image only)
        if args.compare and not args.batch:
            results = compare_processing_methods(
                args.image_path, output_dir=args.compare_dir, application_type=app_type
            )

            print("\nComparison Results:")
            for name, result in results.items():
                if name == "best_method":
                    continue

                if "metrics" in result and result["metrics"]:
                    metrics = result["metrics"]
                    print(f"\n{name}:")
                    print(f"  SSIM: {metrics['ssim']:.4f}")
                    print(f"  PSNR: {metrics['psnr']:.2f} dB")
                    print(f"  Quality Score: {metrics['quality_score']:.4f}")
                elif "error" in result:
                    print(f"\n{name}: Error - {result['error']}")

            if "best_method" in results:
                print(
                    f"\nBest method: {results['best_method']['name']} "
                    f"(Score: {results['best_method']['score']:.4f})"
                )

                # Print comparison results saved
                if not args.output:
                    best_name = results["best_method"]["name"]
                    if "output_path" in results[best_name]:
                        print(f"Best result saved to: {results[best_name]['output_path']}")
            return 0

        # Batch processing mode
        if args.batch:
            if not args.output:
                print("Error: Output directory must be specified for batch processing")
                return 1

            print(f"Batch processing images matching: {args.image_path}")
            print(f"Output directory: {args.output}")
            print(f"Using {args.workers} workers")

            results = batch_process_images(
                args.image_path,
                args.output,
                config=config,
                n_workers=args.workers,
                skip_existing=args.skip_existing,
                generate_report=args.report,
                log_file=args.log_file,
            )

            print(f"\nBatch processing completed in {results['processing_time']:.2f} seconds")
            print(f"Total images: {results['total']}")
            print(f"Processed successfully: {results['successful']}")
            print(f"Failed: {results['failed']}")
            print(f"Skipped: {results['skipped']}")

            if args.report:
                report_path = os.path.join(args.output, "batch_processing_report.json")
                print(f"\nDetailed report saved to: {report_path}")

            return 0

        # Single image processing mode
        if args.output and not args.batch:
            print(f"\nProcessing image with {app_type} application type...")
            processed, metrics = process_image(
                args.image_path,
                output_path=args.output,
                config=config,
            )

            if metrics:
                print("\nQuality Metrics:")
                print(f"SSIM: {metrics['ssim']:.4f}")
                print(f"PSNR: {metrics['psnr']:.2f} dB")
                print(f"Quality Score: {metrics['quality_score']:.4f}")

            print(f"Processed image saved to: {args.output}")

        # Save preset if requested
        if args.save_preset:
            try:
                save_preset(
                    args.save_preset, config.to_dict(), description=args.preset_description or ""
                )
                print(f"\nSaved preset '{args.save_preset}'")
            except ImageProcessingError as e:
                print(f"Error saving preset: {e}")
                return 1

    except ImageProcessingError as e:
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

    return 0


__all__ = ["main"]
