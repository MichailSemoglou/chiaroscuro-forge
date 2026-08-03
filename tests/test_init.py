"""
Tests for __init__.py module exports and import strategies.

Tests cover:
- Public API exports
- Import availability
- Module metadata
"""

import unittest


class TestPublicAPI(unittest.TestCase):
    """Test that all public APIs are accessible."""

    def test_process_image_import(self):
        """Test process_image is importable."""
        from chiaroscuro_forge import process_image

        self.assertTrue(callable(process_image))

    def test_analyze_image_characteristics_import(self):
        """Test analyze_image_characteristics is importable."""
        from chiaroscuro_forge import analyze_image_characteristics

        self.assertTrue(callable(analyze_image_characteristics))

    def test_get_image_statistics_import(self):
        """Test get_image_statistics is importable."""
        from chiaroscuro_forge import get_image_statistics

        self.assertTrue(callable(get_image_statistics))

    def test_batch_process_images_import(self):
        """Test batch_process_images is importable."""
        from chiaroscuro_forge import batch_process_images

        self.assertTrue(callable(batch_process_images))

    def test_analyze_batch_import(self):
        """Test analyze_batch is importable."""
        from chiaroscuro_forge import analyze_batch

        self.assertTrue(callable(analyze_batch))

    def test_compare_processing_methods_import(self):
        """Test compare_processing_methods is importable."""
        from chiaroscuro_forge import compare_processing_methods

        self.assertTrue(callable(compare_processing_methods))

    def test_save_preset_import(self):
        """Test save_preset is importable."""
        from chiaroscuro_forge import save_preset

        self.assertTrue(callable(save_preset))

    def test_load_preset_import(self):
        """Test load_preset is importable."""
        from chiaroscuro_forge import load_preset

        self.assertTrue(callable(load_preset))

    def test_list_presets_import(self):
        """Test list_presets is importable."""
        from chiaroscuro_forge import list_presets

        self.assertTrue(callable(list_presets))

    def test_exception_import(self):
        """Test ImageProcessingError is importable."""
        from chiaroscuro_forge import ImageProcessingError

        self.assertTrue(issubclass(ImageProcessingError, Exception))


class TestModuleMetadata(unittest.TestCase):
    """Test module metadata attributes."""

    def test_version_attribute(self):
        """Test __version__ is defined."""
        import chiaroscuro_forge

        self.assertTrue(hasattr(chiaroscuro_forge, "__version__"))
        self.assertIsInstance(chiaroscuro_forge.__version__, str)
        self.assertRegex(chiaroscuro_forge.__version__, r"\d+\.\d+\.\d+")

    def test_author_attribute(self):
        """Test __author__ is defined."""
        import chiaroscuro_forge

        self.assertTrue(hasattr(chiaroscuro_forge, "__author__"))
        self.assertIsInstance(chiaroscuro_forge.__author__, str)

    def test_email_attribute(self):
        """Test __email__ is defined."""
        import chiaroscuro_forge

        self.assertTrue(hasattr(chiaroscuro_forge, "__email__"))
        self.assertIsInstance(chiaroscuro_forge.__email__, str)
        self.assertIn("@", chiaroscuro_forge.__email__)

    def test_license_attribute(self):
        """Test __license__ is defined."""
        import chiaroscuro_forge

        self.assertTrue(hasattr(chiaroscuro_forge, "__license__"))
        self.assertEqual(chiaroscuro_forge.__license__, "MIT")

    def test_all_attribute(self):
        """Test __all__ is defined with correct exports."""
        import chiaroscuro_forge

        self.assertTrue(hasattr(chiaroscuro_forge, "__all__"))

        expected_exports = [
            "process_image",
            "ProcessingConfig",
            "analyze_image_characteristics",
            "get_image_statistics",
            "batch_process_images",
            "analyze_batch",
            "compare_processing_methods",
            "save_preset",
            "load_preset",
            "list_presets",
            "ImageProcessingError",
            "get_cache_manager",
            "invalidate_preset_cache",
            "invalidate_stats_cache",
            "process_image_tiled",
            "should_use_tiling",
            "ServiceContainer",
            "get_container",
            "inject",
            "setup_default_services",
            "GPUContext",
            "gpu_available",
            "get_gpu_info",
            "get_gpu_backend",
            "GPUBackend",
            "GPUInfo",
            "benchmark_operation",
            "safe_gpu",
            "TaskQueue",
            "LocalQueue",
            "DistributedBatchProcessor",
            "TaskStatus",
            "TaskResult",
            "QueueConfig",
            "QueueHealth",
            "create_queue",
            "app",
            "api_key_manager",
            "job_manager",
            "run_server",
            "APIJobStatus",
            "JobInfo",
            "APIResponse",
            "ProcessingParams",
        ]

        self.assertIsInstance(chiaroscuro_forge.__all__, list)
        self.assertEqual(len(chiaroscuro_forge.__all__), len(set(chiaroscuro_forge.__all__)))
        self.assertEqual(set(chiaroscuro_forge.__all__), set(expected_exports))

        for export_name in expected_exports:
            self.assertEqual(
                chiaroscuro_forge.__all__.count(export_name),
                1,
                msg=f"Duplicate export in __all__: {export_name}",
            )
            self.assertTrue(
                hasattr(chiaroscuro_forge, export_name),
                msg=f"Expected export not present on chiaroscuro_forge: {export_name}",
            )


class TestImportStrategies(unittest.TestCase):
    """Test import fallback strategies."""

    def test_modular_imports_work(self):
        """Test that modular structure imports work."""
        # These should all succeed
        from chiaroscuro_forge.analysis import analyze_image_characteristics
        from chiaroscuro_forge.batch import batch_process_images
        from chiaroscuro_forge.comparison import compare_processing_methods
        from chiaroscuro_forge.exceptions import ImageProcessingError
        from chiaroscuro_forge.presets import list_presets, load_preset, save_preset
        from chiaroscuro_forge.processing import process_image

        # Verify all are callable/classes
        self.assertTrue(callable(process_image))
        self.assertTrue(callable(analyze_image_characteristics))
        self.assertTrue(callable(batch_process_images))
        self.assertTrue(callable(compare_processing_methods))
        self.assertTrue(callable(save_preset))
        self.assertTrue(callable(load_preset))
        self.assertTrue(callable(list_presets))
        self.assertTrue(issubclass(ImageProcessingError, Exception))


if __name__ == "__main__":
    unittest.main()
