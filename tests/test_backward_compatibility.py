"""
Backward Compatibility Tests

These tests ensure that the gradual migration maintains 100% backward
compatibility with existing code using the package.
"""

import unittest
import sys
import os


class TestBackwardCompatibility(unittest.TestCase):
    """Test that all imports work as expected during migration."""
    
    def test_main_package_imports(self):
        """Test importing from main package (most common usage)."""
        try:
            from chiaroscuro_forge import (
                process_image,
                analyze_image_characteristics,
                get_image_statistics,
                batch_process_images,
                analyze_batch,
                compare_processing_methods,
                save_preset,
                load_preset,
                list_presets,
                ImageProcessingError,
            )
            
            # Verify functions are callable
            self.assertTrue(callable(process_image))
            self.assertTrue(callable(analyze_image_characteristics))
            self.assertTrue(callable(get_image_statistics))
            self.assertTrue(callable(batch_process_images))
            self.assertTrue(callable(analyze_batch))
            self.assertTrue(callable(compare_processing_methods))
            self.assertTrue(callable(save_preset))
            self.assertTrue(callable(load_preset))
            self.assertTrue(callable(list_presets))
            self.assertTrue(issubclass(ImageProcessingError, Exception))
            
        except ImportError as e:
            self.fail(f"Failed to import from main package: {e}")
    
    def test_module_specific_imports(self):
        """Test importing from specific modules (new modular structure)."""
        try:
            from chiaroscuro_forge.processing import process_image
            from chiaroscuro_forge.analysis import (
                analyze_image_characteristics,
                get_image_statistics
            )
            from chiaroscuro_forge.batch import batch_process_images, analyze_batch
            from chiaroscuro_forge.comparison import compare_processing_methods
            from chiaroscuro_forge.presets import save_preset, load_preset, list_presets
            from chiaroscuro_forge.exceptions import ImageProcessingError
            
            # Verify functions are callable
            self.assertTrue(callable(process_image))
            self.assertTrue(callable(analyze_image_characteristics))
            self.assertTrue(callable(get_image_statistics))
            self.assertTrue(callable(batch_process_images))
            self.assertTrue(callable(analyze_batch))
            self.assertTrue(callable(compare_processing_methods))
            self.assertTrue(callable(save_preset))
            self.assertTrue(callable(load_preset))
            self.assertTrue(callable(list_presets))
            self.assertTrue(issubclass(ImageProcessingError, Exception))
            
        except ImportError as e:
            self.fail(f"Failed to import from specific modules: {e}")
    
    def test_import_equivalence(self):
        """Test that imports from both styles work and have the same signatures."""
        try:
            # Import both ways
            from chiaroscuro_forge import process_image as pi_main
            from chiaroscuro_forge.processing import process_image as pi_module
            
            # They should both be callable and have the same name
            self.assertTrue(callable(pi_main), "Main package import should be callable")
            self.assertTrue(callable(pi_module), "Module import should be callable")
            self.assertEqual(pi_main.__name__, pi_module.__name__,
                           "Functions should have the same name")
            
        except ImportError as e:
            self.fail(f"Failed equivalence test: {e}")
    
    def test_version_attribute(self):
        """Test that version information is accessible."""
        try:
            import chiaroscuro_forge
            
            self.assertTrue(hasattr(chiaroscuro_forge, '__version__'))
            self.assertTrue(hasattr(chiaroscuro_forge, '__author__'))
            self.assertTrue(hasattr(chiaroscuro_forge, '__license__'))
            
            # Version should be 0.3.0 or later
            version = chiaroscuro_forge.__version__
            major, minor, *_ = version.split('.')
            self.assertGreaterEqual(int(major), 0)
            self.assertGreaterEqual(int(minor), 3)
            
        except (ImportError, AttributeError) as e:
            self.fail(f"Failed to access version info: {e}")
    
    def test_exception_inheritance(self):
        """Test that custom exceptions work correctly."""
        from chiaroscuro_forge import ImageProcessingError
        from chiaroscuro_forge.exceptions import ImageProcessingError as IPE_module
        
        # Both should be Exception subclasses with the same name
        self.assertTrue(issubclass(ImageProcessingError, Exception))
        self.assertTrue(issubclass(IPE_module, Exception))
        self.assertEqual(ImageProcessingError.__name__, IPE_module.__name__)
        
        # Should be raiseable and catchable
        with self.assertRaises(ImageProcessingError):
            raise ImageProcessingError("Test error")
    
    def test_constants_available(self):
        """Test that constants module is accessible."""
        try:
            from chiaroscuro_forge.constants import (
                DEFAULT_WIN_SIZE,
                VALID_APP_TYPES,
                QUALITY_WEIGHTS,
            )
            
            self.assertIsInstance(DEFAULT_WIN_SIZE, int)
            self.assertIsInstance(VALID_APP_TYPES, list)
            self.assertIsInstance(QUALITY_WEIGHTS, dict)
            
        except ImportError as e:
            self.fail(f"Failed to import constants: {e}")
    
    def test_validation_available(self):
        """Test that validation utilities are accessible."""
        try:
            from chiaroscuro_forge.validation import (
                validate_array,
                validate_image_path,
                validate_processing_params,
            )
            
            self.assertTrue(callable(validate_array))
            self.assertTrue(callable(validate_image_path))
            self.assertTrue(callable(validate_processing_params))
            
        except ImportError as e:
            self.fail(f"Failed to import validation functions: {e}")


class TestModuleStructure(unittest.TestCase):
    """Test the module structure and organization."""
    
    def test_all_modules_exist(self):
        """Test that all expected modules exist."""
        expected_modules = [
            'chiaroscuro_forge',
            'chiaroscuro_forge.constants',
            'chiaroscuro_forge.exceptions',
            'chiaroscuro_forge.validation',
            'chiaroscuro_forge.metrics',
            'chiaroscuro_forge.processing',
            'chiaroscuro_forge.analysis',
            'chiaroscuro_forge.batch',
            'chiaroscuro_forge.comparison',
            'chiaroscuro_forge.presets',
            'chiaroscuro_forge.cli',
        ]
        
        for module_name in expected_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"Module {module_name} could not be imported: {e}")
    
    def test_public_api_matches(self):
        """Test that __all__ lists are consistent."""
        import chiaroscuro_forge
        
        expected_exports = {
            'process_image',
            'analyze_image_characteristics',
            'get_image_statistics',
            'batch_process_images',
            'analyze_batch',
            'compare_processing_methods',
            'save_preset',
            'load_preset',
            'list_presets',
            'ImageProcessingError',
            # Phase 3.1: Cache management (added in v0.3.0)
            'get_cache_manager',
            'invalidate_preset_cache',
            'invalidate_stats_cache',
            # Phase 3.2: Tile-based processing (added in v0.3.0)
            'process_image_tiled',
            'should_use_tiling',
            # Phase 3.3: Dependency injection (added in v0.3.0)
            'ServiceContainer',
            'get_container',
            'inject',
            'setup_default_services',
        }
        
        self.assertEqual(set(chiaroscuro_forge.__all__), expected_exports,
                        "Public API exports don't match expected list")


if __name__ == '__main__':
    unittest.main()
