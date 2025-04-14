import os
import unittest
import numpy as np
from skimage import data
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main script as a module
import chiaroscuro_forge


class TestChiaroscuroForge(unittest.TestCase):
    def setUp(self):
        # Create a test image
        self.test_image = data.astronaut()
        self.test_image_path = os.path.join(os.path.dirname(__file__), "test_image.png")
        self.output_path = os.path.join(os.path.dirname(__file__), "test_output.png")
        
        # Save test image
        from skimage.io import imsave
        imsave(self.test_image_path, self.test_image)
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
    
    def test_image_analysis(self):
        # Test image characteristic analysis
        analysis = chiaroscuro_forge.analyze_image_characteristics(self.test_image_path)
        self.assertIsInstance(analysis, dict)
        self.assertIn('characteristics', analysis)
        self.assertIn('suggested_params', analysis)
        self.assertIn('suggested_application', analysis)
        
        # Check specific characteristics
        chars = analysis['characteristics']
        self.assertIn('is_color', chars)
        self.assertIn('brightness', chars)
        self.assertIn('contrast', chars)
        self.assertIn('noise_level', chars)
        self.assertIn('edge_density', chars)
        
        # Check suggested parameters
        params = analysis['suggested_params']
        self.assertIn('denoise_type', params)
        self.assertIn('denoise_sigma', params)
        self.assertIn('equalize_method', params)
    
    def test_image_processing(self):
        # Test basic image processing
        processed_image, metrics = chiaroscuro_forge.process_image(
            self.test_image_path,
            output_path=self.output_path
        )
        
        # Check results
        self.assertIsInstance(processed_image, np.ndarray)
        self.assertIsInstance(metrics, dict)
        self.assertIn('ssim', metrics)
        self.assertIn('psnr', metrics)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.output_path))


if __name__ == '__main__':
    unittest.main()
