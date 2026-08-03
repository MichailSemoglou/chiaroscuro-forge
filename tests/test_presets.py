"""
Comprehensive tests for presets module.

Tests cover:
- Loading presets from disk
- Saving presets with parameters
- Listing available presets
- Error handling for missing/invalid presets
- Directory creation and cleanup
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from chiaroscuro_forge.config import ProcessingConfig
from chiaroscuro_forge.exceptions import ImageProcessingError
from chiaroscuro_forge.presets import list_presets, load_preset, save_preset


class TestLoadPreset(unittest.TestCase):
    """Test loading presets from disk."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_dir = os.path.join(self.temp_dir, "presets")
        os.makedirs(self.preset_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_valid_preset(self):
        """Test loading a valid preset file through the real config path."""
        preset_data = {
            "name": "test_preset",
            "description": "Test preset",
            "params": {
                "equalize_method": "clahe",
                "sharpen_amount": 1.5,
                "denoise_type": "gaussian",
                "contrast_stretch_percentiles": [5, 95],
            },
        }

        preset_path = os.path.join(self.preset_dir, "test_preset.json")
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            params = load_preset("test_preset")

        config = ProcessingConfig.from_dict(params)
        merged = config.merge({"sharpen": False})

        self.assertEqual(params["equalize_method"], "clahe")
        self.assertEqual(params["sharpen_amount"], 1.5)
        self.assertEqual(params["denoise_type"], "gaussian")
        self.assertEqual(params["contrast_stretch_percentiles"], [5, 95])
        self.assertEqual(merged.contrast_stretch_percentiles, (5, 95))
        self.assertEqual(merged.to_context()["contrast_stretch_percentiles"], (5, 95))
        self.assertFalse(merged.sharpen)

    def test_load_preset_not_found(self):
        """Test loading a non-existent preset."""
        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                load_preset("nonexistent")

        self.assertIn("Preset not found", str(ctx.exception))

    def test_load_preset_rejects_absolute_path(self):
        """Absolute preset paths should be rejected before lookup."""
        absolute_path = os.path.join(self.temp_dir, "evil")
        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                load_preset(absolute_path)

        self.assertIn("Invalid preset name", str(ctx.exception))

    def test_load_preset_rejects_traversal(self):
        """Path traversal preset names should be rejected before lookup."""
        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                load_preset("../evil")

        self.assertIn("Invalid preset name", str(ctx.exception))

    def test_load_preset_invalid_json(self):
        """Test loading a preset with invalid JSON."""
        preset_path = os.path.join(self.preset_dir, "invalid.json")
        with open(preset_path, "w") as f:
            f.write("{ invalid json }")

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                load_preset("invalid")

        self.assertIn("Failed to parse preset file", str(ctx.exception))

    def test_load_preset_missing_name_field(self):
        """Test loading a preset without 'name' field."""
        preset_data = {"params": {"equalize_method": "clahe"}}

        preset_path = os.path.join(self.preset_dir, "missing_name.json")
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                load_preset("missing_name")

        self.assertIn("Invalid preset format", str(ctx.exception))
        self.assertIn("missing 'name' or 'params' fields", str(ctx.exception))

    def test_load_preset_missing_params_field(self):
        """Test loading a preset without 'params' field."""
        preset_data = {"name": "test", "description": "Test"}

        preset_path = os.path.join(self.preset_dir, "missing_params.json")
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                load_preset("missing_params")

        self.assertIn("Invalid preset format", str(ctx.exception))

    def test_load_preset_general_error(self):
        """Test error handling during preset loading."""
        # Create file first
        preset_path = os.path.join(self.preset_dir, "error.json")
        with open(preset_path, "w") as f:
            json.dump({"name": "test", "params": {}}, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                with self.assertRaises(ImageProcessingError) as ctx:
                    load_preset("error")

        self.assertIn("Error loading preset", str(ctx.exception))


class TestSavePreset(unittest.TestCase):
    """Test saving presets to disk."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_dir = os.path.join(self.temp_dir, "presets")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_preset_basic(self):
        """Test saving a basic preset."""
        params = {"equalize_method": "clahe", "sharpen_amount": 1.8}

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            save_preset("test_save", params)

        # Verify file was created
        preset_path = os.path.join(self.preset_dir, "test_save.json")
        self.assertTrue(os.path.exists(preset_path))

        # Verify content
        with open(preset_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["name"], "test_save")
        self.assertEqual(saved_data["params"], params)

    def test_save_preset_with_description(self):
        """Test saving a preset with description."""
        params = {"denoise_type": "gaussian"}
        description = "Test description"

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            save_preset("described", params, description)

        preset_path = os.path.join(self.preset_dir, "described.json")
        with open(preset_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["description"], description)

    def test_save_preset_rejects_absolute_path(self):
        """Absolute preset paths should be rejected before writing."""
        params = {"test": "value"}

        absolute_path = os.path.join(self.temp_dir, "evil")
        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                save_preset(absolute_path, params)

        self.assertIn("Invalid preset name", str(ctx.exception))

    def test_save_preset_rejects_traversal(self):
        """Traversal preset names should be rejected before writing."""
        params = {"test": "value"}

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with self.assertRaises(ImageProcessingError) as ctx:
                save_preset("../evil", params)

        self.assertIn("Invalid preset name", str(ctx.exception))

    def test_save_preset_creates_directory(self):
        """Test that save_preset creates directory if needed."""
        # Don't create preset_dir beforehand
        params = {"test": "value"}

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            save_preset("new_preset", params)

        # Directory should now exist
        self.assertTrue(os.path.exists(self.preset_dir))

    def test_save_preset_directory_creation_fails(self):
        """Test error handling when directory creation fails."""
        params = {"test": "value"}

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with patch("os.makedirs", side_effect=PermissionError("Access denied")):
                with self.assertRaises(ImageProcessingError) as ctx:
                    save_preset("fail_mkdir", params)

        self.assertIn("Failed to create presets directory", str(ctx.exception))

    def test_save_preset_file_write_fails(self):
        """Test error handling when file write fails."""
        params = {"test": "value"}

        # Create directory first
        os.makedirs(self.preset_dir)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                with self.assertRaises(ImageProcessingError) as ctx:
                    save_preset("fail_write", params)

        self.assertIn("Failed to save preset", str(ctx.exception))

    def test_save_preset_overwrites_existing(self):
        """Test that saving overwrites existing preset."""
        os.makedirs(self.preset_dir)

        # Save initial preset
        initial_params = {"value": 1}
        preset_path = os.path.join(self.preset_dir, "overwrite.json")
        preset_data = {"name": "overwrite", "description": "", "params": initial_params}
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        # Overwrite with new params
        new_params = {"value": 2}
        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            save_preset("overwrite", new_params, "Updated")

        # Verify overwrite
        with open(preset_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["params"]["value"], 2)
        self.assertEqual(saved_data["description"], "Updated")


class TestListPresets(unittest.TestCase):
    """Test listing available presets."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_dir = os.path.join(self.temp_dir, "presets")
        os.makedirs(self.preset_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_list_empty_directory(self):
        """Test listing presets when directory is empty."""
        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 0)

    def test_list_presets_directory_not_exists(self):
        """Test listing presets when directory doesn't exist."""
        # Remove preset directory
        os.rmdir(self.preset_dir)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 0)

    def test_list_single_preset(self):
        """Test listing a single preset."""
        preset_data = {
            "name": "preset1",
            "description": "First preset",
            "params": {"test": "value"},
        }

        preset_path = os.path.join(self.preset_dir, "preset1.json")
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "preset1")
        self.assertEqual(presets[0]["description"], "First preset")
        self.assertEqual(presets[0]["filename"], "preset1.json")

    def test_list_multiple_presets(self):
        """Test listing multiple presets."""
        for i in range(3):
            preset_data = {
                "name": f"preset{i}",
                "description": f"Preset {i}",
                "params": {"index": i},
            }
            preset_path = os.path.join(self.preset_dir, f"preset{i}.json")
            with open(preset_path, "w") as f:
                json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 3)
        names = [p["name"] for p in presets]
        self.assertIn("preset0", names)
        self.assertIn("preset1", names)
        self.assertIn("preset2", names)

    def test_list_presets_ignores_non_json(self):
        """Test that non-JSON files are ignored."""
        # Create JSON preset
        preset_data = {"name": "valid", "description": "", "params": {}}
        with open(os.path.join(self.preset_dir, "valid.json"), "w") as f:
            json.dump(preset_data, f)

        # Create non-JSON files
        with open(os.path.join(self.preset_dir, "readme.txt"), "w") as f:
            f.write("Not a preset")
        with open(os.path.join(self.preset_dir, "config.yaml"), "w") as f:
            f.write("also: not a preset")

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "valid")

    def test_list_presets_handles_corrupt_files(self):
        """Test that corrupt JSON files are skipped."""
        # Create valid preset
        preset_data = {"name": "valid", "description": "", "params": {}}
        with open(os.path.join(self.preset_dir, "valid.json"), "w") as f:
            json.dump(preset_data, f)

        # Create corrupt JSON
        with open(os.path.join(self.preset_dir, "corrupt.json"), "w") as f:
            f.write("{ invalid json }")

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        # Should only return valid preset
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "valid")

    def test_list_presets_missing_name_uses_filename(self):
        """Test that missing 'name' field uses filename."""
        preset_data = {"description": "No name field", "params": {}}

        preset_path = os.path.join(self.preset_dir, "noname.json")
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 1)
        # Should use filename without .json extension
        self.assertEqual(presets[0]["name"], "noname")

    def test_list_presets_missing_description(self):
        """Test that missing 'description' field defaults to empty string."""
        preset_data = {"name": "nodesc", "params": {}}

        preset_path = os.path.join(self.preset_dir, "nodesc.json")
        with open(preset_path, "w") as f:
            json.dump(preset_data, f)

        with patch("chiaroscuro_forge.presets.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            presets = list_presets()

        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["description"], "")


if __name__ == "__main__":
    unittest.main()
