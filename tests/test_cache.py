"""Tests for the caching module."""

import os
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from PIL import Image

from chiaroscuro_forge.cache import (
    CacheManager,
    get_cache_manager,
    compute_file_hash,
    get_file_cache_key,
    cached_image_stats,
    cached_preset,
    invalidate_preset_cache,
    invalidate_stats_cache,
)
from chiaroscuro_forge.exceptions import ImageProcessingError


class TestCacheManager(unittest.TestCase):
    """Tests for CacheManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CacheManager(max_stats_cache=5, max_preset_cache=3)

    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        self.assertEqual(self.cache.max_stats_cache, 5)
        self.assertEqual(self.cache.max_preset_cache, 3)
        self.assertEqual(len(self.cache._stats_cache), 0)
        self.assertEqual(len(self.cache._preset_cache), 0)

    def test_get_stats_empty(self):
        """Test getting stats from empty cache."""
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["stats_cached"], 0)
        self.assertEqual(stats["presets_cached"], 0)

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        # Add some data
        self.cache._stats_cache["key1"] = (time.time(), {"data": 1})
        self.cache._preset_cache["preset1"] = (time.time(), {"param": 1})
        self.cache._cache_hits = 5
        self.cache._cache_misses = 3

        self.cache.clear()

        self.assertEqual(len(self.cache._stats_cache), 0)
        self.assertEqual(len(self.cache._preset_cache), 0)
        self.assertEqual(self.cache._cache_hits, 0)
        self.assertEqual(self.cache._cache_misses, 0)

    def test_clear_stats_cache_only(self):
        """Test clearing stats cache while preserving preset cache."""
        self.cache._stats_cache["key1"] = (time.time(), {"data": 1})
        self.cache._preset_cache["preset1"] = (time.time(), {"param": 1})

        self.cache.clear_stats_cache()

        self.assertEqual(len(self.cache._stats_cache), 0)
        self.assertEqual(len(self.cache._preset_cache), 1)

    def test_clear_preset_cache_only(self):
        """Test clearing preset cache while preserving stats cache."""
        self.cache._stats_cache["key1"] = (time.time(), {"data": 1})
        self.cache._preset_cache["preset1"] = (time.time(), {"param": 1})

        self.cache.clear_preset_cache()

        self.assertEqual(len(self.cache._stats_cache), 1)
        self.assertEqual(len(self.cache._preset_cache), 0)


class TestGlobalCacheManager(unittest.TestCase):
    """Tests for global cache manager."""

    def test_get_global_cache_manager(self):
        """Test getting global cache manager."""
        cache1 = get_cache_manager()
        cache2 = get_cache_manager()

        # Should return the same instance
        self.assertIs(cache1, cache2)

    def tearDown(self):
        """Clean up after each test."""
        get_cache_manager().clear()


class TestFileHashing(unittest.TestCase):
    """Tests for file hashing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("Test content for hashing")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_compute_file_hash(self):
        """Test computing file hash."""
        hash1 = compute_file_hash(self.test_file)
        hash2 = compute_file_hash(self.test_file)

        # Same file should have same hash
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA256 hash length

    def test_compute_file_hash_different_files(self):
        """Test that different files have different hashes."""
        file2 = os.path.join(self.temp_dir.name, "test2.txt")
        with open(file2, "w") as f:
            f.write("Different content")

        hash1 = compute_file_hash(self.test_file)
        hash2 = compute_file_hash(file2)

        self.assertNotEqual(hash1, hash2)

    def test_compute_file_hash_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with self.assertRaises(ImageProcessingError):
            compute_file_hash("nonexistent_file.txt")

    def test_get_file_cache_key(self):
        """Test generating cache key from file."""
        key1 = get_file_cache_key(self.test_file)
        key2 = get_file_cache_key(self.test_file)

        # Same file without modifications should have same key
        self.assertEqual(key1, key2)
        self.assertIn(self.test_file, key1)

    def test_get_file_cache_key_after_modification(self):
        """Test cache key changes after file modification."""
        key1 = get_file_cache_key(self.test_file)

        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        with open(self.test_file, "a") as f:
            f.write(" Modified")

        key2 = get_file_cache_key(self.test_file)

        # Key should change after modification
        self.assertNotEqual(key1, key2)

    def test_get_file_cache_key_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with self.assertRaises(ImageProcessingError):
            get_file_cache_key("nonexistent_file.txt")


class TestCachedImageStats(unittest.TestCase):
    """Tests for @cached_image_stats decorator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_image = os.path.join(self.temp_dir.name, "test.jpg")

        # Create test image
        image = Image.new("RGB", (100, 100), color="white")
        image.save(self.test_image)

        # Clear cache
        get_cache_manager().clear()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        get_cache_manager().clear()

    def test_cached_function_basic(self):
        """Test basic caching functionality."""
        call_count = [0]

        @cached_image_stats(ttl=60)
        def analyze(image_path):
            call_count[0] += 1
            return {"brightness": 0.5}

        # First call - cache miss
        result1 = analyze(self.test_image)
        self.assertEqual(call_count[0], 1)
        self.assertEqual(result1["brightness"], 0.5)

        # Second call - cache hit
        result2 = analyze(self.test_image)
        self.assertEqual(call_count[0], 1)  # Function not called again
        self.assertEqual(result2["brightness"], 0.5)

        # Verify cache stats
        stats = get_cache_manager().get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_cached_function_ttl_expiration(self):
        """Test TTL expiration."""
        call_count = [0]

        @cached_image_stats(ttl=0.1)  # 100ms TTL
        def analyze(image_path):
            call_count[0] += 1
            return {"brightness": 0.5}

        # First call
        result1 = analyze(self.test_image)
        self.assertEqual(call_count[0], 1)

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should recompute
        result2 = analyze(self.test_image)
        self.assertEqual(call_count[0], 2)

    def test_cached_function_no_ttl(self):
        """Test caching without TTL."""
        call_count = [0]

        @cached_image_stats(ttl=None)
        def analyze(image_path):
            call_count[0] += 1
            return {"brightness": 0.5}

        # Multiple calls
        analyze(self.test_image)
        analyze(self.test_image)
        analyze(self.test_image)

        # Function should only be called once
        self.assertEqual(call_count[0], 1)

    def test_cached_function_different_files(self):
        """Test caching with different files."""
        image2 = os.path.join(self.temp_dir.name, "test2.jpg")
        Image.new("RGB", (100, 100), color="black").save(image2)

        call_count = [0]

        @cached_image_stats(ttl=60)
        def analyze(image_path):
            call_count[0] += 1
            return {"path": image_path}

        result1 = analyze(self.test_image)
        result2 = analyze(image2)

        # Both should be cache misses
        self.assertEqual(call_count[0], 2)
        self.assertNotEqual(result1["path"], result2["path"])

    def test_cached_function_cache_key_error(self):
        """Test graceful handling of cache key generation errors."""
        call_count = [0]

        @cached_image_stats(ttl=60)
        def analyze(image_path):
            call_count[0] += 1
            return {"result": "ok"}

        # Call with nonexistent file - should just compute without caching
        result = analyze("nonexistent.jpg")
        self.assertEqual(result["result"], "ok")
        self.assertEqual(call_count[0], 1)


class TestCachedPreset(unittest.TestCase):
    """Tests for @cached_preset decorator."""

    def setUp(self):
        """Set up test fixtures."""
        get_cache_manager().clear()

    def tearDown(self):
        """Clean up test fixtures."""
        get_cache_manager().clear()

    def test_cached_preset_basic(self):
        """Test basic preset caching."""
        call_count = [0]

        @cached_preset()
        def load(preset_name):
            call_count[0] += 1
            return {"param": f"value_{preset_name}"}

        # First call - cache miss
        result1 = load("test_preset")
        self.assertEqual(call_count[0], 1)
        self.assertEqual(result1["param"], "value_test_preset")

        # Second call - cache hit
        result2 = load("test_preset")
        self.assertEqual(call_count[0], 1)  # Not called again

        # Verify cache stats
        stats = get_cache_manager().get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_cached_preset_returns_copy(self):
        """Test that cached presets return copies to prevent mutations."""

        @cached_preset()
        def load(preset_name):
            return {"param": 1}

        result1 = load("test")
        result1["param"] = 999  # Mutate result

        result2 = load("test")  # Get from cache
        self.assertEqual(result2["param"], 1)  # Should be original value

    def test_cached_preset_different_presets(self):
        """Test caching different presets."""
        call_count = [0]

        @cached_preset()
        def load(preset_name):
            call_count[0] += 1
            return {"name": preset_name}

        load("preset1")
        load("preset2")
        load("preset1")  # Cache hit

        self.assertEqual(call_count[0], 2)  # Only 2 unique presets loaded


class TestCacheInvalidation(unittest.TestCase):
    """Tests for cache invalidation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_image = os.path.join(self.temp_dir.name, "test.jpg")
        Image.new("RGB", (100, 100), color="white").save(self.test_image)
        get_cache_manager().clear()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        get_cache_manager().clear()

    def test_invalidate_preset_cache_specific(self):
        """Test invalidating specific preset."""
        cache = get_cache_manager()
        cache._preset_cache["preset1"] = (time.time(), {"param": 1})
        cache._preset_cache["preset2"] = (time.time(), {"param": 2})

        invalidate_preset_cache("preset1")

        self.assertNotIn("preset1", cache._preset_cache)
        self.assertIn("preset2", cache._preset_cache)

    def test_invalidate_preset_cache_all(self):
        """Test invalidating all presets."""
        cache = get_cache_manager()
        cache._preset_cache["preset1"] = (time.time(), {"param": 1})
        cache._preset_cache["preset2"] = (time.time(), {"param": 2})

        invalidate_preset_cache()

        self.assertEqual(len(cache._preset_cache), 0)

    def test_invalidate_stats_cache_specific(self):
        """Test invalidating specific image stats."""
        cache = get_cache_manager()
        key = get_file_cache_key(self.test_image)
        cache._stats_cache[key] = (time.time(), {"stats": 1})

        invalidate_stats_cache(self.test_image)

        self.assertNotIn(key, cache._stats_cache)

    def test_invalidate_stats_cache_all(self):
        """Test invalidating all image stats."""
        cache = get_cache_manager()
        cache._stats_cache["key1"] = (time.time(), {"stats": 1})
        cache._stats_cache["key2"] = (time.time(), {"stats": 2})

        invalidate_stats_cache()

        self.assertEqual(len(cache._stats_cache), 0)

    def test_invalidate_stats_cache_nonexistent_file(self):
        """Test invalidating cache for nonexistent file."""
        # Should not raise error
        invalidate_stats_cache("nonexistent.jpg")

    def test_cached_image_stats_propagates_unrelated_cache_key_errors(self):
        """Unrelated cache-key failures should bubble up instead of being swallowed."""
        cache = CacheManager()
        call_count = [0]

        @cached_image_stats(ttl=None)
        def analyze(image_path):
            call_count[0] += 1
            return {"result": image_path}

        with patch("chiaroscuro_forge.cache.get_cache_manager", return_value=cache):
            with patch(
                "chiaroscuro_forge.cache.get_file_cache_key", side_effect=RuntimeError("boom")
            ):
                with self.assertRaises(RuntimeError):
                    analyze("file1")

        self.assertEqual(call_count[0], 0)


class TestCacheEviction(unittest.TestCase):
    """Tests for cache eviction policies."""

    def setUp(self):
        """Set up test fixtures."""
        get_cache_manager().clear()

    def tearDown(self):
        """Clean up test fixtures."""
        get_cache_manager().clear()

    def test_stats_cache_eviction(self):
        """Test that stats cache evicts oldest entries when full."""
        cache = CacheManager(max_stats_cache=3)

        # Mock function
        call_count = [0]

        @cached_image_stats(ttl=None)
        def analyze(image_path):
            call_count[0] += 1
            return {"result": image_path}

        # Fill cache
        with patch("chiaroscuro_forge.cache.get_cache_manager", return_value=cache):
            with patch("chiaroscuro_forge.cache.get_file_cache_key", side_effect=lambda p: p):
                analyze("file1")
                time.sleep(0.01)
                analyze("file2")
                time.sleep(0.01)
                analyze("file3")
                time.sleep(0.01)

                # This should evict file1 (oldest)
                analyze("file4")

                # file1 should be evicted, others should be cached
                self.assertEqual(len(cache._stats_cache), 3)

    def test_preset_cache_eviction(self):
        """Test that preset cache evicts oldest entries when full."""
        cache = CacheManager(max_preset_cache=2)

        call_count = [0]

        @cached_preset()
        def load(preset_name):
            call_count[0] += 1
            return {"name": preset_name}

        with patch("chiaroscuro_forge.cache.get_cache_manager", return_value=cache):
            load("preset1")
            time.sleep(0.01)
            load("preset2")
            time.sleep(0.01)

            # This should evict preset1 (oldest)
            load("preset3")

            self.assertEqual(len(cache._preset_cache), 2)
            self.assertNotIn("preset1", cache._preset_cache)


if __name__ == "__main__":
    unittest.main()
