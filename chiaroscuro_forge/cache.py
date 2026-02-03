"""
Caching Module for Chiaroscuro Forge

This module provides caching functionality for image statistics and presets
to improve performance by avoiding redundant computations.

Features:
- LRU cache for image statistics
- Preset caching with automatic invalidation
- File hash-based cache keys
- Configurable cache sizes
"""

import hashlib
import time
import copy
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import json

from .exceptions import ImageProcessingError


class CacheManager:
    """
    Central cache manager for image processing operations.
    
    This class provides a unified interface for managing caches across
    different operations, with configurable sizes and TTL support.
    """
    
    def __init__(self, max_stats_cache: int = 128, max_preset_cache: int = 32):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        max_stats_cache : int, optional
            Maximum number of image statistics to cache (default: 128)
        max_preset_cache : int, optional
            Maximum number of presets to cache (default: 32)
        """
        self.max_stats_cache = max_stats_cache
        self.max_preset_cache = max_preset_cache
        self._stats_cache: Dict[str, Tuple[float, Any]] = {}
        self._preset_cache: Dict[str, Tuple[float, Dict]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary with cache hits and misses
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "stats_cached": len(self._stats_cache),
            "presets_cached": len(self._preset_cache),
        }
    
    def clear(self):
        """Clear all caches."""
        self._stats_cache.clear()
        self._preset_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_stats_cache(self):
        """Clear image statistics cache only."""
        self._stats_cache.clear()
    
    def clear_preset_cache(self):
        """Clear preset cache only."""
        self._preset_cache.clear()


# Global cache manager instance
_global_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns
    -------
    CacheManager
        The global cache manager
    """
    return _global_cache_manager


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a file for cache key generation.
    
    Uses streaming to handle large files efficiently.
    
    Parameters
    ----------
    file_path : str
        Path to the file
    chunk_size : int, optional
        Size of chunks to read (default: 8192 bytes)
    
    Returns
    -------
    str
        Hexadecimal SHA256 hash of the file
    
    Raises
    ------
    ImageProcessingError
        If file cannot be read
    """
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        raise ImageProcessingError(f"Failed to compute file hash: {e}")


def get_file_cache_key(file_path: str) -> str:
    """
    Generate cache key for a file based on path and modification time.
    
    This avoids expensive hash computation while still detecting file changes.
    
    Parameters
    ----------
    file_path : str
        Path to the file
    
    Returns
    -------
    str
        Cache key string
    """
    try:
        path = Path(file_path)
        mtime = path.stat().st_mtime
        size = path.stat().st_size
        return f"{file_path}:{mtime}:{size}"
    except Exception as e:
        raise ImageProcessingError(f"Failed to generate cache key: {e}")


def cached_image_stats(ttl: Optional[float] = 3600):
    """
    Decorator for caching image statistics with TTL.
    
    Parameters
    ----------
    ttl : float, optional
        Time-to-live in seconds (default: 3600 = 1 hour)
        Set to None for no expiration
    
    Returns
    -------
    callable
        Decorated function with caching
    
    Examples
    --------
    >>> @cached_image_stats(ttl=1800)
    ... def analyze_image(image_path):
    ...     # Expensive computation
    ...     return stats
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(image_path: str, *args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            try:
                cache_key = get_file_cache_key(image_path)
            except Exception:
                # If cache key generation fails, just compute without caching
                return func(image_path, *args, **kwargs)
            
            # Check cache
            if cache_key in cache._stats_cache:
                timestamp, result = cache._stats_cache[cache_key]
                
                # Check TTL
                if ttl is None or (time.time() - timestamp) < ttl:
                    cache._cache_hits += 1
                    return result
                else:
                    # Expired, remove from cache
                    del cache._stats_cache[cache_key]
            
            # Cache miss - compute result
            cache._cache_misses += 1
            result = func(image_path, *args, **kwargs)
            
            # Store in cache with timestamp
            cache._stats_cache[cache_key] = (time.time(), result)
            
            # Evict oldest entries if cache is full (simple FIFO)
            if len(cache._stats_cache) > cache.max_stats_cache:
                # Remove oldest entry
                oldest_key = min(cache._stats_cache.keys(), 
                               key=lambda k: cache._stats_cache[k][0])
                del cache._stats_cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator


def cached_preset(ttl: Optional[float] = None):
    """
    Decorator for caching preset loading with TTL.
    
    Parameters
    ----------
    ttl : float, optional
        Time-to-live in seconds (default: None = no expiration)
    
    Returns
    -------
    callable
        Decorated function with caching
    
    Examples
    --------
    >>> @cached_preset()
    ... def load_preset(name):
    ...     # Load from disk
    ...     return preset_dict
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(preset_name: str, *args, **kwargs):
            cache = get_cache_manager()
            
            # Check cache
            if preset_name in cache._preset_cache:
                timestamp, result = cache._preset_cache[preset_name]
                
                # Check TTL
                if ttl is None or (time.time() - timestamp) < ttl:
                    cache._cache_hits += 1
                    return copy.deepcopy(result)  # Return deep copy to prevent mutations
                else:
                    # Expired, remove from cache
                    del cache._preset_cache[preset_name]
            
            # Cache miss - compute result
            cache._cache_misses += 1
            result = func(preset_name, *args, **kwargs)
            
            # Store in cache with timestamp (store a copy to prevent mutations)
            cache._preset_cache[preset_name] = (time.time(), copy.deepcopy(result))
            
            # Evict oldest entries if cache is full
            if len(cache._preset_cache) > cache.max_preset_cache:
                oldest_key = min(cache._preset_cache.keys(),
                               key=lambda k: cache._preset_cache[k][0])
                del cache._preset_cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator


def invalidate_preset_cache(preset_name: Optional[str] = None):
    """
    Invalidate preset cache.
    
    Parameters
    ----------
    preset_name : str, optional
        Name of specific preset to invalidate.
        If None, invalidates entire preset cache.
    
    Examples
    --------
    >>> invalidate_preset_cache("photography")  # Invalidate one preset
    >>> invalidate_preset_cache()  # Invalidate all presets
    """
    cache = get_cache_manager()
    
    if preset_name is None:
        cache.clear_preset_cache()
    elif preset_name in cache._preset_cache:
        del cache._preset_cache[preset_name]


def invalidate_stats_cache(image_path: Optional[str] = None):
    """
    Invalidate image statistics cache.
    
    Parameters
    ----------
    image_path : str, optional
        Path to specific image to invalidate.
        If None, invalidates entire stats cache.
    
    Examples
    --------
    >>> invalidate_stats_cache("image.jpg")  # Invalidate one image
    >>> invalidate_stats_cache()  # Invalidate all stats
    """
    cache = get_cache_manager()
    
    if image_path is None:
        cache.clear_stats_cache()
    else:
        try:
            cache_key = get_file_cache_key(image_path)
            if cache_key in cache._stats_cache:
                del cache._stats_cache[cache_key]
        except Exception:
            pass  # Ignore errors in cache invalidation


__all__ = [
    "CacheManager",
    "get_cache_manager",
    "compute_file_hash",
    "get_file_cache_key",
    "cached_image_stats",
    "cached_preset",
    "invalidate_preset_cache",
    "invalidate_stats_cache",
]
