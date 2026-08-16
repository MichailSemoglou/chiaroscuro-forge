"""
Preset management for image processing parameters.

This module provides functionality to save, load, and list processing parameter presets,
allowing users to reuse successful enhancement configurations across multiple images.
"""

import json
import logging
import os
from typing import Any, Dict, List

from .cache import cached_preset, invalidate_preset_cache
from .exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


def _resolve_preset_path(preset_name: str) -> str:
    """Validate and resolve the preset path for load/save operations."""
    if not preset_name or not isinstance(preset_name, str):
        raise ImageProcessingError("Invalid preset name")

    if os.path.isabs(preset_name):
        raise ImageProcessingError("Invalid preset name")

    normalized = os.path.normpath(preset_name)
    if normalized in {".", ".."} or normalized.startswith("../") or normalized.startswith("..\\"):
        raise ImageProcessingError("Invalid preset name")

    if os.path.sep in preset_name or (os.path.altsep and os.path.altsep in preset_name):
        raise ImageProcessingError("Invalid preset name")

    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
    preset_path = os.path.join(preset_dir, f"{preset_name}.json")
    resolved_path = os.path.realpath(preset_path)
    preset_dir_real = os.path.realpath(preset_dir)

    if os.path.commonpath([preset_dir_real, resolved_path]) != preset_dir_real:
        raise ImageProcessingError("Invalid preset name")

    return preset_path


@cached_preset()  # Cache indefinitely unless invalidated
def load_preset(preset_name: str) -> Dict[str, Any]:
    """
    Load a preset configuration from disk.

    Results are cached indefinitely to avoid repeated disk I/O.
    Cache is automatically invalidated when presets are saved
    Load a preset configuration from disk.

    Parameters
    ----------
    preset_name : str
        Name of the preset to load (without .json extension)

    Returns
    -------
    dict
        Dictionary of processing parameters

    Raises
    ------
    ImageProcessingError
        If preset file not found or invalid format

    Examples
    --------
    >>> params = load_preset("photography")
    >>> process_image("photo.jpg", **params)
    """
    preset_path = _resolve_preset_path(preset_name)

    if not os.path.exists(preset_path):
        raise ImageProcessingError(f"Preset not found: {preset_name}")

    try:
        with open(preset_path, "r") as f:
            preset_data = json.load(f)

        if "name" not in preset_data or "params" not in preset_data:
            raise ImageProcessingError("Invalid preset format: missing 'name' or 'params' fields")

        params: Dict[str, Any] = preset_data["params"]
        return params
    except json.JSONDecodeError as e:
        raise ImageProcessingError(f"Failed to parse preset file: {e}")
    except Exception as e:
        raise ImageProcessingError(f"Error loading preset: {e}")


def save_preset(preset_name: str, params: Dict[str, Any], description: str = "") -> None:
    """
    Save processing parameters as a preset.

    Automatically invalidates the cache for this preset to ensure
    fresh data is loaded on next access.

    Parameters
    ----------
    preset_name : str
        Name for the preset (without .json extension)
    params : dict
        Dictionary of processing parameters to save
    description : str, optional
        Human-readable description of the preset

    Raises
    ------
    ImageProcessingError
        If unable to create preset directory or save file

    Examples
    --------
    >>> params = {"equalize_method": "clahe", "sharpen_amount": 1.8}
    >>> save_preset("my_preset", params, "High contrast for documents")
    """
    preset_path = _resolve_preset_path(preset_name)
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

    if not os.path.exists(preset_dir):
        try:
            os.makedirs(preset_dir)
        except Exception as e:
            raise ImageProcessingError(f"Failed to create presets directory: {e}")

    preset_data = {"name": preset_name, "description": description, "params": params}

    try:
        with open(preset_path, "w") as f:
            json.dump(preset_data, f, indent=4)

        # Invalidate cache for this preset
        invalidate_preset_cache(preset_name)
    except Exception as e:
        raise ImageProcessingError(f"Failed to save preset: {e}")


def list_presets() -> List[Dict[str, Any]]:
    """
    List all available presets.

    Returns
    -------
    list of dict
        List of preset metadata dictionaries with keys:
        - name: Preset name
        - description: Preset description
        - filename: JSON filename

    Examples
    --------
    >>> presets = list_presets()
    >>> for preset in presets:
    ...     print(f"{preset['name']}: {preset['description']}")
    """
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

    if not os.path.exists(preset_dir):
        return []

    presets = []

    for filename in os.listdir(preset_dir):
        if filename.endswith(".json"):
            preset_path = os.path.join(preset_dir, filename)
            try:
                with open(preset_path, "r") as f:
                    preset_data = json.load(f)

                presets.append(
                    {
                        "name": preset_data.get("name", filename[:-5]),
                        "description": preset_data.get("description", ""),
                        "filename": filename,
                    }
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Skipping unreadable preset %s: %s", filename, exc)
                continue

    return presets


__all__ = ["load_preset", "save_preset", "list_presets"]
