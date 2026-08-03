"""
Shared processing configuration for Chiaroscuro Forge.

This module provides a single, consistent configuration layer for all
processing entry points: CLI, API, presets, batch runner, and
distributed processing.

All parameters that affect image processing behavior live in one
dataclass, with defaults, validation, and serialization in one place.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ProcessingConfig:
    """Immutable-by-convention configuration for image processing.

    Every field carries a default that matches the existing
    ``process_image`` signature, so existing callers that pass
    individual keyword arguments continue to work unchanged.

    Parameters
    ----------
    application_type : str
        One of ``"general"``, ``"photography"``, ``"medical"``,
        ``"document"``, ``"art"``.
    scale_factor : float
        Scaling factor for resizing (> 0).
    order : int
        Interpolation order for general operations (0-5).
    order_rescale : int
        Interpolation order for rescaling (0-5).
    order_rotate : int
        Interpolation order for rotation (0-5).
    rotation_angle : float
        Rotation angle in degrees to apply before metrics evaluation.
    denoise_type : str
        One of ``"gaussian"``, ``"median"``, ``"bilateral"``,
        ``"none"``.
    denoise_sigma : float
        Sigma parameter for denoising (>= 0).
    sharpen : bool
        Whether to apply sharpening.
    sharpen_amount : float
        Amount of sharpening (> 0).
    equalize : bool
        Whether to apply histogram equalization.
    equalize_method : str
        One of ``"standard"``, ``"clahe"``, ``"stretch"``,
        ``"adaptive_gamma"``.
    clip_limit : float
        CLAHE clip limit (> 0).
    clip_limit_kernel_size : int
        CLAHE kernel size (> 0).
    contrast_stretch_percentiles : tuple
        Low and high percentiles for contrast stretching (0-100).
    gamma_correction : float
        Gamma correction value (> 0).
    color_preservation : str
        One of ``"none"``, ``"lab"``, ``"rgb"``, ``"ratio"``.
    color_preservation_strength : float
        Strength of color preservation (0.0-1.0).
    calculate_metrics : bool
        Whether to calculate quality metrics.
    calculate_advanced_metrics : bool
        Whether to calculate advanced metrics (MS-SSIM, etc.).
    use_tiling : Optional[bool]
        Force tiling on/off. ``None`` means auto-detect.
    tile_size : int
        Tile size in pixels for large-image processing.
    tile_overlap : int
        Overlap between tiles in pixels.
    """

    application_type: str = "general"

    scale_factor: float = 1.0
    order: int = 1
    order_rescale: int = 1
    order_rotate: int = 1

    denoise_type: str = "gaussian"
    denoise_sigma: float = 1.0

    sharpen: bool = True
    sharpen_amount: float = 1.5

    equalize: bool = True
    equalize_method: str = "stretch"
    clip_limit: float = 0.03
    clip_limit_kernel_size: int = 8
    contrast_stretch_percentiles: Tuple[int, int] = (2, 98)

    gamma_correction: float = 1.0

    color_preservation: str = "lab"
    color_preservation_strength: float = 0.7

    calculate_metrics: bool = True
    calculate_advanced_metrics: bool = True

    use_tiling: Optional[bool] = None
    tile_size: int = 512
    tile_overlap: int = 64
    rotation_angle: float = 0.0

    def merge(self, overrides: Dict[str, Any]) -> "ProcessingConfig":
        """Return a new config with fields replaced by *overrides*.

        Parameters
        ----------
        overrides : dict
            Mapping of field names to new values.

        Returns
        -------
        ProcessingConfig
            New instance with merged values.

        Raises
        ------
        ValueError
            If any override key is not a recognized field.
        """
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(
                f"Unknown config fields: {sorted(unknown)}. " f"Known fields: {sorted(known)}"
            )
        current = asdict(self)
        current.update(overrides)
        return ProcessingConfig(**current)

    def to_context(self) -> Dict:
        """Build the context dictionary consumed by the processing pipeline.

        Returns
        -------
        dict
            Context dict with keys expected by ``ImageProcessingPipeline``
            stages.
        """
        return {
            "scale_factor": self.scale_factor,
            "order": self.order,
            "order_rescale": self.order_rescale,
            "order_rotate": self.order_rotate,
            "rotation_angle": self.rotation_angle,
            "denoise_type": self.denoise_type,
            "denoise_sigma": self.denoise_sigma,
            "sharpen": self.sharpen,
            "sharpen_amount": self.sharpen_amount,
            "equalize": self.equalize,
            "equalize_method": self.equalize_method,
            "clip_limit": self.clip_limit,
            "clip_limit_kernel_size": self.clip_limit_kernel_size,
            "contrast_stretch_percentiles": self.contrast_stretch_percentiles,
            "gamma_correction": self.gamma_correction,
            "color_preservation": self.color_preservation,
            "color_preservation_strength": self.color_preservation_strength,
        }

    # -- application-type presets ---------------------------------------

    @classmethod
    def preset(cls, application_type: str = "general") -> "ProcessingConfig":
        """Return a config with sensible defaults per application type.

        Parameters
        ----------
        application_type : str
            One of the valid application types.

        Returns
        -------
        ProcessingConfig
        """
        overrides: dict = {}
        if application_type == "document":
            overrides["equalize_method"] = "clahe"
            overrides["clip_limit"] = 0.02
            overrides["sharpen_amount"] = 1.8
            overrides["denoise_sigma"] = 0.5
            overrides["gamma_correction"] = 1.1
        elif application_type == "photography":
            overrides["color_preservation"] = "lab"
            overrides["color_preservation_strength"] = 0.9
            overrides["sharpen_amount"] = 1.2
            overrides["clip_limit"] = 0.02
        elif application_type == "medical":
            overrides["equalize_method"] = "clahe"
            overrides["clip_limit"] = 0.04
            overrides["denoise_type"] = "bilateral"
            overrides["denoise_sigma"] = 0.4
            overrides["sharpen_amount"] = 0.5
            overrides["color_preservation"] = "none"
        elif application_type == "art":
            overrides["color_preservation"] = "ratio"
            overrides["color_preservation_strength"] = 0.5
            overrides["gamma_correction"] = 1.15
            overrides["sharpen"] = False
        else:
            raise ValueError(f"Unknown application type: {application_type}")
        return cls(application_type=application_type, **overrides)

    # -- serialization --------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize to a plain dictionary suitable for JSON.

        ``contrast_stretch_percentiles`` is stored as a list for JSON
        compatibility.

        Returns
        -------
        dict
        """
        d = asdict(self)
        d["contrast_stretch_percentiles"] = list(d["contrast_stretch_percentiles"])
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ProcessingConfig":
        """Deserialize from a plain dictionary.

        ``contrast_stretch_percentiles`` may be a list (from JSON) or a
        tuple.

        Parameters
        ----------
        data : dict

        Returns
        -------
        ProcessingConfig
        """
        data = dict(data)
        if "contrast_stretch_percentiles" in data and isinstance(
            data["contrast_stretch_percentiles"], list
        ):
            data["contrast_stretch_percentiles"] = tuple(data["contrast_stretch_percentiles"])
        return cls(**data)
