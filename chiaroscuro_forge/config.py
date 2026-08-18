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
    """Configuration container for the image processing pipeline.

    This dataclass centralizes the options exposed by the CLI, API, batch
    runner, and direct Python calls. The defaults are chosen to preserve the
    historical behavior of ``process_image`` while allowing advanced modes such
    as the opt-in linear-light workflow to be enabled explicitly.

    Notes
    -----
    The class is intentionally ``immutable-by-convention``. Callers should use
    ``merge()`` or construct a new ``ProcessingConfig`` when they need a derived
    configuration instead of mutating an instance in place.

    Parameters
    ----------
    application_type : str
        One of ``"general"``, ``"photography"``, ``"medical"``,
        ``"document"``, or ``"art"``.
    scale_factor : float
        Scaling factor for resizing. Values greater than 1.0 enlarge the image,
        while values between 0 and 1 shrink it.
    order : int
        Interpolation order used for general image operations.
    order_rescale : int
        Interpolation order used during rescaling.
    order_rotate : int
        Interpolation order used during rotation.
    rotation_angle : float
        Rotation angle in degrees applied before metrics evaluation.
    denoise_type : str
        One of ``"gaussian"``, ``"median"``, ``"bilateral"``, or ``"none"``.
    denoise_sigma : float
        Noise parameter for the selected denoising method.
    sharpen : bool
        Whether to apply a sharpening step.
    sharpen_amount : float
        Sharpening strength. Higher values increase enhancement intensity.
    equalize : bool
        Whether histogram equalization is enabled.
    equalize_method : str
        One of ``"standard"``, ``"clahe"``, ``"stretch"``, or
        ``"adaptive_gamma"``.
    clip_limit : float
        CLAHE clip limit. Higher values preserve more detail but may increase
        contrast amplification.
    clip_limit_kernel_size : int
        Kernel size used by CLAHE when the method requires local context.
    contrast_stretch_percentiles : tuple
        Low and high percentiles used for contrast stretching in normalized
        percentage space.
    gamma_correction : float
        Gamma-adjustment factor applied during processing.
    linear_light : bool
        Whether to convert the input into linear-light space before processing
        and map the output back to sRGB with a tone-mapping stage. This remains
        opt-in to preserve the default workflow.
    color_preservation : str
        One of ``"none"``, ``"lab"``, ``"rgb"``, or ``"ratio"``.
    color_preservation_strength : float
        Relative strength of the selected color-preservation strategy.
    calculate_metrics : bool
        Whether to compute the standard quality metrics for the processed result.
    calculate_advanced_metrics : bool
        Whether to compute advanced metrics such as MS-SSIM and related
        perceptual comparisons.
    use_tiling : Optional[bool]
        Whether to force tiling on or off. ``None`` allows the library to choose
        auto-detection based on image size and memory constraints.
    tile_size : int
        Tile side length used for large-image processing.
    tile_overlap : int
        Pixel overlap between neighboring tiles.
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
    contrast_stretch_percentiles: Tuple[float, float] = (2.0, 98.0)

    gamma_correction: float = 1.0

    color_preservation: str = "lab"
    color_preservation_strength: float = 0.7

    calculate_metrics: bool = True
    calculate_advanced_metrics: bool = True

    use_tiling: Optional[bool] = None
    tile_size: int = 512
    tile_overlap: int = 64
    rotation_angle: float = 0.0
    linear_light: bool = False

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
            "linear_light": self.linear_light,
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
        if application_type == "general":
            pass
        elif application_type == "document":
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
