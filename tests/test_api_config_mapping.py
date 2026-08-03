"""Unit tests for API processing-parameter mapping."""

from chiaroscuro_forge.api import _build_processing_config_from_params


class DummyParams:
    def dict(self, exclude_none=True):
        return {
            "application_type": "photography",
            "gamma": 1.2,
            "scale_factor": 2.0,
            "denoise_type": "bilateral",
            "equalize_method": "clahe",
            "clip_limit": 0.02,
            "color_preservation": "lab",
            "use_tiling": True,
            "tile_size": 256,
            "tile_overlap": 32,
        }


def test_build_processing_config_maps_api_params():
    config = _build_processing_config_from_params(DummyParams())

    assert config.application_type == "photography"
    assert config.gamma_correction == 1.2
    assert config.scale_factor == 2.0
    assert config.denoise_type == "bilateral"
    assert config.equalize_method == "clahe"
    assert config.clip_limit == 0.02
    assert config.color_preservation == "lab"
    assert config.use_tiling is True
    assert config.tile_size == 256
    assert config.tile_overlap == 32
