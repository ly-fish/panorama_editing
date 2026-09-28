import itertools

import pytest

from editpano import ConfigError, EditPanoConfig
from editpano.geometry.tiling import angular_extent, split_box_into_tiles


def test_small_box_is_one_tile():
    assert split_box_into_tiles([100, 100, 150, 140], 1024, 512) == [[100, 100, 150, 140]]


def test_tiles_cover_box_with_overlap_and_bounded_span():
    box = [0.0, 100.0, 600.0, 300.0]  # ~211 deg wide
    tiles = split_box_into_tiles(box, 1024, 512, tile_span_deg=60, overlap=0.2, max_tiles=6)
    assert 1 < len(tiles) <= 6
    assert min(t[0] for t in tiles) == box[0] and max(t[2] for t in tiles) == box[2]
    xs = sorted(tiles, key=lambda t: t[0])
    assert all(a[2] > b[0] for a, b in itertools.pairwise(xs))  # overlapping neighbours
    assert all(angular_extent(t, 1024, 512)[1] <= 60 + 1e-9 for t in tiles)


def test_tiling_validation():
    with pytest.raises(ValueError):
        split_box_into_tiles([0, 0, 10, 10], 100, 50, overlap=1.0)


def test_config_defaults_match_thesis_operating_point():
    c = EditPanoConfig()
    assert (
        c.projection.patch_size,
        c.projection.fov_margin,
        c.projection.fov_min,
        c.projection.fov_max,
    ) == (512, 1.5, 30.0, 150.0)
    assert (c.composite.mode, c.composite.feather_px, c.composite.patch_mask_feather) == (
        "inward",
        8,
        3,
    )
    assert (c.editor.name, c.editor.seed, c.gate.confidence_threshold) == ("qwen", 0, 0.7)


def test_config_overrides_and_roundtrip():
    c = EditPanoConfig().with_overrides(
        ["composite.feather_px=16", "tiling.mode=off", "editor.name=fake:pattern"]
    )
    assert (
        c.composite.feather_px == 16 and c.tiling.mode == "off" and c.editor.name == "fake:pattern"
    )
    assert EditPanoConfig.from_dict(c.to_dict()) == c
    assert c.fingerprint() != EditPanoConfig().fingerprint()


@pytest.mark.parametrize(
    "bad",
    [
        {"composite": {"mode": "blur"}},
        {"projection": {"patch_size": "big"}},
        {"nope": 1},
        {"composite": {"feathr_px": 3}},
        {"tiling": {"mode": False}},
    ],
)
def test_config_rejects_invalid(bad):
    with pytest.raises(ConfigError):
        EditPanoConfig.from_dict(bad)


def test_unknown_override_key():
    with pytest.raises(ConfigError):
        EditPanoConfig().with_overrides({"composite.nope": 1})


def test_yaml_file_equals_builtin_defaults(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert EditPanoConfig.from_yaml(root / "configs/editpano.yaml") == EditPanoConfig()
