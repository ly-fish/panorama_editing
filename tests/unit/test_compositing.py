import numpy as np
import pytest
from PIL import Image

from editpano.geometry.compositing import (
    composite_reprojection,
    constrain_patch_to_mask,
    inward_feather_alpha,
    inward_feather_blend,
    poisson_blend,
    seam_unwrap_shift,
)
from tests._helpers import wrapped_mask


def _content(base):
    return (255 - base).astype(np.uint8)


@pytest.mark.parametrize("feather", [0, 4, 8, 16])
def test_inward_blend_background_bit_exact(pano, feather):
    m = np.zeros(pano.shape[:2], np.uint8)
    m[100:220, 300:500] = 255
    out = inward_feather_blend(pano, _content(pano), m, feather)
    assert np.array_equal(out[m == 0], pano[m == 0])
    assert np.array_equal(out[160, 400], _content(pano)[160, 400])  # deep interior fully replaced


def test_inward_blend_handles_wrapped_support(pano):
    H, W = pano.shape[:2]
    m = wrapped_mask(W, H, 60, 60, 200, 300)
    out = inward_feather_blend(pano, _content(pano), m, 8)
    assert np.array_equal(out[m == 0], pano[m == 0])
    # the ramp is measured across the seam: the columns next to u=0 are interior, not boundary
    assert np.array_equal(out[250, 0], _content(pano)[250, 0])


def test_inward_alpha_is_zero_on_boundary_and_one_inside():
    m = np.zeros((64, 64), np.uint8)
    m[16:48, 16:48] = 1
    a = inward_feather_alpha(m, 8)
    assert a[0, 0] == 0 and a[32, 32] == 1.0 and 0 < a[17, 32] < 1


def test_seam_unwrap_shift():
    m = np.zeros((4, 10), np.uint8)
    m[:, [0, 1, 9]] = 1
    s = seam_unwrap_shift(m)
    assert s and not (np.roll(m, s, axis=1)[:, [0, -1]].any(axis=0).all())
    assert seam_unwrap_shift(np.ones((2, 5), np.uint8)) is None
    assert seam_unwrap_shift(np.zeros((2, 5), np.uint8)) == 0


def test_poisson_background_exact(pano):
    m = np.zeros(pano.shape[:2], np.uint8)
    m[100:200, 300:420] = 255
    out = poisson_blend(pano, _content(pano), m)
    assert np.array_equal(out[m == 0], pano[m == 0])


def test_feather_mode_leaks_outside_support(pano):
    sup = np.zeros(pano.shape[:2], np.float32)
    sup[100:200, 300:420] = 1
    vis = np.ones(pano.shape[:2], bool)
    content = _content(pano).astype(np.float32)
    inward = composite_reprojection(pano, content, sup, vis, "inward", 8)
    feather = composite_reprojection(pano, content, sup, vis, "feather", 8)
    outside = sup == 0
    assert np.array_equal(inward[outside], pano[outside])
    assert not np.array_equal(feather[outside], pano[outside])


def test_constrain_patch_restores_outside_soft_mask(small_pano):
    src = Image.fromarray(small_pano[:128, :128])
    edited = Image.fromarray(255 - small_pano[:128, :128])
    mask = np.zeros((128, 128), np.uint8)
    mask[40:90, 40:90] = 255
    out = np.asarray(constrain_patch_to_mask(src, edited, Image.fromarray(mask), feather=3))
    assert np.array_equal(out[:20, :20], np.asarray(src)[:20, :20])
    assert np.array_equal(out[64, 64], np.asarray(edited)[64, 64])
