import numpy as np
import pytest

from editpano.geometry import (
    ViewParams,
    erp_to_perspective,
    perspective_grid,
    project_mask_to_perspective,
)
from editpano.geometry.projection import bilinear_sample, remap_patch_to_erp, scatter_patch_to_erp


def test_optical_axis_samples_view_centre(pano):
    H, W = pano.shape[:2]
    for lat, lon in [(0, 0), (30, -60), (-45, 120)]:
        _, smap = erp_to_perspective(pano, ViewParams(lat, lon, 60), 64, 64)
        u, v = smap[31:33, 31:33].reshape(-1, 2).mean(axis=0)
        assert u == pytest.approx((lon / 360 + 0.5) * W, abs=0.6)
        assert v == pytest.approx((0.5 - lat / 180) * H, abs=0.6)


def test_inverse_grid_inverts_forward_sampling():
    view = ViewParams(20.0, 45.0, 70.0)
    _, smap = erp_to_perspective(np.zeros((512, 1024, 3), np.uint8), view, 128, 128)
    mx, my, valid = perspective_grid(view, 128, 128, 512, 1024)
    # Each patch pixel's ERP sample lands back (approximately) on that patch pixel.
    for i, j in [(10, 10), (64, 64), (100, 30)]:
        u, v = smap[i, j]
        ui, vi = round(float(u)) % 1024, round(float(v))
        if valid[vi, ui]:
            assert mx[vi, ui] == pytest.approx(j, abs=1.5)
            assert my[vi, ui] == pytest.approx(i, abs=1.5)


def test_analytic_inverse_is_hole_free_scatter_is_not(pano):
    view = ViewParams(50.0, 0.0, 140.0)
    patch, smap = erp_to_perspective(pano, view, 256, 256)
    _, support, visible = remap_patch_to_erp(patch, view, 512, 1024)
    assert support.sum() == visible.sum() > 0  # every visible pixel receives a value
    _, written = scatter_patch_to_erp(patch, smap, pano)
    assert (visible & ~written).mean() > 0.0  # forward scatter leaves holes


def test_periodic_bilinear_blends_seam_columns():
    img = np.zeros((4, 8, 1), np.float32)
    img[:, 0] = 100.0
    img[:, -1] = 0.0
    u = np.array([[7.5]], np.float32)
    v = np.array([[1.0]], np.float32)
    assert bilinear_sample(img, u, v, wrap_u=True)[0, 0, 0] == pytest.approx(50.0)
    assert bilinear_sample(img, u, v, wrap_u=False)[0, 0, 0] == pytest.approx(0.0)


def test_mask_projection_is_binary_and_centred():
    m = np.zeros((512, 1024), np.uint8)
    m[236:276, 492:532] = 255
    _, smap = erp_to_perspective(np.zeros((512, 1024, 3), np.uint8), ViewParams(0, 0, 60), 128, 128)
    pm = project_mask_to_perspective(m, smap)
    assert set(np.unique(pm)) <= {0, 255}
    assert pm[64, 64] == 255 and pm[2, 2] == 0
