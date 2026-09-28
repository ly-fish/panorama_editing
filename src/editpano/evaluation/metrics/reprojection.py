"""Editor-free reprojection diagnostics (thesis Table 3.4, Eq. A.2).

A patch is sent patch -> ERP -> patch through (a) the dense analytic inverse and
(b) naive forward scatter; the round trip should be near-lossless and the
analytic operator is hole-free by construction.  The boundary-seam ratio
measures the gradient ridge along a composite contour (~1 = seamless).
"""

from __future__ import annotations

import math

import numpy as np

from ...geometry.camera import ViewParams
from ...geometry.compositing import composite_reprojection
from ...geometry.projection import (
    erp_to_perspective,
    perspective_grid,
    remap_patch_to_erp,
    scatter_patch_to_erp,
)

__all__ = ["boundary_seam_ratio", "cycle_consistency", "psnr"]


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    if mask is not None:
        a32, b32 = a32[mask], b32[mask]
    mse = float(np.mean((a32 - b32) ** 2))
    return math.inf if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def cycle_consistency(
    patch: np.ndarray, erp_base: np.ndarray, view: ViewParams, with_ssim: bool = True
) -> dict[str, float]:
    """Round-trip ``patch`` through ERP with both inverse operators."""
    Hp, Wp = patch.shape[:2]
    H, W = erp_base.shape[:2]
    _, sample_map = erp_to_perspective(erp_base, view, Wp, Hp)
    _, _, visible = perspective_grid(view, Wp, Hp, H, W)
    full = np.full((Hp, Wp), 255, np.uint8)

    content, support, vis = remap_patch_to_erp(patch, view, H, W, full)
    erp_inv = composite_reprojection(erp_base, content, support, vis, "feather", 0)
    p_inv, _ = erp_to_perspective(erp_inv, view, Wp, Hp)
    erp_fwd, written = scatter_patch_to_erp(patch, sample_map, erp_base)
    p_fwd, _ = erp_to_perspective(erp_fwd, view, Wp, Hp)

    out = {
        "cycle_psnr_analytic": psnr(p_inv, patch),
        "cycle_psnr_scatter": psnr(p_fwd, patch),
        "scatter_hole_frac": float((visible & ~written).sum() / max(1, int(visible.sum()))),
    }
    if with_ssim:
        out["cycle_ssim_analytic"] = _ssim(p_inv, patch)
        out["cycle_ssim_scatter"] = _ssim(p_fwd, patch)
    return out


def boundary_seam_ratio(edited: np.ndarray, support: np.ndarray, band: int = 8) -> float:
    """Mean gradient magnitude on the support contour / in the surrounding band."""
    import cv2

    support = support.astype(np.uint8)
    if support.sum() == 0 or support.sum() == support.size:
        return math.nan
    gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gmag = np.sqrt(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2
        + cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    )

    def k(r: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    ring = (cv2.dilate(support, k(1)) > 0) & ~(cv2.erode(support, k(1)) > 0)
    wide = (cv2.dilate(support, k(band)) > 0) & ~(cv2.erode(support, k(band)) > 0)
    band_only = wide & ~ring
    if ring.sum() == 0 or band_only.sum() == 0:
        return math.nan
    return float(gmag[ring].mean() / (gmag[band_only].mean() + 1e-6))
