"""Seam-position equivariance metrics (thesis Eq. 3.30-3.31).

YE-LPIPS compares a method's output for a shifted input -- after undoing the
shift -- with its centre-position output, inside fixed target-centred tangent
views weighted by the projected reference support (spatial AlexNet LPIPS).
The Discontinuity Score (DS) and boundary-averaging attack are kept as
anti-gaming diagnostics: smoothing the two seam columns lowers DS without
restoring the intended edit.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from ...geometry.camera import ViewParams
from ...geometry.circular import circular_roi_from_mask
from ...geometry.projection import erp_to_perspective, project_mask_to_perspective

__all__ = [
    "SpatialLPIPS",
    "boundary_average_attack",
    "build_eval_views",
    "ds_score",
    "roi_ds_score",
    "sample_view",
    "seam_band_ref_error",
    "seam_rgb_gap",
    "ye_lpips",
]

C_DS = 0.1
DS_BAND = 6
ROI_DS_PAD = 16
EVAL_MARGIN = 1.3
EVAL_FOV_MIN, EVAL_FOV_MAX = 30.0, 150.0
EVAL_SIZE = 512
TILE_TRIGGER_LON = 150.0
TILE_TRIGGER_LAT = 120.0
TILE_FOV = 90.0

_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64) / 16.0


def ds_score(rgb: np.ndarray, rows: slice | None = None, c_ds: float = C_DS) -> float:
    """Discontinuity Score at the ERP seam (continuous image ~ 1; seam step >> 1)."""
    arr = np.asarray(rgb, dtype=np.float64)
    grey = (
        (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) if arr.ndim == 3 else arr
    )
    grey = grey / 255.0
    if rows is not None:
        grey = grey[rows]
    half = DS_BAND // 2
    band = np.concatenate([grey[:, -half:], grey[:, :half]], axis=1)
    L = band.shape[0]
    padded = np.pad(band, ((1, 1), (0, 0)), mode="edge")
    deriv = np.zeros((L, DS_BAND - 2))
    for j in range(1, DS_BAND - 1):
        window = np.stack(
            [padded[:-2, j - 1 : j + 2], padded[1:-1, j - 1 : j + 2], padded[2:, j - 1 : j + 2]],
            axis=1,
        )
        deriv[:, j - 1] = np.tensordot(window, _SCHARR_X, axes=([1, 2], [0, 1]))
    a1, a2, a3, a4 = (np.abs(deriv[:, k]) for k in range(4))
    return float(np.mean(a2 / (a1 + c_ds) + a3 / (a4 + c_ds)) / 2.0)


def roi_ds_score(rgb: np.ndarray, y1: int, y2: int, pad: int = ROI_DS_PAD) -> float:
    H = rgb.shape[0]
    return ds_score(rgb, rows=slice(max(0, y1 - pad), min(H, y2 + pad)))


def seam_rgb_gap(rgb: np.ndarray, rows: slice | None = None) -> float:
    arr = np.asarray(rgb, dtype=np.float64)
    if rows is not None:
        arr = arr[rows]
    return float(np.abs(arr[:, 0] - arr[:, -1]).mean())


def seam_band_ref_error(out: np.ndarray, ref: np.ndarray, rows: slice | None = None) -> float:
    half = DS_BAND // 2
    o = np.concatenate([out[:, -half:], out[:, :half]], axis=1).astype(np.float64)
    r = np.concatenate([ref[:, -half:], ref[:, :half]], axis=1).astype(np.float64)
    if rows is not None:
        o, r = o[rows], r[rows]
    return float(np.abs(o - r).mean())


def boundary_average_attack(rgb: np.ndarray) -> np.ndarray:
    """Replace the first and last columns by their average (metric-sanity attack)."""
    out = rgb.copy()
    avg = (rgb[:, 0].astype(np.float64) + rgb[:, -1].astype(np.float64)) / 2.0
    out[:, 0] = np.round(avg).astype(rgb.dtype)
    out[:, -1] = np.round(avg).astype(rgb.dtype)
    return out


def build_eval_views(source_mask: np.ndarray) -> list[dict[str, Any]]:
    """Method-independent tangent views fixed by the source reference support."""
    roi = circular_roi_from_mask(source_mask)
    H, W = source_mask.shape[:2]
    span_lon = roi.span_deg
    span_lat = (roi.y2 - roi.y1) / H * 180.0
    lat_c = (0.5 - (roi.y1 + roi.y2) / 2.0 / H) * 180.0
    views: list[dict[str, float]] = []
    if span_lon <= TILE_TRIGGER_LON and span_lat <= TILE_TRIGGER_LAT:
        lon_c = ((roi.center / W - 0.5) * 360.0 + 180.0) % 360.0 - 180.0
        fov_w = span_lon * EVAL_MARGIN * math.cos(math.radians(min(abs(lat_c), 85.0)))
        fov = float(np.clip(max(fov_w, span_lat * EVAL_MARGIN), EVAL_FOV_MIN, EVAL_FOV_MAX))
        views.append({"lat": lat_c, "lon": lon_c, "fov": fov})
    else:
        n_lon, n_lat = (
            max(1, math.ceil(span_lon / TILE_FOV)),
            max(1, math.ceil(span_lat / TILE_FOV)),
        )
        for j in range(n_lat):
            lat_j = float(
                np.clip(
                    (0.5 - (roi.y1 + (j + 0.5) * (roi.y2 - roi.y1) / n_lat) / H) * 180.0,
                    -80.0,
                    80.0,
                )
            )
            for k in range(n_lon):
                col = (roi.x_start + (k + 0.5) * roi.width / n_lon) % W
                views.append(
                    {
                        "lat": lat_j,
                        "lon": ((col / W - 0.5) * 360.0 + 180.0) % 360.0 - 180.0,
                        "fov": TILE_FOV,
                    }
                )
    out = []
    blank = np.zeros((H, W, 3), np.uint8)
    for v in views:
        _, sample_map = erp_to_perspective(
            blank, ViewParams(v["lat"], v["lon"], v["fov"]), EVAL_SIZE, EVAL_SIZE
        )
        w = project_mask_to_perspective(source_mask, sample_map).astype(np.float32) / 255.0
        if w.sum() > 0:
            out.append({**v, "weight": w})
    return out


def sample_view(erp_rgb: np.ndarray, view: dict[str, Any]) -> np.ndarray:
    patch, _ = erp_to_perspective(
        erp_rgb, ViewParams(view["lat"], view["lon"], view["fov"]), EVAL_SIZE, EVAL_SIZE
    )
    return patch


class SpatialLPIPS:
    """Lazy spatial AlexNet LPIPS; ``weighted(a, b, w)`` returns ``(sum w*d, sum w)``."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._net: Any = None
        self._lock = threading.Lock()

    def weighted(self, a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
        import torch

        with self._lock:
            if self._net is None:
                import lpips

                self._net = (
                    lpips.LPIPS(net="alex", spatial=True, verbose=False).to(self.device).eval()
                )
        ta = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        tb = torch.from_numpy(b).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            d = self._net(ta.to(self.device), tb.to(self.device)).squeeze().cpu().numpy()
        return float((d * weight).sum()), float(weight.sum())


def ye_lpips(
    erp_ref: np.ndarray,
    erp_test: np.ndarray,
    views: list[dict[str, Any]],
    lpips_model: SpatialLPIPS,
) -> float:
    """Support-weighted LPIPS between two source-frame outputs across the eval views."""
    num = den = 0.0
    for v in views:
        n, d = lpips_model.weighted(sample_view(erp_ref, v), sample_view(erp_test, v), v["weight"])
        num, den = num + n, den + d
    return num / den if den > 0 else math.nan
