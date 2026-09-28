"""Gnomonic projection ERP -> perspective patch and its dense analytic inverse.

Forward (thesis Eq. 3.6-3.11)::

    d_cam = normalise(x_j, y_i, 1),   x_j = (j - W_p/2 + 0.5) / f,  y_i = -(i - H_p/2 + 0.5) / f
    d     = R_y(lon) R_x(lat) d_cam
    lon   = atan2(d_x, d_z),  lat = asin(d_y)

Inverse (thesis Eq. 3.14-3.16) evaluates, for *every* ERP pixel, the camera
ray ``R_x(lat)^T R_y(lon)^T d_world`` and its patch coordinates, so the
edited patch can be resampled onto the ERP grid with ``cv2.remap`` -- dense and
hole-free by construction.  The operation order and dtypes intentionally
mirror the thesis implementation so results are bit-identical.
"""

from __future__ import annotations

import math

import numpy as np

from .camera import ViewParams

__all__ = [
    "bilinear_sample",
    "erp_to_perspective",
    "perspective_grid",
    "project_mask_to_perspective",
    "remap_patch_to_erp",
    "scatter_patch_to_erp",
]


def bilinear_sample(
    img: np.ndarray, u: np.ndarray, v: np.ndarray, wrap_u: bool = True
) -> np.ndarray:
    """Bilinear interpolation of ``img`` at fractional ``(u, v)``.

    With ``wrap_u`` the horizontal neighbours are periodic (``u1 = (u0 + 1) mod W``),
    so samples between the last and first columns blend both seam columns
    (thesis Eq. 3.24) instead of clamping to one border.
    """
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    du = (u - u0.astype(np.float32))[..., np.newaxis]
    dv = (v - v0.astype(np.float32))[..., np.newaxis]
    if wrap_u:
        u0 = u0 % W
        u1 = (u0 + 1) % W
    else:
        u1 = np.clip(u0 + 1, 0, W - 1)
        u0 = np.clip(u0, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    v0 = np.clip(v0, 0, H - 1)
    top = img[v0, u0] * (1 - du) + img[v0, u1] * du
    bottom = img[v1, u0] * (1 - du) + img[v1, u1] * du
    return top * (1 - dv) + bottom * dv


def erp_to_perspective(
    erp: np.ndarray,
    view: ViewParams,
    out_w: int = 512,
    out_h: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rectilinear patch from an ERP panorama.

    Args:
        erp: ``[H, W, 3]`` uint8 panorama.
        view: optical axis and horizontal FoV.
        out_w, out_h: patch size in pixels.

    Returns:
        ``(patch, sample_map)`` -- ``patch`` is ``[out_h, out_w, 3]`` uint8 and
        ``sample_map`` is ``[out_h, out_w, 2]`` float32 ERP coordinates ``(u, v)``
        of every patch pixel (``S`` in the thesis).
    """
    erp_arr = np.asarray(erp).astype(np.float32)
    H_erp, W_erp = erp_arr.shape[:2]

    lat_c = math.radians(view.lat)
    lon_c = math.radians(view.lon)
    f = (out_w / 2.0) / math.tan(math.radians(view.fov) / 2.0)

    xs = (np.arange(out_w) - out_w / 2.0 + 0.5) / f
    ys = -(np.arange(out_h) - out_h / 2.0 + 0.5) / f
    xv, yv = np.meshgrid(xs, ys)
    zv = np.ones_like(xv)
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xv, yv, zv = xv / norm, yv / norm, zv / norm

    # pitch: R_x(lat)
    cos_lat, sin_lat = math.cos(lat_c), math.sin(lat_c)
    xw = xv
    yw = yv * cos_lat + zv * sin_lat
    zw = -yv * sin_lat + zv * cos_lat
    # yaw: R_y(lon)
    cos_lon, sin_lon = math.cos(lon_c), math.sin(lon_c)
    xr = xw * cos_lon + zw * sin_lon
    yr = yw
    zr = -xw * sin_lon + zw * cos_lon

    lat_map = np.arcsin(np.clip(yr, -1.0, 1.0))
    lon_map = np.arctan2(xr, zr)
    u_map = (lon_map / (2 * math.pi) + 0.5) * W_erp
    v_map = (0.5 - lat_map / math.pi) * H_erp
    u_map = (u_map % W_erp).astype(np.float32)
    v_map = np.clip(v_map, 0, H_erp - 1).astype(np.float32)

    patch = bilinear_sample(erp_arr, u_map, v_map).astype(np.uint8)
    return patch, np.stack([u_map, v_map], axis=-1)


def project_mask_to_perspective(erp_mask: np.ndarray, sample_map: np.ndarray) -> np.ndarray:
    """Project a binary ERP mask (0/255) into patch space (bilinear, threshold 127)."""
    H_erp, W_erp = erp_mask.shape[:2]
    mask_f = erp_mask.astype(np.float32)
    u = sample_map[..., 0]
    v = sample_map[..., 1]
    u0 = np.floor(u).astype(np.int32) % W_erp
    u1 = (u0 + 1) % W_erp
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H_erp - 1)
    v1 = np.clip(v0 + 1, 0, H_erp - 1)
    du = (u - np.floor(u))[..., np.newaxis]
    dv = (v - np.floor(v))[..., np.newaxis]
    top = mask_f[v0, u0, np.newaxis] * (1 - du) + mask_f[v0, u1, np.newaxis] * du
    bottom = mask_f[v1, u0, np.newaxis] * (1 - du) + mask_f[v1, u1, np.newaxis] * du
    interp = (top * (1 - dv) + bottom * dv)[..., 0]
    return (interp >= 127).astype(np.uint8) * 255


def perspective_grid(
    view: ViewParams,
    persp_w: int,
    persp_h: int,
    erp_h: int,
    erp_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense analytic inverse: patch coordinates of every ERP pixel.

    Returns:
        ``(map_x, map_y, valid)`` -- ``[H, W]`` float32 patch coordinates (``-1``
        outside the frustum) and a bool mask of ERP pixels whose ray lies in
        front of the camera and inside the patch bounds.
    """
    lat_c = math.radians(view.lat)
    lon_c = math.radians(view.lon)
    f = (persp_w / 2.0) / math.tan(math.radians(view.fov) / 2.0)

    uu, vv = np.meshgrid(np.arange(erp_w, dtype=np.float64), np.arange(erp_h, dtype=np.float64))
    lon_map = (uu / erp_w - 0.5) * (2.0 * math.pi)
    lat_map = (0.5 - vv / erp_h) * math.pi

    cos_lat_m = np.cos(lat_map)
    xr = cos_lat_m * np.sin(lon_map)
    yr = np.sin(lat_map)
    zr = cos_lat_m * np.cos(lon_map)

    cos_lon, sin_lon = math.cos(lon_c), math.sin(lon_c)  # R_y(lon)^T
    x_p = xr * cos_lon - zr * sin_lon
    y_p = yr
    z_p = xr * sin_lon + zr * cos_lon
    cos_lat, sin_lat = math.cos(lat_c), math.sin(lat_c)  # R_x(lat)^T
    xv = x_p
    yv = y_p * cos_lat - z_p * sin_lat
    zv = y_p * sin_lat + z_p * cos_lat

    valid = zv > 1e-6
    safe_zv = np.where(valid, zv, 1.0)
    map_x = (xv / safe_zv * f + persp_w / 2.0 - 0.5).astype(np.float32)
    map_y = (-yv / safe_zv * f + persp_h / 2.0 - 0.5).astype(np.float32)
    valid = valid & (map_x >= 0) & (map_x < persp_w - 1) & (map_y >= 0) & (map_y < persp_h - 1)
    map_x = np.where(valid, map_x, np.float32(-1.0))
    map_y = np.where(valid, map_y, np.float32(-1.0))
    return map_x, map_y, valid


def remap_patch_to_erp(
    patch: np.ndarray,
    view: ViewParams,
    erp_h: int,
    erp_w: int,
    persp_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a patch onto the ERP grid with the analytic inverse.

    Returns:
        ``(content, support, visible)`` -- float32 ``[H, W, 3]`` reprojected
        patch ``R``; float32 ``[H, W]`` binary support (patch mask remapped and
        thresholded at 0.5, or the full frustum if ``persp_mask`` is None); and
        the bool frustum mask.
    """
    import cv2

    H_p, W_p = patch.shape[:2]
    map_x, map_y, visible = perspective_grid(view, W_p, H_p, erp_h, erp_w)
    content = cv2.remap(
        patch.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if persp_mask is not None:
        m = cv2.remap(
            persp_mask.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        support = ((m > 127) & visible).astype(np.float32)
    else:
        support = visible.astype(np.float32)
    return content, support, visible


def scatter_patch_to_erp(
    patch: np.ndarray, sample_map: np.ndarray, erp_base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Naive forward scatter (diagnostic baseline): nearest ERP pixel per patch pixel.

    Returns ``(erp_image, written)``; unwritten visible pixels are *holes*.
    """
    H, W = erp_base.shape[:2]
    result = erp_base.copy().astype(np.float32)
    written = np.zeros((H, W), dtype=bool)
    u = np.clip(np.round(sample_map[..., 0]).astype(np.int64), 0, W - 1).ravel()
    v = np.clip(np.round(sample_map[..., 1]).astype(np.int64), 0, H - 1).ravel()
    result[v, u] = patch.reshape(-1, 3)
    written[v, u] = True
    return result.clip(0, 255).astype(np.uint8), written
