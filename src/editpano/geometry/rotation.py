"""Whole-sphere rotation of an ERP panorama (for controlled distortion studies)."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["rotate_erp"]


def rotate_erp(
    img: np.ndarray,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
    nearest: bool = False,
) -> np.ndarray:
    """Rotate the viewing sphere by ``R = R_x(pitch) R_y(yaw) R_z(roll)``.

    Each output pixel samples the input at the inverse-rotated direction with
    longitude wrapping.  ``nearest`` is intended for label/mask images.
    """
    import cv2

    H, W = img.shape[:2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    lon = (uu / W - 0.5) * (2 * math.pi)
    lat = (0.5 - vv / H) * math.pi
    cl = np.cos(lat)
    d = np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)
    p, y, r = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), math.sin(r), 0], [-math.sin(r), math.cos(r), 0], [0, 0, 1]])
    din = d @ (Rx @ Ry @ Rz)
    lon_i = np.arctan2(din[..., 0], din[..., 2])
    lat_i = np.arcsin(np.clip(din[..., 1], -1.0, 1.0))
    map_x = ((lon_i / (2 * math.pi) + 0.5) * W).astype(np.float32)
    map_y = ((0.5 - lat_i / math.pi) * H).astype(np.float32)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(img, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_WRAP)
