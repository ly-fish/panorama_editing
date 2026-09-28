"""Target-support rasterisation in ERP and patch space."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "bbox_to_binary_mask",
    "box_mask_in_perspective",
    "center_oval_mask",
    "clamp_box",
    "dilate_mask",
    "instance_union_mask",
]


def clamp_box(box: Sequence[float], W: int, H: int) -> tuple[int, int, int, int]:
    """Round and clamp a box to a valid, non-empty integer rectangle."""
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
    x2, y2 = max(x1 + 1, min(W, x2)), max(y1 + 1, min(H, y2))
    return x1, y1, x2, y2


def bbox_to_binary_mask(
    box: Sequence[float], H: int, W: int, margin_frac: float = 0.0
) -> np.ndarray:
    """``[H, W]`` uint8 mask, 255 inside ``box`` (coordinates clipped to ``W - 1``/``H - 1``)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if margin_frac > 0:
        dx, dy = (x2 - x1) * margin_frac, (y2 - y1) * margin_frac
        x1, x2, y1, y2 = x1 - dx, x2 + dx, y1 - dy, y2 + dy
    x1i, y1i = int(np.clip(x1, 0, W - 1)), int(np.clip(y1, 0, H - 1))
    x2i, y2i = int(np.clip(x2, 0, W - 1)), int(np.clip(y2, 0, H - 1))
    mask = np.zeros((H, W), dtype=np.uint8)
    if x2i > x1i and y2i > y1i:
        mask[y1i:y2i, x1i:x2i] = 255
    return mask


def instance_union_mask(boxes: Iterable[Sequence[float]], W: int, H: int) -> np.ndarray:
    """Union of tight instance boxes (thesis Eq. 3.28); x2/y2 may reach ``W``/``H``."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        x1, y1 = int(np.clip(b[0], 0, W - 1)), int(np.clip(b[1], 0, H - 1))
        x2, y2 = int(np.clip(b[2], 0, W)), int(np.clip(b[3], 0, H))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def box_mask_in_perspective(box: Sequence[float], erp_w: int, sample_map: np.ndarray) -> np.ndarray:
    """Patch-space support ``m_p`` of an ERP box via the sample map (thesis Eq. 3.12).

    The horizontal test is wrap-aware: ``x2 > W`` denotes a circular interval and
    membership is ``(u - x1) mod W <= width``.
    """
    x1, y1, x2, y2 = box
    u = sample_map[..., 0]
    v = sample_map[..., 1]
    inside_u = (u - x1) % erp_w <= (x2 - x1) if x2 > erp_w else (u >= x1) & (u <= x2)
    inside = inside_u & (v >= y1) & (v <= y2)
    return inside.astype(np.uint8) * 255


def center_oval_mask(w: int, h: int) -> np.ndarray:
    """Rounded-rectangle support used for manual view hints (no box available)."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", (w, h), 0)
    mx, my = max(8, w // 8), max(8, h // 8)
    ImageDraw.Draw(mask).rounded_rectangle(
        (mx, my, w - mx, h - my), radius=min(w, h) // 10, fill=255
    )
    return np.asarray(mask, dtype=np.uint8)


def dilate_mask(binary_mask: np.ndarray, radius: int | tuple[int, int]) -> np.ndarray:
    """Elliptical dilation; ``radius`` may be ``(horizontal, vertical)``."""
    import cv2

    if isinstance(radius, (tuple, list)):
        rx, ry = int(radius[0]), int(radius[1])
    else:
        rx = ry = int(radius)
    if rx <= 0 and ry <= 0:
        return binary_mask
    rx, ry = max(rx, 1), max(ry, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
    return cv2.dilate(binary_mask, kernel)
