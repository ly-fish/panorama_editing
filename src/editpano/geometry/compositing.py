"""Support-restricted compositing (thesis Eq. 3.12, 3.17-3.18).

The default ``inward`` mode keeps the *entire* alpha ramp inside the edit
support: ``M = 1_Omega * smoothstep(clip(DT(1_Omega) / r, 0, 1))``, so every
pixel outside ``Omega`` is copied from the input bit-exactly.  ``feather``
(outward Gaussian, legacy) and ``poisson`` (gradient domain) are kept as
ablation arms.  All modes are seam-aware: a support that wraps across the
+-180 deg meridian is rolled contiguous before any neighbourhood operation.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "CompositeMode",
    "composite_reprojection",
    "constrain_patch_to_mask",
    "gaussian_feather_blend",
    "inward_feather_alpha",
    "inward_feather_blend",
    "paste_box",
    "poisson_blend",
    "seam_unwrap_shift",
]


class CompositeMode(str, Enum):
    INWARD = "inward"
    FEATHER = "feather"
    POISSON = "poisson"


def seam_unwrap_shift(mask: np.ndarray) -> int | None:
    """Roll amount that makes a seam-touching mask contiguous.

    Returns ``0`` when the mask does not touch both borders, the positive roll
    otherwise, and ``None`` when every column is occupied (cannot unwrap).
    """
    occupied = mask.max(axis=0) > 0
    if not (occupied[0] and occupied[-1]):
        return 0
    empty = np.flatnonzero(~occupied)
    if empty.size == 0:
        return None
    return int(mask.shape[1] - int(empty[0]) - 1)


def inward_feather_alpha(mask01: np.ndarray, feather_px: int) -> np.ndarray:
    """Chamfer-DT smoothstep alpha that is 0 on/outside the boundary, 1 deep inside."""
    import cv2

    if feather_px <= 0:
        return mask01.astype(np.float32)
    dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 3)
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def inward_feather_blend(
    base: np.ndarray, content: np.ndarray, mask: np.ndarray, feather_px: int = 8
) -> np.ndarray:
    """Composite ``content`` over ``base`` inside ``mask`` (uint8 0/255) with an inward ramp."""
    mask = (mask > 127).astype(np.uint8)
    if mask.max() == 0:
        return base.copy()
    shift = seam_unwrap_shift(mask) or 0
    if shift:
        mask = np.roll(mask, shift, axis=1)
        base = np.roll(base, shift, axis=1)
        content = np.roll(content, shift, axis=1)
    alpha = inward_feather_alpha(mask, feather_px)
    out = content.astype(np.float32) * alpha[..., None] + base.astype(np.float32) * (
        1.0 - alpha[..., None]
    )
    out = out.clip(0, 255).astype(np.uint8)
    keep = mask == 0
    out[keep] = base[keep]
    if shift:
        out = np.roll(out, -shift, axis=1)
    return out


def poisson_blend(base: np.ndarray, content: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gradient-domain composite (``cv2.seamlessClone``); background bit-exact."""
    import cv2

    mask = (mask > 127).astype(np.uint8) * 255
    if mask.max() == 0:
        return base.copy()
    shift = seam_unwrap_shift(mask)
    if shift is None:
        return base.copy()
    if shift:
        mask = np.roll(mask, shift, axis=1)
        base_r, content_r = np.roll(base, shift, axis=1), np.roll(content, shift, axis=1)
    else:
        base_r, content_r = base, content
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    if mask.max() == 0:
        return base.copy()
    x, y, w, h = cv2.boundingRect(mask)
    out = cv2.seamlessClone(
        np.ascontiguousarray(content_r),
        np.ascontiguousarray(base_r),
        mask,
        (x + w // 2, y + h // 2),
        cv2.NORMAL_CLONE,
    )
    keep = mask == 0
    out[keep] = base_r[keep]
    if shift:
        out = np.roll(out, -shift, axis=1)
    return out


def _smooth_horizontal_seam(img: np.ndarray) -> np.ndarray:
    result = img.copy()
    avg = (img[:, 0, :].astype(np.float32) + img[:, -1, :].astype(np.float32)) / 2.0
    result[:, 0, :] = avg
    result[:, -1, :] = avg
    return result


def gaussian_feather_blend(
    base: np.ndarray, content: np.ndarray, support: np.ndarray, feather_px: int
) -> np.ndarray:
    """Legacy outward Gaussian alpha blend (leaks up to ~3 sigma outside the support)."""
    import cv2

    blend: np.ndarray = support.astype(np.float32)
    if feather_px > 0:
        blend = cv2.GaussianBlur(blend, (0, 0), feather_px)
    b3 = blend[..., np.newaxis]
    out = content * b3 + base.astype(np.float32) * (1.0 - b3)
    if blend[:, 0].max() > 0.05 or blend[:, -1].max() > 0.05:
        out = _smooth_horizontal_seam(out)
    return out.clip(0, 255).astype(np.uint8)


def composite_reprojection(
    base: np.ndarray,
    content: np.ndarray,
    support: np.ndarray,
    visible: np.ndarray,
    mode: CompositeMode | str = CompositeMode.INWARD,
    feather_px: int = 8,
) -> np.ndarray:
    """Composite an analytically reprojected patch into the panorama.

    Args:
        base: ``[H, W, 3]`` uint8 input panorama ``I``.
        content: ``[H, W, 3]`` float32 reprojected patch ``R``.
        support: ``[H, W]`` float32 binary ERP support.
        visible: ``[H, W]`` bool camera frustum.
        mode: composite operator.
        feather_px: ramp depth ``r`` (inward) or Gaussian sigma (feather).
    """
    import cv2

    mode = CompositeMode(mode)
    if mode in (CompositeMode.INWARD, CompositeMode.POISSON):
        # Keep the composite strictly inside the eroded frustum so the blend
        # never references the black remap border.
        inner = (support > 0.5) & (
            cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        )
        bin_mask = inner.astype(np.uint8) * 255
        content_u8 = content.clip(0, 255).astype(np.uint8)
        if mode is CompositeMode.INWARD:
            return inward_feather_blend(base, content_u8, bin_mask, feather_px=feather_px)
        try:
            return poisson_blend(base, content_u8, bin_mask)
        except cv2.error:
            pass
    return gaussian_feather_blend(base, content, support, feather_px)


def constrain_patch_to_mask(
    source: Image.Image, edited: Image.Image, mask: Image.Image, feather: int = 3
) -> Image.Image:
    """Patch-space output constraint ``P_hat = A_p * P_E + (1 - A_p) * P`` (thesis Eq. 3.12)."""
    source = source.convert("RGB")
    edited = edited.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    soft = mask.convert("L").resize(source.size, Image.Resampling.NEAREST)
    if feather > 0:
        soft = soft.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return Image.composite(edited, source, soft)


def paste_box(
    base: Image.Image, patch: Image.Image, box: tuple[int, int, int, int], feather: int
) -> Image.Image:
    """Feathered rectangular paste (used by the ERP-crop / whole+paste baselines)."""
    base = base.convert("RGB").copy()
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    patch = patch.convert("RGB")
    if patch.size != (w, h):
        patch = patch.resize((w, h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (w, h), 255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    region = base.crop((x1, y1, x2, y2))
    base.paste(Image.composite(patch, region, mask), (x1, y1))
    return base
