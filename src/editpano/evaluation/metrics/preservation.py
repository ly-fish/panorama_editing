"""Preservation and edit-activity metrics (thesis Eq. 3.29, A.1).

All metrics take ``[H, W, 3]`` uint8 arrays and a boolean target support ``F``;
the background is ``B = ~F``.  Edited images of a different size are resized to
the original with Lanczos (as in the evaluated protocol).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

__all__ = [
    "BG_PSNR_CAP",
    "COVERAGE_THRESHOLD",
    "background_changed_rate",
    "background_psnr",
    "background_ssim",
    "match_size",
    "target_coverage",
]

BG_PSNR_CAP = 99.0
COVERAGE_THRESHOLD = 12.0  # mean |dRGB| in uint8 units (= 12/255)


def match_size(original: np.ndarray, edited: np.ndarray) -> np.ndarray:
    """Resize ``edited`` to ``original``'s spatial size if they differ."""
    if original.shape == edited.shape:
        return edited
    h, w = original.shape[:2]
    return np.asarray(Image.fromarray(edited).resize((w, h), Image.Resampling.LANCZOS))


def background_psnr(
    original: np.ndarray,
    edited: np.ndarray,
    target: np.ndarray,
    cap: float = BG_PSNR_CAP,
    min_background: float = 0.05,
) -> float | None:
    """PSNR over background pixels, capped at ``cap`` dB (pixel-identical -> cap).

    Returns ``None`` when the background covers less than ``min_background`` of the image.
    """
    edited = match_size(original, edited)
    bg = ~np.asarray(target, dtype=bool)
    if bg.mean() < min_background:
        return None
    mse = float(np.mean((original[bg].astype(np.float32) - edited[bg].astype(np.float32)) ** 2))
    return cap if mse == 0 else min(cap, float(10 * np.log10(255.0**2 / mse)))


def background_ssim(original: np.ndarray, edited: np.ndarray, target: np.ndarray) -> float:
    """Mean of the full SSIM map over background pixels (saturation-prone diagnostic)."""
    from skimage.metrics import structural_similarity

    edited = match_size(original, edited)
    bg = ~np.asarray(target, dtype=bool)
    _, smap = structural_similarity(
        original.astype(np.float32),
        edited.astype(np.float32),
        channel_axis=2,
        data_range=255,
        full=True,
    )
    return float(smap[bg].mean())


def target_coverage(
    original: np.ndarray,
    edited: np.ndarray,
    target: np.ndarray,
    threshold: float = COVERAGE_THRESHOLD,
) -> float | None:
    """Fraction of target pixels whose mean absolute RGB change exceeds ``threshold``."""
    edited = match_size(original, edited)
    fg = np.asarray(target, dtype=bool)
    if not fg.any():
        return None
    delta = np.abs(original.astype(np.float32) - edited.astype(np.float32)).mean(axis=2)
    return float((delta[fg] > threshold).mean())


def background_changed_rate(
    original: np.ndarray,
    edited: np.ndarray,
    target: np.ndarray,
    threshold: float = COVERAGE_THRESHOLD,
) -> float:
    """Fraction of background pixels changed by more than ``threshold`` (leakage)."""
    edited = match_size(original, edited)
    bg = ~np.asarray(target, dtype=bool)
    delta = np.abs(original.astype(np.float32) - edited.astype(np.float32)).mean(axis=2)
    return float((delta[bg] > threshold).mean()) if bg.any() else math.nan
