"""Deterministic synthetic panoramas and supports shared by the tests."""

from __future__ import annotations

import numpy as np


def synthetic_pano(W: int = 1024, H: int = 512, seed: int = 0) -> np.ndarray:
    """Longitude-periodic smooth panorama plus noise (the seam itself is unremarkable)."""
    rng = np.random.default_rng(seed)
    u = np.arange(W)[None, :, None]
    v = np.arange(H)[:, None, None]
    base = 120 + 60 * np.sin(2 * np.pi * u / W + np.array([0, 2, 4])) + 40 * np.cos(np.pi * v / H)
    return np.clip(base + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)


def wrapped_mask(W: int, H: int, left: int, right: int, y1: int, y2: int) -> np.ndarray:
    """Support covering ``[W-left, W) U [0, right)`` columns and rows ``[y1, y2)``."""
    m = np.zeros((H, W), np.uint8)
    m[y1:y2, W - left :] = 255
    m[y1:y2, :right] = 255
    return m
