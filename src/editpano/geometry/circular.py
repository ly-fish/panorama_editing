"""Circular (S^1) target support and canonical longitude shifts (thesis §3.3.3).

ERP longitude is periodic, so the horizontal extent of a target is an *arc*.
:class:`CircularROI` stores the shortest covering arc of a support mask:

* ``x_start`` -- first arc column in ``[0, W)``;
* ``width``   -- arc length in pixels, half-open range ``[x_start, x_start + width) mod W``;
* ``wrap``    -- whether the arc crosses the ``u = 0`` seam.

The canonical shift ``r = round_half_up(W/2 - c) mod W`` moves the arc centre to
``u = W/2`` and satisfies ``r(T_q M) = r(M) - q (mod W)`` exactly, which yields
integer seam-position equivariance of the canonical operator
``F_can = T_{-r} . L . T_r`` (thesis Eq. 3.21-3.23).

Rolls follow ``np.roll(img, shift, axis=1)``: column ``u`` moves to ``(u + shift) mod W``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "REASON_EMPTY",
    "REASON_FULL_WIDTH",
    "REASON_OK",
    "REASON_TIE_MAX_GAP",
    "CircularROI",
    "circular_distance",
    "circular_roi_from_box",
    "circular_roi_from_mask",
    "normalize_shift",
    "roll_image",
    "round_half_up",
]

REASON_OK = "ok"
REASON_EMPTY = "empty_mask"
REASON_FULL_WIDTH = "full_width"
REASON_TIE_MAX_GAP = "tie_max_gap"


def round_half_up(x: float) -> int:
    """``floor(x + 0.5)`` -- unlike banker's rounding it commutes with integer shifts."""
    return math.floor(x + 0.5)


def normalize_shift(shift: int, W: int) -> int:
    """Normalise a roll amount to the signed range ``[-W/2, W/2)``."""
    s = int(shift) % W
    if s >= W - W // 2:
        s -= W
    return s


def roll_image(arr: np.ndarray, shift: int) -> np.ndarray:
    """Interpolation-free horizontal circular shift ``T_q``."""
    if shift % arr.shape[1] == 0:
        return arr.copy()
    return np.roll(arr, shift, axis=1)


def circular_distance(u: np.ndarray | float, c: float, W: int) -> np.ndarray:
    """Periodic distance ``min(|u - c|, W - |u - c|)``."""
    d = np.abs(np.asarray(u, dtype=np.float64) - c) % W
    return np.minimum(d, W - d)


@dataclass(frozen=True)
class CircularROI:
    """Shortest covering arc of a binary target support on the ERP circle."""

    x_start: int
    width: int
    y1: int
    y2: int
    erp_width: int
    erp_height: int
    reason: str = REASON_OK
    tie_count: int = field(default=1)

    @property
    def wrap(self) -> bool:
        return self.x_start + self.width > self.erp_width

    @property
    def center(self) -> float:
        """Circular centre column (multiple of 0.5) in ``[0, W)``."""
        return (self.x_start + self.width / 2.0) % self.erp_width

    @property
    def span_deg(self) -> float:
        return self.width / self.erp_width * 360.0

    @property
    def degenerate(self) -> bool:
        return self.reason in (REASON_EMPTY, REASON_FULL_WIDTH)

    def circular_box(self) -> list[int]:
        """``[x_start, y1, x_start + width, y2]`` -- ``x2`` may exceed ``W``."""
        return [self.x_start, self.y1, self.x_start + self.width, self.y2]

    def split_boxes(self) -> list[list[int]]:
        """Half-open ordinary boxes; two fragments when the arc wraps."""
        W = self.erp_width
        if self.reason == REASON_EMPTY:
            return []
        if not self.wrap:
            return [[self.x_start, self.y1, self.x_start + self.width, self.y2]]
        right_w = W - self.x_start
        return [[self.x_start, self.y1, W, self.y2], [0, self.y1, self.width - right_w, self.y2]]

    def naive_box(self, mask: np.ndarray | None = None) -> list[int]:
        """Linear min/max box (seam-unaware; degenerates to full width for wrapped arcs)."""
        if mask is not None:
            cols = np.flatnonzero(np.asarray(mask).max(axis=0) > 0)
            if cols.size == 0:
                return [0, self.y1, 0, self.y2]
            return [int(cols[0]), self.y1, int(cols[-1]) + 1, self.y2]
        if not self.wrap:
            return [self.x_start, self.y1, self.x_start + self.width, self.y2]
        return [0, self.y1, self.erp_width, self.y2]

    def canonical_shift(self) -> int:
        """Integer roll moving the arc centre to ``u = W/2`` (signed, ``[-W/2, W/2)``)."""
        W = self.erp_width
        return normalize_shift(round_half_up(W / 2.0 - self.center), W)

    def rolled(self, shift: int) -> CircularROI:
        W = self.erp_width
        return CircularROI(
            int((self.x_start + shift) % W),
            self.width,
            self.y1,
            self.y2,
            W,
            self.erp_height,
            self.reason,
            self.tie_count,
        )

    def contains_u(self, u: np.ndarray | float) -> np.ndarray:
        """Wrap-aware horizontal membership ``(u - x_start) mod W < width``."""
        return (np.asarray(u) - self.x_start) % self.erp_width < self.width

    def rasterize(self) -> np.ndarray:
        mask = np.zeros((self.erp_height, self.erp_width), dtype=np.uint8)
        for x1, y1, x2, y2 in self.split_boxes():
            mask[y1:y2, x1:x2] = 255
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_start": int(self.x_start),
            "width": int(self.width),
            "y1": int(self.y1),
            "y2": int(self.y2),
            "erp_width": int(self.erp_width),
            "erp_height": int(self.erp_height),
            "wrap": bool(self.wrap),
            "center": float(self.center),
            "span_deg": float(self.span_deg),
            "reason": self.reason,
            "tie_count": int(self.tie_count),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CircularROI:
        return CircularROI(
            int(d["x_start"]),
            int(d["width"]),
            int(d["y1"]),
            int(d["y2"]),
            int(d["erp_width"]),
            int(d["erp_height"]),
            str(d.get("reason", REASON_OK)),
            int(d.get("tie_count", 1)),
        )


def circular_roi_from_mask(mask: np.ndarray) -> CircularROI:
    """Shortest covering arc: complement of the longest circular run of empty columns.

    Degenerate supports keep an explicit reason (empty / full width); tied maximal
    gaps are resolved deterministically by the smallest arc start column.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    H, W = mask.shape[:2]
    cols = np.flatnonzero(mask.max(axis=0) > 0)
    rows = np.flatnonzero(mask.max(axis=1) > 0)
    y1 = int(rows[0]) if rows.size else 0
    y2 = int(rows[-1]) + 1 if rows.size else 0
    if cols.size == 0:
        return CircularROI(0, 0, y1, y2, W, H, reason=REASON_EMPTY)
    if cols.size == W:
        return CircularROI(0, W, y1, y2, W, H, reason=REASON_FULL_WIDTH)

    nxt = np.roll(cols, -1)
    gaps = (nxt - cols - 1) % W  # empty run after each occupied column
    max_gap = int(gaps.max())
    tied = np.flatnonzero(gaps == max_gap)
    starts = (cols[tied] + gaps[tied] + 1) % W
    x_start = int(starts[int(np.argmin(starts))])
    return CircularROI(
        x_start,
        W - max_gap,
        y1,
        y2,
        W,
        H,
        reason=REASON_OK if tied.size == 1 else REASON_TIE_MAX_GAP,
        tie_count=int(tied.size),
    )


def circular_roi_from_box(box: Sequence[float], erp_width: int, erp_height: int) -> CircularROI:
    """CircularROI from a half-open box; ``x1 > x2`` or ``x2 > W`` denotes wrapping."""
    x1, y1, x2, y2 = box
    x1i = math.floor(x1) % erp_width
    y1i = max(0, math.floor(y1))
    y2i = min(erp_height, math.ceil(y2))
    if x2 > x1:
        width = math.ceil(x2) - math.floor(x1)
    else:
        width = math.ceil(x2) % erp_width + erp_width - math.floor(x1)
    width = min(width, erp_width)
    if width <= 0:
        return CircularROI(0, 0, y1i, y2i, erp_width, erp_height, reason=REASON_EMPTY)
    if width >= erp_width:
        return CircularROI(0, erp_width, y1i, y2i, erp_width, erp_height, reason=REASON_FULL_WIDTH)
    return CircularROI(x1i, width, y1i, y2i, erp_width, erp_height)
