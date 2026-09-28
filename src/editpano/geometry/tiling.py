"""Narrow-FoV tiling of wide targets (thesis Eq. 3.26).

A patch near the 150 deg FoV clamp is itself strongly stretched towards its
borders.  :func:`split_box_into_tiles` divides a wide ordinary box into an
overlapping grid whose cells each span at most ``tile_span_deg``, so every
editor call sees a near-rectilinear view.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["angular_extent", "split_box_into_tiles"]


def angular_extent(box: Sequence[float], W: int, H: int) -> tuple[float, float]:
    """(horizontal, vertical) angular span of an ERP box in degrees."""
    return (box[2] - box[0]) / W * 360.0, (box[3] - box[1]) / H * 180.0


def _tile_count(span: float, tile_span_deg: float, overlap: float) -> int:
    # n overlapping tiles of width t cover t * (n - overlap * (n - 1)).
    if span <= tile_span_deg:
        return 1
    ratio = span / tile_span_deg
    return max(2, math.ceil((ratio - overlap) / (1.0 - overlap)))


def _intervals(start: float, end: float, n: int, overlap: float) -> list[tuple[float, float]]:
    if n == 1:
        return [(start, end)]
    width = (end - start) / (n - overlap * (n - 1))
    step = width * (1.0 - overlap)
    spans = []
    for i in range(n):
        lo = start + i * step
        hi = end if i == n - 1 else lo + width
        spans.append((max(start, lo), min(end, hi)))
    return spans


def split_box_into_tiles(
    box: Sequence[float],
    W: int,
    H: int,
    tile_span_deg: float = 60.0,
    overlap: float = 0.2,
    max_tiles: int = 6,
) -> list[list[float]]:
    """Split an ERP box into a row-major grid of overlapping sub-boxes.

    Only axes exceeding ``tile_span_deg`` are split; ``max_tiles`` caps the grid
    (shrinking the more-split axis first).  A box that already fits is returned
    unchanged as a single tile.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if tile_span_deg <= 0:
        raise ValueError("tile_span_deg must be positive")
    ang_w, ang_h = angular_extent(box, W, H)
    ncols = _tile_count(ang_w, tile_span_deg, overlap)
    nrows = _tile_count(ang_h, tile_span_deg, overlap)
    while ncols * nrows > max_tiles:
        if ncols >= nrows and ncols > 1:
            ncols -= 1
        elif nrows > 1:
            nrows -= 1
        else:
            break
    if ncols == 1 and nrows == 1:
        return [list(box)]
    x1, y1, x2, y2 = box
    tiles: list[list[float]] = []
    for ty1, ty2 in _intervals(y1, y2, nrows, overlap):
        for tx1, tx2 in _intervals(x1, x2, ncols, overlap):
            if tx2 - tx1 > 1 and ty2 - ty1 > 1:
                tiles.append([tx1, ty1, tx2, ty2])
    return tiles
