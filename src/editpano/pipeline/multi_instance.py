"""Repeated-target editing (thesis Eq. 3.27-3.28).

Detected instances are greedily merged into compact groups whose joint view
stays inside a FoV budget; each group is edited once with the single-ROI
operator on the running panorama.  The editor sees the whole group patch but
only the union of tight instance boxes is composited, so gaps between
instances keep their original pixels.  A containment gate collapses to one
edit when the dominant instance's view already contains most instance centres.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import MultiInstanceConfig
from ..geometry.camera import ViewParams, great_circle_deg, pixel_to_lonlat
from ..geometry.masks import instance_union_mask
from ..types import PatchRecord
from .local import LocalEditOperator, OperatorOutput

__all__ = ["MultiInstanceEditor", "cluster_instances", "contained_fraction", "group_bbox"]

Box = Sequence[float]


def group_bbox(group: Sequence[Box]) -> list[float]:
    return [
        min(b[0] for b in group),
        min(b[1] for b in group),
        max(b[2] for b in group),
        max(b[3] for b in group),
    ]


def _group_fov(group: Sequence[Box], W: int, H: int, margin: float) -> float:
    # Grouping uses the plain (non latitude-aware) ERP span, as evaluated.
    return ViewParams.from_box(group_bbox(group), W, H, margin=margin, lat_aware=False).fov


def _center_latlon(box: Box, W: int, H: int) -> tuple[float, float]:
    lon, lat = pixel_to_lonlat(0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]), W, H)
    return lat, lon


def cluster_instances(
    boxes: Sequence[Box],
    W: int,
    H: int,
    fov_budget: float = 110.0,
    margin: float = 1.5,
    seam_frac: float = 0.6,
) -> list[list[list[float]]]:
    """Greedy agglomerative grouping: always take the merge with the smallest FoV,
    provided it stays below ``fov_budget`` and does not straddle the seam."""
    clusters: list[list[list[float]]] = [[list(b)] for b in boxes]
    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                merged = clusters[i] + clusters[j]
                xs = [b[0] for b in merged] + [b[2] for b in merged]
                if max(xs) - min(xs) > seam_frac * W:
                    continue
                fov = _group_fov(merged, W, H, margin)
                if fov < fov_budget and (best is None or fov < best[0]):
                    best = (fov, i, j)
        if best is None:
            break
        _, i, j = best
        merged = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    return clusters


def contained_fraction(boxes: Sequence[Box], dominant: Box, W: int, H: int, margin: float) -> float:
    """Fraction of instance centres inside the dominant box's adaptive view."""
    half = 0.5 * ViewParams.from_box(dominant, W, H, margin=margin, lat_aware=False).fov
    lat0, lon0 = _center_latlon(dominant, W, H)
    n = sum(1 for b in boxes if great_circle_deg(*_center_latlon(b, W, H), lat0, lon0) <= half)
    return n / max(1, len(boxes))


class MultiInstanceEditor:
    def __init__(self, operator: LocalEditOperator, config: MultiInstanceConfig | None = None):
        self.op = operator
        self.cfg = config or operator.config.multi_instance

    def plan(
        self, boxes: Sequence[Box], W: int, H: int, primary_box: Box | None = None
    ) -> list[list[list[float]]]:
        c = self.cfg
        groups = cluster_instances(boxes, W, H, c.fov_budget, c.margin, c.seam_frac)
        if len(boxes) > 1 and len(groups) > 1:
            dominant = primary_box or max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            if contained_fraction(boxes, dominant, W, H, c.margin) >= c.gate_thresh:
                groups = [[list(b) for b in boxes]]
        if len(groups) > c.max_groups:

            def area(g: list[list[float]]) -> float:
                bb = group_bbox(g)
                return (bb[2] - bb[0]) * (bb[3] - bb[1])

            groups = sorted(groups, key=area, reverse=True)[: c.max_groups]
        return groups

    def edit(
        self,
        image: np.ndarray,
        boxes: Sequence[Box],
        prompt: str,
        seed: int,
        primary_box: Box | None = None,
    ) -> OperatorOutput:
        H, W = image.shape[:2]
        groups = self.plan(boxes, W, H, primary_box)
        current, patches = image, list[PatchRecord]()
        for g in groups:
            mask = instance_union_mask(g, W, H) if self.cfg.use_instance_mask else None
            out = self.op.edit_box(
                current, [float(v) for v in group_bbox(g)], prompt, seed, erp_mask=mask
            )
            current, patches = out.image, patches + out.patches
        return OperatorOutput(current, patches, {"groups": groups, "n_instances": len(boxes)})
