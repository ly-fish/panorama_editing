"""Seam-aware execution strategies built on the single-ROI operator.

``canonical`` is the EditPano path for supports that may wrap across the ERP
seam: roll image and support so the circular arc centre sits at ``u = W/2``,
edit once as an ordinary box, roll back (thesis Eq. 3.22)::

    F_can(I, t, m) = T_{-r(m)} L(T_{r(m)} I, t, B(T_{r(m)} m))

Given a deterministic editor this is *exactly* equivariant to integer longitude
shifts: ``F_can(T_q I, t, T_q m) == T_q F_can(I, t, m)`` (Eq. 3.24).

The other strategies are the controlled comparison arms of the seam study:

* ``identity``        -- no edit (negative control);
* ``current``         -- linear min/max box of the support (degenerates to full
  width when the support wraps);
* ``split``           -- edit the two stored fragments sequentially (right -> left);
* ``direct_circular`` -- circular centre/FoV and wrapped mask, no image roll.

Each strategy optionally tiles (``tiled=True``) wide boxes with narrow-FoV views.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np

from ..config import EditPanoConfig
from ..errors import DegenerateROIError
from ..geometry.circular import CircularROI, circular_roi_from_mask, roll_image
from ..geometry.tiling import split_box_into_tiles
from ..types import PatchRecord
from .local import LocalEditOperator, OperatorOutput

__all__ = ["SeamStrategy", "SeamStrategyRunner"]


class SeamStrategy(str, Enum):
    IDENTITY = "identity"
    CURRENT = "current"
    SPLIT = "split"
    DIRECT_CIRCULAR = "direct_circular"
    CANONICAL = "canonical"


class SeamStrategyRunner:
    """Execute a :class:`SeamStrategy` on an ERP image and a (possibly wrapped) support."""

    def __init__(self, operator: LocalEditOperator):
        self.op = operator

    @property
    def config(self) -> EditPanoConfig:
        return self.op.config

    # ------------------------------------------------------------ building blocks
    def edit_box(
        self,
        image: np.ndarray,
        box: Sequence[float],
        prompt: str,
        seed: int,
        tiled: bool = False,
        erp_mask: np.ndarray | None = None,
    ) -> OperatorOutput:
        """One ordinary box, optionally as a sequential chain of narrow tiles."""
        if not tiled:
            return self.op.edit_box(image, box, prompt, seed, erp_mask=erp_mask)
        H, W = image.shape[:2]
        t = self.config.tiling
        tiles = split_box_into_tiles(
            [float(v) for v in box], W, H, t.tile_span_deg, t.overlap, t.max_tiles
        )
        current, patches, views = image, [], []
        for tbox in tiles:
            out = self.op.edit_box(current, tbox, prompt, seed, erp_mask=erp_mask)
            current = out.image
            patches.extend(out.patches)
            views.append(out.meta["view"])
        return OperatorOutput(
            current, patches, {"box": [float(v) for v in box], "tiles": tiles, "views": views}
        )

    # ------------------------------------------------------------ strategies
    def run(
        self,
        strategy: SeamStrategy | str,
        image: np.ndarray,
        support: np.ndarray,
        prompt: str,
        seed: int = 0,
        tiled: bool = False,
    ) -> OperatorOutput:
        """Apply ``strategy``.

        Args:
            image: ``[H, W, 3]`` uint8 panorama.
            support: ``[H, W]`` binary target support; may wrap across the seam.
            prompt: effective editor prompt.
            seed: editor seed (matched across strategies for fair comparison).
            tiled: tile wide boxes with narrow-FoV views.
        """
        strategy = SeamStrategy(strategy)
        support = np.asarray(support)
        if support.ndim == 3:
            support = support.max(axis=-1)
        if support.shape != image.shape[:2]:
            raise ValueError(f"support shape {support.shape} != image {image.shape[:2]}")
        roi = circular_roi_from_mask(support)
        meta: dict = {"strategy": strategy.value, "tiled": tiled, "circular_roi": roi.to_dict()}
        if strategy is SeamStrategy.IDENTITY:
            return OperatorOutput(image.copy(), [], meta)
        if roi.degenerate:
            raise DegenerateROIError(roi.reason)

        if strategy is SeamStrategy.CURRENT:
            box = roi.naive_box(support)
            out = self.edit_box(image, box, prompt, seed, tiled)
            meta["naive_box"] = box
        elif strategy is SeamStrategy.SPLIT:
            boxes = sorted(roi.split_boxes(), key=lambda b: -b[0])  # right -> left
            current, patches = image, list[PatchRecord]()
            for b in boxes:
                o = self.edit_box(current, b, prompt, seed, tiled)
                current, patches = o.image, patches + o.patches
            out = OperatorOutput(current, patches, {})
            meta["split_boxes"] = boxes
        elif strategy is SeamStrategy.DIRECT_CIRCULAR:
            box = roi.circular_box()
            wrapped = np.where(support > 0, np.uint8(255), np.uint8(0))
            out = self.op.edit_box(image, box, prompt, seed, erp_mask=wrapped)
            meta["circular_box"] = box
        else:
            out = self.canonical(image, support, roi, prompt, seed, tiled)
            meta.update(out.meta)
        out.meta = {**out.meta, **meta}
        return out

    def canonical(
        self,
        image: np.ndarray,
        support: np.ndarray,
        roi: CircularROI,
        prompt: str,
        seed: int,
        tiled: bool = False,
        erp_mask: np.ndarray | None = None,
    ) -> OperatorOutput:
        """``T_{-r} . L . T_r`` -- edit the support once as a contiguous region."""
        r = roi.canonical_shift()
        support_c = roll_image(support, r)
        roi_c = circular_roi_from_mask(support_c)
        if roi_c.wrap:  # pragma: no cover - guarded by construction
            raise RuntimeError(f"canonical roll failed to unwrap the ROI (r={r})")
        box_c = roi_c.naive_box(support_c)
        mask_c = roll_image(erp_mask, r) if erp_mask is not None else None
        out = self.edit_box(roll_image(image, r), box_c, prompt, seed, tiled, erp_mask=mask_c)
        out.image = roll_image(out.image, -r)
        out.meta.update({"canonical_shift": int(r), "canonical_box": box_c})
        return out
