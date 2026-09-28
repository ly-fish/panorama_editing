"""The single-ROI EditPano operator ``L`` (thesis Eq. 3.1).

    I*, patch = L(I, t', b):
        (lat, lon, fov) = adaptive_view(b)                 # Eq. 3.3-3.5
        P, S            = gnomonic(I; lat, lon, fov)       # Eq. 3.6-3.11
        m_p             = support(b | m_erp; S)            # Eq. 3.12
        P_E             = E([P, Q(P, m_p)], t')            # frozen editor
        P_hat           = A_p * P_E + (1 - A_p) * P        # patch constraint
        R, Omega        = analytic_inverse(P_hat, m_p)     # Eq. 3.14-3.16
        I*              = M * R + (1 - M) * I              # inward composite, Eq. 3.17-3.18

Everything except the editor call is deterministic geometry.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..config import EditPanoConfig
from ..editors.base import EditRequest, PatchEditor
from ..geometry.camera import ViewParams
from ..geometry.compositing import composite_reprojection, constrain_patch_to_mask
from ..geometry.masks import box_mask_in_perspective, center_oval_mask
from ..geometry.projection import (
    erp_to_perspective,
    project_mask_to_perspective,
    remap_patch_to_erp,
)
from ..types import PatchRecord

__all__ = ["LocalEditOperator", "OperatorOutput"]


@dataclass
class OperatorOutput:
    """Result of one operator application (image is ``[H, W, 3]`` uint8)."""

    image: np.ndarray
    patches: list[PatchRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def editor_calls(self) -> int:
        return len(self.patches)


class LocalEditOperator:
    """Project -> edit -> constrain -> reproject -> composite, for one view."""

    def __init__(self, editor: PatchEditor, config: EditPanoConfig | None = None):
        self.editor = editor
        self.config = config or EditPanoConfig()

    # ------------------------------------------------------------ views
    def view_for_box(self, box: Sequence[float], width: int, height: int) -> ViewParams:
        p = self.config.projection
        make = ViewParams.from_circular_box if box[2] > width else ViewParams.from_box
        return make(
            box,
            width,
            height,
            margin=p.fov_margin,
            lat_aware=p.lat_aware_fov,
            fov_min=p.fov_min,
            fov_max=p.fov_max,
        )

    # ------------------------------------------------------------ public
    def edit_box(
        self,
        image: np.ndarray,
        box: Sequence[float],
        prompt: str,
        seed: int,
        erp_mask: np.ndarray | None = None,
    ) -> OperatorOutput:
        """Edit the target inside ``box`` (``x2 > W`` allowed for wrapped boxes).

        ``erp_mask`` (``[H, W]`` uint8) restricts the support; otherwise the box
        itself is the geometric support.
        """
        H, W = image.shape[:2]
        view = self.view_for_box(box, W, H)
        size = self.config.projection.patch_size
        patch, sample_map = erp_to_perspective(image, view, size, size)
        if erp_mask is not None:
            persp_mask = project_mask_to_perspective(erp_mask, sample_map)
        else:
            persp_mask = box_mask_in_perspective(box, W, sample_map)
        out = self._edit_view(image, view, patch, persp_mask, prompt, seed)
        out.patches[0].box = [float(v) for v in box]
        out.meta.update({"box": [float(v) for v in box], "used_erp_mask": erp_mask is not None})
        return out

    def edit_view(
        self, image: np.ndarray, view: ViewParams, prompt: str, seed: int
    ) -> OperatorOutput:
        """Edit a manually specified view (support = centred rounded rectangle)."""
        size = self.config.projection.patch_size
        patch, _ = erp_to_perspective(image, view, size, size)
        return self._edit_view(image, view, patch, center_oval_mask(size, size), prompt, seed)

    # ------------------------------------------------------------ core
    def _edit_view(
        self,
        image: np.ndarray,
        view: ViewParams,
        patch: np.ndarray,
        persp_mask: np.ndarray,
        prompt: str,
        seed: int,
    ) -> OperatorOutput:
        cfg = self.config
        H, W = image.shape[:2]
        src_img = Image.fromarray(patch)
        mask_img = Image.fromarray(persp_mask, mode="L")

        t0 = time.perf_counter()
        edited = self.editor.edit(EditRequest(src_img, prompt, mask_img, int(seed)))
        t_edit = time.perf_counter() - t0
        if cfg.composite.constrain_patch_to_mask:
            edited = constrain_patch_to_mask(
                src_img, edited, mask_img, feather=cfg.composite.patch_mask_feather
            )
        edited = edited.convert("RGB")
        if edited.size != src_img.size:
            edited = edited.resize(src_img.size, Image.Resampling.BICUBIC)

        content, support, visible = remap_patch_to_erp(np.asarray(edited), view, H, W, persp_mask)
        final = composite_reprojection(
            image, content, support, visible, cfg.composite.mode, cfg.composite.feather_px
        )
        record = PatchRecord(view=view, source=src_img, edited=edited, mask=mask_img)
        return OperatorOutput(
            final, [record], {"view": view.to_dict(), "edit_seconds": round(t_edit, 3)}
        )
