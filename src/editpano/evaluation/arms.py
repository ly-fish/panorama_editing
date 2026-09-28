"""Comparison arms of the reference-ROI benchmark (thesis §3.4.1).

All arms share the same inputs, instruction, reference box and frozen editor;
only the geometry around the editor changes.

* ``whole_erp``         -- the editor rewrites the complete ERP (no support constraint)
* ``erp_crop``          -- edit the axis-aligned box crop without un-distortion, paste back
* ``whole_then_paste``  -- whole-ERP edit, then copy back only the box
* ``editpano``          -- project -> edit -> analytic reproject -> inward composite
* ``editpano_tiled``    -- EditPano with narrow-FoV tiles for every target
* ``editpano_detected`` -- EditPano with the ROI predicted by frozen Grounding DINO
* ``editpano_multi``    -- EditPano over every annotated instance of the target label
* ``cubemap``           -- edit the fixed 90 deg cube face closest to the target
* ``sd_inpaint_roi``    -- mask-conditioned SD inpainting on the ERP, box pasted back
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..editors.base import EditRequest
from ..errors import ConfigError
from ..geometry.camera import ViewParams, great_circle_deg, pixel_to_lonlat
from ..geometry.compositing import composite_reprojection, paste_box
from ..geometry.masks import bbox_to_binary_mask, clamp_box
from ..geometry.projection import (
    erp_to_perspective,
    project_mask_to_perspective,
    remap_patch_to_erp,
)
from ..pipeline.editpano import EditPano
from .datasets import Case

__all__ = ["ARMS", "Arm", "ArmContext", "cube_face_for_box"]

CUBE_FACES = [(0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, -90.0), (90.0, 0.0), (-90.0, 0.0)]


@dataclass
class ArmContext:
    """Everything an arm needs: the configured pipeline and outputs of prerequisite arms."""

    pano: EditPano
    image: np.ndarray
    box: tuple[float, float, float, float]
    case: Case
    seed: int
    upstream: dict[str, np.ndarray]
    artifacts: dict[str, Image.Image] = field(default_factory=dict)  # saved next to edited.png


@dataclass(frozen=True)
class Arm:
    name: str
    run: Callable[[ArmContext], np.ndarray]
    requires: tuple[str, ...] = ()
    description: str = ""


def _whole(ctx: ArmContext) -> np.ndarray:
    return ctx.pano._whole(ctx.image, ctx.pano.prompt_policy(ctx.case.prompt), ctx.seed).image


def _editpano(ctx: ArmContext) -> np.ndarray:
    result = ctx.pano.edit(ctx.image, ctx.case.prompt, box=ctx.box, seed=ctx.seed)
    for i, rec in enumerate(result.patches):
        ctx.artifacts["local_patch.png" if i == 0 else f"local_patch_{i}.png"] = rec.edited
    return np.asarray(result.image)


def _editpano_tiled(ctx: ArmContext) -> np.ndarray:
    tiled = EditPano(
        ctx.pano.config.with_overrides({"tiling.mode": "always"}),
        editor=ctx.pano.editor,
        prompt_policy=ctx.pano.prompt_policy,
    )
    return np.asarray(tiled.edit(ctx.image, ctx.case.prompt, box=ctx.box, seed=ctx.seed).image)


def _erp_crop(ctx: ArmContext) -> np.ndarray:
    H, W = ctx.image.shape[:2]
    x1, y1, x2, y2 = clamp_box(ctx.box, W, H)
    orig = Image.fromarray(ctx.image)
    edited = ctx.pano.editor.edit(
        EditRequest(
            orig.crop((x1, y1, x2, y2)), ctx.pano.prompt_policy(ctx.case.prompt), None, ctx.seed
        )
    )
    return np.asarray(
        paste_box(orig, edited, (x1, y1, x2, y2), ctx.pano.config.composite.feather_px)
    )


def _whole_then_paste(ctx: ArmContext) -> np.ndarray:
    H, W = ctx.image.shape[:2]
    whole = Image.fromarray(ctx.upstream["whole_erp"])
    x1, y1, x2, y2 = clamp_box(ctx.box, W, H)
    return np.asarray(
        paste_box(
            Image.fromarray(ctx.image),
            whole.crop((x1, y1, x2, y2)),
            (x1, y1, x2, y2),
            ctx.pano.config.composite.feather_px,
        )
    )


def cube_face_for_box(box: tuple[float, float, float, float], W: int, H: int) -> ViewParams:
    """The fixed 90 deg cube face whose centre is closest to the box centre."""
    lon0, lat0 = pixel_to_lonlat((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, W, H)
    flat, flon = min(CUBE_FACES, key=lambda f: great_circle_deg(lat0, lon0, f[0], f[1]))
    return ViewParams(lat=flat, lon=flon, fov=90.0)


def _cubemap(ctx: ArmContext) -> np.ndarray:
    H, W = ctx.image.shape[:2]
    size = ctx.pano.config.projection.patch_size
    view = cube_face_for_box(ctx.box, W, H)
    face, sample_map = erp_to_perspective(ctx.image, view, size, size)
    persp_mask = project_mask_to_perspective(bbox_to_binary_mask(ctx.box, H, W), sample_map)
    edited = ctx.pano.editor.edit(
        EditRequest(Image.fromarray(face), ctx.pano.prompt_policy(ctx.case.prompt), None, ctx.seed)
    )
    edited = edited.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    content, support, visible = remap_patch_to_erp(np.asarray(edited), view, H, W, persp_mask)
    return composite_reprojection(
        ctx.image, content, support, visible, "feather", ctx.pano.config.composite.feather_px
    )


def _sd_inpaint_roi(ctx: ArmContext) -> np.ndarray:
    """Needs an ``sd_inpaint`` editor; prompt = ``case.extra['tgt']`` when present (English)."""
    H, W = ctx.image.shape[:2]
    x1, y1, x2, y2 = clamp_box(ctx.box, W, H)
    h = max(8, round(min(512, H) / 8.0) * 8)
    w = max(8, round(int(h * W / H) / 8.0) * 8)
    orig = Image.fromarray(ctx.image)
    mask = Image.new("L", (W, H), 0)
    mask.paste(255, (x1, y1, x2, y2))
    prompt = str(ctx.case.extra.get("tgt") or ctx.case.prompt)
    inpaint = getattr(ctx.pano.editor, "inpaint", None)
    if inpaint is None:
        raise ConfigError("the sd_inpaint_roi arm needs the 'sd_inpaint' editor")
    small = inpaint(
        orig.resize((w, h), Image.Resampling.LANCZOS),
        mask.resize((w, h), Image.Resampling.NEAREST),
        prompt,
        ctx.seed,
    )
    out = small.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    return np.asarray(
        paste_box(
            orig, out.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), ctx.pano.config.composite.feather_px
        )
    )


def _editpano_detected(ctx: ArmContext) -> np.ndarray:
    """End-to-end: the target is localised by the frozen detector from ``object_label``."""
    result = ctx.pano.edit(ctx.image, ctx.case.prompt, target=ctx.case.object_label, seed=ctx.seed)
    return np.asarray(result.image)


def _editpano_multi(ctx: ArmContext) -> np.ndarray:
    """Every annotated instance sharing the reference label (score with ``instance_support``)."""
    from ..localization.annotations import load_annotations, reference_detection

    label = reference_detection(ctx.case.annotations).label
    boxes = [list(a.box) for a in load_annotations(ctx.case.annotations) if a.label == label]
    return np.asarray(
        ctx.pano.edit(ctx.image, ctx.case.prompt, instances=boxes, seed=ctx.seed).image
    )


ARMS: dict[str, Arm] = {
    a.name: a
    for a in [
        Arm("whole_erp", _whole, description="editor on the complete ERP"),
        Arm("erp_crop", _erp_crop, description="box crop without un-distortion"),
        Arm("whole_then_paste", _whole_then_paste, ("whole_erp",), "whole edit, box pasted back"),
        Arm("editpano", _editpano, description="EditPano (single ROI)"),
        Arm("editpano_tiled", _editpano_tiled, description="EditPano with narrow-FoV tiles"),
        Arm("editpano_detected", _editpano_detected, description="EditPano, Grounding DINO ROI"),
        Arm("editpano_multi", _editpano_multi, description="EditPano, all annotated instances"),
        Arm("cubemap", _cubemap, description="fixed 90 deg cube face"),
        Arm("sd_inpaint_roi", _sd_inpaint_roi, description="SD inpainting on the ERP box"),
    ]
}


def arm_order(names: list[str]) -> list[str]:
    """Dependency-respecting order (prerequisites first)."""
    ordered: list[str] = []

    def visit(n: str) -> None:
        if n in ordered:
            return
        for r in ARMS[n].requires:
            visit(r)
        ordered.append(n)

    for n in names:
        visit(n)
    return ordered


def describe() -> dict[str, Any]:
    return {n: a.description for n, a in ARMS.items()}
