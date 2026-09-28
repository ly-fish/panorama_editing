"""Reference ROIs from Grounded-SAM benchmark annotations (``result_*_mask.json``).

Each annotation file is a list ``[{"value", "label", "logit", "box"}, ...]``
where ``value == 0`` is background.  The *reference ROI* of the controlled
protocol is the highest-logit non-background box, selected once per scene and
held constant across methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import DatasetError
from ..types import Detection

__all__ = ["Annotation", "annotation_mask", "load_annotations", "reference_detection"]


@dataclass(frozen=True)
class Annotation:
    value: int
    label: str
    logit: float
    box: tuple[float, float, float, float]


def load_annotations(path: str | Path) -> list[Annotation]:
    """Foreground annotations with a box (background entries are dropped)."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"annotation file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for d in rows:
        if isinstance(d, dict) and d.get("value", 0) != 0 and "box" in d:
            b = d["box"]
            out.append(
                Annotation(
                    int(d["value"]),
                    str(d.get("label", "object")),
                    float(d.get("logit", 0.0)),
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                )
            )
    return out


def reference_detection(
    path: str | Path, object_value: int | None = None, *, with_mask: bool = False
) -> Detection:
    """The protocol's reference ROI: highest-logit foreground box.

    ``with_mask`` additionally recovers a tight support from the colour-coded
    mask visualisation next to the JSON (see :func:`annotation_mask`); the
    thesis protocol uses box support only (``with_mask=False``).
    """
    anns = load_annotations(path)
    if object_value is not None:
        anns = [a for a in anns if a.value == object_value]
    if not anns:
        raise DatasetError(f"no foreground annotation with a box in {path}")
    best = max(anns, key=lambda a: a.logit)
    mask = annotation_mask(Path(path), best.box) if with_mask else None
    return Detection(best.box, score=best.logit, label=best.label, mask=mask, source="annotation")


def annotation_mask(
    json_path: Path, box: tuple[float, float, float, float], dilation_frac: float = 0.04
) -> np.ndarray | None:
    """Recover a tight ERP mask from the Grounded-SAM visualisation (G channel > 8).

    Foreground is kept only inside the (3 %-padded) box, dilated by
    ``dilation_frac`` of the box size and clipped to an 8 %-padded guard box.
    Returns ``None`` when the visualisation/panorama is missing or the recovered
    support is implausibly small (< 1 % of the box).
    """
    from ..geometry.masks import dilate_mask

    mask_path = json_path.with_suffix(".jpg")
    pano_path = json_path.parent / "panorama.jpg"
    if not mask_path.exists() or not pano_path.exists():
        return None
    with Image.open(pano_path) as pano:
        W, H = pano.size
    with Image.open(mask_path) as src:
        vis = src.convert("RGB")
        if vis.size != (W, H):
            vis = vis.resize((W, H), Image.Resampling.NEAREST)
        rgb = np.asarray(vis)
    fg = rgb[..., 1].astype(np.int16) > 8
    if fg.mean() <= 0.001:
        return None
    foreground = fg.astype(np.uint8) * 255
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    ix1, iy1 = int(np.clip(x1 - 0.03 * bw, 0, W - 1)), int(np.clip(y1 - 0.03 * bh, 0, H - 1))
    ix2, iy2 = int(np.clip(x2 + 0.03 * bw, 1, W)), int(np.clip(y2 + 0.03 * bh, 1, H))
    selected = np.zeros((H, W), dtype=np.uint8)
    selected[iy1:iy2, ix1:ix2] = foreground[iy1:iy2, ix1:ix2]
    if np.count_nonzero(selected) / max(1, (ix2 - ix1) * (iy2 - iy1)) < 0.01:
        return None
    selected = dilate_mask(
        selected, (max(2, round(bw * dilation_frac)), max(2, round(bh * dilation_frac)))
    )
    guard = np.zeros_like(selected)
    guard[
        int(np.clip(y1 - 0.08 * bh, 0, H - 1)) : int(np.clip(y2 + 0.08 * bh, 1, H)),
        int(np.clip(x1 - 0.08 * bw, 0, W - 1)) : int(np.clip(x2 + 0.08 * bw, 1, W)),
    ] = 255
    return np.where(guard > 0, selected, 0).astype(np.uint8)
