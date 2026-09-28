"""Domain objects exchanged between localisation, operators and callers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry.camera import ViewParams

__all__ = ["Detection", "EditResult", "PatchRecord", "ViewHint"]


@dataclass(frozen=True)
class Detection:
    """A localised target in ERP pixel coordinates.

    ``box`` is ``[x1, y1, x2, y2]``; ``x2 > W`` denotes a box wrapping across the
    seam.  ``mask`` optionally restricts the composite support (``[H, W]`` uint8).
    """

    box: tuple[float, float, float, float]
    score: float = 1.0
    label: str = "object"
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    source: str = "caller"  # caller | annotation | grounding_dino | ...

    @staticmethod
    def from_box(
        box: Sequence[float],
        score: float = 1.0,
        label: str = "object",
        mask: np.ndarray | None = None,
        source: str = "caller",
    ) -> Detection:
        if len(box) != 4:
            raise ValueError(f"box must have 4 values, got {box!r}")
        x1, y1, x2, y2 = (float(v) for v in box)
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"box must satisfy x2 > x1 and y2 > y1, got {box!r}")
        return Detection((x1, y1, x2, y2), float(score), label, mask, source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [float(v) for v in self.box],
            "score": float(self.score),
            "label": self.label,
            "has_mask": self.mask is not None,
            "source": self.source,
        }


@dataclass(frozen=True)
class ViewHint:
    """Manual spherical target direction (degrees) when no box is available."""

    lon: float
    lat: float
    fov: float | None = None


@dataclass
class PatchRecord:
    """One editor call: the view, the rectilinear patch pair and its support."""

    view: ViewParams
    source: Image.Image
    edited: Image.Image
    mask: Image.Image
    box: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"view": self.view.to_dict(), "box": self.box}


@dataclass
class EditResult:
    """Output of :meth:`editpano.EditPano.edit`.

    Attributes:
        image: the edited ERP panorama ``I*``.
        branch: which operator produced it (``local``, ``canonical``, ``tiled``,
            ``canonical_tiled``, ``multi_instance`` or ``whole``).
        patches: every editor call made on a rectilinear patch.
        editor_calls: total number of editor invocations.
        meta: JSON-serialisable provenance (config fingerprint, ROI, views, timing).
    """

    image: Image.Image
    branch: str
    patches: list[PatchRecord] = field(default_factory=list)
    editor_calls: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path, *, intermediates: bool = False) -> Path:
        """Save the panorama (and optionally patches + ``*_meta.json``) next to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)
        if intermediates:
            for i, rec in enumerate(self.patches):
                rec.source.save(path.with_name(f"{path.stem}_patch{i}_source.png"))
                rec.edited.save(path.with_name(f"{path.stem}_patch{i}_edited.png"))
                rec.mask.save(path.with_name(f"{path.stem}_patch{i}_mask.png"))
            path.with_name(f"{path.stem}_meta.json").write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "editor_calls": self.editor_calls,
            "patches": [p.to_dict() for p in self.patches],
            "meta": self.meta,
        }
