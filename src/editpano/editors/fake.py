"""Deterministic CPU editors for tests, dry runs and CI (no model weights)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .base import EditorInfo, EditRequest
from .registry import register_editor

__all__ = ["FakeEditor"]


class FakeEditor:
    """``identity`` returns the input, ``invert`` inverts it, ``pattern`` paints a
    seeded diagonal pattern inside the mask.  Used to prove geometric properties
    (preservation, equivariance) independently of any generative model."""

    KINDS = ("identity", "invert", "pattern")

    def __init__(self, kind: str = "pattern", **_: Any):
        if kind not in self.KINDS:
            raise ValueError(f"fake editor kind must be one of {self.KINDS}")
        self.kind = kind

    @property
    def info(self) -> EditorInfo:
        return EditorInfo(name=f"fake:{self.kind}")

    def edit(self, request: EditRequest) -> Image.Image:
        image = request.image
        if self.kind == "identity":
            return image.copy()
        arr = np.asarray(image.convert("RGB")).astype(np.int32)
        if self.kind == "invert":
            return Image.fromarray((255 - arr).astype(np.uint8))
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        pattern = ((xx + yy + int(request.seed)) % 32 < 16).astype(np.uint8) * 255
        out = arr.copy()
        out[..., 0] = pattern
        out[..., 1] = 255 - pattern
        out[..., 2] = pattern // 2 + 64
        if request.mask is not None:
            m = np.asarray(request.mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
            keep = m <= 127
            out[keep] = arr[keep]
        return Image.fromarray(out.astype(np.uint8))


for _kind in FakeEditor.KINDS:
    register_editor(f"fake:{_kind}")(lambda _k=_kind, **kw: FakeEditor(_k, **kw))
