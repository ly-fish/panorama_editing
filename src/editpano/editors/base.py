"""The editor port: EditPano treats any frozen perspective editor as a black box.

An editor only has to map ``(patch, prompt, optional patch mask, seed)`` to an
edited patch.  No gradients, activations or weight updates are required, which
is what makes the geometric framework editor-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

__all__ = ["EditRequest", "EditorInfo", "PatchEditor"]


@dataclass(frozen=True)
class EditRequest:
    """Input to a frozen editor.

    Attributes:
        image: RGB image to edit -- a rectilinear patch, or the whole ERP for
            the whole-image baseline.
        prompt: effective editor prompt ``t'``.
        mask: editable support in ``image`` coordinates (mode ``L``, 255 = edit),
            or ``None`` for unconstrained whole-image editing.
        seed: generator seed; editors must be deterministic given the seed.
    """

    image: Image.Image
    prompt: str
    mask: Image.Image | None = None
    seed: int = 0


@dataclass(frozen=True)
class EditorInfo:
    """Provenance recorded in run signatures."""

    name: str
    model_id: str | None = None
    revision: str | None = None


@runtime_checkable
class PatchEditor(Protocol):
    """Structural interface every editor backend implements."""

    @property
    def info(self) -> EditorInfo: ...

    def edit(self, request: EditRequest) -> Image.Image: ...
