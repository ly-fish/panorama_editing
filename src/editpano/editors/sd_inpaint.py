"""Frozen Stable-Diffusion-1.5 inpainting adapter (second editor in the thesis).

Mask-conditioned: the projected patch support is passed to the inpainting
model and also retained as the geometric compositing support.  A seeded
generator makes inference deterministic (required by the equivariance study).
Expects English prompts.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from PIL import Image

from ..errors import EditPanoError
from ._torch import hf_snapshot_revision, require_torch, resolve_device
from .base import EditorInfo, EditRequest
from .registry import register_editor

__all__ = ["StableDiffusionInpaintEditor"]

DEFAULT_MODEL_ID = os.environ.get(
    "SD_INPAINT_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting"
)
_PIPELINES: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def _round8(x: int) -> int:
    return max(8, round(x / 8.0) * 8)


class StableDiffusionInpaintEditor:
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seeded: bool = True,
        **_: Any,
    ):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self._device = device
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seeded = seeded

    @property
    def info(self) -> EditorInfo:
        return EditorInfo("sd_inpaint", self.model_id, hf_snapshot_revision(self.model_id))

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(None)
        return self._device

    def pipeline(self) -> Any:
        torch = require_torch()
        from diffusers import StableDiffusionInpaintPipeline

        key = (self.model_id, self.device)
        with _LOCK:
            if key not in _PIPELINES:
                dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_id, torch_dtype=dtype, safety_checker=None
                )
                _PIPELINES[key] = pipe.to(self.device)
            return _PIPELINES[key]

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Run the inpainting model at ``image``'s own size (must be a multiple of 8)."""
        torch = require_torch()
        kwargs: dict[str, Any] = {}
        if self.seeded:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(int(seed))
        return self.pipeline()(
            prompt=prompt,
            image=image.convert("RGB"),
            mask_image=mask.convert("L"),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            **kwargs,
        ).images[0]

    def edit(self, request: EditRequest) -> Image.Image:
        if request.mask is None:
            raise EditPanoError("sd_inpaint needs a patch mask; it cannot edit a whole panorama")
        W, H = request.image.size
        w, h = _round8(min(512, W)), _round8(min(512, H))
        out = self.inpaint(
            request.image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS),
            request.mask.convert("L").resize((w, h), Image.Resampling.NEAREST),
            request.prompt,
            request.seed,
        )
        return out.resize((W, H), Image.Resampling.LANCZOS)


register_editor("sd_inpaint")(StableDiffusionInpaintEditor)
