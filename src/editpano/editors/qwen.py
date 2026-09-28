"""Frozen Qwen-Image-Edit-2511 adapter (the thesis default editor).

Uses the upstream ``diffusers.QwenImageEditPlusPipeline`` unchanged.  When a
patch mask is given, a second reference image marks the editable support in
cyan (``Q(P, m_p)`` in the thesis) and a fixed instruction is appended to the
prompt; both are byte-identical to the evaluated implementation.
"""

from __future__ import annotations

import threading
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from ._torch import hf_snapshot_revision, require_torch, resolve_device
from .base import EditorInfo, EditRequest
from .registry import register_editor

__all__ = ["MASK_REFERENCE_INSTRUCTION", "QwenImageEditor", "make_mask_reference"]

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

MASK_REFERENCE_INSTRUCTION = (
    "There are two reference images. The first is the clean source view. "
    "The second marks the only editable region in cyan. Apply the requested "
    "replacement only inside that marked region. Keep the camera, object scale, "
    "perspective, illumination and every unmarked structure unchanged. Return a "
    "clean photorealistic image without the cyan overlay."
)

_PIPELINES: dict[tuple[str, str, str], Any] = {}
_LOCK = threading.Lock()


def make_mask_reference(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Cyan-tinted copy of ``image`` with a saturated outline around ``mask``."""
    base = image.convert("RGB")
    m = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", base.size, (0, 220, 235))
    marked = Image.composite(tint, base, m.point(lambda v: 92 if v > 0 else 0))
    outer = m.filter(ImageFilter.MaxFilter(9))
    inner = m.filter(ImageFilter.MinFilter(9))
    edge = ImageChops.subtract(outer, inner).point(lambda v: 255 if v > 8 else 0)
    marked.paste((0, 255, 255), mask=edge)
    return marked


class QwenImageEditor:
    """Qwen-Image-Edit-2511 with the thesis inference settings."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        use_mask_reference: bool = True,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        negative_prompt: str = " ",
        **_: Any,
    ):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self._device = device
        self.use_mask_reference = use_mask_reference
        self.num_inference_steps = int(num_inference_steps)
        self.true_cfg_scale = float(true_cfg_scale)
        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = negative_prompt

    @property
    def info(self) -> EditorInfo:
        return EditorInfo("qwen", self.model_id, hf_snapshot_revision(self.model_id))

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(None)
        return self._device

    def pipeline(self) -> Any:
        torch = require_torch()
        from diffusers import QwenImageEditPlusPipeline

        dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
        key = (self.model_id, self.device, str(dtype))
        with _LOCK:
            if key not in _PIPELINES:
                pipe = QwenImageEditPlusPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
                pipe.to(self.device)
                pipe.set_progress_bar_config(disable=None)
                _PIPELINES[key] = pipe
            return _PIPELINES[key]

    def edit(self, request: EditRequest) -> Image.Image:
        torch = require_torch()
        images = [request.image]
        prompt = request.prompt
        if self.use_mask_reference and request.mask is not None:
            images.append(make_mask_reference(request.image, request.mask))
            prompt = f"{prompt}\n\n{MASK_REFERENCE_INSTRUCTION}"
        out = self.pipeline()(
            image=images,
            prompt=prompt,
            generator=torch.Generator(device=self.device).manual_seed(int(request.seed)),
            true_cfg_scale=self.true_cfg_scale,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
        )
        return out.images[0]


register_editor("qwen")(QwenImageEditor)
