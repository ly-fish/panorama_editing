"""Region-CLIP: prompt/crop cosine similarity with frozen CLIP ViT-B/32 (x100).

Global directional CLIP is diluted when an edit touches ~1 % of a panorama, so
edit alignment is measured on an ROI crop instead.  The default backend is the
reference OpenAI ``clip`` package (``pip install git+https://github.com/openai/CLIP``);
a Hugging Face ``transformers`` backend is available as a fallback.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

__all__ = ["RegionCLIP", "roi_crop"]


def roi_crop(
    image: Image.Image, box: Sequence[float], ref_size: tuple[int, int], pad: float = 1.5
) -> Image.Image:
    """Square-ish ROI crop padded by ``pad`` for context (box in ``ref_size`` coordinates)."""
    W, H = image.size
    sx, sy = W / ref_size[0], H / ref_size[1]
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(max(x2 - x1, y2 - y1) * (1 + pad) / 2, 40)
    L, T = max(0, int(cx - half)), max(0, int(cy - half))
    R, B = min(W, int(cx + half)), min(H, int(cy + half))
    return image.crop((L, T, R, B))


class RegionCLIP:
    """Lazy CLIP scorer; ``score(crop, prompt)`` returns ``100 * cos(z_I, z_t)``."""

    def __init__(self, device: str = "cpu", backend: str = "openai", model: str = "ViT-B/32"):
        self.device, self.backend, self.model_name = device, backend, model
        self._model: Any = None
        self._pre: Any = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if self.backend == "openai":
                import clip

                self._model, self._pre = clip.load(self.model_name, device=self.device)
                self._model.eval()
            else:
                from transformers import CLIPModel, CLIPProcessor

                hf_id = "openai/clip-vit-base-patch32"
                model: Any = CLIPModel.from_pretrained(hf_id)
                self._model = model.to(self.device).eval()
                self._pre = CLIPProcessor.from_pretrained(hf_id)

    def score(self, image: Image.Image | np.ndarray, prompt: str) -> float:
        import torch

        self._ensure()
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        with torch.no_grad():
            if self.backend == "openai":
                import clip

                img = self._model.encode_image(self._pre(pil).unsqueeze(0).to(self.device))
                txt = self._model.encode_text(
                    clip.tokenize([prompt], truncate=True).to(self.device)
                )
            else:
                inp = self._pre(
                    text=[prompt], images=pil, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                img = self._model.get_image_features(pixel_values=inp["pixel_values"])
                txt = self._model.get_text_features(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                )
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            return float((img * txt).sum().item()) * 100.0
