"""Text-to-panorama generation with DiT360 (FLUX.1-dev + panorama LoRA).

DiT360 differs from a stock FLUX pipeline in one place: the packed latent token
grid is *circularly padded* by one column on each side during denoising (and
the padding removed before decoding), so the left and right ERP borders are
generated as neighbours and the panorama closes seamlessly.  Rather than
vendoring a modified copy of the whole FLUX pipeline, :class:`DiT360Pipeline`
subclasses ``diffusers.FluxPipeline`` and overrides only the two latent hooks.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from ..editors._torch import require_torch, resolve_device

__all__ = ["PanoramaGenerator", "circular_pad_tokens", "circular_unpad_tokens"]

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_ID = "Insta360-Research/DiT360-Panorama-Image-Generation"


def circular_pad_tokens(tokens: Any, n_h: int, n_w: int) -> Any:
    """``[B, n_h*n_w, C] -> [B, n_h*(n_w+2), C]``: wrap the last/first token columns."""
    torch = require_torch()
    squeeze = tokens.dim() == 2  # latent_image_ids are [n_h*n_w, 3]
    t = tokens.unsqueeze(0) if squeeze else tokens
    b, _, c = t.shape
    grid = t.reshape(b, n_h, n_w, c)
    grid = torch.cat([grid[:, :, -1:], grid, grid[:, :, :1]], dim=2).reshape(b, -1, c)
    return grid[0] if squeeze else grid


def circular_unpad_tokens(tokens: Any, n_h: int, n_w: int) -> Any:
    """Inverse of :func:`circular_pad_tokens`."""
    b, _, c = tokens.shape
    return tokens.reshape(b, n_h, n_w + 2, c)[:, :, 1:-1].reshape(b, -1, c)


def _make_pipeline_class() -> Any:
    import diffusers

    FluxPipeline: Any = diffusers.FluxPipeline

    class DiT360Pipeline(FluxPipeline):
        """FLUX pipeline with circular latent padding (seamless ERP longitude)."""

        def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        ):
            latents, ids = super().prepare_latents(
                batch_size, num_channels_latents, height, width, dtype, device, generator, latents
            )
            n_h = int(height) // (self.vae_scale_factor * 2)
            n_w = int(width) // (self.vae_scale_factor * 2)
            return circular_pad_tokens(latents, n_h, n_w), circular_pad_tokens(ids, n_h, n_w)

        @staticmethod
        def _unpack_latents(latents, height, width, vae_scale_factor):
            n_h = int(height) // (vae_scale_factor * 2)
            n_w = int(width) // (vae_scale_factor * 2)
            if latents.shape[1] == n_h * (n_w + 2):
                latents = circular_unpad_tokens(latents, n_h, n_w)
            return FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)

    return DiT360Pipeline


class PanoramaGenerator:
    """Generate a 2:1 ERP panorama from text (defaults follow the DiT360 release)."""

    _cache: ClassVar[dict[tuple[str, str, str], Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self, base_model: str = BASE_MODEL_ID, lora: str | None = LORA_ID, device: str | None = None
    ):
        self.base_model, self.lora, self._device = base_model, lora, device

    def pipeline(self) -> Any:
        torch = require_torch()
        device = resolve_device(self._device)
        key = (self.base_model, str(self.lora), device)
        with self._lock:
            if key not in self._cache:
                dtype = torch.float16 if device.startswith("cuda") else torch.float32
                pipe = (
                    _make_pipeline_class()
                    .from_pretrained(self.base_model, torch_dtype=dtype)
                    .to(device)
                )
                if self.lora:
                    pipe.load_lora_weights(self.lora)
                self._cache[key] = pipe
            return self._cache[key]

    def generate(
        self,
        prompt: str,
        width: int = 2048,
        height: int = 1024,
        steps: int = 28,
        guidance_scale: float = 2.8,
        seed: int = 0,
    ) -> Any:
        torch = require_torch()
        pipe = self.pipeline()
        gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
        return pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]
