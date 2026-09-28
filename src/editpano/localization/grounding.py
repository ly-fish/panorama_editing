"""Frozen open-vocabulary detection on the whole ERP (Grounding DINO).

Detection is far more distortion-robust than pixel synthesis, so the target is
localised directly on the raw panorama (thesis: mIoU 0.957 val / 0.862 test
against reference ROIs); only *editing* needs the gnomonic un-distortion.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PIL import Image

from ..editors._torch import require_torch, resolve_device
from ..types import Detection

__all__ = ["GroundingDinoDetector"]

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _normalise_query(text: str) -> str:
    t = (text or "").strip().lower() or "object"
    return t if t.endswith(".") else t + "."


class GroundingDinoDetector:
    """Lazy, frozen Grounding DINO returning ERP-pixel detections sorted by score."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_boxes: int = 20,
    ):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_boxes = max_boxes
        self._device = device
        self._proc: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            require_torch()
            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            self._device = resolve_device(self._device)
            self._proc = AutoProcessor.from_pretrained(self.model_id)
            model: Any = GroundingDinoForObjectDetection.from_pretrained(self.model_id)
            self._model = model.to(self._device).eval()

    def detect(self, image: Image.Image | np.ndarray, phrase: str) -> list[Detection]:
        torch = require_torch()
        self._ensure()
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
        W, H = pil.size
        inputs = self._proc(images=pil, text=_normalise_query(phrase), return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        res = self._post_process(outputs, inputs, (H, W))
        boxes, scores = res.get("boxes"), res.get("scores")
        if boxes is None or scores is None or len(boxes) == 0:
            return []
        dets = [
            Detection(
                (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                float(s),
                phrase,
                source="grounding_dino",
            )
            for b, s in zip(
                boxes.detach().cpu().tolist(), scores.detach().cpu().tolist(), strict=True
            )
        ]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.max_boxes]

    def _post_process(
        self, outputs: Any, inputs: Any, target_size: tuple[int, int]
    ) -> dict[str, Any]:
        """Bridge the ``threshold`` / ``box_threshold`` keyword drift across transformers versions."""
        fn = self._proc.post_process_grounded_object_detection
        ids = inputs["input_ids"]
        for kwargs in ({"threshold": self.box_threshold}, {"box_threshold": self.box_threshold}):
            try:
                return fn(
                    outputs,
                    ids,
                    text_threshold=self.text_threshold,
                    target_sizes=[target_size],
                    **kwargs,
                )[0]
            except TypeError:
                continue
        return fn(outputs, ids, target_sizes=[target_size])[0]

    def detect_roi(self, image: Image.Image | np.ndarray, phrase: str) -> Detection | None:
        """Top-scoring box (single-target editing)."""
        dets = self.detect(image, phrase)
        return dets[0] if dets else None

    def detect_all(
        self, image: Image.Image | np.ndarray, phrase: str, min_score: float = 0.20
    ) -> list[Detection]:
        """Every instance above ``min_score`` (multi-instance editing)."""
        return [d for d in self.detect(image, phrase) if d.score >= min_score]
