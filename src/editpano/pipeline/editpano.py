"""High-level facade: resolve the target, gate on confidence, route to an operator.

Routing (first match wins)::

    no usable ROI / confidence < tau  ->  whole-ERP edit (or ROIRequiredError)
    several instance boxes            ->  multi-instance grouped editing
    support wraps the seam            ->  circular canonical  (tiled if policy says so)
    tiling policy triggers            ->  narrow-FoV tiles
    otherwise                         ->  single-ROI operator L
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .._version import __version__
from ..config import EditPanoConfig
from ..editors.base import EditRequest, PatchEditor
from ..editors.registry import create_editor
from ..errors import ROIRequiredError
from ..geometry.camera import ViewParams
from ..geometry.circular import circular_roi_from_mask
from ..geometry.tiling import angular_extent
from ..types import Detection, EditResult, ViewHint
from .local import LocalEditOperator, OperatorOutput
from .multi_instance import MultiInstanceEditor
from .seam import SeamStrategyRunner

__all__ = ["EditPano", "PromptPolicy", "load_image"]

log = logging.getLogger(__name__)

PromptPolicy = Callable[[str], str]
"""Maps the raw instruction ``t`` to the effective editor prompt ``t'`` (identity by default)."""

ImageLike = str | Path | Image.Image | np.ndarray


def load_image(image: ImageLike) -> np.ndarray:
    """Any image-like input -> contiguous ``[H, W, 3]`` uint8 RGB array."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 array, got shape={arr.shape} dtype={arr.dtype}")
    return np.ascontiguousarray(arr)


class EditPano:
    """Instruction-guided local editing of ERP panoramas around a frozen editor.

    Example::

        pano = EditPano()                                   # Qwen-Image-Edit-2511
        result = pano.edit("room.jpg", "Replace the sofa with a leather armchair",
                           box=[812, 530, 1104, 690])
        result.save("room_edited.png", intermediates=True)
    """

    def __init__(
        self,
        config: EditPanoConfig | None = None,
        editor: PatchEditor | str | None = None,
        detector: Any | None = None,
        prompt_policy: PromptPolicy | None = None,
    ):
        self.config = config or EditPanoConfig()
        if editor is None or isinstance(editor, str):
            ec = self.config.editor
            kwargs = {
                k: v for k, v in (("model_id", ec.model_id), ("device", ec.device)) if v is not None
            }
            editor = create_editor(editor or ec.name, **kwargs, **ec.options)
        self.editor: PatchEditor = editor
        self.detector = detector
        self.prompt_policy: PromptPolicy = prompt_policy or (lambda t: t)
        self.local = LocalEditOperator(self.editor, self.config)
        self.seam = SeamStrategyRunner(self.local)
        self.multi = MultiInstanceEditor(self.local)

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> EditPano:
        return cls(EditPanoConfig.from_yaml(path), **kwargs)

    # ------------------------------------------------------------------ API
    def edit(
        self,
        image: ImageLike,
        instruction: str,
        *,
        box: Sequence[float] | None = None,
        mask: np.ndarray | None = None,
        detection: Detection | None = None,
        instances: Sequence[Sequence[float]] | None = None,
        hint: ViewHint | None = None,
        target: str | None = None,
        multi_instance: bool = False,
        seed: int | None = None,
    ) -> EditResult:
        """Edit one panorama.

        Args:
            image: path, PIL image or ``[H, W, 3]`` uint8 array (ERP, 2:1).
            instruction: natural-language edit ``t``.
            box: ERP target box ``[x1, y1, x2, y2]``; ``x2 > W`` wraps the seam.
            mask: optional ``[H, W]`` support restricting the composite; if given
                without ``box`` it defines the (possibly wrapped) target itself.
            detection: a ready-made :class:`Detection` (score is gated).
            instances: several instance boxes -> multi-instance editing.
            hint: manual view direction when no box is available.
            target: phrase for the configured detector (end-to-end mode).
            multi_instance: with ``target``, edit every detected instance.
            seed: overrides ``config.editor.seed``.
        """
        t_start = time.perf_counter()
        erp = load_image(image)
        H, W = erp.shape[:2]
        prompt = self.prompt_policy(instruction)
        seed = self.config.editor.seed if seed is None else int(seed)
        meta: dict[str, Any] = {
            "editpano_version": __version__,
            "config": self.config.fingerprint(),
            "editor": self.editor.info.__dict__,
            "seed": seed,
            "prompt": prompt,
            "erp_size": [W, H],
        }

        if target is not None and box is None and detection is None and instances is None:
            detection, instances = self._detect(erp, target, multi_instance)
            meta["detector_target"] = target
        if box is not None:
            detection = Detection.from_box(box, mask=mask)
        if detection is None and mask is not None and instances is None:
            detection = self._detection_from_mask(mask)
        if instances is not None and len(instances) == 1 and detection is None:
            detection, instances = Detection.from_box(instances[0]), None

        if instances:
            branch, out = "multi_instance", self.multi.edit(erp, instances, prompt, seed)
            meta["instances"] = [list(map(float, b)) for b in instances]
        elif hint is not None and detection is None:
            view = ViewParams(hint.lat, hint.lon, hint.fov or self.config.projection.hint_fov)
            branch, out = "local", self.local.edit_view(erp, view, prompt, seed)
        elif (
            detection is not None
            and self._confidence(detection) >= self.config.gate.confidence_threshold
        ):
            meta["detection"] = detection.to_dict()
            branch, out = self._route(erp, detection, prompt, seed)
        else:
            reason = (
                "no ROI"
                if detection is None
                else f"confidence {self._confidence(detection):.2f} below gate"
            )
            if self.config.gate.fallback == "raise":
                raise ROIRequiredError(f"cannot edit locally ({reason})")
            log.warning("falling back to whole-ERP editing (%s)", reason)
            branch, out = "whole", self._whole(erp, prompt, seed)
            meta["fallback_reason"] = reason

        meta.update(out.meta)
        meta["seconds"] = round(time.perf_counter() - t_start, 3)
        n_calls = 1 if branch == "whole" else out.editor_calls
        return EditResult(Image.fromarray(out.image), branch, out.patches, n_calls, meta)

    # ------------------------------------------------------------ internals
    def _confidence(self, det: Detection) -> float:
        # Explicit detections are clamped up so that a supplied ROI is always used
        # at the default gate (thesis: the gate is not a calibrated detector gate).
        return max(self.config.gate.detection_min_confidence, min(1.0, float(det.score)))

    def _detect(
        self, erp: np.ndarray, target: str, multi: bool
    ) -> tuple[Detection | None, list[list[float]] | None]:
        if self.detector is None:
            from ..localization.grounding import GroundingDinoDetector

            self.detector = GroundingDinoDetector(device=self.config.editor.device)
        if multi:
            dets = self.detector.detect_all(erp, target)
            boxes = [list(d.box) for d in dets]
            if len(boxes) > 1:
                return None, boxes
            return (dets[0], None) if dets else (None, None)
        return self.detector.detect_roi(erp, target), None

    @staticmethod
    def _detection_from_mask(mask: np.ndarray) -> Detection:
        roi = circular_roi_from_mask(mask)
        return Detection.from_box(roi.circular_box(), mask=mask, source="mask")

    def _route(
        self, erp: np.ndarray, det: Detection, prompt: str, seed: int
    ) -> tuple[str, OperatorOutput]:
        H, W = erp.shape[:2]
        wrapped = det.box[2] > W
        tiled = self._should_tile(det.box, W, H)
        if wrapped and self.config.canonicalize_seam:
            support = det.mask if det.mask is not None else self._wrapped_box_mask(det.box, W, H)
            roi = circular_roi_from_mask(support)
            out = self.seam.canonical(
                erp, support, roi, prompt, seed, tiled=tiled, erp_mask=det.mask
            )
            return ("canonical_tiled" if tiled else "canonical"), out
        if tiled:
            return "tiled", self.seam.edit_box(
                erp, det.box, prompt, seed, tiled=True, erp_mask=det.mask
            )
        return "local", self.local.edit_box(erp, det.box, prompt, seed, erp_mask=det.mask)

    def _should_tile(self, box: Sequence[float], W: int, H: int) -> bool:
        t = self.config.tiling
        if t.mode == "off":
            return False
        if t.mode == "always":
            return True
        span_lon, span_lat = angular_extent(box, W, H)
        return max(span_lon, span_lat) > t.auto_min_span_deg

    @staticmethod
    def _wrapped_box_mask(box: Sequence[float], W: int, H: int) -> np.ndarray:
        x1, y1, x2, y2 = (int(np.floor(v)) for v in box)
        mask = np.zeros((H, W), dtype=np.uint8)
        rows = slice(max(0, y1), min(H, y2))
        mask[rows, x1:W] = 255
        mask[rows, 0 : max(0, x2 - W)] = 255
        return mask

    def _whole(self, erp: np.ndarray, prompt: str, seed: int) -> OperatorOutput:
        src = Image.fromarray(erp)
        out = self.editor.edit(EditRequest(src, prompt, None, seed)).convert("RGB")
        meta = {"editor_output_size": list(out.size)}
        if out.size != src.size:
            out = out.resize(src.size, Image.Resampling.LANCZOS)
        return OperatorOutput(np.asarray(out, dtype=np.uint8).copy(), [], meta)
