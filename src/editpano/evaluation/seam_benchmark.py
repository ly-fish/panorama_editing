"""Controlled longitude-origin stress benchmark (thesis §3.4.1, Fig. 3.3).

Each reference ROI is rasterised to a half-open rectangle; panorama and mask
are moved by *interpolation-free integer rolls* into three conditions:

* ``center`` -- circular ROI centre at ``u = W/2``;
* ``near``   -- nearest ROI boundary ``round(W * 5 deg / 360)`` px from the seam;
* ``cross``  -- circular centre at ``u = 0`` (the ROI straddles the seam).

Undoing a roll reproduces the source bit-exactly; :func:`validate_seam_benchmark`
is the QA gate that must pass before any GPU run.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..errors import DatasetError
from ..geometry.circular import (
    CircularROI,
    circular_roi_from_mask,
    normalize_shift,
    roll_image,
    round_half_up,
)
from ..localization.annotations import load_annotations
from .datasets import load_cases

__all__ = [
    "VARIANTS",
    "build_seam_benchmark",
    "rasterize_box",
    "validate_seam_benchmark",
    "variant_shift",
]

GENERATOR_VERSION = "seam_set_v1.0"
VARIANTS = ("center", "near", "cross")


def sha256_pixels(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def rasterize_box(box: Sequence[float], W: int, H: int) -> tuple[np.ndarray, list[int]]:
    """Half-open integer rectangle mask (floor coordinates) from a float box."""
    x1, y1, x2, y2 = (int(np.floor(v)) for v in box)
    x1 = max(0, min(W - 1, x1))
    x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(y1 + 1, min(H, y2))
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask, [x1, y1, x2, y2]


def variant_shift(roi: CircularROI, variant: str, near_gap_px: int) -> int:
    """Signed integer roll placing the ROI in the requested condition."""
    W = roi.erp_width
    if variant == "center":
        return normalize_shift(round_half_up(W / 2.0 - roi.center), W)
    if variant == "near":
        if roi.width > W - near_gap_px:
            raise ValueError(f"ROI width {roi.width} cannot keep a {near_gap_px}px seam gap")
        return normalize_shift(W - near_gap_px - roi.width - roi.x_start, W)
    if variant == "cross":
        return normalize_shift(round_half_up(0.0 - roi.center), W)
    raise ValueError(f"unknown variant {variant!r}")


def build_seam_benchmark(
    data_root: str | Path,
    prompts: str | Path,
    out_dir: str | Path,
    near_gap_deg: float = 5.0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Write ``cases/<case>/{input,roi_mask}.png`` and ``manifest.json`` under ``out_dir``."""
    out = Path(out_dir)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    scenes = load_cases(data_root, prompts, limit=limit)
    width = 0
    for sc in scenes:
        src = np.asarray(Image.open(sc.panorama).convert("RGB"), dtype=np.uint8)
        H, W = src.shape[:2]
        width = W
        best = max(load_annotations(sc.annotations), key=lambda a: a.logit)
        src_mask, src_box = rasterize_box(best.box, W, H)
        roi = circular_roi_from_mask(src_mask)
        if roi.reason != "ok":
            raise DatasetError(f"{sc.scene_id}: source ROI degenerate ({roi.reason})")
        gap = round_half_up(W * near_gap_deg / 360.0)
        for variant in VARIANTS:
            s = variant_shift(roi, variant, gap)
            img_v, mask_v = roll_image(src, s), roll_image(src_mask, s)
            roi_v = circular_roi_from_mask(mask_v)
            cid = f"{sc.scene_id}__{variant}"
            cdir = out / "cases" / cid
            cdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img_v).save(cdir / "input.png")
            Image.fromarray(mask_v).save(cdir / "roi_mask.png")
            cases.append(
                {
                    "case_id": cid,
                    "source_scene_id": sc.scene_id,
                    "variant": variant,
                    "input": f"cases/{cid}/input.png",
                    "roi_mask": f"cases/{cid}/roi_mask.png",
                    "width": W,
                    "height": H,
                    "shift_px": int(s),
                    "shift_deg": round(s / W * 360.0, 4),
                    "source_box": src_box,
                    "circular_roi": {
                        k: roi_v.to_dict()[k] for k in ("x_start", "width", "y1", "y2", "wrap")
                    },
                    "split_boxes": roi_v.split_boxes(),
                    "naive_box": roi_v.naive_box(mask_v),
                    "prompt": sc.prompt,
                    "prompt_sha256": hashlib.sha256(sc.prompt.encode()).hexdigest(),
                    "object_label": sc.object_label,
                    "object_value": best.value,
                    "source_pixel_sha256": sha256_pixels(src),
                    "generator_version": GENERATOR_VERSION,
                }
            )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "near_gap_deg": near_gap_deg,
        "near_gap_px": round_half_up((width or 2048) * near_gap_deg / 360.0),
        "data_root": str(data_root),
        "prompts": str(prompts),
        "num_source_scenes": len(scenes),
        "num_cases": len(cases),
        "variants": list(VARIANTS),
        "cases": cases,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def validate_seam_benchmark(
    manifest_path: str | Path,
    expected_scenes: int | None = None,
    data_root: str | Path | None = None,
) -> list[str]:
    """QA gate; returns a list of errors (empty = pass).

    Checks: cardinality; bit-exact inverse roll and source hash; identical area,
    circular width, vertical range, object value and prompt across variants;
    centre/near/cross placement rules; PNG/RGB format.
    """
    mpath = Path(manifest_path)
    man = json.loads(mpath.read_text(encoding="utf-8"))
    root, g = mpath.parent, int(man["near_gap_px"])
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    by_scene: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for c in man["cases"]:
        by_scene[c["source_scene_id"]][c["variant"]] = c
    if expected_scenes is not None:
        check(
            len(by_scene) == expected_scenes,
            f"expected {expected_scenes} scenes, got {len(by_scene)}",
        )
    check(len(man["cases"]) == 3 * len(by_scene), "every scene needs exactly 3 variants")
    src_root = Path(data_root or man["data_root"])
    for sid, variants in sorted(by_scene.items()):
        check(set(variants) == set(VARIANTS), f"{sid}: variants {sorted(variants)}")
        src_path = src_root / sid / "panorama.jpg"
        src = (
            np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
            if src_path.exists()
            else None
        )
        stats = {}
        for variant, c in variants.items():
            with Image.open(root / c["input"]) as im:
                check(
                    im.format == "PNG" and im.mode == "RGB",
                    f"{c['case_id']}: input must be RGB PNG",
                )
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(root / c["roi_mask"]).convert("L"), dtype=np.uint8)
            back = roll_image(arr, -int(c["shift_px"]))
            check(
                sha256_pixels(back) == c["source_pixel_sha256"],
                f"{c['case_id']}: inverse roll not bit-exact",
            )
            if src is not None:
                check(np.array_equal(back, src), f"{c['case_id']}: differs from re-decoded source")
            roi = circular_roi_from_mask(mask)
            check(
                roi.x_start == c["circular_roi"]["x_start"]
                and roi.width == c["circular_roi"]["width"],
                f"{c['case_id']}: manifest circular_roi mismatch",
            )
            stats[variant] = (
                int((mask > 0).sum()),
                roi.width,
                (roi.y1, roi.y2),
                c["object_value"],
                c["prompt_sha256"],
            )
            first, last = bool((mask[:, 0] > 0).any()), bool((mask[:, -1] > 0).any())
            if variant == "center":
                check(
                    not roi.wrap and not first and not last,
                    f"{c['case_id']}: centre touches the seam",
                )
                check(
                    abs(roi.center - c["width"] / 2.0) <= 1.0, f"{c['case_id']}: centre not at W/2"
                )
            elif variant == "near":
                d = min(roi.x_start, c["width"] - (roi.x_start + roi.width))
                check(not roi.wrap and abs(d - g) <= 1, f"{c['case_id']}: near gap {d}px != {g}+-1")
            else:
                check(roi.wrap and first and last, f"{c['case_id']}: cross must wrap both borders")
        if set(stats) == set(VARIANTS):
            check(
                stats["near"] == stats["center"] == stats["cross"],
                f"{sid}: invariants differ across variants",
            )
    return errors
