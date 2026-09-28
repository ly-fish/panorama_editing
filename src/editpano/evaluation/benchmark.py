"""Reference-ROI benchmark: run arms -> score -> report.

Outputs are cached under ``<out_dir>/<arm>/<case_id>/{edited.png, meta.json}``.
A cache hit requires a ``meta.json`` whose *run signature* (config fingerprint,
editor identity + snapshot revision, arm, package version, prompt hash) matches;
the mere existence of an image is never trusted.  Cases are sharded
deterministically (``i % num_shards == shard_index``) for SLURM arrays.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import traceback
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .._version import __version__
from ..geometry.masks import instance_union_mask
from ..localization.annotations import reference_detection
from ..pipeline.editpano import EditPano, load_image
from .arms import ARMS, ArmContext, arm_order
from .datasets import Case
from .metrics.preservation import background_psnr, background_ssim, match_size, target_coverage
from .stats import paired_comparison, stars

__all__ = ["report", "run_benchmark", "run_signature", "score_benchmark"]

log = logging.getLogger(__name__)


def run_signature(pano: EditPano, arm: str, case: Case) -> str:
    payload = {
        "config": pano.config.to_dict(),
        "editor": pano.editor.info.__dict__,
        "arm": arm,
        "version": __version__,
        "prompt": hashlib.sha256(case.prompt.encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _atomic_save_image(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    Image.fromarray(arr).save(tmp, format="PNG")
    os.replace(tmp, path)


def run_benchmark(
    cases: Sequence[Case],
    arms: Sequence[str],
    pano: EditPano,
    out_dir: str | Path,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
    overwrite: bool = False,
    retries: int = 1,
) -> list[dict[str, Any]]:
    """Run every requested arm on this shard's cases; returns the failure ledger."""
    out = Path(out_dir)
    order = arm_order(list(arms))
    failures: list[dict[str, Any]] = []
    shard = [c for i, c in enumerate(cases) if i % num_shards == shard_index]
    for ci, case in enumerate(shard, 1):
        image = load_image(case.panorama)
        box = reference_detection(case.annotations).box
        upstream: dict[str, np.ndarray] = {}
        for arm in order:
            cdir = out / arm / case.case_id
            cdir.mkdir(parents=True, exist_ok=True)
            img_p, meta_p = cdir / "edited.png", cdir / "meta.json"
            sig = run_signature(pano, arm, case)
            if not overwrite and img_p.exists() and meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta.get("run_signature") == sig and meta.get("completed"):
                        if arm in {r for a in order for r in ARMS[a].requires}:
                            upstream[arm] = load_image(img_p)
                        continue
                except json.JSONDecodeError:
                    pass
            t0, result, reason = time.perf_counter(), None, None
            for _attempt in range(retries + 1):
                ctx = ArmContext(pano, image, box, case, pano.config.editor.seed, upstream)
                try:
                    result = ARMS[arm].run(ctx)
                    reason = None
                    break
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    log.error(
                        "%s/%s failed: %s\n%s", arm, case.case_id, reason, traceback.format_exc()
                    )
            meta = {
                "run_signature": sig,
                "case_id": case.case_id,
                "arm": arm,
                "completed": result is not None,
                "reason": reason,
                "seconds": round(time.perf_counter() - t0, 2),
                "box": list(box),
            }
            if result is not None:
                result = match_size(image, result)
                for name, art in ctx.artifacts.items():
                    art.save(cdir / name)
                _atomic_save_image(img_p, result)
                upstream[arm] = result
            else:
                failures.append({"case_id": case.case_id, "arm": arm, "reason": reason})
            _atomic_write_text(meta_p, json.dumps(meta, ensure_ascii=False, indent=2))
            log.info(
                "[%d/%d] %s %s %s (%.1fs)",
                ci,
                len(shard),
                case.case_id,
                arm,
                "ok" if result is not None else "FAIL",
                meta["seconds"],
            )
    _atomic_write_text(out / f"failures_shard{shard_index}.json", json.dumps(failures, indent=2))
    return failures


def score_benchmark(
    cases: Sequence[Case],
    arms: Sequence[str],
    out_dir: str | Path,
    *,
    region_clip: Any | None = None,
    with_ssim: bool = False,
    instance_support: bool = False,
) -> Path:
    """Score cached outputs -> ``per_case.csv``.

    The target support is the reference box (or, with ``instance_support``, the
    union of tight boxes of every instance sharing the reference label).
    """
    from .metrics.semantic import roi_crop

    out = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for case in cases:
        src = load_image(case.panorama)
        H, W = src.shape[:2]
        det = reference_detection(case.annotations)
        if instance_support:
            from ..localization.annotations import load_annotations

            boxes = [a.box for a in load_annotations(case.annotations) if a.label == det.label]
            support = instance_union_mask(boxes, W, H) > 0
        else:
            support = instance_union_mask([det.box], W, H) > 0
        for arm in arms:
            p = out / arm / case.case_id / "edited.png"
            row: dict[str, Any] = {
                "case_id": case.case_id,
                "scene_id": case.scene_id,
                "arm": arm,
                "edit_type": case.edit_type or "",
                "completed": int(p.exists()),
            }
            if p.exists():
                ed = load_image(p)
                row["bg_psnr"] = background_psnr(src, ed, support)
                row["coverage"] = target_coverage(src, ed, support)
                if with_ssim:
                    row["bg_ssim"] = background_ssim(src, ed, support)
                if region_clip is not None and case.clip_text:
                    crop = roi_crop(Image.fromarray(ed), det.box, (W, H))
                    row["region_clip"] = region_clip.score(crop, case.clip_text)
            rows.append(row)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = out / "per_case.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return path


def _load(csv_path: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    data: dict[str, dict[str, dict[str, float | None]]] = defaultdict(lambda: defaultdict(dict))
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("case_id", "scene_id", "arm", "edit_type", "completed"):
                    continue
                data[r["arm"]][k][r["case_id"]] = float(v) if v not in ("", None) else None
    return data


def report(
    csv_path: str | Path,
    reference: str = "editpano",
    metrics: Sequence[str] = ("bg_psnr", "coverage", "region_clip"),
) -> str:
    """Markdown table of means plus paired sign-flip tests of every arm vs ``reference``."""
    data = _load(Path(csv_path))
    metrics = [m for m in metrics if any(m in data[a] for a in data)]

    def mean(vals: dict[str, float | None]) -> float:
        f = [v for v in vals.values() if v is not None and math.isfinite(v)]
        return float(np.mean(f)) if f else math.nan

    lines = ["| Arm | " + " | ".join(metrics) + " |", "| --- |" + " --- |" * len(metrics)]
    for arm in data:
        lines.append(
            f"| {arm} | " + " | ".join(f"{mean(data[arm].get(m, {})):.3f}" for m in metrics) + " |"
        )
    lines += [
        "",
        f"Paired sign-flip tests vs `{reference}` (difference = arm - reference):",
        "",
        "| Arm | Metric | n | Δ mean | 95% CI | p | d_z |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in data:
        if arm == reference or reference not in data:
            continue
        for m in metrics:
            res = paired_comparison(data[arm].get(m, {}), data[reference].get(m, {}))
            if res.n:
                lines.append(
                    f"| {arm} | {m} | {res.n} | {res.mean_diff:+.3f} | "
                    f"[{res.ci95[0]:+.3f}, {res.ci95[1]:+.3f}] | {res.p:.1e} {stars(res.p)} | {res.dz:+.2f} |"
                )
    return "\n".join(lines) + "\n"
