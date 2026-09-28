"""Seam-position equivariance study (thesis §3.5.6): run -> score -> report.

Every strategy sees identical pixels, prompt, seed and editor for the three
longitude conditions of each scene.  Outputs are returned to the source frame
(inverse construction roll) and compared with the same strategy's *centre*
output via YE-LPIPS inside fixed, method-independent tangent views.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..errors import DegenerateROIError
from ..geometry.circular import circular_roi_from_mask, roll_image
from ..pipeline.seam import SeamStrategy, SeamStrategyRunner
from .benchmark import _atomic_save_image, _atomic_write_text
from .metrics.preservation import background_changed_rate, background_psnr, target_coverage
from .metrics.seam import SpatialLPIPS, build_eval_views, ds_score, sample_view, ye_lpips
from .stats import holm, one_sided_t_bound, paired_comparison

__all__ = ["report_seam", "run_seam_study", "score_seam_study"]

log = logging.getLogger(__name__)

STRATEGIES = tuple(s.value for s in SeamStrategy)
NI_MARGINS = {"coverage": -0.05, "region_clip": -1.0, "bg_changed_rate": 0.005}
TAU_EQ = 0.05


def _load_case(root: Path, c: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(root / c["input"]).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(root / c["roi_mask"]).convert("L"), dtype=np.uint8)
    return img, mask


def run_seam_study(
    manifest: str | Path,
    runner: SeamStrategyRunner,
    out_dir: str | Path,
    strategies: Sequence[str] = STRATEGIES,
    seeds: Sequence[int] = (0,),
    tiled: bool = False,
    scenes: Sequence[str] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    prompt_key: str = "prompt",
) -> list[dict[str, Any]]:
    """Execute strategies; outputs at ``<out>/<strategy>/<seed>/<case_id>/edited.png``."""
    mpath = Path(manifest)
    man = json.loads(mpath.read_text(encoding="utf-8"))
    all_scenes = sorted({c["source_scene_id"] for c in man["cases"]})
    if scenes:
        all_scenes = [s for s in all_scenes if s in set(scenes)]
    mine = {s for i, s in enumerate(all_scenes) if i % num_shards == shard_index}
    out, failures = Path(out_dir), []
    for c in (c for c in man["cases"] if c["source_scene_id"] in mine):
        img, mask = _load_case(mpath.parent, c)
        for strategy in strategies:
            for seed in seeds:
                cdir = out / strategy / str(seed) / c["case_id"]
                cdir.mkdir(parents=True, exist_ok=True)
                if (cdir / "meta.json").exists() and (cdir / "edited.png").exists():
                    continue
                meta: dict[str, Any] = {"case_id": c["case_id"], "strategy": strategy, "seed": seed}
                try:
                    res = runner.run(
                        strategy, img, mask, c.get(prompt_key) or c["prompt"], seed, tiled=tiled
                    )
                    _atomic_save_image(cdir / "edited.png", res.image)
                    meta.update(
                        completed=True,
                        editor_calls=res.editor_calls,
                        **{k: v for k, v in res.meta.items() if k != "views"},
                    )
                except DegenerateROIError as exc:
                    meta.update(completed=False, reason=f"degenerate:{exc.reason}")
                    failures.append(meta)
                _atomic_write_text(cdir / "meta.json", json.dumps(meta, indent=2, default=str))
                log.info(
                    "%s %s seed=%d %s",
                    c["case_id"],
                    strategy,
                    seed,
                    "ok" if meta["completed"] else meta.get("reason"),
                )
    return failures


def score_seam_study(
    manifest: str | Path,
    runs_dir: str | Path,
    *,
    device: str = "cpu",
    region_clip: Any | None = None,
    attack: bool = True,
) -> Path:
    """Per-case YE-LPIPS, DS, coverage, background leakage (and Region-CLIP) -> ``per_case.csv``."""
    from .metrics.seam import boundary_average_attack

    mpath, runs = Path(manifest), Path(runs_dir)
    man = json.loads(mpath.read_text(encoding="utf-8"))
    lp = SpatialLPIPS(device)
    by_scene: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for c in man["cases"]:
        by_scene[c["source_scene_id"]][c["variant"]] = c
    rows: list[dict[str, Any]] = []
    for sid, variants in sorted(by_scene.items()):
        cc = variants["center"]
        img_c, mask_c = _load_case(mpath.parent, cc)
        src, src_mask = roll_image(img_c, -cc["shift_px"]), roll_image(mask_c, -cc["shift_px"])
        views = build_eval_views(src_mask)
        span = circular_roi_from_mask(src_mask).span_deg
        arms = [d.name for d in sorted(runs.iterdir()) if d.is_dir()]
        for arm in arms:
            for seed_dir in sorted((runs / arm).iterdir()):
                outs = {}
                for v, c in variants.items():
                    p = seed_dir / c["case_id"] / "edited.png"
                    outs[v] = np.asarray(Image.open(p).convert("RGB")) if p.exists() else None
                groups = [(arm, outs)]
                if attack and arm == SeamStrategy.CANONICAL.value:
                    groups.append(
                        (
                            "boundary_avg",
                            {
                                v: None if o is None else boundary_average_attack(o)
                                for v, o in outs.items()
                            },
                        )
                    )
                for name, group in groups:
                    ref = group["center"]
                    ref_src = (
                        None if ref is None else roll_image(ref, -variants["center"]["shift_px"])
                    )
                    for v, c in variants.items():
                        o = group[v]
                        row: dict[str, Any] = {
                            "scene_id": sid,
                            "case_id": c["case_id"],
                            "variant": v,
                            "arm": name,
                            "seed": int(seed_dir.name),
                            "span_deg": round(span, 3),
                            "completed": int(o is not None),
                        }
                        if o is not None:
                            o_src = roll_image(o, -c["shift_px"])
                            row["ye_lpips"] = (
                                ye_lpips(ref_src, o_src, views, lp)
                                if v != "center" and ref_src is not None
                                else None
                            )
                            row["d_ds"] = ds_score(o) - ds_score(_load_case(mpath.parent, c)[0])
                            fg = src_mask > 0
                            row["coverage"] = target_coverage(src, o_src, fg)
                            row["bg_psnr"] = background_psnr(src, o_src, fg)
                            row["bg_changed_rate"] = background_changed_rate(src, o_src, fg)
                            if region_clip is not None:
                                w = [float(vw["weight"].sum()) for vw in views]
                                s = [
                                    region_clip.score(sample_view(o_src, vw), c["prompt"])
                                    for vw in views
                                ]
                                row["region_clip"] = float(np.dot(w, s) / sum(w))
                        rows.append(row)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path = runs / "per_case.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return path


def _per_scene(csv_path: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """data[arm][variant][metric][scene] = seed-mean."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("ye_lpips", "coverage", "region_clip", "bg_changed_rate", "bg_psnr", "d_ds"):
                if r.get(k) not in (None, ""):
                    acc[r["arm"]][r["variant"]][k][r["scene_id"]].append(float(r[k]))
    return {
        a: {
            v: {m: {s: float(np.mean(x)) for s, x in ms.items()} for m, ms in vs.items()}
            for v, vs in av.items()
        }
        for a, av in acc.items()
    }


def report_seam(
    csv_path: str | Path, proposed: str = "canonical", baseline: str = "current"
) -> str:
    """Markdown summary: per-arm table, primary contrast, Holm family, guardrails."""
    d = _per_scene(Path(csv_path))

    def mean(x: dict[str, float]) -> float:
        return float(np.mean(list(x.values()))) if x else math.nan

    def scene_min(arm: str, m: str) -> dict[str, float]:
        per = [d[arm].get(v, {}).get(m, {}) for v in ("center", "near", "cross")]
        scenes = set.intersection(*(set(p) for p in per)) if all(per) else set()
        return {s: min(p[s] for p in per) for s in scenes}

    lines = [
        "| Arm | Near-YE-LPIPS | Cross-YE-LPIPS | Penalty | Coverage (scene-min) | Region-CLIP (scene-min) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in d:
        near, cross = (
            mean(d[arm].get("near", {}).get("ye_lpips", {})),
            mean(d[arm].get("cross", {}).get("ye_lpips", {})),
        )
        lines.append(
            f"| {arm} | {near:.4f} | {cross:.4f} | {cross - near:.4f} | "
            f"{mean(scene_min(arm, 'coverage')):.4f} | {mean(scene_min(arm, 'region_clip')):.3f} |"
        )
    lines.append("")
    if proposed in d and baseline in d:
        prim = paired_comparison(
            d[proposed]["cross"]["ye_lpips"], d[baseline]["cross"]["ye_lpips"], n_flips=100_000
        )
        lines += [
            f"**Primary contrast** Cross-YE-LPIPS `{proposed}` - `{baseline}`: "
            f"Δ={prim.mean_diff:+.4f} [95% CI {prim.ci95[0]:+.4f}, {prim.ci95[1]:+.4f}], "
            f"p={prim.p:.1e}, d_z={prim.dz:+.2f}, wins/ties/losses {prim.wins}/{prim.ties}/{prim.losses}",
            "",
        ]
        family: dict[str, float] = {
            a: paired_comparison(d[proposed]["cross"]["ye_lpips"], d[a]["cross"]["ye_lpips"]).p
            for a in ("split", "direct_circular")
            if a in d
        }
        for a, p in holm(family).items():
            lines.append(f"- vs `{a}`: Holm-adjusted p={p:.1e}")
        lines += ["", "**Guardrails** (cross - centre, one-sided 95% bound vs margin):"]
        for m, margin in NI_MARGINS.items():
            c, x = d[proposed]["center"].get(m, {}), d[proposed]["cross"].get(m, {})
            diffs = [x[s] - c[s] for s in sorted(set(c) & set(x))]
            if not diffs:
                continue
            side = "upper" if margin > 0 else "lower"
            bound = float(np.mean(diffs)) if np.std(diffs) == 0 else one_sided_t_bound(diffs, side)
            ok = bound <= margin if margin > 0 else bound >= margin
            lines.append(
                f"- {m}: mean Δ={np.mean(diffs):+.5f}, bound={bound:+.5f}, margin={margin:+} -> "
                f"{'PASS' if ok else 'FAIL'}"
            )
        cross_ye = mean(d[proposed]["cross"]["ye_lpips"])
        lines.append(
            f"- τ_eq: Cross-YE-LPIPS {cross_ye:.4f} <= {TAU_EQ} -> {'PASS' if cross_ye <= TAU_EQ else 'FAIL'}"
        )
    return "\n".join(lines) + "\n"
