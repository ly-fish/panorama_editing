"""``editpano`` command-line interface.

editpano edit      --input pano.jpg --prompt "..." (--box X1 Y1 X2 Y2 | --annotations a.json | --detect PHRASE)
editpano generate  --prompt "..." --output pano.png
editpano bench     run|score|report   (reference-ROI benchmark)
editpano seam      build|validate|run|score|report   (longitude-origin study)
editpano diagnose  cycle              (editor-free reprojection check)
editpano editors | config | doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__

log = logging.getLogger("editpano")


# ----------------------------------------------------------------- helpers
def _config(args: argparse.Namespace) -> Any:
    from .config import EditPanoConfig

    cfg = (
        EditPanoConfig.from_yaml(args.config) if getattr(args, "config", None) else EditPanoConfig()
    )
    overrides = list(getattr(args, "set", None) or [])
    if getattr(args, "editor", None):
        overrides.append(f"editor.name={args.editor}")
    if getattr(args, "device", None):
        overrides.append(f"editor.device={args.device}")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"editor.seed={args.seed}")
    return cfg.with_overrides(overrides) if overrides else cfg


def _pano(args: argparse.Namespace) -> Any:
    from .pipeline.editpano import EditPano

    return EditPano(_config(args))


def _add_common(p: argparse.ArgumentParser, editor: bool = True) -> None:
    p.add_argument("--config", help="YAML config (default: built-in thesis operating point)")
    p.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set composite.feather_px=16 (repeatable)",
    )
    if editor:
        p.add_argument("--editor", help="editor backend (qwen | sd_inpaint | fake:pattern | ...)")
        p.add_argument("--device", help="cuda | cuda:1 | cpu (default: auto)")
        p.add_argument("--seed", type=int, help="editor seed (default 0)")


def _cases(args: argparse.Namespace) -> Any:
    from .evaluation.datasets import load_cases

    return load_cases(
        args.data_root,
        args.prompts,
        scene_ids=args.scenes,
        edit_types=args.edit_types,
        limit=args.limit,
        prompt_key=args.prompt_key,
    )


def _add_cases(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-root",
        required=True,
        help="folder with <scene_id>/panorama.jpg + result_*_mask.json",
    )
    p.add_argument("--prompts", required=True, help="JSON prompts file (see docs/data_format.md)")
    p.add_argument("--scenes", nargs="*", help="restrict to these scene ids")
    p.add_argument("--edit-types", nargs="*", help="restrict to these edit types")
    p.add_argument(
        "--prompt-key", default="prompt", help="prompts-file field used as the instruction"
    )
    p.add_argument("--limit", type=int)


# ----------------------------------------------------------------- commands
def cmd_edit(args: argparse.Namespace) -> int:
    from .localization.annotations import load_annotations, reference_detection
    from .types import ViewHint

    pano = _pano(args)
    kw: dict[str, Any] = {}
    if args.box:
        kw["box"] = args.box
    elif args.annotations:
        det = reference_detection(
            args.annotations, args.object_value, with_mask=args.annotation_mask
        )
        if args.multi_instance:
            kw["instances"] = [
                list(a.box) for a in load_annotations(args.annotations) if a.label == det.label
            ]
        else:
            kw["detection"] = det
    elif args.detect:
        kw.update(target=args.detect, multi_instance=args.multi_instance)
    elif args.hint:
        kw["hint"] = ViewHint(
            lon=args.hint[0], lat=args.hint[1], fov=args.hint[2] if len(args.hint) > 2 else None
        )
    result = pano.edit(args.input, args.prompt, **kw)
    out = result.save(args.output, intermediates=args.save_intermediates)
    print(
        json.dumps(
            {
                "output": str(out),
                "branch": result.branch,
                "editor_calls": result.editor_calls,
                "seconds": result.meta.get("seconds"),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .generation import PanoramaGenerator

    img = PanoramaGenerator(device=args.device).generate(
        args.prompt, args.width, args.height, args.steps, args.guidance_scale, args.seed
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)
    print(args.output)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .evaluation import benchmark

    if args.action == "run":
        failures = benchmark.run_benchmark(
            _cases(args),
            args.arms,
            _pano(args),
            args.out_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            overwrite=args.overwrite,
        )
        print(f"{len(failures)} failure(s)")
        return 1 if failures else 0
    if args.action == "score":
        rc = None
        if args.region_clip:
            from .evaluation.metrics.semantic import RegionCLIP

            rc = RegionCLIP(device=args.metric_device)
        path = benchmark.score_benchmark(
            _cases(args),
            args.arms,
            args.out_dir,
            region_clip=rc,
            with_ssim=args.ssim,
            instance_support=args.instance_support,
        )
        print(path)
        return 0
    text = benchmark.report(Path(args.out_dir) / "per_case.csv", reference=args.reference)
    (Path(args.out_dir) / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_seam(args: argparse.Namespace) -> int:
    from .evaluation import seam_benchmark, seam_study

    if args.action == "build":
        man = seam_benchmark.build_seam_benchmark(
            args.data_root, args.prompts, args.out_dir, args.near_gap_deg, args.limit
        )
        print(f"{man['num_cases']} cases -> {Path(args.out_dir) / 'manifest.json'}")
        return 0
    if args.action == "validate":
        errors = seam_benchmark.validate_seam_benchmark(args.manifest, args.expected_scenes)
        for e in errors:
            print("ERROR:", e)
        print("QA PASSED" if not errors else f"QA FAILED ({len(errors)} errors)")
        return 0 if not errors else 1
    if args.action == "run":
        errors = seam_benchmark.validate_seam_benchmark(args.manifest)
        if errors:
            print("refusing to run: seam benchmark QA failed; run `editpano seam validate`")
            return 1
        from .pipeline.local import LocalEditOperator
        from .pipeline.seam import SeamStrategyRunner

        pano = _pano(args)
        runner = SeamStrategyRunner(LocalEditOperator(pano.editor, pano.config))
        failures = seam_study.run_seam_study(
            args.manifest,
            runner,
            args.out_dir,
            args.strategies,
            args.seeds,
            args.tiled,
            args.scenes,
            args.num_shards,
            args.shard_index,
            args.prompt_key,
        )
        print(f"{len(failures)} degenerate case(s)")
        return 0
    if args.action == "score":
        rc = None
        if args.region_clip:
            from .evaluation.metrics.semantic import RegionCLIP

            rc = RegionCLIP(device=args.metric_device)
        print(
            seam_study.score_seam_study(
                args.manifest, args.out_dir, device=args.metric_device, region_clip=rc
            )
        )
        return 0
    text = seam_study.report_seam(Path(args.out_dir) / "per_case.csv")
    (Path(args.out_dir) / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    import numpy as np

    from .evaluation.metrics.reprojection import cycle_consistency
    from .geometry.camera import ViewParams
    from .geometry.projection import erp_to_perspective
    from .localization.annotations import reference_detection
    from .pipeline.editpano import load_image

    cfg = _config(args).projection
    rows = []
    for case in _cases(args):
        erp = load_image(case.panorama)
        H, W = erp.shape[:2]
        view = ViewParams.from_box(
            reference_detection(case.annotations).box,
            W,
            H,
            margin=cfg.fov_margin,
            lat_aware=cfg.lat_aware_fov,
            fov_min=cfg.fov_min,
            fov_max=cfg.fov_max,
        )
        patch = (
            load_image(Path(args.patches) / case.case_id / args.patch_name)
            if args.patches
            else erp_to_perspective(erp, view, cfg.patch_size, cfg.patch_size)[0]
        )
        rows.append({"case_id": case.case_id, **cycle_consistency(patch, erp, view)})
        print(json.dumps(rows[-1]))
    keys = [k for k in rows[0] if k != "case_id"] if rows else []
    print(json.dumps({k: float(np.mean([r[k] for r in rows])) for k in keys}, indent=2))
    return 0


def cmd_editors(_: argparse.Namespace) -> int:
    from .editors import available_editors

    print("\n".join(available_editors()))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    import yaml

    print(yaml.safe_dump(_config(args).to_dict(), sort_keys=False, allow_unicode=True))
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    import importlib

    ok = True
    for mod, extra in [
        ("numpy", "core"),
        ("cv2", "core"),
        ("PIL", "core"),
        ("yaml", "core"),
        ("torch", "edit"),
        ("diffusers", "edit"),
        ("transformers", "edit"),
        ("skimage", "eval"),
        ("scipy", "eval"),
        ("lpips", "eval"),
        ("clip", "eval"),
        ("peft", "generation"),
    ]:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {mod:13s} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = ok and extra != "core"
            print(f"  missing  {mod:13s} (pip install editpano[{extra}])")
    try:
        import torch

        print(
            f"  cuda     available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"
        )
    except ImportError:
        pass
    from .editors._torch import hf_snapshot_revision

    for mid in (
        "Qwen/Qwen-Image-Edit-2511",
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "IDEA-Research/grounding-dino-tiny",
    ):
        print(
            f"  weights  {mid}: {'cached ' + str(hf_snapshot_revision(mid))[:10] if hf_snapshot_revision(mid) else 'not cached'}"
        )
    return 0 if ok else 1


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="editpano", description="Geometry-consistent 360° panorama editing."
    )
    ap.add_argument("--version", action="version", version=f"editpano {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("edit", help="edit one panorama")
    _add_common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="ERP target box; X2 > width wraps across the seam",
    )
    g.add_argument("--annotations", help="Grounded-SAM result_*_mask.json (reference ROI)")
    g.add_argument("--detect", metavar="PHRASE", help="localise with frozen Grounding DINO")
    g.add_argument("--hint", type=float, nargs="+", metavar="DEG", help="LON LAT [FOV] view hint")
    p.add_argument(
        "--object-value", type=int, help="with --annotations: pick this annotation value"
    )
    p.add_argument(
        "--annotation-mask", action="store_true", help="recover a tight mask from the visualisation"
    )
    p.add_argument(
        "--multi-instance", action="store_true", help="edit every instance of the target"
    )
    p.add_argument("--save-intermediates", action="store_true")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("generate", help="text -> 2:1 ERP panorama (DiT360)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=2.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    p.set_defaults(func=cmd_generate)

    from .evaluation.arms import ARMS

    p = sub.add_parser("bench", help="reference-ROI benchmark")
    p.add_argument("action", choices=["run", "score", "report"])
    _add_common(p)
    _add_cases(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arms", nargs="+", default=["whole_erp", "editpano"], choices=sorted(ARMS))
    p.add_argument("--reference", default="editpano")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--region-clip", action="store_true", help="score Region-CLIP (needs CLIP)")
    p.add_argument("--ssim", action="store_true", help="also score BG-SSIM")
    p.add_argument(
        "--instance-support", action="store_true", help="support = union of instance boxes"
    )
    p.add_argument("--metric-device", default="cpu")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("seam", help="longitude-origin (seam) equivariance study")
    p.add_argument("action", choices=["build", "validate", "run", "score", "report"])
    _add_common(p)
    p.add_argument("--data-root")
    p.add_argument("--prompts")
    p.add_argument("--manifest")
    p.add_argument("--out-dir")
    p.add_argument("--near-gap-deg", type=float, default=5.0)
    p.add_argument("--expected-scenes", type=int)
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["identity", "current", "split", "direct_circular", "canonical"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--prompt-key", default="prompt")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--region-clip", action="store_true")
    p.add_argument("--metric-device", default="cpu")
    p.set_defaults(func=cmd_seam)

    p = sub.add_parser("diagnose", help="editor-free geometry diagnostics")
    p.add_argument("action", choices=["cycle"])
    _add_common(p, editor=False)
    _add_cases(p)
    p.add_argument(
        "--patches", help="folder <case_id>/<patch-name> of edited patches (default: source patch)"
    )
    p.add_argument("--patch-name", default="local_patch.png")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("editors", help="list registered editor backends")
    p.set_defaults(func=cmd_editors)
    p = sub.add_parser("config", help="print the resolved configuration")
    _add_common(p)
    p.set_defaults(func=cmd_config)
    p = sub.add_parser("doctor", help="check dependencies, CUDA and cached weights")
    p.set_defaults(func=cmd_doctor)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except Exception as exc:
        from .errors import EditPanoError

        if isinstance(exc, EditPanoError) and args.verbose == 0:
            print(f"editpano: error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
