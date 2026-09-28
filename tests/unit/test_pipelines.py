"""Benchmark and seam-study pipelines on a synthetic on-disk dataset."""

import csv
import importlib.util
import json

import pytest

from editpano import EditPano, EditPanoConfig
from editpano.editors.fake import FakeEditor
from editpano.evaluation import benchmark, load_cases
from editpano.evaluation.seam_benchmark import build_seam_benchmark, validate_seam_benchmark
from editpano.localization import load_annotations, reference_detection
from tests._dataset import make_dataset


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


def test_reference_roi_is_highest_logit(dataset):
    data, _ = dataset
    det = reference_detection(data / "scene_a/result_1_mask.json")
    assert det.score == 0.6 and det.source == "annotation"
    assert len(load_annotations(data / "scene_a/result_1_mask.json")) == 2


def test_load_cases(dataset):
    data, prompts = dataset
    cases = load_cases(data, prompts)
    assert [c.case_id for c in cases] == ["scene_a", "scene_b", "scene_c"]
    assert load_cases(data, prompts, scene_ids=["scene_b"])[0].clip_text.startswith("replace")


def test_benchmark_run_cache_score_report(tmp_path, dataset):
    data, prompts = dataset
    cases = load_cases(data, prompts)
    pano = EditPano(EditPanoConfig(), editor=FakeEditor("invert"))
    arms = ["whole_erp", "whole_then_paste", "erp_crop", "editpano", "cubemap"]
    out = tmp_path / "runs"
    assert benchmark.run_benchmark(cases, arms, pano, out) == []
    stamp = (out / "editpano/scene_a/edited.png").stat().st_mtime_ns
    benchmark.run_benchmark(cases, arms, pano, out)  # cache hit
    assert (out / "editpano/scene_a/edited.png").stat().st_mtime_ns == stamp
    other = EditPano(
        EditPanoConfig().with_overrides({"composite.feather_px": 4}), editor=FakeEditor("invert")
    )
    assert benchmark.run_signature(other, "editpano", cases[0]) != benchmark.run_signature(
        pano, "editpano", cases[0]
    )
    path = benchmark.score_benchmark(cases, arms, out)
    with open(path) as fh:
        rows = {(r["arm"], r["case_id"]): r for r in csv.DictReader(fh)}
    assert (
        float(rows[("editpano", "scene_a")]["bg_psnr"])
        > float(rows[("whole_erp", "scene_a")]["bg_psnr"]) + 20
    )
    assert float(rows[("editpano", "scene_a")]["coverage"]) > 0.8
    text = benchmark.report(path)
    assert "| editpano |" in text and "whole_erp | bg_psnr" in text


def test_sharding_partitions_cases(tmp_path, dataset):
    data, prompts = dataset
    cases = load_cases(data, prompts)
    pano = EditPano(editor=FakeEditor("identity"))
    for k in range(2):
        benchmark.run_benchmark(
            cases, ["editpano"], pano, tmp_path / "r", num_shards=2, shard_index=k
        )
    assert sorted(p.name for p in (tmp_path / "r/editpano").iterdir()) == [
        "scene_a",
        "scene_b",
        "scene_c",
    ]


def test_seam_benchmark_build_and_qa(tmp_path, dataset):
    data, prompts = dataset
    man = build_seam_benchmark(data, prompts, tmp_path / "seam")
    assert man["num_cases"] == 9
    assert validate_seam_benchmark(tmp_path / "seam/manifest.json", expected_scenes=3) == []
    cross = [c for c in man["cases"] if c["variant"] == "cross"]
    assert all(c["circular_roi"]["wrap"] for c in cross)
    # tampering is caught by the QA gate
    mpath = tmp_path / "seam/manifest.json"
    bad = json.loads(mpath.read_text())
    bad["cases"][0]["shift_px"] += 1
    mpath.write_text(json.dumps(bad))
    assert validate_seam_benchmark(mpath)


@pytest.mark.requires_torch
@pytest.mark.skipif(importlib.util.find_spec("lpips") is None, reason="lpips not installed")
def test_seam_study_end_to_end(tmp_path, dataset):
    from editpano.evaluation import seam_study
    from editpano.pipeline import LocalEditOperator, SeamStrategyRunner

    data, prompts = dataset
    build_seam_benchmark(data, prompts, tmp_path / "seam", limit=2)
    runner = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern")))
    seam_study.run_seam_study(
        tmp_path / "seam/manifest.json",
        runner,
        tmp_path / "runs",
        strategies=["current", "canonical"],
    )
    try:
        path = seam_study.score_seam_study(tmp_path / "seam/manifest.json", tmp_path / "runs")
    except Exception as exc:  # pragma: no cover - AlexNet weights unavailable offline
        pytest.skip(f"LPIPS backbone unavailable: {exc}")
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["variant"] == "cross"]
    canon = [float(r["ye_lpips"]) for r in rows if r["arm"] == "canonical"]
    assert canon and max(canon) == 0.0
    assert "Primary contrast" in seam_study.report_seam(path)


def test_detected_and_multi_instance_arms(tmp_path, dataset):
    from editpano import Detection

    class StubDetector:
        def __init__(self):
            self.queries = []

        def detect_roi(self, image, phrase):
            self.queries.append(phrase)
            return Detection.from_box([150, 90, 210, 150], score=0.4, source="stub")

        def detect_all(self, image, phrase):
            return [self.detect_roi(image, phrase)]

    data, prompts = dataset
    cases = load_cases(data, prompts)
    det = StubDetector()
    pano = EditPano(editor=FakeEditor("invert"), detector=det)
    out = tmp_path / "r"
    assert benchmark.run_benchmark(cases, ["editpano_detected", "editpano_multi"], pano, out) == []
    assert det.queries == ["chair"] * 3
    path = benchmark.score_benchmark(cases, ["editpano_multi"], out, instance_support=True)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert all(float(r["coverage"]) > 0.5 for r in rows)
