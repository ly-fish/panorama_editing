import math

import numpy as np
import pytest

from editpano.evaluation.metrics import (
    background_changed_rate,
    background_psnr,
    ds_score,
    target_coverage,
)
from editpano.evaluation.metrics.seam import boundary_average_attack, build_eval_views
from editpano.evaluation.stats import cohens_dz, holm, paired_comparison, sign_flip_p, stars
from tests._helpers import synthetic_pano


def test_preservation_metrics():
    a = synthetic_pano(256, 128)
    fg = np.zeros((128, 256), bool)
    fg[40:80, 100:150] = True
    b = a.copy()
    assert background_psnr(a, b, fg) == 99.0
    b[fg] = 255 - b[fg]
    assert background_psnr(a, b, fg) == 99.0 and target_coverage(a, b, fg) > 0.9
    b[0, 0] = (b[0, 0].astype(int) + 50) % 256
    assert 40 < background_psnr(a, b, fg) < 99 and background_changed_rate(a, b, fg) > 0
    assert background_psnr(a, b, np.ones_like(fg)) is None  # no background left
    assert target_coverage(a, b, np.zeros_like(fg)) is None


def test_coverage_threshold_is_12_over_255():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[:2] = 12
    b[2:] = 13
    assert target_coverage(a, b, np.ones((4, 4), bool)) == 0.5


def test_ds_detects_seam_step_and_attack_hides_it():
    img = synthetic_pano(256, 128)
    assert ds_score(img) < 3
    stepped = img.copy()
    stepped[:, :3] = 255 - stepped[:, :3]
    assert ds_score(stepped) > ds_score(img)
    assert ds_score(boundary_average_attack(stepped)) < ds_score(stepped)


def test_eval_views_tile_super_wide_supports():
    m = np.zeros((256, 512), np.uint8)
    m[100:150, 200:260] = 255
    assert len(build_eval_views(m)) == 1
    m[:, :] = 0
    m[80:160, 0:300] = 255  # > 150 deg
    assert len(build_eval_views(m)) > 1


def test_sign_flip_and_effect_size():
    d = np.full(31, 2.0) + np.linspace(-0.1, 0.1, 31)
    assert sign_flip_p(d, n_flips=20000) == pytest.approx(1 / 20001)  # add-one floor
    assert sign_flip_p(np.array([1.0, -1.0, 1.0, -1.0])) > 0.5
    assert cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_paired_comparison_and_holm():
    a = {f"s{i}": 10.0 + i for i in range(10)}
    b = {f"s{i}": 5.0 + i for i in range(10)}
    b["only_b"] = 1.0
    a["nan"], b["nan"] = math.nan, 1.0
    r = paired_comparison(a, b)
    assert r.n == 10 and r.mean_diff == 5.0 and r.losses == 10 and r.ci95 == (5.0, 5.0)
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj == {"x": 0.03, "z": 0.06, "y": 0.06}
    assert stars(1e-4) == "***" and stars(0.2) == "n.s."
