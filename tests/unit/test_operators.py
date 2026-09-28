"""Operator-level guarantees, proven with deterministic fake editors."""

import numpy as np
import pytest

from editpano import DegenerateROIError, EditPanoConfig
from editpano.editors.fake import FakeEditor
from editpano.geometry import roll_image
from editpano.pipeline import (
    LocalEditOperator,
    MultiInstanceEditor,
    SeamStrategyRunner,
    cluster_instances,
)
from tests._helpers import wrapped_mask


def _changed(a, b):
    return np.any(a != b, axis=2)


def test_local_edit_preserves_everything_outside_the_box(pano):
    op = LocalEditOperator(FakeEditor("invert"))
    box = [300.0, 180.0, 420.0, 300.0]
    out = op.edit_box(pano, box, "p", 0)
    changed = _changed(out.image, pano)
    assert changed[200:280, 320:400].mean() > 0.9
    outside = np.ones_like(changed)
    outside[178:303, 298:423] = False  # box + 2 px rasterisation tolerance
    assert not changed[outside].any()
    assert out.editor_calls == 1 and out.patches[0].box == box


def test_identity_editor_is_near_lossless(pano):
    out = LocalEditOperator(FakeEditor("identity")).edit_box(pano, [300, 180, 420, 300], "p", 0)
    diff = np.abs(out.image.astype(int) - pano.astype(int))
    assert diff.max() <= 40 and diff.mean() < 0.2  # only resampling error inside the box


@pytest.mark.parametrize("q", [1, 17, -250, 511, 700])
def test_canonical_strategy_is_exactly_longitude_equivariant(pano, q):
    """F_can(T_q I, T_q m) == T_q F_can(I, m) bit-exactly (thesis Eq. 3.24)."""
    H, W = pano.shape[:2]
    m = wrapped_mask(W, H, 50, 70, 200, 280)
    run = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern"))).run
    ref = run("canonical", pano, m, "p", 0).image
    shifted = run("canonical", roll_image(pano, q), roll_image(m, q), "p", 0).image
    assert np.array_equal(shifted, roll_image(ref, q))


def test_current_strategy_is_not_equivariant(pano):
    H, W = pano.shape[:2]
    m = wrapped_mask(W, H, 50, 70, 200, 280)
    run = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern"))).run
    centre = run("current", roll_image(pano, 500), roll_image(m, 500), "p", 0).image
    cross = run("current", pano, m, "p", 0).image
    assert not np.array_equal(roll_image(centre, -500), cross)


def test_strategy_call_counts_and_identity(pano):
    H, W = pano.shape[:2]
    m = wrapped_mask(W, H, 50, 70, 200, 280)
    run = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern"))).run
    assert run("identity", pano, m, "p").editor_calls == 0
    assert np.array_equal(run("identity", pano, m, "p").image, pano)
    assert run("split", pano, m, "p").editor_calls == 2
    assert run("canonical", pano, m, "p").editor_calls == 1
    assert run("direct_circular", pano, m, "p").editor_calls == 1


def test_degenerate_supports_raise(pano):
    run = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern"))).run
    with pytest.raises(DegenerateROIError):
        run("canonical", pano, np.zeros(pano.shape[:2], np.uint8), "p")
    with pytest.raises(DegenerateROIError):
        run("canonical", pano, np.full(pano.shape[:2], 255, np.uint8), "p")


def test_tiled_canonical_equivariant_and_multiple_calls(pano):
    H, W = pano.shape[:2]
    m = wrapped_mask(W, H, 300, 200, 150, 330)  # ~176 deg wide
    run = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern"))).run
    a = run("canonical", pano, m, "p", tiled=True)
    b = run("canonical", roll_image(pano, 123), roll_image(m, 123), "p", tiled=True)
    assert a.editor_calls > 1
    assert np.array_equal(b.image, roll_image(a.image, 123))


def test_multi_instance_grouping_and_gap_preservation(pano):
    boxes = [[100, 200, 160, 260], [180, 210, 230, 270], [700, 220, 760, 280]]
    groups = cluster_instances(boxes, 1024, 512, fov_budget=110)
    assert len(groups) == 2 and sorted(len(g) for g in groups) == [1, 2]
    out = MultiInstanceEditor(LocalEditOperator(FakeEditor("invert"))).edit(pano, boxes, "p", 0)
    changed = _changed(out.image, pano)
    assert changed[230, 130] and changed[240, 200] and changed[250, 730]
    assert not changed[240, 170]  # the gap between grouped instances is untouched
    assert out.editor_calls == 2


def test_multi_instance_max_groups_cap(pano):
    boxes = [[i * 100, 240, i * 100 + 20, 260] for i in range(9)]
    cfg = EditPanoConfig().with_overrides(
        {"multi_instance.max_groups": 2, "multi_instance.fov_budget": 20}
    )
    ed = MultiInstanceEditor(LocalEditOperator(FakeEditor("pattern"), cfg))
    assert len(ed.plan(boxes, 1024, 512)) == 2
