import numpy as np
import pytest

from editpano.geometry.circular import (
    REASON_EMPTY,
    REASON_FULL_WIDTH,
    REASON_TIE_MAX_GAP,
    circular_roi_from_box,
    circular_roi_from_mask,
    normalize_shift,
    roll_image,
    round_half_up,
)

W, H = 256, 128


def box_mask(x1: int, x2: int, y1: int = 40, y2: int = 90) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    if x2 <= W:
        m[y1:y2, x1:x2] = 255
    else:
        m[y1:y2, x1:] = 255
        m[y1:y2, : x2 - W] = 255
    return m


def test_width_and_start_covariant_under_every_shift():
    base = box_mask(100, 180)
    r0 = circular_roi_from_mask(base)
    for s in range(W):
        r = circular_roi_from_mask(roll_image(base, s))
        assert (r.width, r.x_start, r.y1, r.y2) == (r0.width, (r0.x_start + s) % W, r0.y1, r0.y2)


@pytest.mark.parametrize("width", [80, 81])  # odd width exercises the .5 centre
def test_canonical_shift_identity_r_of_TqM_equals_r_minus_q(width):
    base = box_mask(100, 100 + width)
    r0 = circular_roi_from_mask(base).canonical_shift()
    for q in range(-W, W, 7):
        rq = circular_roi_from_mask(roll_image(base, q)).canonical_shift()
        assert (rq - (r0 - q)) % W == 0
        canon = circular_roi_from_mask(roll_image(roll_image(base, q), rq))
        assert not canon.wrap and abs(canon.center - W / 2) <= 0.5


def test_wrapped_support_is_short_arc_not_full_width():
    m = box_mask(236, 276)  # 20 px each side of the seam
    roi = circular_roi_from_mask(m)
    assert roi.wrap and roi.width == 40 and roi.x_start == 236
    assert roi.naive_box(m) == [0, 40, 256, 90]  # the linear box degenerates
    assert roi.split_boxes() == [[236, 40, 256, 90], [0, 40, 20, 90]]
    assert np.array_equal(roi.rasterize(), m)


def test_degenerate_reasons():
    assert circular_roi_from_mask(np.zeros((H, W), np.uint8)).reason == REASON_EMPTY
    assert circular_roi_from_mask(np.full((H, W), 255, np.uint8)).reason == REASON_FULL_WIDTH
    m = np.zeros((H, W), np.uint8)
    m[:, [0, 128]] = 255  # two equal gaps
    roi = circular_roi_from_mask(m)
    assert roi.reason == REASON_TIE_MAX_GAP and roi.tie_count == 2 and roi.x_start == 0


def test_round_half_up_commutes_with_integer_shifts():
    for x in np.arange(-5, 5, 0.5):
        for q in range(-3, 4):
            assert round_half_up(x - q) == round_half_up(x) - q
    assert round(2.5) == 2 and round_half_up(2.5) == 3  # banker's rounding would break equivariance


def test_normalize_shift_range():
    for s in range(-600, 600, 13):
        n = normalize_shift(s, W)
        assert -W // 2 <= n < W // 2 and (n - s) % W == 0


def test_from_box_and_membership():
    roi = circular_roi_from_box([240, 10, 272, 50], W, H)
    assert roi.wrap and roi.width == 32
    assert roi.contains_u(np.array([250.0, 3.0, 20.0])).tolist() == [True, True, False]
    assert circular_roi_from_dict_roundtrip(roi)


def circular_roi_from_dict_roundtrip(roi) -> bool:
    from editpano.geometry.circular import CircularROI

    return CircularROI.from_dict(roi.to_dict()) == roi
