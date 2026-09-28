"""Golden tests: outputs must match the thesis implementation bit-for-bit.

``tests/golden/legacy_outputs.json`` holds SHA-256 digests produced by the
original research code on synthetic inputs with the deterministic ``pattern``
fake editor.  Floating-point kernels in OpenCV/NumPy may differ across library
versions, so the exact comparison runs only in the recorded environment.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from editpano import EditPano, EditPanoConfig
from editpano.editors.fake import FakeEditor
from editpano.pipeline import LocalEditOperator, MultiInstanceEditor, SeamStrategyRunner
from tests._helpers import synthetic_pano

GOLD = json.loads((Path(__file__).resolve().parents[1] / "golden/legacy_outputs.json").read_text())
ENV = {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__}
pytestmark = pytest.mark.skipif(
    GOLD["env"] != ENV, reason=f"goldens recorded with {GOLD['env']}, have {ENV}"
)


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def img():
    return synthetic_pano()


def _pano(**over):
    return EditPano(
        EditPanoConfig().with_overrides(over) if over else EditPanoConfig(),
        editor=FakeEditor("pattern"),
    )


def test_local(img):
    assert (
        sha(_pano().edit(img, "p", box=[300.5, 180.2, 420.7, 300.9]).image)
        == GOLD["cases"]["local_box"]
    )
    assert (
        sha(_pano().edit(img, "p", box=[600, 20, 760, 110]).image)
        == GOLD["cases"]["local_box_highlat"]
    )
    assert (
        sha(
            _pano(**{"composite.mode": "feather"})
            .edit(img, "p", box=[300.5, 180.2, 420.7, 300.9])
            .image
        )
        == GOLD["cases"]["local_box_feather"]
    )


def test_seam_strategies(img):
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    m[200:280, W - 50 :] = 255
    m[200:280, :70] = 255
    run = SeamStrategyRunner(LocalEditOperator(FakeEditor("pattern"))).run
    for legacy, new in [
        ("current", "current"),
        ("split", "split"),
        ("direct_circular", "direct_circular"),
        ("proposed", "canonical"),
    ]:
        assert sha(run(new, img, m, "p", 0).image) == GOLD["cases"][f"seam_{legacy}"], new
    wide = np.zeros((H, W), np.uint8)
    wide[150:330, W - 300 :] = 255
    wide[150:330, :200] = 255
    assert (
        sha(run("split", img, wide, "p", 0, tiled=True).image) == GOLD["cases"]["seam_split_tiled"]
    )
    assert (
        sha(run("canonical", img, wide, "p", 0, tiled=True).image)
        == GOLD["cases"]["seam_canonical_tiled"]
    )


def test_tiled_and_multi_instance(img):
    tiled = _pano(**{"tiling.mode": "always"}).edit(img, "p", box=[100.0, 150.0, 700.0, 330.0])
    assert sha(tiled.image) == GOLD["cases"]["tiled_wide_box"]
    boxes = [[100, 200, 160, 260], [180, 210, 230, 270], [700, 220, 760, 280], [400, 60, 440, 100]]
    out = MultiInstanceEditor(LocalEditOperator(FakeEditor("pattern"))).edit(
        img, boxes, "p", 0, primary_box=boxes[0]
    )
    assert sha(out.image) == GOLD["cases"]["multi_instance"]
