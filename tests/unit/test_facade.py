import numpy as np
import pytest

from editpano import Detection, EditPano, EditPanoConfig, ROIRequiredError, ViewHint
from editpano.editors.fake import FakeEditor
from tests._helpers import wrapped_mask


def test_routing(fake_pano, pano):
    H, W = pano.shape[:2]
    assert fake_pano.edit(pano, "p", box=[300, 180, 420, 300]).branch == "local"
    assert fake_pano.edit(pano, "p", box=[1000, 200, 1060, 280]).branch == "canonical"
    assert (
        fake_pano.edit(pano, "p", mask=wrapped_mask(W, H, 40, 40, 200, 260)).branch == "canonical"
    )
    assert (
        fake_pano.edit(pano, "p", instances=[[100, 200, 160, 260], [700, 220, 760, 280]]).branch
        == "multi_instance"
    )
    assert fake_pano.edit(pano, "p", hint=ViewHint(lon=30, lat=10)).branch == "local"
    assert fake_pano.edit(pano, "p").branch == "whole"


def test_tiling_policy(pano):
    auto = EditPano(
        EditPanoConfig().with_overrides({"tiling.mode": "auto"}), editor=FakeEditor("pattern")
    )
    assert auto.edit(pano, "p", box=[300, 180, 420, 300]).branch == "local"
    r = auto.edit(pano, "p", box=[100, 150, 700, 330])
    assert r.branch == "tiled" and r.editor_calls > 1
    assert auto.edit(pano, "p", box=[900, 150, 1400, 330]).branch == "canonical_tiled"


def test_confidence_gate_and_fallback(pano):
    strict = EditPanoConfig().with_overrides({"gate.confidence_threshold": 0.9})
    p = EditPano(strict, editor=FakeEditor("pattern"))
    low = Detection.from_box([300, 180, 420, 300], score=0.5)
    assert p.edit(pano, "p", detection=low).branch == "whole"  # 0.5 -> clamped to 0.7 < 0.9
    assert (
        p.edit(pano, "p", detection=Detection.from_box([300, 180, 420, 300], 0.95)).branch
        == "local"
    )
    raising = EditPano(
        EditPanoConfig().with_overrides({"gate.fallback": "raise"}), editor=FakeEditor("pattern")
    )
    with pytest.raises(ROIRequiredError):
        raising.edit(pano, "p")


def test_result_metadata_and_save(tmp_path, fake_pano, pano):
    r = fake_pano.edit(pano, "replace it", box=[300, 180, 420, 300], seed=3)
    assert r.meta["seed"] == 3 and r.meta["config"] == EditPanoConfig().fingerprint()
    assert r.meta["view"]["fov"] >= 30 and r.editor_calls == 1
    out = r.save(tmp_path / "x.png", intermediates=True)
    assert (
        out.exists()
        and (tmp_path / "x_meta.json").exists()
        and (tmp_path / "x_patch0_edited.png").exists()
    )


def test_prompt_policy_and_image_inputs(tmp_path, pano):
    seen = []

    class Spy(FakeEditor):
        def edit(self, request):
            seen.append(request.prompt)
            return super().edit(request)

    p = EditPano(editor=Spy("identity"), prompt_policy=lambda t: t + " | keep perspective")
    from PIL import Image

    Image.fromarray(pano).save(tmp_path / "in.png")
    p.edit(tmp_path / "in.png", "add a lamp", box=[300, 180, 420, 300])
    assert seen == ["add a lamp | keep perspective"]
    with pytest.raises(ValueError):
        p.edit(np.zeros((10, 10), np.uint8), "x", box=[1, 1, 5, 5])


def test_invalid_box():
    with pytest.raises(ValueError):
        Detection.from_box([10, 10, 5, 20])
