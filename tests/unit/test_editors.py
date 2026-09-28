import numpy as np
import pytest
from PIL import Image

from editpano import (
    ConfigError,
    EditRequest,
    PatchEditor,
    available_editors,
    create_editor,
    register_editor,
)
from editpano.editors.fake import FakeEditor
from editpano.editors.qwen import MASK_REFERENCE_INSTRUCTION, make_mask_reference


def test_builtin_registry():
    names = available_editors()
    assert {"qwen", "sd_inpaint", "fake:identity", "fake:invert", "fake:pattern"} <= set(names)
    assert isinstance(create_editor("fake:pattern"), PatchEditor)
    with pytest.raises(ConfigError):
        create_editor("does-not-exist")


def test_register_custom_editor():
    class Gray:
        info = property(lambda self: __import__("editpano").EditorInfo("gray"))

        def edit(self, request):
            return request.image.convert("L").convert("RGB")

    register_editor("test:gray")(Gray)
    assert create_editor("test:gray").info.name == "gray"
    with pytest.raises(ConfigError):
        register_editor("test:gray")(lambda: None)


def test_fake_pattern_respects_mask_and_seed(small_pano):
    img = Image.fromarray(small_pano[:64, :64])
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 255
    e = FakeEditor("pattern")
    a = np.asarray(e.edit(EditRequest(img, "p", Image.fromarray(mask), seed=0)))
    b = np.asarray(e.edit(EditRequest(img, "p", Image.fromarray(mask), seed=5)))
    assert np.array_equal(a[mask == 0], small_pano[:64, :64][mask == 0])
    assert not np.array_equal(a, b)


def test_mask_reference_is_deterministic_and_marks_support():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (16, 16, 48, 48))
    ref = np.asarray(make_mask_reference(img, mask))
    assert tuple(ref[0, 0]) == (100, 100, 100)
    assert tuple(ref[16, 30]) == (0, 255, 255)  # outline
    assert ref[32, 32, 1] > ref[32, 32, 0]  # cyan tint inside
    assert np.array_equal(ref, np.asarray(make_mask_reference(img, mask)))
    assert "cyan" in MASK_REFERENCE_INSTRUCTION


def test_facade_builds_plugin_editor_without_kwargs(small_pano):
    from editpano import EditorInfo, EditPano

    @register_editor("test:plain")
    class Plain:
        info = EditorInfo("plain")

        def edit(self, request):
            return request.image

    r = EditPano(editor="test:plain").edit(small_pano, "p", box=[100, 80, 160, 140])
    assert r.meta["editor"]["name"] == "plain"
