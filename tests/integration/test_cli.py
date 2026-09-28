import json

import numpy as np
from PIL import Image

from editpano.cli import main
from tests._dataset import make_dataset
from tests._helpers import synthetic_pano


def test_cli_edit_with_box_and_intermediates(tmp_path, capsys):
    Image.fromarray(synthetic_pano(512, 256)).save(tmp_path / "in.png")
    rc = main(
        [
            "edit",
            "--input",
            str(tmp_path / "in.png"),
            "--prompt",
            "x",
            "--editor",
            "fake:pattern",
            "--box",
            "490",
            "100",
            "540",
            "150",
            "--output",
            str(tmp_path / "out.png"),
            "--save-intermediates",
        ]
    )
    assert rc == 0
    info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert info["branch"] == "canonical" and info["editor_calls"] == 1
    assert (tmp_path / "out_meta.json").exists()
    assert np.asarray(Image.open(tmp_path / "out.png")).shape == (256, 512, 3)


def test_cli_edit_with_annotations_and_multi_instance(tmp_path, capsys):
    data, _ = make_dataset(tmp_path)
    base = [
        "edit",
        "--input",
        str(data / "scene_a/panorama.jpg"),
        "--prompt",
        "x",
        "--editor",
        "fake:invert",
        "--annotations",
        str(data / "scene_a/result_1_mask.json"),
        "--output",
        str(tmp_path / "o.png"),
    ]
    assert main(base) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["branch"] == "local"
    assert main([*base, "--multi-instance"]) == 0
    assert (
        json.loads(capsys.readouterr().out.strip().splitlines()[-1])["branch"] == "multi_instance"
    )


def test_cli_bench_and_seam(tmp_path, capsys):
    data, prompts = make_dataset(tmp_path)
    common = ["--data-root", str(data), "--prompts", str(prompts), "--out-dir", str(tmp_path / "b")]
    assert (
        main(
            ["bench", "run", "--editor", "fake:pattern", "--arms", "whole_erp", "editpano", *common]
        )
        == 0
    )
    assert main(["bench", "score", "--arms", "whole_erp", "editpano", *common]) == 0
    assert main(["bench", "report", *common]) == 0
    assert "editpano" in (tmp_path / "b/report.md").read_text()
    assert (tmp_path / "b/editpano/scene_a/local_patch.png").exists()
    assert (
        main(
            [
                "diagnose",
                "cycle",
                "--data-root",
                str(data),
                "--prompts",
                str(prompts),
                "--patches",
                str(tmp_path / "b/editpano"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    summary = json.loads(out[out.rindex("\n{\n") + 1 :])
    assert summary["cycle_psnr_analytic"] > summary["cycle_psnr_scatter"]
    seam = ["--data-root", str(data), "--prompts", str(prompts), "--out-dir", str(tmp_path / "s")]
    assert main(["seam", "build", *seam]) == 0
    assert (
        main(
            [
                "seam",
                "validate",
                "--manifest",
                str(tmp_path / "s/manifest.json"),
                "--expected-scenes",
                "3",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "seam",
                "run",
                "--editor",
                "fake:pattern",
                "--manifest",
                str(tmp_path / "s/manifest.json"),
                "--out-dir",
                str(tmp_path / "r"),
                "--strategies",
                "identity",
                "canonical",
            ]
        )
        == 0
    )
    assert (tmp_path / "r/canonical/0/scene_a__cross/edited.png").exists()


def test_cli_misc(capsys):
    assert main(["editors"]) == 0
    assert "fake:pattern" in capsys.readouterr().out
    assert main(["config", "--set", "composite.feather_px=4"]) == 0
    assert "feather_px: 4" in capsys.readouterr().out
    assert main(["config", "--set", "composite.nope=1"]) == 2
