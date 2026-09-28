"""Build a tiny synthetic benchmark on disk (Grounded-SAM-style layout)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tests._helpers import synthetic_pano

BOXES = {
    "scene_a": [300.4, 180.0, 420.9, 300.2],
    "scene_b": [620.0, 60.0, 700.0, 150.0],
    "scene_c": [80.0, 220.0, 260.0, 330.0],
}


def make_dataset(root: Path, W: int = 512, H: int = 256) -> tuple[Path, Path]:
    data = root / "data"
    rows = []
    for i, (sid, box) in enumerate(BOXES.items()):
        sdir = data / sid
        sdir.mkdir(parents=True)
        Image.fromarray(synthetic_pano(W, H, seed=i)).save(sdir / "panorama.jpg", quality=95)
        b = [v * W / 1024 for v in box]
        anns = [
            {"value": 0, "label": "background"},
            {"value": 1, "label": "chair", "logit": 0.3, "box": [b[0] + 5, b[1], b[2] - 5, b[3]]},
            {"value": 2, "label": "chair", "logit": 0.6, "box": b},
        ]
        (sdir / "result_1_mask.json").write_text(json.dumps(anns))
        rows.append(
            {"scene_id": sid, "prompt": f"replace the chair in {sid}", "object_label": "chair"}
        )
    prompts = root / "prompts.json"
    prompts.write_text(json.dumps(rows))
    return data, prompts
