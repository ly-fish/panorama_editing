# Data format

Datasets, model weights and experiment outputs are **not** part of this
repository. The benchmark tools read scenes laid out as follows.

## Scene folders

```
<data-root>/
└── scene_00/
    ├── panorama.jpg             # ERP panorama, 2:1 (thesis: 2048 × 1024)
    ├── result_42_mask.json      # Grounded-SAM annotations (required)
    ├── result_42_mask.jpg       # colour-coded mask visualisation (optional)
    └── scene_42_instruction.txt # fallback instruction (optional)
```

`result_*_mask.json` is a list of detections; `value == 0` is background:

```json
[{"value": 0, "label": "background"},
 {"value": 1, "label": "sofa", "logit": 0.47, "box": [812.4, 530.0, 1104.9, 690.2]}]
```

The **reference ROI** of a scene is the highest-logit foreground box
(`editpano.localization.reference_detection`). Box coordinates are ERP pixels
`[x1, y1, x2, y2]`; for a box crossing the ±180° seam pass `x2 > W`.

## Prompts file

A JSON list (or a `{scene_id: {...}}` mapping). Only `scene_id` and `prompt`
are required:

```json
[
  {"scene_id": "scene_00", "prompt": "Replace the sofa with a leather armchair",
   "object_label": "sofa"},
  {"scene_id": "scene_00", "prompt": "Remove the sofa", "edit_type": "removal",
   "clip_text": null},
  {"scene_id": "scene_01", "prompt": "将书桌替换为玻璃书桌", "tgt": "a glass desk"}
]
```

| Field | Meaning |
| --- | --- |
| `prompt` | instruction given to the editor (`--prompt-key` selects another field, e.g. `tgt` for English-only editors) |
| `object_label` | name of the source object (reports / seam manifest) |
| `edit_type` | several cases per scene; case id becomes `<scene_id>__<edit_type>` |
| `clip_text` | Region-CLIP text; defaults to `prompt`, `null` disables Region-CLIP (e.g. removal) |
| `tgt` | English target phrase used by the `sd_inpaint_roi` arm |

## Outputs

```
<out-dir>/<arm>/<case_id>/edited.png   # edited ERP (input resolution)
<out-dir>/<arm>/<case_id>/meta.json    # run signature, timing, box, failure reason
<out-dir>/per_case.csv                  # `editpano bench score`
<out-dir>/report.md                     # `editpano bench report`
```

The seam study writes `<out-dir>/<strategy>/<seed>/<case_id>/{edited.png,meta.json}`
next to a benchmark manifest produced by `editpano seam build`.
