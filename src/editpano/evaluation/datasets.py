"""Benchmark scene discovery.

A scene directory holds ``panorama.jpg`` (ERP) and a Grounded-SAM annotation
``result_*_mask.json``.  A *prompts file* lists the cases to evaluate::

    [{"scene_id": "scene_00", "prompt": "...", "object_label": "...",
      "edit_type": "attribute",          # optional: several cases per scene
      "clip_text": "..."}, ...]          # optional: Region-CLIP text (null = skip)

A mapping ``{scene_id: {...}}`` is accepted as well.  See docs/data_format.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DatasetError

__all__ = ["Case", "load_cases"]


@dataclass(frozen=True)
class Case:
    case_id: str
    scene_id: str
    panorama: Path
    annotations: Path
    prompt: str
    object_label: str = "object"
    clip_text: str | None = None
    edit_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


def _rows(prompts: Path) -> list[dict[str, Any]]:
    data = json.loads(prompts.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [{"scene_id": k, **v} for k, v in data.items()]
    if not isinstance(data, list):
        raise DatasetError(f"{prompts}: expected a list or mapping of cases")
    return data


def load_cases(
    data_root: str | Path,
    prompts: str | Path,
    *,
    scene_ids: Iterable[str] | None = None,
    edit_types: Iterable[str] | None = None,
    limit: int | None = None,
    prompt_key: str = "prompt",
) -> list[Case]:
    """Resolve every case of ``prompts`` against the scene folders under ``data_root``."""
    root, prompts = Path(data_root), Path(prompts)
    wanted = set(scene_ids) if scene_ids else None
    types = set(edit_types) if edit_types else None
    cases: list[Case] = []
    for row in _rows(prompts):
        sid = row["scene_id"]
        if (wanted and sid not in wanted) or (types and row.get("edit_type") not in types):
            continue
        sdir = root / sid
        pano = sdir / "panorama.jpg"
        anns = sorted(sdir.glob("result_*_mask.json"))
        if not pano.exists() or not anns:
            raise DatasetError(f"{sid}: missing panorama.jpg or result_*_mask.json under {sdir}")
        prompt = row.get(prompt_key)
        if not prompt:
            instr = sorted(sdir.glob("scene_*_instruction.txt"))
            prompt = instr[0].read_text(encoding="utf-8").strip() if instr else ""
        if not prompt:
            raise DatasetError(f"{sid}: no {prompt_key!r} in prompts file and no instruction.txt")
        et = row.get("edit_type")
        clip_text = row.get("clip_text", prompt)
        cases.append(
            Case(
                f"{sid}__{et}" if et else sid,
                sid,
                pano,
                anns[0],
                prompt,
                row.get("object_label", "object"),
                clip_text,
                et,
                dict(row),
            )
        )
        if limit and len(cases) >= limit:
            break
    return cases
