"""Typed, immutable configuration.

The defaults reproduce the thesis operating point exactly (512 px patch, FoV
margin 1.5 clamped to [30, 150] deg, latitude-aware FoV, 3 px patch-mask
feather, 8 px inward composite, confidence gate 0.7, Qwen-Image-Edit-2511 with
seed 0 / 40 steps / true-CFG 4.0).

Configs are plain frozen dataclasses: they can be loaded from YAML, overridden
with dotted keys (``composite.feather_px=16``), hashed for run signatures and
round-tripped through JSON.  Unknown keys are rejected loudly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .errors import ConfigError

__all__ = [
    "CompositeConfig",
    "EditPanoConfig",
    "EditorConfig",
    "GateConfig",
    "MultiInstanceConfig",
    "ProjectionConfig",
    "TilingConfig",
]

T = TypeVar("T")
_YAML_BOOL_WORDS = {"on", "off", "yes", "no", "y", "n"}


@dataclass(frozen=True)
class ProjectionConfig:
    patch_size: int = 512
    fov_margin: float = 1.5
    fov_min: float = 30.0
    fov_max: float = 150.0
    lat_aware_fov: bool = True
    hint_fov: float = 75.0


@dataclass(frozen=True)
class CompositeConfig:
    mode: str = "inward"  # inward | feather | poisson
    feather_px: int = 8
    constrain_patch_to_mask: bool = True
    patch_mask_feather: int = 3


@dataclass(frozen=True)
class EditorConfig:
    name: str = "qwen"  # registry key, e.g. qwen | sd_inpaint | fake:pattern
    model_id: str | None = None  # None -> backend default
    seed: int = 0
    device: str | None = None  # None -> cuda if available else cpu
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateConfig:
    confidence_threshold: float = 0.7
    detection_min_confidence: float = 0.7  # explicit detections are clamped up to this
    fallback: str = "whole"  # whole | raise


@dataclass(frozen=True)
class TilingConfig:
    mode: str = "off"  # off | auto | always
    tile_span_deg: float = 60.0
    overlap: float = 0.2
    max_tiles: int = 6
    auto_min_span_deg: float = 100.0  # "auto" tiles targets wider than this


@dataclass(frozen=True)
class MultiInstanceConfig:
    fov_budget: float = 110.0
    margin: float = 1.5
    max_groups: int = 6
    gate_thresh: float = 0.8
    use_instance_mask: bool = True
    seam_frac: float = 0.6


@dataclass(frozen=True)
class EditPanoConfig:
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    multi_instance: MultiInstanceConfig = field(default_factory=MultiInstanceConfig)
    canonicalize_seam: bool = True  # use F_can for wrapped supports

    def __post_init__(self) -> None:
        _validate(self)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EditPanoConfig:
        return _build(cls, dict(data or {}), path="")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EditPanoConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Stable 12-hex digest used in run signatures."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def with_overrides(self, overrides: Mapping[str, Any] | Iterable[str]) -> EditPanoConfig:
        """Return a copy with dotted-key overrides applied.

        Accepts ``{"composite.feather_px": 16}`` or ``["composite.feather_px=16"]``
        (string values are parsed as YAML scalars).
        """
        import yaml

        if not isinstance(overrides, Mapping):
            parsed: dict[str, Any] = {}
            for item in overrides:
                if "=" not in item:
                    raise ConfigError(f"override {item!r} must look like key=value")
                key, raw = item.split("=", 1)
                # YAML 1.1 reads on/off/yes/no as booleans; keep them as strings here.
                parsed[key.strip()] = (
                    raw.strip() if raw.strip().lower() in _YAML_BOOL_WORDS else yaml.safe_load(raw)
                )
            overrides = parsed
        data = self.to_dict()
        for key, value in overrides.items():
            node = data
            parts = key.split(".")
            for p in parts[:-1]:
                if not isinstance(node.get(p), dict):
                    raise ConfigError(f"unknown config section {p!r} in {key!r}")
                node = node[p]
            if parts[-1] not in node:
                raise ConfigError(f"unknown config key {key!r}")
            node[parts[-1]] = value
        return EditPanoConfig.from_dict(data)


# ---------------------------------------------------------------------- internals
def _build(cls: type[T], data: dict[str, Any], path: str) -> T:
    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - fields
    if unknown:
        raise ConfigError(
            f"unknown config key(s) {sorted(unknown)} in section {path or '<root>'!r}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path}{name} must be a mapping")
            kwargs[name] = _build(hint, dict(value), path=f"{path}{name}.")  # type: ignore[arg-type]
        else:
            kwargs[name] = _coerce(value, hint, f"{path}{name}")
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, name: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if value is None:
        if type(None) in args:
            return None
        raise ConfigError(f"{name} must not be null")
    target = (
        next((a for a in args if a is not type(None)), hint)
        if args and origin is not dict
        else hint
    )
    if typing.get_origin(target) is dict or target is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{name} must be a mapping")
        return dict(value)
    if target is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be a boolean, got {value!r}")
        return value
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be a number, got {value!r}")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            hint = (
                " (quote it: YAML reads bare on/off/yes/no as booleans)"
                if isinstance(value, bool)
                else ""
            )
            raise ConfigError(f"{name} must be a string, got {value!r}{hint}")
        return value
    return value


def _validate(cfg: EditPanoConfig) -> None:
    p, c, g, t, m = cfg.projection, cfg.composite, cfg.gate, cfg.tiling, cfg.multi_instance
    checks = [
        (p.patch_size >= 64, "projection.patch_size must be >= 64"),
        (p.fov_margin > 0, "projection.fov_margin must be > 0"),
        (0 < p.fov_min <= p.fov_max < 180, "require 0 < fov_min <= fov_max < 180"),
        (0 < p.hint_fov < 180, "projection.hint_fov must be in (0, 180)"),
        (c.mode in ("inward", "feather", "poisson"), f"composite.mode {c.mode!r} invalid"),
        (c.feather_px >= 0 and c.patch_mask_feather >= 0, "feather radii must be >= 0"),
        (0.0 <= g.confidence_threshold <= 1.0, "gate.confidence_threshold must be in [0, 1]"),
        (
            0.0 <= g.detection_min_confidence <= 1.0,
            "gate.detection_min_confidence must be in [0, 1]",
        ),
        (g.fallback in ("whole", "raise"), f"gate.fallback {g.fallback!r} invalid"),
        (t.mode in ("off", "auto", "always"), f"tiling.mode {t.mode!r} invalid"),
        (t.tile_span_deg > 0 and 0 <= t.overlap < 1 and t.max_tiles >= 1, "invalid tiling policy"),
        (m.max_groups >= 1 and 0 < m.fov_budget < 180, "invalid multi_instance policy"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ConfigError(msg)
