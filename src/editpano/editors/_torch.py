"""Shared helpers for torch-backed editors (imported lazily)."""

from __future__ import annotations

from typing import Any

from ..errors import EditorUnavailableError


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EditorUnavailableError(
            "this editor needs PyTorch; install with `pip install editpano[edit]`"
        ) from exc
    return torch


def resolve_device(device: str | None) -> str:
    torch = require_torch()
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def hf_snapshot_revision(model_id: str) -> str | None:
    """Commit hash of the locally cached Hugging Face snapshot, if any."""
    import os
    from pathlib import Path

    hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    ref = hub / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None
