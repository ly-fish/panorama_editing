"""Frozen perspective editors behind a single port (:class:`PatchEditor`)."""

from .base import EditorInfo, EditRequest, PatchEditor
from .registry import available_editors, create_editor, register_editor

__all__ = [
    "EditRequest",
    "EditorInfo",
    "PatchEditor",
    "available_editors",
    "create_editor",
    "register_editor",
]
