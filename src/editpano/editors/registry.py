"""Name -> factory registry for editor backends.

Built-in names: ``qwen``, ``sd_inpaint``, ``fake:identity``, ``fake:invert``,
``fake:pattern``.  Third-party packages can expose additional factories under
the ``editpano.editors`` entry-point group; they are discovered lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ConfigError
from .base import PatchEditor

__all__ = ["available_editors", "create_editor", "register_editor"]

EditorFactory = Callable[..., PatchEditor]
_REGISTRY: dict[str, EditorFactory] = {}
_ENTRY_POINTS_LOADED = False


def register_editor(name: str) -> Callable[[EditorFactory], EditorFactory]:
    """Decorator registering an editor factory under ``name``."""

    def deco(factory: EditorFactory) -> EditorFactory:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ConfigError(f"editor {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def _load_builtins() -> None:
    from . import fake, qwen, sd_inpaint  # noqa: F401  (registration side effects)


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    from importlib.metadata import entry_points

    for ep in entry_points(group="editpano.editors"):
        if ep.name not in _REGISTRY:
            _REGISTRY[ep.name] = ep.load()


def available_editors() -> list[str]:
    _load_builtins()
    _load_entry_points()
    return sorted(_REGISTRY)


def create_editor(name: str, **kwargs: Any) -> PatchEditor:
    """Instantiate a registered editor; heavy weights load lazily on first edit."""
    _load_builtins()
    if name not in _REGISTRY:
        _load_entry_points()
    if name not in _REGISTRY:
        raise ConfigError(f"unknown editor {name!r}; available: {available_editors()}")
    return _REGISTRY[name](**kwargs)
