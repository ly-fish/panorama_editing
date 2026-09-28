"""EditPano -- geometry-consistent local editing of 360° ERP panoramas.

Wrap any frozen perspective image editor; EditPano handles the sphere:
localise -> gnomonic projection -> frozen edit -> analytic inverse
reprojection -> inward-feathered composite, with circular canonicalisation for
targets that cross the ERP seam.

>>> from editpano import EditPano
>>> pano = EditPano(editor="fake:pattern")          # deterministic CPU editor
"""

from ._version import __version__
from .config import EditPanoConfig
from .editors import (
    EditorInfo,
    EditRequest,
    PatchEditor,
    available_editors,
    create_editor,
    register_editor,
)
from .errors import (
    ConfigError,
    DatasetError,
    DegenerateROIError,
    EditorUnavailableError,
    EditPanoError,
    ROIRequiredError,
)
from .geometry import CircularROI, ViewParams
from .pipeline import EditPano, SeamStrategy
from .types import Detection, EditResult, PatchRecord, ViewHint

__all__ = [
    "CircularROI",
    "ConfigError",
    "DatasetError",
    "DegenerateROIError",
    "Detection",
    "EditPano",
    "EditPanoConfig",
    "EditPanoError",
    "EditRequest",
    "EditResult",
    "EditorInfo",
    "EditorUnavailableError",
    "PatchEditor",
    "PatchRecord",
    "ROIRequiredError",
    "SeamStrategy",
    "ViewHint",
    "ViewParams",
    "__version__",
    "available_editors",
    "create_editor",
    "register_editor",
]
