"""Exception hierarchy.  Every error raised on purpose derives from :class:`EditPanoError`."""

from __future__ import annotations


class EditPanoError(Exception):
    """Base class for all EditPano errors."""


class ConfigError(EditPanoError, ValueError):
    """Invalid or unknown configuration value."""


class DegenerateROIError(EditPanoError):
    """The target support has no unique circular centre (empty or full width)."""

    def __init__(self, reason: str):
        super().__init__(f"degenerate circular ROI: {reason}")
        self.reason = reason


class ROIRequiredError(EditPanoError):
    """No usable target region was found and whole-ERP fallback is disabled."""


class EditorUnavailableError(EditPanoError, ImportError):
    """An editor backend was requested whose optional dependencies are missing."""


class DatasetError(EditPanoError):
    """A benchmark scene or manifest is missing or inconsistent."""
