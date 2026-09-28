"""Target localisation: reference annotations and frozen open-vocabulary detection."""

from .annotations import Annotation, annotation_mask, load_annotations, reference_detection
from .grounding import GroundingDinoDetector

__all__ = [
    "Annotation",
    "GroundingDinoDetector",
    "annotation_mask",
    "load_annotations",
    "reference_detection",
]
