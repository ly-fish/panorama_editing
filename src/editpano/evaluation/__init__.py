"""Evaluation: metrics, paired statistics, benchmark construction and runners.

Heavy optional dependencies (CLIP, LPIPS, scikit-image, scipy) are imported
only by the functions that need them.
"""

from .datasets import Case, load_cases
from .stats import PairedResult, bootstrap_ci, cohens_dz, holm, paired_comparison, sign_flip_p

__all__ = [
    "Case",
    "PairedResult",
    "bootstrap_ci",
    "cohens_dz",
    "holm",
    "load_cases",
    "paired_comparison",
    "sign_flip_p",
]
