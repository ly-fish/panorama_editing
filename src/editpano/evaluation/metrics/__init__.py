"""Evaluation metrics; heavy backends (CLIP, LPIPS, scikit-image) load lazily."""

from .preservation import (
    background_changed_rate,
    background_psnr,
    background_ssim,
    target_coverage,
)
from .reprojection import boundary_seam_ratio, cycle_consistency
from .seam import SpatialLPIPS, build_eval_views, ds_score, ye_lpips
from .semantic import RegionCLIP, roi_crop

__all__ = [
    "RegionCLIP",
    "SpatialLPIPS",
    "background_changed_rate",
    "background_psnr",
    "background_ssim",
    "boundary_seam_ratio",
    "build_eval_views",
    "cycle_consistency",
    "ds_score",
    "roi_crop",
    "target_coverage",
    "ye_lpips",
]
