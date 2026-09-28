"""Pure spherical geometry: no deep-learning dependencies.

Everything in this sub-package is deterministic numpy / OpenCV code and can be
unit-tested on a CPU in milliseconds.
"""

from .camera import ViewParams, great_circle_deg, lonlat_to_pixel, pixel_to_lonlat
from .circular import (
    CircularROI,
    circular_roi_from_box,
    circular_roi_from_mask,
    normalize_shift,
    roll_image,
    round_half_up,
)
from .compositing import (
    CompositeMode,
    composite_reprojection,
    constrain_patch_to_mask,
    inward_feather_blend,
)
from .masks import bbox_to_binary_mask, box_mask_in_perspective, instance_union_mask
from .projection import (
    erp_to_perspective,
    perspective_grid,
    project_mask_to_perspective,
    remap_patch_to_erp,
    scatter_patch_to_erp,
)
from .rotation import rotate_erp
from .tiling import split_box_into_tiles

__all__ = [
    "CircularROI",
    "CompositeMode",
    "ViewParams",
    "bbox_to_binary_mask",
    "box_mask_in_perspective",
    "circular_roi_from_box",
    "circular_roi_from_mask",
    "composite_reprojection",
    "constrain_patch_to_mask",
    "erp_to_perspective",
    "great_circle_deg",
    "instance_union_mask",
    "inward_feather_blend",
    "lonlat_to_pixel",
    "normalize_shift",
    "perspective_grid",
    "pixel_to_lonlat",
    "project_mask_to_perspective",
    "remap_patch_to_erp",
    "roll_image",
    "rotate_erp",
    "round_half_up",
    "scatter_patch_to_erp",
    "split_box_into_tiles",
]
