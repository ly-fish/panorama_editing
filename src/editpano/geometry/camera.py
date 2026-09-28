"""Spherical view parameters and the adaptive-FoV rule (thesis Eq. 3.3-3.5).

Conventions (shared by the whole package)::

    ERP pixel (u, v) of an H x W panorama
        longitude  lambda = (u / W - 1/2) * 360 deg        in [-180, 180)
        latitude   phi    = (1/2 - v / H) * 180 deg        in [-90, 90]

Longitude is periodic: column ``u`` and ``u + W`` are the same meridian.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ViewParams", "great_circle_deg", "lonlat_to_pixel", "pixel_to_lonlat"]

FOV_MIN_DEG = 30.0
FOV_MAX_DEG = 150.0
LAT_CLAMP_DEG = 85.0


def pixel_to_lonlat(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    """ERP pixel -> (longitude, latitude) in degrees."""
    return (u / width - 0.5) * 360.0, (0.5 - v / height) * 180.0


def lonlat_to_pixel(lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
    """(longitude, latitude) in degrees -> fractional ERP pixel."""
    return (lon / 360.0 + 0.5) * width, (0.5 - lat / 180.0) * height


def great_circle_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Angular distance between two directions on the sphere, in degrees."""
    a, b = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon1 - lon2)
    c = math.sin(a) * math.sin(b) + math.cos(a) * math.cos(b) * math.cos(dl)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _adaptive_fov(
    width_px: float,
    height_px: float,
    lat: float,
    img_w: int,
    img_h: int,
    margin: float,
    lat_aware: bool,
    fov_min: float,
    fov_max: float,
) -> float:
    fov_from_w = (width_px / img_w) * 360.0 * margin
    if lat_aware:
        fov_from_w *= math.cos(math.radians(min(abs(lat), LAT_CLAMP_DEG)))
    fov_from_h = (height_px / img_h) * 180.0 * margin
    return float(max(fov_min, min(fov_max, max(fov_from_w, fov_from_h))))


@dataclass(frozen=True)
class ViewParams:
    """A rectilinear (gnomonic) camera looking into the panorama sphere.

    Attributes:
        lat: latitude of the optical axis, degrees ``[-90, 90]``.
        lon: longitude of the optical axis, degrees ``[-180, 180]``.
        fov: horizontal field of view, degrees ``(0, 180)``.
    """

    lat: float
    lon: float
    fov: float

    def focal_length(self, patch_width: int) -> float:
        return (patch_width / 2.0) / math.tan(math.radians(self.fov) / 2.0)

    def to_dict(self) -> dict[str, float]:
        return {"lat": float(self.lat), "lon": float(self.lon), "fov": float(self.fov)}

    @staticmethod
    def from_box(
        box: Sequence[float],
        img_w: int,
        img_h: int,
        *,
        fov: float | None = None,
        margin: float = 1.5,
        lat_aware: bool = True,
        fov_min: float = FOV_MIN_DEG,
        fov_max: float = FOV_MAX_DEG,
    ) -> ViewParams:
        """View centred on an ordinary (non-wrapping) ERP box ``[x1, y1, x2, y2]``.

        With ``fov=None`` the FoV is ``clip(margin * max(span_lon', span_lat))``
        where ``span_lon' = span_lon * cos(min(|lat|, 85 deg))`` if ``lat_aware``.
        """
        x1, y1, x2, y2 = (float(c) for c in box)
        lon, lat = pixel_to_lonlat((x1 + x2) / 2.0, (y1 + y2) / 2.0, img_w, img_h)
        if fov is None:
            fov = _adaptive_fov(
                x2 - x1, y2 - y1, lat, img_w, img_h, margin, lat_aware, fov_min, fov_max
            )
        return ViewParams(lat=lat, lon=lon, fov=float(fov))

    @staticmethod
    def from_circular_box(
        box: Sequence[float],
        img_w: int,
        img_h: int,
        *,
        fov: float | None = None,
        margin: float = 1.5,
        lat_aware: bool = True,
        fov_min: float = FOV_MIN_DEG,
        fov_max: float = FOV_MAX_DEG,
    ) -> ViewParams:
        """Like :meth:`from_box` but the horizontal interval lives on S^1.

        ``box = [x_start, y1, x_start + width, y2]`` where ``x2`` may exceed the
        image width (the interval wraps across the +-180 deg seam).  The view
        centre is the *circular* centre ``(x_start + width / 2) mod W``.
        """
        x1, y1, x2, y2 = (float(c) for c in box)
        width = x2 - x1
        cx = (x1 + width / 2.0) % img_w
        lon, lat = pixel_to_lonlat(cx, (y1 + y2) / 2.0, img_w, img_h)
        lon = ((lon + 180.0) % 360.0) - 180.0
        if fov is None:
            fov = _adaptive_fov(
                width, y2 - y1, lat, img_w, img_h, margin, lat_aware, fov_min, fov_max
            )
        return ViewParams(lat=lat, lon=lon, fov=float(fov))
