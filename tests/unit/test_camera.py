import math

import pytest

from editpano.geometry.camera import ViewParams, great_circle_deg, lonlat_to_pixel, pixel_to_lonlat


def test_pixel_lonlat_roundtrip():
    for u, v in [(0, 0), (512, 256), (1023.5, 511.0), (17.25, 400.5)]:
        lon, lat = pixel_to_lonlat(u, v, 1024, 512)
        assert lonlat_to_pixel(lon, lat, 1024, 512) == pytest.approx((u, v))


def test_center_is_origin():
    assert pixel_to_lonlat(512, 256, 1024, 512) == (0.0, 0.0)


def test_adaptive_fov_margin_and_clamps():
    # 64 px of a 1024 px ERP = 22.5 deg; x1.5 = 33.75 deg at the equator.
    v = ViewParams.from_box([480, 240, 544, 272], 1024, 512)
    assert v.fov == pytest.approx(33.75)
    assert ViewParams.from_box([510, 254, 514, 258], 1024, 512).fov == 30.0  # lower clamp
    assert ViewParams.from_box([0, 100, 1000, 400], 1024, 512).fov == 150.0  # upper clamp


def test_lat_aware_fov_shrinks_high_latitude_spans():
    box = [480, 40, 580, 80]  # centre latitude ~ 73 deg
    plain = ViewParams.from_box(box, 1024, 512, lat_aware=False)
    aware = ViewParams.from_box(box, 1024, 512, lat_aware=True)
    assert aware.fov < plain.fov
    assert plain.lat == pytest.approx(aware.lat)


def test_circular_box_centres_on_the_seam():
    v = ViewParams.from_circular_box([1000, 200, 1048, 260], 1024, 512)  # 24 px each side of u=0
    assert v.lon == pytest.approx(-180.0)
    fov_w = 48 / 1024 * 360 * 1.5 * math.cos(math.radians(abs(v.lat)))
    fov_h = 60 / 512 * 180 * 1.5
    assert v.fov == pytest.approx(max(30.0, fov_w, fov_h))
    # the linear min/max centre of the same support would point at the wrong meridian
    assert ViewParams.from_box([0, 200, 1024, 260], 1024, 512).lon == pytest.approx(0.0)


def test_great_circle():
    assert great_circle_deg(0, 0, 0, 90) == pytest.approx(90)
    assert great_circle_deg(90, 0, -90, 0) == pytest.approx(180)
    assert great_circle_deg(10, 170, 10, -170) == pytest.approx(great_circle_deg(10, 0, 10, 20))
