"""Shared fixtures."""

from __future__ import annotations

import os

# Oversubscribed BLAS/OpenMP threads make CPU inference 10-20x slower on quota-limited
# machines (e.g. shared login nodes); keep tests predictable.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

from editpano import EditPano, EditPanoConfig
from editpano.editors.fake import FakeEditor
from tests._helpers import synthetic_pano


@pytest.fixture
def pano() -> np.ndarray:
    return synthetic_pano()


@pytest.fixture
def small_pano() -> np.ndarray:
    return synthetic_pano(512, 256, seed=1)


@pytest.fixture
def fake_pano() -> EditPano:
    return EditPano(EditPanoConfig(), editor=FakeEditor("pattern"))
