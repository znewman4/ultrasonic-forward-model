"""Geometry definitions for the ultrasonic array and reflector models."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .data_loader import load_fmc
    from .models.boundary_circle import CircularReflector, circular_boundary_points
    from .models.point_reflector import PointReflector, reflector_coordinate
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from data_loader import load_fmc
    from models.boundary_circle import CircularReflector, circular_boundary_points
    from models.point_reflector import PointReflector, reflector_coordinate


EXPECTED_ELEMENT_COUNT = 64
EXPECTED_APERTURE_M = 39.69e-3
APERTURE_TOLERANCE_M = 0.10e-3


def array_coordinates(mat_path: str | Path) -> np.ndarray:
    """Return measured element-centre coordinates as an ``(64, 2)`` x-z array.

    The x coordinates come from the MAT file. The modelled array surface lies at
    z=0, consistent with the stored element-centre z coordinates.
    """
    fmc = load_fmc(mat_path)
    centres = np.asarray(fmc.metadata.array.element_centres_m, dtype=float)

    if centres.shape != (EXPECTED_ELEMENT_COUNT, 3):
        raise ValueError(
            f"Expected {EXPECTED_ELEMENT_COUNT} element centres with x/y/z coordinates; "
            f"got shape {centres.shape}"
        )
    if not np.allclose(centres[:, 2], 0.0, atol=1e-12):
        raise ValueError("MAT-file element centres are not all located at z=0")

    aperture_m = float(np.ptp(centres[:, 0]))
    if not np.isclose(aperture_m, EXPECTED_APERTURE_M, atol=APERTURE_TOLERANCE_M):
        raise ValueError(
            f"Expected an aperture of approximately {EXPECTED_APERTURE_M * 1e3:.2f} mm; "
            f"got {aperture_m * 1e3:.3f} mm"
        )

    coordinates = np.column_stack((centres[:, 0], np.zeros(EXPECTED_ELEMENT_COUNT)))
    return coordinates
