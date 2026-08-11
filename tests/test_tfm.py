"""Synthetic point-reflector tests for Total Focusing Method imaging."""
from pathlib import Path

import numpy as np
import pytest

from src.forward_model import simulate_point_reflector_fmc
from src.geometry import PointReflector, array_coordinates, reflector_coordinate
from src.imaging.tfm import tfm_image


MAT_PATH = Path(__file__).parents[1] / "data" / "raw" / "5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat"
ELEMENTS = array_coordinates(MAT_PATH)
TIME = np.arange(1000, dtype=float) * 20e-9
X_GRID = np.linspace(-0.01, 0.01, 41)
Z_GRID = np.linspace(0.032, 0.048, 33)


def _image(x_m: float, z_m: float) -> np.ndarray:
    reflector = reflector_coordinate(PointReflector(x_m=x_m, z_m=z_m))
    fmc, _ = simulate_point_reflector_fmc(
        TIME, ELEMENTS, reflector, 6300.0, 5e6, 0.35e-6
    )
    return tfm_image(fmc, TIME, ELEMENTS, X_GRID, Z_GRID, 6300.0)


@pytest.mark.parametrize("x_m,z_m", [(0.0, 0.04), (0.005, 0.044)])
def test_tfm_localises_and_moves_with_reflector(x_m: float, z_m: float) -> None:
    image = _image(x_m, z_m)
    peak_z_index, peak_x_index = np.unravel_index(np.argmax(image), image.shape)

    assert image.shape == (Z_GRID.size, X_GRID.size)
    assert np.all(np.isfinite(image))
    assert abs(X_GRID[peak_x_index] - x_m) <= 0.0005
    assert abs(Z_GRID[peak_z_index] - z_m) <= 0.0005
