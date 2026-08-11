"""Tests for the first point-reflector propagation model."""
from pathlib import Path

import numpy as np

from src.geometry import PointReflector, array_coordinates, reflector_coordinate
from src.propagation import path_length_matrix, travel_time_matrix


MAT_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "raw"
    / "5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat"
)
WAVE_SPEED_M_S = 6300.0


def test_real_array_matrix_shapes_symmetry_and_positive_values() -> None:
    elements = array_coordinates(MAT_PATH)
    reflector = reflector_coordinate(PointReflector(x_m=0.0, z_m=0.04))

    paths = path_length_matrix(elements, reflector)
    times = travel_time_matrix(elements, reflector, WAVE_SPEED_M_S)

    assert paths.shape == (64, 64)
    assert times.shape == (64, 64)
    assert np.allclose(paths, paths.T)
    assert np.allclose(times, times.T)
    assert np.all(paths > 0.0)
    assert np.all(times > 0.0)


def test_central_ideal_element_pulse_echo_time() -> None:
    element = np.array([[0.0, 0.0]])
    reflector = np.array([0.0, 0.04])

    travel_time = travel_time_matrix(element, reflector, WAVE_SPEED_M_S)[0, 0]

    expected_s = 2.0 * 0.04 / WAVE_SPEED_M_S
    assert np.isclose(travel_time, expected_s)
    assert np.isclose(travel_time * 1e6, 12.70, atol=0.01)


def test_deeper_reflector_increases_every_travel_time() -> None:
    elements = array_coordinates(MAT_PATH)
    shallow = reflector_coordinate(PointReflector(x_m=0.0, z_m=0.04))
    deep = reflector_coordinate(PointReflector(x_m=0.0, z_m=0.05))

    shallow_times = travel_time_matrix(elements, shallow, WAVE_SPEED_M_S)
    deep_times = travel_time_matrix(elements, deep, WAVE_SPEED_M_S)

    assert np.all(deep_times > shallow_times)
