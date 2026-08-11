"""Tests for the geometric circular-reflector boundary approximation."""
import numpy as np

from src.forward_model import simulate_circular_reflector_fmc as legacy_circular_fmc
from src.geometry import CircularReflector as LegacyCircularReflector
from src.models.boundary_circle import (
    CircularReflector,
    circular_boundary_points,
    simulate_circular_reflector_fmc,
)
from src.models.point_reflector import simulate_point_reflector_fmc
from src.propagation import boundary_travel_times


ELEMENTS = np.column_stack((np.linspace(-3.0e-3, 3.0e-3, 5), np.zeros(5)))
TIME = np.arange(0.0, 18.0e-6, 20.0e-9)
CENTRE = np.array([0.4e-3, 40.0e-3])
SPEED = 6300.0
FREQUENCY = 5.0e6
SIGMA = 0.35e-6


def test_legacy_imports_are_compatibility_aliases() -> None:
    assert LegacyCircularReflector is CircularReflector
    assert legacy_circular_fmc is simulate_circular_reflector_fmc


def _circular_fmc(radius_m: float, n_points: int) -> np.ndarray:
    reflector = CircularReflector(CENTRE[0], CENTRE[1], radius_m, n_points)
    boundary = circular_boundary_points(reflector)
    fmc, _ = simulate_circular_reflector_fmc(
        TIME, ELEMENTS, boundary, SPEED, FREQUENCY, SIGMA
    )
    return fmc


def test_boundary_points_have_correct_shape_and_radius() -> None:
    reflector = CircularReflector(1.2e-3, 38.0e-3, 0.75e-3, 24)
    points = circular_boundary_points(reflector)
    radii = np.linalg.norm(points - np.array([reflector.x_m, reflector.z_m]), axis=1)

    assert points.shape == (24, 2)
    assert np.allclose(radii, reflector.radius_m)
    assert np.allclose(points[0], [reflector.x_m + reflector.radius_m, reflector.z_m])


def test_boundary_travel_time_and_fmc_shapes_and_reciprocity() -> None:
    boundary = circular_boundary_points(
        CircularReflector(CENTRE[0], CENTRE[1], 0.5e-3, 12)
    )
    times = boundary_travel_times(ELEMENTS, boundary, SPEED)
    fmc, returned_times = simulate_circular_reflector_fmc(
        TIME, ELEMENTS, boundary, SPEED, FREQUENCY, SIGMA
    )

    assert times.shape == (5, 5, 12)
    assert fmc.shape == (5, 5, TIME.size)
    assert np.allclose(returned_times, times)
    assert np.allclose(times, times.transpose(1, 0, 2))
    assert np.allclose(fmc, fmc.transpose(1, 0, 2))


def test_boundary_quadrature_converges_with_increasing_point_count() -> None:
    reference = _circular_fmc(0.5e-3, 256)
    errors = [
        np.linalg.norm(_circular_fmc(0.5e-3, n_points) - reference)
        for n_points in (8, 16, 32)
    ]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_radius_approaching_zero_approaches_point_reflector_model() -> None:
    point_fmc, _ = simulate_point_reflector_fmc(
        TIME, ELEMENTS, CENTRE, SPEED, FREQUENCY, SIGMA
    )
    circular_fmc = _circular_fmc(radius_m=1.0e-9, n_points=32)

    relative_error = np.linalg.norm(circular_fmc - point_fmc) / np.linalg.norm(point_fmc)
    assert relative_error < 1.0e-5
