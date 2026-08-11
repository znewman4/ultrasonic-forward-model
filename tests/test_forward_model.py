"""Tests for the first point-reflector FMC forward model."""
from pathlib import Path

import numpy as np

from src.forward_model import simulate_point_reflector_ascan, simulate_point_reflector_fmc
from src.geometry import PointReflector, array_coordinates, reflector_coordinate
from src.propagation import travel_time_matrix


MAT_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "raw"
    / "5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat"
)
TIME_S = np.arange(1000, dtype=float) * 20.0e-9
WAVE_SPEED_M_S = 6300.0
CENTRE_FREQUENCY_HZ = 5.0e6
SIGMA_S = 0.35e-6


def _simulate(depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elements = array_coordinates(MAT_PATH)
    reflector = reflector_coordinate(PointReflector(x_m=0.0, z_m=depth_m))
    fmc, travel_times = simulate_point_reflector_fmc(
        TIME_S,
        elements,
        reflector,
        WAVE_SPEED_M_S,
        CENTRE_FREQUENCY_HZ,
        SIGMA_S,
    )
    assert travel_times.shape == (64, 64)
    return fmc, elements, reflector


def test_output_shape_reciprocity_and_finite_values() -> None:
    fmc, elements, reflector = _simulate(0.04)
    expected = travel_time_matrix(elements, reflector, WAVE_SPEED_M_S)
    measured = TIME_S[np.argmax(np.abs(fmc), axis=2)]

    assert fmc.shape == (64, 64, 1000)
    assert np.allclose(fmc, fmc.transpose(1, 0, 2))
    assert np.all(np.isfinite(fmc))
    assert np.all(np.abs(measured - expected) <= 20.0e-9)


def test_centre_pulse_echo_peaks_near_predicted_arrival() -> None:
    fmc, elements, reflector = _simulate(0.04)
    centre_index = int(np.argmin(np.abs(elements[:, 0])))
    predicted = travel_time_matrix(
        elements, reflector, WAVE_SPEED_M_S
    )[centre_index, centre_index]
    measured = TIME_S[np.argmax(fmc[centre_index, centre_index])]

    assert abs(measured - predicted) <= 20.0e-9


def test_deeper_reflector_produces_later_pulses() -> None:
    shallow_fmc, elements, _ = _simulate(0.04)
    deep_fmc, _, _ = _simulate(0.05)
    centre_index = int(np.argmin(np.abs(elements[:, 0])))
    shallow_peak = TIME_S[np.argmax(np.abs(shallow_fmc), axis=2)]
    deep_peak = TIME_S[np.argmax(np.abs(deep_fmc), axis=2)]

    assert np.all(deep_peak > shallow_peak)


def test_one_ascan_returns_analytical_arrival_time() -> None:
    time = TIME_S
    trace, arrival = simulate_point_reflector_ascan(
        time,
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.04]),
        WAVE_SPEED_M_S,
        CENTRE_FREQUENCY_HZ,
        SIGMA_S,
    )
    assert trace.shape == time.shape
    assert np.isclose(arrival, 2.0 * 0.04 / WAVE_SPEED_M_S)
    assert abs(time[np.argmax(np.abs(trace))] - arrival) <= 20.0e-9
