"""Tests for the plate (back-wall) image-source ray model."""
import numpy as np
from scipy.signal import hilbert

from src.experiment_io import nominal_array_coordinates
from src.models.plate_ray import back_wall_paths, scatterer_paths, synthesize
from src.models.point_reflector_physics import (
    PropagationPhysics, pair_transfer, simulate_point_reflector_fmc_fd,
)

C, T, F0, SIGMA = 6300.0, 0.0494, 6.4e6, 80e-9
ELEMENTS = nominal_array_coordinates()
TIME = np.arange(1000) * 40e-9 - 0.4e-6
PLAIN = PropagationPhysics(geometric_spreading=True)


def test_back_wall_time_and_spreading_follow_image_source_geometry() -> None:
    paths = {p.label: p for p in back_wall_paths(ELEMENTS, T, C, F0, PropagationPhysics(), 1.0)}
    assert np.allclose(np.diag(paths["BW1"].travel_time_s), 2 * T / C)
    assert np.allclose(np.diag(paths["BW2"].travel_time_s), 4 * T / C)
    dx = ELEMENTS[5, 0] - ELEMENTS[40, 0]
    assert np.isclose(paths["BW1"].travel_time_s[5, 40], np.hypot(dx, 2 * T) / C)
    assert np.allclose(paths["BW1"].travel_time_s, paths["BW1"].travel_time_s.T)
    # lossless 2D spreading: the second echo is 1/sqrt(2) of the first (pulse-echo)
    ratio = np.diag(paths["BW2"].amplitude) / np.diag(paths["BW1"].amplitude)
    assert np.allclose(ratio, 1 / np.sqrt(2))


def test_round_trip_factor_scales_later_echoes_only() -> None:
    base = {p.label: p for p in back_wall_paths(ELEMENTS, T, C, F0, PropagationPhysics(), 2.0)}
    lossy = {p.label: p for p in back_wall_paths(ELEMENTS, T, C, F0, PropagationPhysics(), 2.0,
                                                 round_trip_factor=0.8)}
    assert np.allclose(lossy["BW1"].amplitude, base["BW1"].amplitude)
    assert np.allclose(lossy["BW2"].amplitude, 0.8 * base["BW2"].amplitude)


def test_direct_scatterer_path_matches_existing_point_model() -> None:
    s = np.array([0.003, 0.029])
    physics = PropagationPhysics(geometric_spreading=True, spreading_reference_amplitude_m=2e-3,
                                 element_width_m=0.53e-3)
    dd = scatterer_paths(ELEMENTS, s, T, C, F0, physics, families=("DD",))[0]
    amp, travel = pair_transfer(ELEMENTS, s, C, F0, physics)
    assert np.allclose(dd.amplitude, amp) and np.allclose(dd.travel_time_s, travel)


def test_scatterer_back_wall_paths_use_the_mirror_image() -> None:
    s = np.array([0.0, 0.03])
    paths = {p.label: p for p in scatterer_paths(ELEMENTS, s, T, C, F0, PLAIN)}
    i, j = 31, 32
    d_direct = np.hypot(ELEMENTS[:, 0] - s[0], ELEMENTS[:, 1] - s[1])
    d_image = np.hypot(ELEMENTS[:, 0] - s[0], ELEMENTS[:, 1] - (2 * T - s[1]))
    assert np.isclose(paths["DB"].travel_time_s[i, j], (d_direct[i] + d_image[j]) / C)
    assert np.isclose(paths["BD"].travel_time_s[i, j], (d_image[i] + d_direct[j]) / C)
    assert np.isclose(paths["BB"].travel_time_s[i, j], (d_image[i] + d_image[j]) / C)
    assert np.allclose(paths["DB"].travel_time_s, paths["BD"].travel_time_s.T)
    # a back-wall leg with |R| = 0.5 halves the amplitude per leg
    half = {p.label: p for p in scatterer_paths(ELEMENTS, s, T, C, F0, PLAIN, back_wall_leg_factor=0.5)}
    assert np.allclose(half["DB"].amplitude, 0.5 * paths["DB"].amplitude)
    assert np.allclose(half["BB"].amplitude, 0.25 * paths["BB"].amplitude)


def test_synthesis_puts_envelope_peak_at_arrival_and_matches_point_model() -> None:
    s = np.array([0.0, 0.03])
    physics = PropagationPhysics(geometric_spreading=True, spreading_reference_amplitude_m=1e-3)
    fmc, dropped = synthesize(TIME, scatterer_paths(ELEMENTS, s, T, C, F0, physics, families=("DD",)),
                              F0, SIGMA)
    reference = simulate_point_reflector_fmc_fd(TIME, ELEMENTS, s, C, F0, SIGMA, physics).fmc
    assert dropped == 0 and np.allclose(fmc, reference, atol=1e-12)
    env = np.abs(hilbert(fmc[10, 50]))
    expected = (np.hypot(*(ELEMENTS[10] - s)) + np.hypot(*(ELEMENTS[50] - s))) / C
    assert abs(TIME[np.argmax(env)] - expected) < 25e-9


def test_synthesis_drops_arrivals_beyond_the_record_instead_of_wrapping() -> None:
    paths = back_wall_paths(ELEMENTS, T, C, F0, PropagationPhysics(), 1.0, max_round_trips=3)
    fmc, dropped = synthesize(TIME, paths, F0, SIGMA)
    third = np.diag(paths[2].travel_time_s)[0]
    assert third > TIME[-1] and dropped >= ELEMENTS.shape[0] ** 2   # BW3 is entirely outside
    late = TIME > 0.9 * TIME[-1]
    early = (TIME > 0) & (TIME < 10e-6)
    assert np.abs(fmc[0, 0, early]).max() < 1e-3 * np.abs(fmc[0, 0]).max()  # nothing wrapped to the start
    assert np.abs(fmc[0, 0, late]).max() < np.abs(fmc[0, 0]).max()
