"""Tests for the M0 physics ladder: spreading, directivity and attenuation."""
import numpy as np
import pytest

from src.models.point_reflector import simulate_point_reflector_fmc
from src.models.point_reflector_physics import (
    PropagationPhysics,
    pair_transfer,
    physics_ladder,
    simulate_ladder_step,
    simulate_point_reflector_fmc_fd,
    simulate_point_reflector_fmc_td,
)
from src.propagation import (
    attenuation_coefficient,
    beam_spread_amplitude,
    element_directivity,
    element_ray_geometry,
)
from src.pulse import gaussian_pulse_spectrum, shifted_pulse


# Holmes, Drinkwater & Wilcox (2005) Table 1 array, matching the measured MAT file.
N_ELEMENTS = 64
PITCH_M = 0.63e-3
WIDTH_M = 0.53e-3
ELEMENTS = np.column_stack(
    ((np.arange(N_ELEMENTS) - (N_ELEMENTS - 1) / 2.0) * PITCH_M, np.zeros(N_ELEMENTS))
)
REFLECTOR = np.array([0.0, 0.04])
SPEED = 6300.0
F0 = 5.0e6
SIGMA = 0.35e-6
TIME = np.arange(1000) * 20.0e-9
ALPHA = 1.15  # Np/m


def _spectral_centroid(trace: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(trace)) ** 2
    frequency = np.fft.rfftfreq(trace.size, TIME[1] - TIME[0])
    return float(np.sum(frequency * spectrum) / np.sum(spectrum))


# --- Individual physics factors -------------------------------------------


def test_ray_geometry_angle_convention() -> None:
    distances, sine = element_ray_geometry(
        np.array([[0.0, 0.0], [0.03, 0.0], [-0.03, 0.0]]), np.array([0.0, 0.04])
    )
    assert np.allclose(distances, [0.04, 0.05, 0.05])
    # Reflector straight below -> normal incidence; offset elements see +/-36.87 deg.
    assert np.allclose(sine, [0.0, -0.6, 0.6])


def test_directivity_matches_holmes_eq5() -> None:
    theta = np.deg2rad(np.linspace(-89.0, 89.0, 41))
    wavelength = SPEED / F0
    x = np.pi * WIDTH_M * np.sin(theta) / wavelength
    expected = np.where(x == 0.0, 1.0, np.sin(x) / np.where(x == 0.0, 1.0, x))
    result = element_directivity(np.sin(theta), F0, WIDTH_M, SPEED)

    assert np.allclose(result, expected)
    assert np.isclose(element_directivity(0.0, F0, WIDTH_M, SPEED), 1.0)
    assert np.allclose(result, result[::-1])


def test_directivity_first_null_and_frequency_broadcast() -> None:
    wide = 3.0e-3  # wider than a wavelength so a null exists in real angles
    null_sine = (SPEED / F0) / wide
    assert abs(element_directivity(null_sine, F0, wide, SPEED)) < 1.0e-12

    table = element_directivity(
        np.array([[0.0], [0.5]]), np.array([2.5e6, 5.0e6, 7.5e6]), WIDTH_M, SPEED
    )
    assert table.shape == (2, 3)
    assert np.allclose(table[0], 1.0)
    # At an oblique angle the element is more directive at higher frequency.
    assert np.all(np.diff(table[1]) < 0.0)


def test_beam_spread_amplitude() -> None:
    assert np.isclose(beam_spread_amplitude(0.04, 0.04), 1.0 / 0.04)
    assert np.isclose(beam_spread_amplitude(0.02, 0.08, 2.0), 2.0 / 0.04)
    with pytest.raises(ValueError):
        beam_spread_amplitude(0.0, 0.04)


def test_attenuation_coefficient_power_law() -> None:
    frequency = np.array([0.0, 2.5e6, 5.0e6, 10.0e6])
    assert np.allclose(attenuation_coefficient(frequency, ALPHA, F0, 0.0), ALPHA)
    assert np.allclose(
        attenuation_coefficient(frequency, ALPHA, F0, 1.0), ALPHA * frequency / F0
    )
    assert np.allclose(
        attenuation_coefficient(frequency, ALPHA, F0, 2.0), ALPHA * (frequency / F0) ** 2
    )
    with pytest.raises(ValueError):
        attenuation_coefficient(frequency, -1.0, F0)


def test_pulse_spectrum_round_trip_reproduces_shifted_pulse() -> None:
    frequency, spectrum = gaussian_pulse_spectrum(TIME.size, TIME[1], F0, SIGMA)
    delay = 12.345e-6  # deliberately not on a sample
    trace = np.fft.irfft(spectrum * np.exp(-2j * np.pi * frequency * delay), n=TIME.size)
    assert np.allclose(trace, shifted_pulse(TIME, delay, F0, SIGMA), atol=1.0e-9)


# --- Ladder assembly -------------------------------------------------------


def test_default_physics_reproduces_m0_exactly() -> None:
    reference, reference_times = simulate_point_reflector_fmc(
        TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA
    )
    fmc, times, amplitude = simulate_point_reflector_fmc_td(
        TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, PropagationPhysics()
    )
    assert np.array_equal(fmc, reference)
    assert np.array_equal(times, reference_times)
    assert np.all(amplitude == 1.0)


def test_td_plus_amplitudes_match_closed_form() -> None:
    physics = PropagationPhysics(geometric_spreading=True, attenuation_np_per_m=ALPHA)
    amplitude, _ = pair_transfer(ELEMENTS, REFLECTOR, SPEED, F0, physics)
    d = np.linalg.norm(ELEMENTS - REFLECTOR, axis=1)
    expected = np.exp(-ALPHA * (d[:, None] + d[None, :])) / np.sqrt(d[:, None] * d[None, :])
    assert np.allclose(amplitude, expected)


def test_frequency_domain_reproduces_time_domain_td_plus() -> None:
    physics = PropagationPhysics(geometric_spreading=True, attenuation_np_per_m=ALPHA)
    td, _, _ = simulate_point_reflector_fmc_td(
        TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, physics
    )
    fd = simulate_point_reflector_fmc_fd(TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, physics)
    assert np.max(np.abs(fd.fmc - td)) <= 1.0e-8 * np.max(np.abs(td))


def test_time_domain_rejects_frequency_dependent_physics() -> None:
    for physics in (
        PropagationPhysics(element_width_m=WIDTH_M),
        PropagationPhysics(attenuation_np_per_m=ALPHA, attenuation_frequency_exponent=1.0),
    ):
        with pytest.raises(ValueError):
            simulate_point_reflector_fmc_td(
                TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, physics
            )


def test_every_ladder_step_is_reciprocal_and_peaks_on_time() -> None:
    for step in physics_ladder(WIDTH_M, ALPHA, 1.0):
        fmc = simulate_ladder_step(step, TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA)
        assert fmc.shape == (N_ELEMENTS, N_ELEMENTS, TIME.size)
        assert np.all(np.isfinite(fmc))
        assert np.allclose(fmc, fmc.transpose(1, 0, 2), atol=1.0e-12 * np.max(np.abs(fmc)))
        _, travel_time = pair_transfer(ELEMENTS, REFLECTOR, SPEED, F0, step.physics)
        peak = TIME[np.argmax(np.abs(fmc), axis=2)]
        assert np.all(np.abs(peak - travel_time) <= 40.0e-9), step.label


def test_directivity_downshifts_oblique_pairs_only() -> None:
    base = PropagationPhysics(geometric_spreading=True)
    directive = PropagationPhysics(geometric_spreading=True, element_width_m=WIDTH_M)
    plain = simulate_point_reflector_fmc_fd(TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, base).fmc
    shaped = simulate_point_reflector_fmc_fd(
        TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, directive
    ).fmc
    edge = 0
    centre = N_ELEMENTS // 2
    # Edge pulse-echo ray is ~26 deg off-normal: reduced amplitude, lower centroid.
    assert np.max(np.abs(shaped[edge, edge])) < np.max(np.abs(plain[edge, edge]))
    assert _spectral_centroid(shaped[edge, edge]) < _spectral_centroid(plain[edge, edge])
    # Centre pulse-echo ray is almost normal: nearly unchanged.
    assert np.isclose(
        np.max(np.abs(shaped[centre, centre])), np.max(np.abs(plain[centre, centre])), rtol=1e-3
    )


def test_frequency_dependent_attenuation_downshifts_spectrum() -> None:
    flat = PropagationPhysics(attenuation_np_per_m=50.0)
    sloped = PropagationPhysics(attenuation_np_per_m=50.0, attenuation_frequency_exponent=1.0)
    centre = N_ELEMENTS // 2
    flat_trace = simulate_point_reflector_fmc_fd(
        TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, flat
    ).fmc[centre, centre]
    sloped_trace = simulate_point_reflector_fmc_fd(
        TIME, ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, sloped
    ).fmc[centre, centre]
    assert _spectral_centroid(sloped_trace) < _spectral_centroid(flat_trace) - 10.0e3


def test_fd_rejects_wrapping_record() -> None:
    with pytest.raises(ValueError):
        simulate_point_reflector_fmc_fd(
            TIME[:600], ELEMENTS, REFLECTOR, SPEED, F0, SIGMA, PropagationPhysics()
        )
