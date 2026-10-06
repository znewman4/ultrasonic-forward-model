"""Tests for half-matrix loading, time-zero fitting, pulse fitting and amplitude fitting."""
import numpy as np
import pytest
from scipy.io import savemat

from src.calibration import fit_gaussian_cosine, fit_global_amplitude, gaussian_cosine, stack_echoes
from src.experiment_io import fit_time_zero_offset, load_hmc, nominal_array_coordinates
from src.pulse import gaussian_pulse


def test_nominal_array_is_centred_with_requested_pitch() -> None:
    elements = nominal_array_coordinates(64, 0.63e-3)
    assert elements.shape == (64, 2)
    assert np.isclose(np.ptp(elements[:, 0]), 63 * 0.63e-3)
    assert np.isclose(elements[:, 0].mean(), 0.0, atol=1e-12)
    assert np.all(elements[:, 1] == 0.0)


def test_load_hmc_fills_reciprocal_traces(tmp_path) -> None:
    n, nt = 4, 20
    rng = np.random.default_rng(0)
    pairs = [(i + 1, j + 1) for i in range(n) for j in range(i, n)]
    data = rng.normal(size=(nt, len(pairs)))
    exp = {
        "time_data": data,
        "tx": np.array([p[0] for p in pairs], dtype=np.uint8),
        "rx": np.array([p[1] for p in pairs], dtype=np.uint8),
        "time": np.arange(nt) * 4e-8,
        "array": {"el_xc": np.arange(n) * 1e-3, "el_zc": np.zeros(n)},
    }
    path = tmp_path / "hmc.mat"
    savemat(path, {"exp_data": exp})
    fmc, time, elements, recorded, traces = load_hmc(path)
    assert fmc.shape == (n, n, nt)
    assert np.array_equal(recorded, np.triu(np.ones((n, n), dtype=bool)))
    assert np.array_equal(fmc, fmc.transpose(1, 0, 2))
    assert np.array_equal(fmc[1, 2], data[:, pairs.index((2, 3))])
    assert elements.shape == (n, 2)


def test_load_hmc_rejects_incomplete_matrix(tmp_path) -> None:
    exp = {
        "time_data": np.zeros((5, 2)),
        "tx": np.array([1, 2], dtype=np.uint8),
        "rx": np.array([1, 2], dtype=np.uint8),
        "time": np.arange(5) * 4e-8,
        "array": {"el_xc": np.arange(3) * 1e-3, "el_zc": np.zeros(3)},
    }
    path = tmp_path / "bad.mat"
    savemat(path, {"exp_data": {**exp, "tx": np.array([1, 3], dtype=np.uint8), "rx": np.array([1, 3], dtype=np.uint8)}})
    with pytest.raises(ValueError, match="complete half matrix"):
        load_hmc(path)


def test_fit_time_zero_offset_recovers_known_offset() -> None:
    elements = nominal_array_coordinates()
    n = elements.shape[0]
    c, thickness, offset = 6300.0, 0.0494, 0.47e-6
    time = np.arange(1000) * 40e-9
    dx = np.abs(elements[:, 0][:, None] - elements[:, 0][None, :])
    fmc = np.zeros((n, n, time.size))
    for multiple, amp in ((2.0, 1.0), (4.0, 0.5)):
        arrival = np.sqrt(dx**2 + (multiple * thickness) ** 2) / c + offset
        fmc += amp * gaussian_pulse(time[None, None, :], 5e6, 0.15e-6, arrival[:, :, None])
    fit = fit_time_zero_offset(fmc, time, elements[:, 0], c)
    assert abs(fit["offset_s"] - offset) < 5e-9
    assert abs(fit["thickness_m"] - thickness) < 0.1e-3
    assert fit["inlier_fraction_100ns"] > 0.95


def test_pulse_fit_recovers_parameters_from_jittered_noisy_echoes() -> None:
    rng = np.random.default_rng(1)
    dt, f0, sigma = 40e-9, 4.8e6, 0.18e-6
    time = np.arange(1000) * dt
    arrivals = 10e-6 + rng.uniform(-0.1e-6, 0.1e-6, 12)
    traces = np.array([
        0.3 * gaussian_pulse(time, f0, sigma, a) + 0.003 * rng.normal(size=time.size) for a in arrivals
    ])
    lag, stack = stack_echoes(traces, time, arrivals + 0.03e-6)
    fit = fit_gaussian_cosine(lag, stack)
    assert abs(fit.centre_frequency_hz - f0) / f0 < 0.03
    assert abs(fit.sigma_s - sigma) / sigma < 0.08
    assert fit.relative_residual < 0.1
    model = gaussian_cosine(lag, fit.amplitude, fit.centre_frequency_hz, fit.sigma_s, fit.phase_rad)
    assert np.corrcoef(model, stack)[0, 1] > 0.99


def test_global_amplitude_is_least_squares_scale() -> None:
    unit = np.array([1.0, 2.0, 4.0])
    a0, spread = fit_global_amplitude(unit, 3.0 * unit)
    assert np.isclose(a0, 3.0) and np.isclose(spread, 0.0)
    a0, spread = fit_global_amplitude(unit, np.array([3.3, 5.4, 12.6]))
    assert 2.9 < a0 < 3.2 and spread > 0
