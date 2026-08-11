"""Tests for the Gaussian-windowed sinusoidal pulse model."""
import numpy as np
import pytest

from src.pulse import gaussian_pulse, shifted_pulse


FREQUENCY_HZ = 5.0e6
SIGMA_S = 0.8e-6


def test_output_shape_and_zero_centred_maximum() -> None:
    time = np.arange(-4.0e-6, 4.0e-6 + 10.0e-9, 10.0e-9)
    pulse = gaussian_pulse(time, FREQUENCY_HZ, SIGMA_S)

    assert pulse.shape == time.shape
    peak_time = time[np.argmax(pulse)]
    assert abs(peak_time) <= 10.0e-9


def test_shifted_pulse_maximum_is_at_arrival_time() -> None:
    time = np.arange(0.0, 20.0e-6, 10.0e-9)
    arrival_time = 12.7e-6
    pulse = shifted_pulse(time, arrival_time, FREQUENCY_HZ, SIGMA_S)

    assert pulse.shape == time.shape
    peak_time = time[np.argmax(pulse)]
    assert abs(peak_time - arrival_time) <= 10.0e-9


def test_centre_time_and_amplitude_scaling() -> None:
    time = np.arange(0.0, 5.0e-6, 10.0e-9)
    pulse = gaussian_pulse(
        time, FREQUENCY_HZ, SIGMA_S, centre_time_s=2.0e-6, amplitude=2.5
    )
    assert np.all(np.isfinite(pulse))
    assert np.isclose(np.max(pulse), 2.5)
    assert abs(time[np.argmax(pulse)] - 2.0e-6) <= 10.0e-9


def test_carrier_frequency_is_approximately_five_mhz() -> None:
    sample_interval = 10.0e-9
    time = np.arange(-5.0e-6, 5.0e-6, sample_interval)
    pulse = gaussian_pulse(time, FREQUENCY_HZ, 2.0e-6)
    frequencies = np.fft.rfftfreq(time.size, sample_interval)
    peak_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(pulse)))]

    assert np.isclose(peak_frequency, FREQUENCY_HZ, rtol=0.02)


@pytest.mark.parametrize(
    ("frequency", "sigma"),
    [(0.0, SIGMA_S), (-FREQUENCY_HZ, SIGMA_S), (FREQUENCY_HZ, 0.0), (FREQUENCY_HZ, -SIGMA_S)],
)
def test_invalid_frequency_or_sigma_raises(frequency: float, sigma: float) -> None:
    with pytest.raises(ValueError):
        gaussian_pulse(np.array([0.0]), frequency, sigma)


@pytest.mark.parametrize(
    ("centre_time", "amplitude"),
    [(np.inf, 1.0), (0.0, np.nan)],
)
def test_invalid_centre_time_or_amplitude_raises(
    centre_time: float, amplitude: float
) -> None:
    with pytest.raises(ValueError):
        gaussian_pulse(
            np.array([0.0]),
            FREQUENCY_HZ,
            SIGMA_S,
            centre_time_s=centre_time,
            amplitude=amplitude,
        )
