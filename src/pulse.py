"""Band-limited pulse definitions for the point-reflector forward model."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _positive_finite(value: float, name: str) -> float:
    """Return ``value`` as a float after validating it as positive and finite."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _time_array(time_s: np.ndarray) -> FloatArray:
    """Validate and return a floating-point time array."""
    if not isinstance(time_s, np.ndarray):
        raise TypeError("time_s must be a NumPy array")
    if not np.issubdtype(time_s.dtype, np.number) or np.iscomplexobj(time_s):
        raise TypeError("time_s must contain real numeric values")
    time = np.asarray(time_s, dtype=float)
    if not np.all(np.isfinite(time)):
        raise ValueError("time_s must contain only finite values")
    return time


def gaussian_pulse(
    time_s: np.ndarray,
    centre_frequency_hz: float,
    sigma_s: float,
    centre_time_s: float = 0.0,
    amplitude: float = 1.0,
) -> FloatArray:
    """Return a Gaussian-windowed sinusoid centred at ``centre_time_s``.

    The pulse is ``exp(-t**2 / (2*sigma**2)) * cos(2*pi*f0*t)``.

    Args:
        time_s: Real NumPy array of time values in seconds.
        centre_frequency_hz: Positive carrier frequency in hertz.
        sigma_s: Positive Gaussian standard deviation in seconds.

    Returns:
        Pulse samples with the same shape as ``time_s``.
    """
    time = _time_array(time_s)
    frequency = _positive_finite(centre_frequency_hz, "centre_frequency_hz")
    sigma = _positive_finite(sigma_s, "sigma_s")
    try:
        centre_time = np.asarray(centre_time_s, dtype=float)
        scale = float(amplitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("centre_time_s and amplitude must be finite") from exc
    if not np.all(np.isfinite(centre_time)) or not np.isfinite(scale):
        raise ValueError("centre_time_s and amplitude must be finite")
    relative_time = time - centre_time
    envelope = np.exp(-(relative_time**2) / (2.0 * sigma**2))
    return scale * envelope * np.cos(2.0 * np.pi * frequency * relative_time)


def shifted_pulse(
    time_s: np.ndarray,
    arrival_time_s: float | np.ndarray,
    centre_frequency_hz: float,
    sigma_s: float,
) -> FloatArray:
    """Return Gaussian pulses centred at one or more arrival times.

    Scalar arrival time preserves the shape of ``time_s``. Array arrival times
    may be broadcast against ``time_s`` to create multiple shifted pulses.
    """
    time = _time_array(time_s)
    try:
        arrival_time = np.asarray(arrival_time_s, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("arrival_time_s must contain finite numeric values") from exc
    if not np.all(np.isfinite(arrival_time)):
        raise ValueError("arrival_time_s must contain finite numeric values")
    return gaussian_pulse(
        time, centre_frequency_hz, sigma_s, centre_time_s=arrival_time
    )


def gaussian_pulse_spectrum(
    n_samples: int,
    sample_interval_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
) -> tuple[FloatArray, NDArray[np.complex128]]:
    """Return ``(frequency_hz, P)`` for the zero-centred Gaussian pulse.

    ``P`` is the NumPy ``rfft`` of :func:`gaussian_pulse` sampled on a periodic
    lag grid ``[0, dt, ..., -2dt, -dt]``, so the pulse is centred at ``t = 0``
    and ``P`` is (to numerical precision) real. A delay ``tau`` is then applied
    in the frequency domain as ``P(omega) * exp(-i*omega*tau)`` and ``irfft``
    recovers ``p(t - tau)``. This sign follows NumPy's ``exp(+i*omega*t)``
    synthesis convention, which is the convention of the frequency-domain
    pulse ``exp(-i*omega*(d_tx + d_rx)/c_L)`` used by the ray models.
    """
    if (
        not isinstance(n_samples, (int, np.integer))
        or isinstance(n_samples, (bool, np.bool_))
        or n_samples < 2
    ):
        raise ValueError("n_samples must be an integer >= 2")
    dt = _positive_finite(sample_interval_s, "sample_interval_s")
    indices = np.arange(n_samples)
    signed_indices = np.where(indices <= n_samples // 2, indices, indices - n_samples)
    pulse = gaussian_pulse(signed_indices * dt, centre_frequency_hz, sigma_s)
    frequency = np.fft.rfftfreq(n_samples, dt)
    return frequency, np.asarray(np.fft.rfft(pulse), dtype=complex)
