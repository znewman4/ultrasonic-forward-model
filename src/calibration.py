"""Calibrate the ray-model pulse and reference amplitude against an experimental echo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import hilbert


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PulseFit:
    """Gaussian-windowed cosine ``A exp(-t^2/2s^2) cos(2 pi f0 t + phase)``.

    ``t`` is measured from the envelope centre. The ray models use ``phase = 0``;
    a large fitted phase means the real echo is not of that form (reported, but
    it does not affect an envelope/Hilbert TFM).
    """

    amplitude: float
    centre_frequency_hz: float
    sigma_s: float
    phase_rad: float
    rms_residual: float
    relative_residual: float


def gaussian_cosine(t: FloatArray, amplitude, f0, sigma, phase) -> FloatArray:
    return amplitude * np.exp(-(t**2) / (2.0 * sigma**2)) * np.cos(2.0 * np.pi * f0 * t + phase)


def stack_echoes(
    traces: FloatArray,
    time_s: FloatArray,
    arrival_guess_s: FloatArray,
    half_width_s: float = 1.2e-6,
    step_s: float = 5e-9,
) -> tuple[FloatArray, FloatArray]:
    """Align echoes on their envelope peak, normalise, and average them.

    ``traces`` is ``(K, Nt)``; ``arrival_guess_s`` gives each echo's approximate
    arrival. Each echo is windowed, upsampled with a cubic spline, shifted so
    its envelope maximum is at 0, scaled to unit envelope peak, then averaged.

    Returns:
        ``(lag_s, stacked)`` on a common lag axis.
    """
    dt = time_s[1] - time_s[0]
    lag = np.arange(-half_width_s, half_width_s + step_s / 2, step_s)
    stack = []
    for trace, guess in zip(traces, arrival_guess_s):
        lo = max(int((guess - half_width_s * 1.5 - time_s[0]) / dt), 0)
        hi = min(int((guess + half_width_s * 1.5 - time_s[0]) / dt), time_s.size)
        if hi - lo < 16:
            continue
        seg = trace[lo:hi]
        env = np.abs(hilbert(seg))
        t_seg = time_s[lo:hi]
        spline = CubicSpline(t_seg, seg)
        fine = np.arange(t_seg[0], t_seg[-1], step_s / 5)
        fine_env = np.abs(hilbert(spline(fine)))
        centre = fine[int(np.argmax(np.where(np.abs(fine - guess) < half_width_s / 2, fine_env, 0)))]
        peak = fine_env.max()
        if peak <= 0:
            continue
        query = centre + lag
        if query[0] < t_seg[0] or query[-1] > t_seg[-1]:
            continue
        stack.append(spline(query) / peak)
    if not stack:
        raise ValueError("no usable echoes to stack")
    return lag, np.mean(stack, axis=0)


def fit_gaussian_cosine(lag_s: FloatArray, echo: FloatArray) -> PulseFit:
    """Least-squares fit of the Gaussian-cosine model to a stacked echo."""
    spectrum = np.abs(np.fft.rfft(echo * np.hanning(echo.size)))
    freq = np.fft.rfftfreq(echo.size, lag_s[1] - lag_s[0])
    f0 = float(freq[int(np.argmax(spectrum))])
    envelope = np.abs(hilbert(echo))
    start = [float(envelope.max()), f0, 0.2e-6, 0.0]

    def residual(p):
        return gaussian_cosine(lag_s, p[0], p[1], p[2], p[3]) - echo

    fit = least_squares(
        residual, start, bounds=([0, 1e5, 5e-9, -np.pi], [np.inf, 5e7, 5e-6, np.pi]),
        x_scale=[1.0, 1e6, 1e-7, 1.0],
    )
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    return PulseFit(
        amplitude=float(fit.x[0]),
        centre_frequency_hz=float(fit.x[1]),
        sigma_s=float(fit.x[2]),
        phase_rad=float(fit.x[3]),
        rms_residual=rms,
        relative_residual=rms / float(np.sqrt(np.mean(echo**2))),
    )


def fit_global_amplitude(unit_model: FloatArray, measured: FloatArray) -> tuple[float, float]:
    """Least-squares scale ``A0`` with ``measured ~ A0 * unit_model``.

    Returns ``(A0, relative_spread)`` where the spread is the standard deviation
    of ``measured / (A0 * unit_model)`` about 1.
    """
    model = np.asarray(unit_model, dtype=float)
    data = np.asarray(measured, dtype=float)
    a0 = float(np.dot(model, data) / np.dot(model, model))
    ratio = data / (a0 * model)
    return a0, float(np.std(ratio))
