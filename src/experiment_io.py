"""Loading and timing calibration for BRAIN1 half-matrix experiment files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.signal import hilbert


FloatArray = NDArray[np.float64]

ELEMENT_COUNT = 64
ELEMENT_PITCH_M = 0.63e-3


def nominal_array_coordinates(
    n_elements: int = ELEMENT_COUNT, pitch_m: float = ELEMENT_PITCH_M
) -> FloatArray:
    """Return ``(N, 2)`` x-z centres of a linear array centred on x=0 at z=0."""
    x = (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch_m
    return np.column_stack((x, np.zeros(n_elements)))


def load_hmc(mat_path: str | Path):
    """Load a BRAIN1 ``exp_data`` file and expand half-matrix capture.

    BRAIN1 stores only pairs with ``tx <= rx``. The missing ``(rx, tx)`` traces
    are filled by reciprocity, which for delay-and-sum is equivalent to BRAIN1's
    ``tt_weight = 2`` on the recorded off-diagonal traces.

    Returns:
        ``(fmc, time_s, elements, recorded, traces)`` where ``fmc`` is
        ``(N, N, Nt)``, ``elements`` is the ``(N, 2)`` array from the file,
        ``recorded`` marks the ``(tx, rx)`` pairs that were actually stored and
        ``traces`` is the raw ``(Nt, Ntraces)`` data.
    """
    exp = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["exp_data"]
    tx = np.asarray(exp.tx, dtype=int) - 1
    rx = np.asarray(exp.rx, dtype=int) - 1
    traces = np.asarray(exp.time_data, dtype=float)
    n = int(max(tx.max(), rx.max()) + 1)
    fmc = np.zeros((n, n, traces.shape[0]))
    recorded = np.zeros((n, n), dtype=bool)
    for k, (i, j) in enumerate(zip(tx, rx)):
        fmc[i, j] = traces[:, k]
        recorded[i, j] = True
    if not np.all(recorded | recorded.T):
        raise ValueError("pairs are not a complete half matrix")
    missing = ~recorded
    fmc[missing] = fmc.transpose(1, 0, 2)[missing]
    elements = np.column_stack(
        (np.asarray(exp.array.el_xc, float), np.asarray(exp.array.el_zc, float))
    )
    return fmc, np.asarray(exp.time, float), elements, recorded, traces


def envelope_peak_time(envelope: FloatArray, time_s: FloatArray, index: int) -> float:
    """Parabolic sub-sample time of the envelope maximum near ``index``."""
    dt = time_s[1] - time_s[0]
    y0, y1, y2 = envelope[index - 1 : index + 2]
    denominator = y0 - 2.0 * y1 + y2
    shift = 0.5 * (y0 - y2) / denominator if denominator != 0.0 else 0.0
    return float(time_s[index] + shift * dt)


def pair_envelope_peaks(
    envelope: FloatArray,
    time_s: FloatArray,
    guess_s: FloatArray,
    half_window: int,
    pairs: tuple[FloatArray, FloatArray],
) -> FloatArray:
    """Envelope-peak time of each ``(i, j)`` pair within ``guess +- half_window`` samples."""
    dt = time_s[1] - time_s[0]
    out = np.full(len(guess_s), np.nan)
    for k, (i, j) in enumerate(zip(*pairs)):
        lo = max(int((guess_s[k] - time_s[0]) / dt) - half_window, 1)
        hi = min(lo + 2 * half_window, time_s.size - 1)
        if hi - lo < 3:
            continue
        index = lo + int(np.argmax(envelope[i, j, lo:hi]))
        if 1 <= index < time_s.size - 1:
            out[k] = envelope_peak_time(envelope[i, j], time_s, index)
    return out


def fit_time_zero_offset(
    fmc: FloatArray,
    time_s: FloatArray,
    element_x_m: FloatArray,
    wave_speed_m_s: float,
    thickness_guess_m: float = 0.0494,
    offset_guess_s: float = 0.44e-6,
    half_window: int = 25,
    fit_speed: bool = False,
) -> dict[str, float]:
    """Fit acquisition time-zero offset from the first two back-wall echoes.

    For a plane back wall of thickness ``T`` the specular echo of pair
    ``(i, j)`` arrives at ``sqrt(dx^2 + (2T)^2)/c + t_off`` (first echo) and
    ``sqrt(dx^2 + (4T)^2)/c + t_off`` (second echo), with ``dx = |x_i - x_j|``.
    All ``N(N+1)/2`` recorded pairs and both echoes are fitted together with a
    robust (soft-L1) loss, giving ``t_off`` independent of the thickness.
    """
    n = fmc.shape[0]
    x = np.asarray(element_x_m, dtype=float)
    pairs = np.triu_indices(n)
    dx = np.abs(x[pairs[0]] - x[pairs[1]])
    envelope = np.abs(hilbert(fmc, axis=2))
    c0 = float(wave_speed_m_s)
    peaks = []
    for multiple in (2.0, 4.0):
        guess = np.sqrt(dx**2 + (multiple * thickness_guess_m) ** 2) / c0 + offset_guess_s
        peaks.append(pair_envelope_peaks(envelope, time_s, guess, half_window, pairs))
    e1, e2 = peaks
    ok = np.isfinite(e1) & np.isfinite(e2)

    # Work in mm, m/s, us and ns so the solver's absolute tolerances are meaningful.
    def residual_ns(p):
        thickness, speed, offset = p[0] * 1e-3, p[1], p[2] * 1e-6
        r1 = np.sqrt(dx**2 + (2.0 * thickness) ** 2) / speed + offset - e1
        r2 = np.sqrt(dx**2 + (4.0 * thickness) ** 2) / speed + offset - e2
        return np.concatenate((r1[ok], r2[ok])) * 1e9

    if fit_speed:
        start = [thickness_guess_m * 1e3, c0, offset_guess_s * 1e6]
        fun = residual_ns
    else:
        start = [thickness_guess_m * 1e3, offset_guess_s * 1e6]
        fun = lambda p: residual_ns([p[0], c0, p[1]])  # noqa: E731
    solution = least_squares(
        fun, start, x_scale=[1.0, 100.0, 0.1][: len(start)] if fit_speed else [1.0, 0.1],
        loss="soft_l1", f_scale=20.0, ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    solution.x = np.array([solution.x[0] * 1e-3] + ([solution.x[1]] if fit_speed else []) + [solution.x[-1] * 1e-6])
    thickness = float(solution.x[0])
    speed = float(solution.x[1]) if fit_speed else c0
    offset = float(solution.x[-1])
    final = np.abs(residual_ns([thickness * 1e3, speed, offset * 1e6])) * 1e-9
    return {
        "offset_s": offset,
        "thickness_m": thickness,
        "wave_speed_m_s": speed,
        "median_abs_residual_s": float(np.median(final)),
        "inlier_fraction_100ns": float(np.mean(final < 100e-9)),
    }
