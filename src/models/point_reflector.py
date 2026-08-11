"""Model M0: the original point-reflector FMC implementation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ..propagation import travel_time_matrix
    from ..pulse import shifted_pulse
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from propagation import travel_time_matrix
    from pulse import shifted_pulse


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PointReflector:
    """A point reflector in the two-dimensional x-z inspection plane."""

    x_m: float
    z_m: float


def reflector_coordinate(reflector: PointReflector) -> np.ndarray:
    """Return a point reflector coordinate as an ``(2,)`` x-z array."""
    coordinate = np.asarray((reflector.x_m, reflector.z_m), dtype=float)
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("Reflector coordinates must be finite")
    return coordinate


def simulate_point_reflector_ascan(
    time_s: np.ndarray,
    transmitter_coordinate: ArrayLike,
    receiver_coordinate: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
    amplitude: float = 1.0,
) -> tuple[FloatArray, float]:
    """Simulate one point-reflector A-scan and return its arrival time."""
    transmitter = np.asarray(transmitter_coordinate, dtype=float)
    receiver = np.asarray(receiver_coordinate, dtype=float)
    reflector = np.asarray(reflector_coordinate, dtype=float)
    for name, coordinate in (
        ("transmitter_coordinate", transmitter),
        ("receiver_coordinate", receiver),
        ("reflector_coordinate", reflector),
    ):
        if coordinate.shape != (2,) or not np.all(np.isfinite(coordinate)):
            raise ValueError(f"{name} must have shape (2,) and finite values")
    arrival_time = float(
        travel_time_matrix(
            np.vstack((transmitter, receiver)), reflector, wave_speed_m_s
        )[0, 1]
    )
    trace = shifted_pulse(time_s, arrival_time, centre_frequency_hz, sigma_s)
    try:
        scale = float(amplitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("amplitude must be a finite scalar") from exc
    if not np.isfinite(scale):
        raise ValueError("amplitude must be a finite scalar")
    return scale * trace, arrival_time


def simulate_point_reflector_fmc(
    time_s: np.ndarray,
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
    amplitude: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Simulate constant-amplitude FMC traces for one point reflector.

    This is Model M0. Its numerical behaviour and public signature are retained
    from the original point-reflector implementation.
    """
    if not isinstance(time_s, np.ndarray):
        raise TypeError("time_s must be a NumPy array")
    if time_s.ndim != 1:
        raise ValueError(f"time_s must have shape (Nt,); got {time_s.shape}")
    try:
        scale = float(amplitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("amplitude must be a finite scalar") from exc
    if not np.isfinite(scale):
        raise ValueError("amplitude must be a finite scalar")

    arrival_times = travel_time_matrix(
        element_coordinates, reflector_coordinate, wave_speed_m_s
    )
    pulses = shifted_pulse(
        time_s[np.newaxis, np.newaxis, :],
        arrival_times[:, :, np.newaxis],
        centre_frequency_hz,
        sigma_s,
    )
    return scale * pulses, arrival_times
