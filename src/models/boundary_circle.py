"""Model M1: equal-weight geometric circular-boundary reflector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ..propagation import boundary_travel_times
    from ..pulse import shifted_pulse
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from propagation import boundary_travel_times
    from pulse import shifted_pulse


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CircularReflector:
    """A circle defined by its centre, radius, and boundary discretisation."""

    x_m: float
    z_m: float
    radius_m: float
    n_points: int


def circular_boundary_points(reflector: CircularReflector) -> FloatArray:
    """Return ``M`` evenly spaced x-z points on a circular boundary."""
    if not isinstance(reflector, CircularReflector):
        raise TypeError("reflector must be a CircularReflector")
    centre = np.asarray((reflector.x_m, reflector.z_m), dtype=float)
    try:
        radius = float(reflector.radius_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius_m must be a non-negative finite scalar") from exc
    if not np.all(np.isfinite(centre)):
        raise ValueError("Reflector centre coordinates must be finite")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_m must be a non-negative finite scalar")
    if (
        not isinstance(reflector.n_points, (int, np.integer))
        or isinstance(reflector.n_points, (bool, np.bool_))
        or reflector.n_points <= 0
    ):
        raise ValueError("n_points must be a positive integer")

    phi = 2.0 * np.pi * np.arange(reflector.n_points) / reflector.n_points
    offsets = radius * np.column_stack((np.cos(phi), np.sin(phi)))
    return centre + offsets


def simulate_circular_reflector_fmc(
    time_s: np.ndarray,
    element_coordinates: ArrayLike,
    boundary_points: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
    amplitude: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Simulate Model M1 with coherent equal boundary weights ``1/M``.

    This geometric finite-extent approximation captures the distribution and
    interference of boundary path lengths. It is not an elastic cylinder model:
    traction-free boundary conditions and mode conversion are not included.
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

    arrival_times = boundary_travel_times(
        element_coordinates, boundary_points, wave_speed_m_s
    )
    n_transmitters, n_receivers, n_boundary_points = arrival_times.shape
    fmc = np.zeros((n_transmitters, n_receivers, time_s.size), dtype=float)
    for point_index in range(n_boundary_points):
        fmc += shifted_pulse(
            time_s[np.newaxis, np.newaxis, :],
            arrival_times[:, :, point_index, np.newaxis],
            centre_frequency_hz,
            sigma_s,
        )
    fmc *= scale / n_boundary_points
    return fmc, arrival_times
