"""Vectorized propagation calculations for reflector forward models."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _validated_coordinates(
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Convert and validate the x-z coordinates used by propagation functions."""
    elements = np.asarray(element_coordinates, dtype=float)
    reflector = np.asarray(reflector_coordinate, dtype=float)
    if elements.ndim != 2 or elements.shape[1] != 2:
        raise ValueError(
            f"element_coordinates must have shape (N, 2); got {elements.shape}"
        )
    if elements.shape[0] == 0:
        raise ValueError("element_coordinates must contain at least one element")
    if reflector.shape != (2,):
        raise ValueError(
            f"reflector_coordinate must have shape (2,); got {reflector.shape}"
        )
    if not np.all(np.isfinite(elements)) or not np.all(np.isfinite(reflector)):
        raise ValueError("element and reflector coordinates must be finite")
    return elements, reflector


def element_to_reflector_distances(
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
) -> FloatArray:
    """Return Euclidean distance from each array element to one reflector.

    Args:
        element_coordinates: Element x-z coordinates with shape ``(N, 2)``.
        reflector_coordinate: Reflector x-z coordinate with shape ``(2,)``.

    Returns:
        Distances in metres with shape ``(N,)``.
    """
    elements, reflector = _validated_coordinates(
        element_coordinates, reflector_coordinate
    )
    return np.linalg.norm(elements - reflector, axis=1)


def path_length_matrix(
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
) -> FloatArray:
    """Return transmitter-reflector-receiver path lengths in metres.

    Entry ``(i, j)`` is ``r_i + r_j``. The returned matrix has shape
    ``(N, N)``.
    """
    distances = element_to_reflector_distances(
        element_coordinates, reflector_coordinate
    )
    return distances[:, np.newaxis] + distances[np.newaxis, :]


def travel_time_matrix(
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
) -> FloatArray:
    """Return transmitter-reflector-receiver travel times in seconds.

    Args:
        element_coordinates: Element x-z coordinates with shape ``(N, 2)``.
        reflector_coordinate: Reflector x-z coordinate with shape ``(2,)``.
        wave_speed_m_s: Positive propagation speed in metres per second.
    """
    try:
        wave_speed = float(wave_speed_m_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("wave_speed_m_s must be a positive finite scalar") from exc
    if not np.isfinite(wave_speed) or wave_speed <= 0.0:
        raise ValueError("wave_speed_m_s must be a positive finite scalar")
    return path_length_matrix(element_coordinates, reflector_coordinate) / wave_speed


def boundary_travel_times(
    element_coordinates: ArrayLike,
    boundary_points: ArrayLike,
    wave_speed_m_s: float,
) -> FloatArray:
    """Return travel times through every boundary point of a reflector.

    Entry ``(i, j, m)`` is the transmitter-boundary-receiver time for
    transmitter ``i``, receiver ``j``, and boundary point ``m``.

    Args:
        element_coordinates: Element x-z coordinates with shape ``(N, 2)``.
        boundary_points: Boundary x-z coordinates with shape ``(M, 2)``.
        wave_speed_m_s: Positive propagation speed in metres per second.

    Returns:
        Travel times in seconds with shape ``(N, N, M)``.
    """
    elements = np.asarray(element_coordinates, dtype=float)
    boundary = np.asarray(boundary_points, dtype=float)
    if elements.ndim != 2 or elements.shape[1] != 2 or elements.shape[0] == 0:
        raise ValueError("element_coordinates must have non-empty shape (N, 2)")
    if boundary.ndim != 2 or boundary.shape[1] != 2 or boundary.shape[0] == 0:
        raise ValueError("boundary_points must have non-empty shape (M, 2)")
    if not np.all(np.isfinite(elements)) or not np.all(np.isfinite(boundary)):
        raise ValueError("element and boundary coordinates must be finite")
    try:
        wave_speed = float(wave_speed_m_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("wave_speed_m_s must be a positive finite scalar") from exc
    if not np.isfinite(wave_speed) or wave_speed <= 0.0:
        raise ValueError("wave_speed_m_s must be a positive finite scalar")

    distances = np.linalg.norm(
        elements[:, np.newaxis, :] - boundary[np.newaxis, :, :], axis=2
    )
    return (
        distances[:, np.newaxis, :] + distances[np.newaxis, :, :]
    ) / wave_speed
