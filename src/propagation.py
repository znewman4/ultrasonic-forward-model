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


# ---------------------------------------------------------------------------
# Amplitude physics for the ray model (Holmes, Drinkwater & Wilcox, 2005).
#
# The functions below supply the multiplicative factors of
#
#     H_ij(omega) = P(omega) * A_ij * D(theta_i, f) * D(theta_j, f)
#                   * exp(-alpha(f) * (d_i + d_j)) * exp(-i*omega*(d_i + d_j)/c_L)
#
# They are deliberately independent of each other so that each factor can be
# switched on separately and its effect on FMC and TFM data isolated.
# ---------------------------------------------------------------------------


def element_ray_geometry(
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return ray length and ``sin(theta)`` from every element to a reflector.

    ``theta`` is the angle between the element normal (+z, into the specimen)
    and the straight ray to the reflector, positive towards +x. It is the angle
    ``theta_tx``/``theta_rx`` in Holmes et al. (2005), eq. (5).

    Returns:
        ``(distances_m, sin_theta)``, each with shape ``(N,)``.
    """
    elements, reflector = _validated_coordinates(
        element_coordinates, reflector_coordinate
    )
    offsets = reflector - elements
    distances = np.linalg.norm(offsets, axis=1)
    if np.any(distances <= 0.0):
        raise ValueError("reflector must not coincide with an array element")
    return distances, offsets[:, 0] / distances


def element_directivity(
    sin_theta: ArrayLike,
    frequency_hz: ArrayLike,
    element_width_m: float,
    wave_speed_m_s: float,
) -> FloatArray:
    """Return the 2D far-field directivity of one rectangular array element.

    Holmes, Drinkwater and Wilcox (2005), eq. (5), after McNab and Stumpf::

        p(theta) = sinc(pi * a * sin(theta) / lambda),   sinc(x) = sin(x)/x

    with element width ``a`` and wavelength ``lambda = c / f``. The elevation
    term of their eq. (4) is dropped because the element length ``L >> a``.
    NumPy's ``np.sinc(x) = sin(pi x)/(pi x)`` already contains the ``pi``, so
    the argument passed to it is ``a * sin(theta) * f / c``.

    ``sin_theta`` and ``frequency_hz`` broadcast against each other, so passing
    shapes ``(N, 1)`` and ``(Nf,)`` returns a frequency-dependent ``(N, Nf)``
    table. The result is real and may be negative beyond the first null.
    """
    width = float(element_width_m)
    speed = float(wave_speed_m_s)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("element_width_m must be a positive finite scalar")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("wave_speed_m_s must be a positive finite scalar")
    sine = np.asarray(sin_theta, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(np.abs(sine) > 1.0 + 1.0e-12):
        raise ValueError("sin_theta values must lie in [-1, 1]")
    if np.any(frequency < 0.0) or not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be finite and non-negative")
    return np.sinc(width * sine * frequency / speed)


def beam_spread_amplitude(
    transmit_distance_m: ArrayLike,
    receive_distance_m: ArrayLike,
    reference_amplitude: float = 1.0,
) -> FloatArray:
    """Return the 2D geometric-spreading amplitude ``A0 / sqrt(d_tx * d_rx)``.

    Holmes, Drinkwater and Wilcox (2005). Each leg is a cylindrical (2D) wave
    whose amplitude decays as ``1/sqrt(d)``. ``A0`` has units of metres so that
    the returned factor is dimensionless; it is an uncalibrated scale.
    """
    d_tx = np.asarray(transmit_distance_m, dtype=float)
    d_rx = np.asarray(receive_distance_m, dtype=float)
    scale = float(reference_amplitude)
    if not np.isfinite(scale):
        raise ValueError("reference_amplitude must be finite")
    if np.any(d_tx <= 0.0) or np.any(d_rx <= 0.0):
        raise ValueError("distances must be strictly positive")
    return scale / np.sqrt(d_tx * d_rx)


def attenuation_coefficient(
    frequency_hz: ArrayLike,
    attenuation_np_per_m: float,
    reference_frequency_hz: float,
    frequency_exponent: float = 0.0,
) -> FloatArray:
    """Return the power-law amplitude attenuation ``alpha(f)`` in Np/m.

    ``alpha(f) = alpha_ref * (f / f_ref) ** n``. ``n = 0`` gives the
    frequency-independent ("simple") attenuation used by M0-TD+; ``n > 0``
    gives frequency-dependent attenuation that preferentially removes high
    frequencies. Velocity dispersion required by causality is neglected.
    """
    alpha_ref = float(attenuation_np_per_m)
    f_ref = float(reference_frequency_hz)
    exponent = float(frequency_exponent)
    if not np.isfinite(alpha_ref) or alpha_ref < 0.0:
        raise ValueError("attenuation_np_per_m must be non-negative and finite")
    if not np.isfinite(f_ref) or f_ref <= 0.0:
        raise ValueError("reference_frequency_hz must be positive and finite")
    if not np.isfinite(exponent) or exponent < 0.0:
        raise ValueError("frequency_exponent must be non-negative and finite")
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(frequency < 0.0) or not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be finite and non-negative")
    if exponent == 0.0:
        return np.full(frequency.shape, alpha_ref)
    return alpha_ref * (frequency / f_ref) ** exponent
