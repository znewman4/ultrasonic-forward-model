"""Ray model of a parallel-sided plate: back-wall echoes and scatterer paths via the back wall.

The array sits on the front face ``z = 0`` and the back wall is the plane ``z = T``.
Longitudinal waves only (no mode conversion), straight rays, the method of images:

* **Back-wall echo n** (``n = 1, 2, ...``): the echo that has travelled ``n`` round trips
  of the plate. Unfolding the reflections gives a straight ray from element ``i`` to the
  image of element ``j`` at depth ``2 n T``, so

      D_n = sqrt((x_i - x_j)^2 + (2 n T)^2),     tau = D_n / c,
      A_n = A_bw / sqrt(D_n) * D(theta_i, f) D(theta_j, f) * rho^(n-1) * exp(-alpha D_n)

  with ``sin(theta_i) = (x_j - x_i) / D_n``. ``A_bw`` is calibrated, and ``rho`` is the
  amplitude factor lost per extra round trip (1 = lossless baseline).
* **Scatterer paths**: a point scatterer at ``S = (x, z)`` has the image ``S' = (x, 2T - z)``.
  The transmit leg may go direct (``S``) or via the back wall (``S'``), and the same for the
  receive leg, giving four families (DD, DB, BD, BB). Each back-wall leg multiplies the
  amplitude by ``rho_bw`` (1 = ``|R| = 1``, longitudinal only).

Amplitudes use the same factors as :mod:`src.models.point_reflector_physics`
(2D spreading, rectangular-element directivity, optional attenuation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ..propagation import (
        attenuation_coefficient, beam_spread_amplitude, element_directivity, element_ray_geometry,
    )
    from ..pulse import gaussian_pulse_spectrum
    from .point_reflector_physics import PropagationPhysics
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from propagation import (
        attenuation_coefficient, beam_spread_amplitude, element_directivity, element_ray_geometry,
    )
    from pulse import gaussian_pulse_spectrum
    from models.point_reflector_physics import PropagationPhysics


FloatArray = NDArray[np.float64]

SCATTERER_FAMILIES = ("DD", "DB", "BD", "BB")  # transmit leg, receive leg: D = direct, B = via back wall


@dataclass(frozen=True)
class PathSet:
    """A family of rays: real amplitude ``(N, N)`` or ``(N, N, Nf)`` and travel time ``(N, N)``."""

    label: str
    amplitude: FloatArray
    travel_time_s: FloatArray


def _factors(
    sin_tx, sin_rx, frequency_hz, physics: PropagationPhysics, wave_speed_m_s: float,
    centre_frequency_hz: float, path_m: FloatArray,
) -> FloatArray:
    """Directivity x attenuation for broadcast ``(N, 1)`` / ``(1, N)`` sines; ``(N, N[, Nf])``."""
    shape = np.broadcast_shapes(sin_tx.shape, sin_rx.shape)
    if frequency_hz is None:
        frequency = np.array(centre_frequency_hz)
        amplitude = np.ones(shape)
        path = path_m
    else:
        frequency = np.asarray(frequency_hz, dtype=float)
        amplitude = np.ones(shape + (1,))
        sin_tx, sin_rx, path = sin_tx[..., None], sin_rx[..., None], path_m[..., None]
    if physics.element_width_m is not None:
        f_dir = frequency if physics.directivity_frequency_dependent else np.array(centre_frequency_hz)
        amplitude = amplitude * (
            element_directivity(sin_tx, f_dir, physics.element_width_m, wave_speed_m_s)
            * element_directivity(sin_rx, f_dir, physics.element_width_m, wave_speed_m_s)
        )
    if physics.attenuation_np_per_m > 0.0:
        reference = physics.attenuation_reference_frequency_hz or centre_frequency_hz
        alpha = attenuation_coefficient(
            frequency, physics.attenuation_np_per_m, reference, physics.attenuation_frequency_exponent
        )
        amplitude = amplitude * np.exp(-alpha * path)
    return amplitude


def back_wall_paths(
    element_coordinates: ArrayLike,
    thickness_m: float,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    physics: PropagationPhysics,
    reference_amplitude: float,
    max_round_trips: int = 3,
    round_trip_factor: float = 1.0,
    frequency_hz: ArrayLike | None = None,
) -> list[PathSet]:
    """Back-wall echoes 1..``max_round_trips`` by the method of images (see module docstring)."""
    elements = np.asarray(element_coordinates, dtype=float)
    x = elements[:, 0]
    dx = x[None, :] - x[:, None]                        # x_j - x_i
    out = []
    for n in range(1, int(max_round_trips) + 1):
        distance = np.sqrt(dx**2 + (2.0 * n * thickness_m) ** 2)
        sin_i = dx / distance                           # element i -> image of element j
        sin_j = -dx / distance                          # element j -> image of element i
        amp = reference_amplitude / np.sqrt(distance) * round_trip_factor ** (n - 1)
        factors = _factors(sin_i, sin_j, frequency_hz, physics, wave_speed_m_s,
                           centre_frequency_hz, distance)
        amp = amp[..., None] * factors if frequency_hz is not None else amp * factors
        out.append(PathSet(f"BW{n}", amp, distance / wave_speed_m_s))
    return out


def scatterer_paths(
    element_coordinates: ArrayLike,
    scatterer_xz: ArrayLike,
    thickness_m: float,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    physics: PropagationPhysics,
    back_wall_leg_factor: float = 1.0,
    families: tuple[str, ...] = SCATTERER_FAMILIES,
    frequency_hz: ArrayLike | None = None,
) -> list[PathSet]:
    """Direct and back-wall paths to one point scatterer.

    ``physics.spreading_reference_amplitude_m`` is the scatterer's ``A0``.
    """
    elements = np.asarray(element_coordinates, dtype=float)
    s = np.asarray(scatterer_xz, dtype=float)
    mirror = np.array([s[0], 2.0 * thickness_m - s[1]])
    legs = {"D": element_ray_geometry(elements, s), "B": element_ray_geometry(elements, mirror)}
    out = []
    for family in families:
        d_tx, sin_tx = legs[family[0]]
        d_rx, sin_rx = legs[family[1]]
        path = d_tx[:, None] + d_rx[None, :]
        amp = beam_spread_amplitude(d_tx[:, None], d_rx[None, :], physics.spreading_reference_amplitude_m)
        amp = amp * back_wall_leg_factor ** family.count("B")
        factors = _factors(sin_tx[:, None], sin_rx[None, :], frequency_hz, physics,
                           wave_speed_m_s, centre_frequency_hz, path)
        amp = amp[..., None] * factors if frequency_hz is not None else amp * factors
        out.append(PathSet(family, amp, path / wave_speed_m_s))
    return out


def synthesize(
    time_s: FloatArray,
    path_sets: list[PathSet],
    centre_frequency_hz: float,
    sigma_s: float,
    guard_sigma: float = 5.0,
) -> tuple[FloatArray, int]:
    """Frequency-domain FMC synthesis for a list of path sets.

    Uses the convention of :func:`simulate_point_reflector_fmc_fd`:
    ``S(omega) = P(omega) T(f) exp(-i omega (tau - t0))`` and ``irfft``. Because the DFT is
    periodic, any arrival within ``guard_sigma`` pulse widths of the record ends (including
    later-than-record multiples that would wrap round) is dropped; the number of dropped
    pair-paths is returned.

    ``PathSet.amplitude`` must be ``(N, N)`` (applied at every frequency) or ``(N, N, Nf)``.
    """
    time = np.asarray(time_s, dtype=float)
    dt = float(time[1] - time[0])
    frequency, pulse = gaussian_pulse_spectrum(time.size, dt, centre_frequency_hz, sigma_s)
    guard = guard_sigma * sigma_s
    total = None
    dropped = 0
    omega = (2.0 * np.pi * frequency)[None, None, :]
    for ps in path_sets:
        keep = (ps.travel_time_s - guard >= time[0]) & (ps.travel_time_s + guard <= time[-1])
        dropped += int(keep.size - keep.sum())
        if not keep.any():
            continue
        amp = ps.amplitude if ps.amplitude.ndim == 3 else ps.amplitude[:, :, None]
        amp = amp * keep[:, :, None]
        phase = np.exp(-1j * omega * (ps.travel_time_s - time[0])[:, :, None])
        spectrum = pulse[None, None, :] * amp * phase
        total = spectrum if total is None else total + spectrum
    if total is None:
        return np.zeros(path_sets[0].travel_time_s.shape + (time.size,)), dropped
    if time.size % 2 == 0:
        total[:, :, -1] = total[:, :, -1].real
    return np.fft.irfft(total, n=time.size, axis=2), dropped
