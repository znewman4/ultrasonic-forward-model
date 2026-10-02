"""M0 physics ladder: the point reflector with ray-amplitude physics added.

The ladder adds one physical effect at a time to Model M0 so that the effect of
each on FMC and TFM data can be isolated (see ``docs/propagation_physics.md``):

1. ``M0``        timing only (identical to :mod:`src.models.point_reflector`);
2. ``M0-TD+``    timing + 2D geometric spreading + frequency-independent
                 attenuation, synthesized in the time domain;
3. ``M0-FD``     exactly the same physics as M0-TD+, synthesized in the
                 frequency domain (must reproduce M0-TD+);
4. ``M0-FD+D``   M0-FD + frequency-dependent element directivity;
5. ``M0-FD+D+A`` M0-FD+D + frequency-dependent (power-law) attenuation.

For transmitter ``i`` and receiver ``j`` the frequency-domain received pulse is

    H_ij(omega) = P(omega) * A0/sqrt(d_i d_j) * D(theta_i, f) D(theta_j, f)
                  * exp(-alpha(f) (d_i + d_j)) * exp(-i omega (d_i + d_j)/c_L)

following Holmes, Drinkwater and Wilcox (2005). The reflector itself is an
isotropic point (scattering amplitude 1): the angle- and frequency-dependent
scatterer response is reserved for the side-drilled-hole model M2.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ..propagation import (
        attenuation_coefficient,
        beam_spread_amplitude,
        element_directivity,
        element_ray_geometry,
    )
    from ..pulse import gaussian_pulse_spectrum, shifted_pulse
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from propagation import (
        attenuation_coefficient,
        beam_spread_amplitude,
        element_directivity,
        element_ray_geometry,
    )
    from pulse import gaussian_pulse_spectrum, shifted_pulse


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PropagationPhysics:
    """Switches for the amplitude factors applied to each M0 ray path.

    The default instance switches everything off and reproduces M0.

    Attributes:
        geometric_spreading: Apply ``A0 / sqrt(d_tx d_rx)``.
        spreading_reference_amplitude_m: ``A0`` in metres (uncalibrated).
        attenuation_np_per_m: Amplitude attenuation at the reference frequency.
        attenuation_reference_frequency_hz: ``f_ref``; ``None`` means the pulse
            centre frequency.
        attenuation_frequency_exponent: ``n`` in ``alpha_ref (f/f_ref)**n``;
            ``0`` is frequency-independent attenuation.
        element_width_m: Element width ``a`` for the directivity; ``None``
            disables directivity.
        directivity_frequency_dependent: Evaluate the directivity at every
            frequency (``True``) or only at the centre frequency (``False``).
    """

    geometric_spreading: bool = False
    spreading_reference_amplitude_m: float = 1.0
    attenuation_np_per_m: float = 0.0
    attenuation_reference_frequency_hz: float | None = None
    attenuation_frequency_exponent: float = 0.0
    element_width_m: float | None = None
    directivity_frequency_dependent: bool = True

    @property
    def is_frequency_dependent(self) -> bool:
        """True when any factor varies across the pulse bandwidth."""
        directivity = (
            self.element_width_m is not None and self.directivity_frequency_dependent
        )
        attenuation = (
            self.attenuation_np_per_m > 0.0
            and self.attenuation_frequency_exponent > 0.0
        )
        return directivity or attenuation


@dataclass(frozen=True)
class LadderStep:
    """One rung of the physics ladder: a label, a domain, and its physics."""

    label: str
    domain: str  # "time" or "frequency"
    physics: PropagationPhysics
    description: str


def physics_ladder(
    element_width_m: float,
    attenuation_np_per_m: float,
    attenuation_frequency_exponent: float = 1.0,
) -> tuple[LadderStep, ...]:
    """Return the five ladder steps requested in ``TODO.md``.

    Every step adds exactly one change to the previous step.
    """
    m0 = PropagationPhysics()
    td_plus = replace(
        m0, geometric_spreading=True, attenuation_np_per_m=attenuation_np_per_m
    )
    fd_directivity = replace(
        td_plus, element_width_m=element_width_m, directivity_frequency_dependent=True
    )
    fd_attenuation = replace(
        fd_directivity, attenuation_frequency_exponent=attenuation_frequency_exponent
    )
    return (
        LadderStep("M0", "time", m0, "timing only"),
        LadderStep(
            "M0-TD+", "time", td_plus,
            "timing + spreading + frequency-independent attenuation",
        ),
        LadderStep("M0-FD", "frequency", td_plus, "M0-TD+ synthesized in frequency"),
        LadderStep(
            "M0-FD+D", "frequency", fd_directivity,
            "+ frequency-dependent element directivity",
        ),
        LadderStep(
            "M0-FD+D+A", "frequency", fd_attenuation,
            "+ frequency-dependent attenuation",
        ),
    )


def pair_transfer(
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    physics: PropagationPhysics,
    frequency_hz: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return the real amplitude factor and travel time for every element pair.

    This single function is used by both synthesizers, so the time- and
    frequency-domain models apply identical physics.

    Args:
        frequency_hz: ``None`` evaluates all factors at the centre frequency and
            returns amplitudes of shape ``(N, N)``. A 1D array returns a
            frequency-dependent table of shape ``(N, N, Nf)``.

    Returns:
        ``(amplitude, travel_time_s)``; ``travel_time_s`` has shape ``(N, N)``.
    """
    if not isinstance(physics, PropagationPhysics):
        raise TypeError("physics must be a PropagationPhysics instance")
    distances, sin_theta = element_ray_geometry(
        element_coordinates, reflector_coordinate
    )
    speed = float(wave_speed_m_s)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("wave_speed_m_s must be a positive finite scalar")
    f0 = float(centre_frequency_hz)
    if not np.isfinite(f0) or f0 <= 0.0:
        raise ValueError("centre_frequency_hz must be a positive finite scalar")

    d_tx = distances[:, np.newaxis]
    d_rx = distances[np.newaxis, :]
    path = d_tx + d_rx
    travel_time = path / speed

    if frequency_hz is None:
        frequency = np.array(f0)
        tx_axis = d_tx
        rx_axis = d_rx
    else:
        frequency = np.asarray(frequency_hz, dtype=float)
        if frequency.ndim != 1:
            raise ValueError("frequency_hz must be one-dimensional")
        # Append a trailing frequency axis: (N, 1, 1) and (1, N, 1).
        tx_axis = d_tx[:, :, np.newaxis]
        rx_axis = d_rx[:, :, np.newaxis]
        path = path[:, :, np.newaxis]

    amplitude = np.ones(np.broadcast_shapes(path.shape, frequency.shape))

    # Factor 1: 2D cylindrical beam spreading, A0 / sqrt(d_tx d_rx).
    if physics.geometric_spreading:
        amplitude = amplitude * beam_spread_amplitude(
            tx_axis, rx_axis, physics.spreading_reference_amplitude_m
        )

    # Factor 2: element directivity, D(theta_tx, f) * D(theta_rx, f).
    if physics.element_width_m is not None:
        directivity_frequency = (
            frequency if physics.directivity_frequency_dependent else np.array(f0)
        )
        # sin(theta) laid out like the distances: (N, 1[, 1]) and (1, N[, 1]).
        tx_sine = sin_theta.reshape(tx_axis.shape)
        rx_sine = sin_theta.reshape(rx_axis.shape)
        amplitude = amplitude * (
            element_directivity(
                tx_sine, directivity_frequency, physics.element_width_m, speed
            )
            * element_directivity(
                rx_sine, directivity_frequency, physics.element_width_m, speed
            )
        )

    # Factor 3: material attenuation, exp(-alpha(f) * (d_tx + d_rx)).
    if physics.attenuation_np_per_m > 0.0:
        reference = physics.attenuation_reference_frequency_hz or f0
        alpha = attenuation_coefficient(
            frequency,
            physics.attenuation_np_per_m,
            reference,
            physics.attenuation_frequency_exponent,
        )
        amplitude = amplitude * np.exp(-alpha * path)

    return np.asarray(amplitude, dtype=float), travel_time


def simulate_point_reflector_fmc_td(
    time_s: np.ndarray,
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
    physics: PropagationPhysics,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Time-domain synthesis: ``s_ij(t) = a_ij * p(t - t_ij)`` (M0, M0-TD+).

    A time-domain trace can only carry one scalar amplitude per pair, so every
    factor is evaluated at the centre frequency. Frequency-dependent physics is
    rejected rather than silently approximated; use
    :func:`simulate_point_reflector_fmc_fd` instead.

    Returns:
        ``(fmc, travel_time_s, amplitude)`` with shapes ``(N, N, Nt)``,
        ``(N, N)`` and ``(N, N)``.
    """
    if not isinstance(time_s, np.ndarray) or time_s.ndim != 1:
        raise ValueError("time_s must be a one-dimensional NumPy array")
    if physics.is_frequency_dependent:
        raise ValueError(
            "frequency-dependent physics needs simulate_point_reflector_fmc_fd"
        )
    amplitude, travel_time = pair_transfer(
        element_coordinates,
        reflector_coordinate,
        wave_speed_m_s,
        centre_frequency_hz,
        physics,
    )
    pulses = shifted_pulse(
        time_s[np.newaxis, np.newaxis, :],
        travel_time[:, :, np.newaxis],
        centre_frequency_hz,
        sigma_s,
    )
    return amplitude[:, :, np.newaxis] * pulses, travel_time, amplitude


@dataclass(frozen=True)
class FrequencyDomainFMCResult:
    """Frequency-domain M0-ladder FMC.

    ``spectrum`` has shape ``(N, N, Nf)`` and uses NumPy's ``exp(+i omega t)``
    synthesis convention: ``fmc = irfft(spectrum)`` on the record time axis.
    ``transfer`` holds the real amplitude factors ``(N, N, Nf)`` before the
    pulse spectrum and delay phase are applied.
    """

    fmc: FloatArray
    spectrum: ComplexArray
    transfer: FloatArray
    frequency_hz: FloatArray
    travel_time_s: FloatArray


def simulate_point_reflector_fmc_fd(
    time_s: np.ndarray,
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
    physics: PropagationPhysics,
    *,
    wrap_guard_sigma: float = 5.0,
) -> FrequencyDomainFMCResult:
    """Frequency-domain synthesis of the M0 ladder (M0-FD and above).

    For each pair and each ``rfft`` frequency bin::

        S_ij(omega) = P(omega) * T_ij(f) * exp(-i omega (t_ij - t_0))

    where ``T_ij`` is :func:`pair_transfer` evaluated per frequency and
    ``t_0 = time_s[0]``. ``irfft`` then returns the FMC on ``time_s``. The
    delay phasor is the requested ``exp(-i omega (d_tx + d_rx) / c_L)``.

    Raises:
        ValueError: If ``time_s`` is not uniform, or an arrival lies within
            ``wrap_guard_sigma`` pulse widths of the record ends (where the
            periodic DFT would wrap the pulse around).
    """
    if not isinstance(time_s, np.ndarray) or time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be a one-dimensional NumPy array")
    time = np.asarray(time_s, dtype=float)
    increments = np.diff(time)
    dt = float(increments[0])
    if dt <= 0.0 or not np.allclose(increments, dt, rtol=1.0e-10, atol=0.0):
        raise ValueError("time_s must be uniformly sampled and increasing")

    frequency, pulse_spectrum = gaussian_pulse_spectrum(
        time.size, dt, centre_frequency_hz, sigma_s
    )
    transfer, travel_time = pair_transfer(
        element_coordinates,
        reflector_coordinate,
        wave_speed_m_s,
        centre_frequency_hz,
        physics,
        frequency_hz=frequency,
    )

    guard = float(wrap_guard_sigma) * float(sigma_s)
    if np.min(travel_time) - guard < time[0] or np.max(travel_time) + guard > time[-1]:
        raise ValueError("time record is too short for wrap-free FFT synthesis")

    phase = np.exp(
        -1j
        * (2.0 * np.pi * frequency)[np.newaxis, np.newaxis, :]
        * (travel_time - time[0])[:, :, np.newaxis]
    )
    spectrum = pulse_spectrum[np.newaxis, np.newaxis, :] * transfer * phase
    if time.size % 2 == 0:
        # The Nyquist bin of a real signal must itself be real.
        spectrum[:, :, -1] = spectrum[:, :, -1].real
    fmc = np.fft.irfft(spectrum, n=time.size, axis=2)
    return FrequencyDomainFMCResult(
        fmc=np.asarray(fmc, dtype=float),
        spectrum=spectrum,
        transfer=transfer,
        frequency_hz=np.asarray(frequency, dtype=float),
        travel_time_s=travel_time,
    )


def simulate_ladder_step(
    step: LadderStep,
    time_s: np.ndarray,
    element_coordinates: ArrayLike,
    reflector_coordinate: ArrayLike,
    wave_speed_m_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
) -> FloatArray:
    """Return the time-domain FMC for one :class:`LadderStep`."""
    arguments = (
        time_s,
        element_coordinates,
        reflector_coordinate,
        wave_speed_m_s,
        centre_frequency_hz,
        sigma_s,
        step.physics,
    )
    if step.domain == "time":
        return simulate_point_reflector_fmc_td(*arguments)[0]
    if step.domain == "frequency":
        return simulate_point_reflector_fmc_fd(*arguments).fmc
    raise ValueError(f"unknown ladder domain {step.domain!r}")
