"""Idealized array coupling for the M2-2D elastic SDH kernel.

This module converts the exact plane-strain P--SV cavity solution into an LL
FMC. It uses phase-only point-to-centre propagation and unit transmit/receive
transfer functions. It is therefore an elastic-scattering FMC, not a calibrated
probe-voltage prediction and not the full finite-probe 3D model of Bostrom and
Bovik (2003).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ..pulse import gaussian_pulse_spectrum
    from ..scattering.elastic_sdh import (
        ElasticMaterial,
        elastic_sdh_scattering,
        recommended_n_max,
    )
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from pulse import gaussian_pulse_spectrum
    from scattering.elastic_sdh import (
        ElasticMaterial,
        elastic_sdh_scattering,
        recommended_n_max,
    )


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ElasticSideDrilledHole:
    """Circular cylindrical cavity in the repository x-z inspection plane."""

    x_m: float
    z_m: float
    radius_m: float

    def __post_init__(self) -> None:
        values = np.asarray((self.x_m, self.z_m, self.radius_m), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("SDH geometry must contain finite values")
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")


@dataclass(frozen=True)
class ElasticSDHFMCResult:
    """M2-2D idealized LL FMC plus the unmeasured P-to-SV kernel.

    ``frequency_domain_ll`` and ``scattering_kernel_ls`` have shape
    ``(N, N, Nf)``. The latter retains paper-convention ``F_PS`` values but is
    not included in ``fmc_ll`` because SV propagation and receive sensitivity
    have not been modelled.
    """

    fmc_ll: FloatArray
    frequency_domain_ll: ComplexArray
    scattering_kernel_ls: ComplexArray
    frequency_hz: FloatArray
    time_s: FloatArray
    metadata: dict[str, object]


def _validated_time(time_s: np.ndarray) -> tuple[FloatArray, float]:
    if not isinstance(time_s, np.ndarray):
        raise TypeError("time_s must be a NumPy array")
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be finite with shape (Nt,), Nt >= 2")
    increments = np.diff(time)
    if np.any(increments <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    dt = float(increments[0])
    if not np.allclose(increments, dt, rtol=1.0e-10, atol=0.0):
        raise ValueError("time_s must be uniformly sampled for FFT synthesis")
    return time, dt


def _validated_elements(element_coordinates: ArrayLike) -> FloatArray:
    elements = np.asarray(element_coordinates, dtype=float)
    if (
        elements.ndim != 2
        or elements.shape[1] != 2
        or elements.shape[0] == 0
        or not np.all(np.isfinite(elements))
    ):
        raise ValueError("element_coordinates must be finite with shape (N, 2)")
    return elements


def _zero_centred_pulse_spectrum(
    n_samples: int,
    dt_s: float,
    centre_frequency_hz: float,
    sigma_s: float,
) -> ComplexArray:
    """DFT of the existing pulse centred at zero on a periodic lag grid."""
    return gaussian_pulse_spectrum(n_samples, dt_s, centre_frequency_hz, sigma_s)[1]


def simulate_elastic_sdh_fmc(
    time_s: np.ndarray,
    element_coordinates: ArrayLike,
    defect: ElasticSideDrilledHole,
    material: ElasticMaterial,
    centre_frequency_hz: float,
    sigma_s: float,
    n_max: int,
    *,
    spectrum_relative_cutoff: float = 1.0e-10,
    wrap_guard_sigma: float = 5.0,
) -> ElasticSDHFMCResult:
    """Synthesize the first idealized M2-2D P-to-P FMC.

    For pair ``(i,j)`` and positive angular frequency ``omega``, the NumPy FFT
    coefficient is

    ``P(omega) * conj(F_PP(beta_j,alpha_i)) * exp(-i*omega*tau_ij)``.

    ``F_PP`` is conjugated because the scattering kernel follows the paper's
    ``exp(-i*omega*t)`` phasor, whereas NumPy ``irfft`` synthesizes positive
    frequencies with ``exp(+i*omega*t)``. Propagation is phase-only,
    ``tau_ij=(r_i+r_j)/c_p``; ``H_tx=H_rx=1``. No spreading, directivity,
    attenuation, surface reflection, probe response, or voltage calibration is
    introduced.
    """
    time, dt = _validated_time(time_s)
    elements = _validated_elements(element_coordinates)
    if not isinstance(defect, ElasticSideDrilledHole):
        raise TypeError("defect must be an ElasticSideDrilledHole")
    if not isinstance(material, ElasticMaterial):
        raise TypeError("material must be an ElasticMaterial")
    if (
        not isinstance(n_max, (int, np.integer))
        or isinstance(n_max, (bool, np.bool_))
        or n_max < 0
    ):
        raise ValueError("n_max must be a non-negative integer")
    cutoff = float(spectrum_relative_cutoff)
    guard = float(wrap_guard_sigma)
    if not np.isfinite(cutoff) or not 0.0 <= cutoff < 1.0:
        raise ValueError("spectrum_relative_cutoff must lie in [0, 1)")
    if not np.isfinite(guard) or guard < 0.0:
        raise ValueError("wrap_guard_sigma must be non-negative and finite")

    centre = np.array((defect.x_m, defect.z_m), dtype=float)
    incident_vectors = centre - elements
    distances = np.linalg.norm(incident_vectors, axis=1)
    if np.any(distances <= defect.radius_m):
        raise ValueError("array elements must lie outside the circular cavity")
    incident_angles = np.arctan2(incident_vectors[:, 1], incident_vectors[:, 0])
    outgoing_vectors = elements - centre
    outgoing_angles = np.arctan2(outgoing_vectors[:, 1], outgoing_vectors[:, 0])
    relative_angles = outgoing_angles[np.newaxis, :] - incident_angles[:, np.newaxis]
    travel_times = (distances[:, np.newaxis] + distances[np.newaxis, :]) / (
        material.longitudinal_speed_m_s
    )

    pulse_spectrum = _zero_centred_pulse_spectrum(
        time.size, dt, centre_frequency_hz, sigma_s
    )
    frequency = np.fft.rfftfreq(time.size, dt)
    angular_frequency = 2.0 * np.pi * frequency
    active = np.abs(pulse_spectrum) > cutoff * np.max(np.abs(pulse_spectrum))
    active[0] = False
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise ValueError("pulse spectrum has no positive-frequency bins above cutoff")

    sigma = float(sigma_s)
    earliest_supported = float(np.min(travel_times) - guard * sigma)
    latest_supported = float(np.max(travel_times) + guard * sigma)
    if earliest_supported < time[0] or latest_supported > time[-1]:
        raise ValueError(
            "time record is too short for wrap-free synthesis: require "
            f"[{earliest_supported:.6e}, {latest_supported:.6e}] s within "
            f"[{time[0]:.6e}, {time[-1]:.6e}] s"
        )

    n = elements.shape[0]
    n_frequency = frequency.size
    frequency_domain = np.zeros((n, n, n_frequency), dtype=complex)
    kernel_ps = np.zeros_like(frequency_domain)
    maximum_boundary_residual = 0.0
    maximum_condition_number = 0.0
    maximum_guide_n_max = 0
    for frequency_index in active_indices:
        frequency_hz = float(frequency[frequency_index])
        response = elastic_sdh_scattering(
            frequency_hz,
            defect.radius_m,
            material,
            0.0,
            relative_angles.ravel(),
            int(n_max),
        )
        f_pp = response.f_pp.reshape(n, n)
        f_ps = response.f_ps.reshape(n, n)
        phase = np.exp(
            -1j
            * angular_frequency[frequency_index]
            * (travel_times - float(time[0]))
        )
        frequency_domain[:, :, frequency_index] = (
            pulse_spectrum[frequency_index] * np.conjugate(f_pp) * phase
        )
        kernel_ps[:, :, frequency_index] = f_ps
        maximum_boundary_residual = max(
            maximum_boundary_residual,
            float(
                response.traction_residual_metadata[
                    "maximum_normalized_boundary_residual"
                ]
            ),
        )
        maximum_condition_number = max(
            maximum_condition_number,
            float(response.traction_residual_metadata["maximum_condition_number"]),
        )
        maximum_guide_n_max = max(
            maximum_guide_n_max,
            recommended_n_max(frequency_hz, defect.radius_m, material),
        )

    if time.size % 2 == 0:
        frequency_domain[:, :, -1] = frequency_domain[:, :, -1].real
    fmc = np.fft.irfft(frequency_domain, n=time.size, axis=2)
    wrap_edge_samples = max(1, int(np.ceil(guard * sigma / dt)))
    edge_peak = max(
        float(np.max(np.abs(fmc[:, :, :wrap_edge_samples]))),
        float(np.max(np.abs(fmc[:, :, -wrap_edge_samples:]))),
    )
    global_peak = max(float(np.max(np.abs(fmc))), np.finfo(float).tiny)

    return ElasticSDHFMCResult(
        fmc_ll=np.asarray(fmc, dtype=float),
        frequency_domain_ll=frequency_domain,
        scattering_kernel_ls=kernel_ps,
        frequency_hz=np.asarray(frequency, dtype=float),
        time_s=time.copy(),
        metadata={
            "model_id": "M2-2D_exact_plane_strain_P-SV_cavity",
            "measured_channel": "LL/P-to-P only",
            "amplitude_calibration": "uncalibrated",
            "paper_phasor": "exp(-i omega t)",
            "numpy_irfft_phasor": "exp(+i omega t)",
            "paper_to_fft_conversion": "complex conjugate F_PP",
            "propagation": "phase-only point-to-centre; no spreading",
            "H_tx": 1.0,
            "H_rx": 1.0,
            "n_max": int(n_max),
            "maximum_recommended_n_max_over_active_bins": maximum_guide_n_max,
            "active_frequency_bin_count": int(active_indices.size),
            "active_frequency_range_hz": [
                float(frequency[active_indices[0]]),
                float(frequency[active_indices[-1]]),
            ],
            "maximum_normalized_boundary_residual": maximum_boundary_residual,
            "maximum_condition_number": maximum_condition_number,
            "travel_time_s": travel_times,
            "incident_angle_rad": incident_angles,
            "outgoing_angle_rad": outgoing_angles,
            "wrap_guard_sigma": guard,
            "edge_to_global_peak_ratio": edge_peak / global_peak,
            "omitted": [
                "full h!=0 3D T-matrix",
                "finite probe aperture and transfer functions",
                "P-to-SV propagation and receive sensitivity",
                "SH",
                "geometric spreading",
                "directivity",
                "attenuation",
                "free-surface multiple scattering",
            ],
        },
    )
