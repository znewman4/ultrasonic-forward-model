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
    from ..pulse import gaussian_pulse
    from ..scattering.elastic_sdh import (
        ElasticMaterial,
        elastic_sdh_modal_scattering,
        elastic_sdh_scattering,
        recommended_n_max,
    )
except ImportError:  # Supports importing when ``src`` is directly on sys.path.
    from pulse import gaussian_pulse
    from scattering.elastic_sdh import (
        ElasticMaterial,
        elastic_sdh_modal_scattering,
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


@dataclass(frozen=True)
class ElasticSDHModalFMCResult:
    """Four separate ideal wavefield FMC branches from the complete matrix.

    The two letters specify incident/transmit and outgoing/receive mode in that
    order. Every array has shape ``(N,N,Nt)``. These are flux-normalized ideal
    elastic wavefield responses with ``H_tx=H_rx=1``; they are deliberately not
    summed and are not calibrated probe-voltage predictions.
    """

    fmc_pp: FloatArray
    fmc_ps: FloatArray
    fmc_sp: FloatArray
    fmc_ss: FloatArray
    frequency_domain_pp: ComplexArray
    frequency_domain_ps: ComplexArray
    frequency_domain_sp: ComplexArray
    frequency_domain_ss: ComplexArray
    frequency_hz: FloatArray
    time_s: FloatArray
    travel_time_pp_s: FloatArray
    travel_time_ps_s: FloatArray
    travel_time_sp_s: FloatArray
    travel_time_ss_s: FloatArray
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
    indices = np.arange(n_samples)
    signed_indices = np.where(indices <= n_samples // 2, indices, indices - n_samples)
    lag_s = signed_indices * dt_s
    pulse = gaussian_pulse(lag_s, centre_frequency_hz, sigma_s)
    return np.asarray(np.fft.rfft(pulse), dtype=complex)


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


def modal_travel_times(
    element_coordinates: ArrayLike,
    defect: ElasticSideDrilledHole,
    material: ElasticMaterial,
) -> dict[str, FloatArray]:
    """Return PP, PS, SP and SS centre-path delays with mode-specific speeds."""
    elements = _validated_elements(element_coordinates)
    if not isinstance(defect, ElasticSideDrilledHole):
        raise TypeError("defect must be an ElasticSideDrilledHole")
    if not isinstance(material, ElasticMaterial):
        raise TypeError("material must be an ElasticMaterial")
    centre = np.array((defect.x_m, defect.z_m), dtype=float)
    distance = np.linalg.norm(elements - centre, axis=1)
    if np.any(distance <= defect.radius_m):
        raise ValueError("array elements must lie outside the circular cavity")
    p = distance / material.longitudinal_speed_m_s
    s = distance / material.shear_speed_m_s
    return {
        "PP": p[:, None] + p[None, :],
        "PS": p[:, None] + s[None, :],
        "SP": s[:, None] + p[None, :],
        "SS": s[:, None] + s[None, :],
    }


def simulate_elastic_sdh_modal_fmc(
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
) -> ElasticSDHModalFMCResult:
    """Synthesize separate PP, PS, SP and SS ideal elastic-wavefield FMCs.

    The scattering matrix is energy-flux normalized. For path ``AB`` the
    incident leg uses mode ``A`` and the outgoing leg mode ``B``. Each positive
    FFT coefficient is

    ``P(omega)*conj(F_flux_BA)*exp(-i*omega*tau_AB)``.

    The conjugation converts the paper's ``exp(-i*omega*t)`` amplitudes to
    NumPy's positive-frequency ``irfft`` representation. ``H_tx=H_rx=1`` for
    both modes. The four responses remain separate because a physical probe
    polarization/voltage operator has not been introduced.
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
    distance = np.linalg.norm(incident_vectors, axis=1)
    if np.any(distance <= defect.radius_m):
        raise ValueError("array elements must lie outside the circular cavity")
    incident_angles = np.arctan2(incident_vectors[:, 1], incident_vectors[:, 0])
    outgoing_vectors = elements - centre
    outgoing_angles = np.arctan2(outgoing_vectors[:, 1], outgoing_vectors[:, 0])
    relative_angles = outgoing_angles[None, :] - incident_angles[:, None]
    delays = modal_travel_times(elements, defect, material)

    sigma = float(sigma_s)
    all_delays = np.stack(tuple(delays.values()))
    earliest_supported = float(np.min(all_delays) - guard * sigma)
    latest_supported = float(np.max(all_delays) + guard * sigma)
    if earliest_supported < time[0] or latest_supported > time[-1]:
        raise ValueError(
            "time record is too short for wrap-free multimode synthesis: require "
            f"[{earliest_supported:.6e}, {latest_supported:.6e}] s within "
            f"[{time[0]:.6e}, {time[-1]:.6e}] s"
        )

    pulse_spectrum = _zero_centred_pulse_spectrum(
        time.size, dt, centre_frequency_hz, sigma_s
    )
    frequency = np.fft.rfftfreq(time.size, dt)
    omega = 2.0 * np.pi * frequency
    active = np.abs(pulse_spectrum) > cutoff * np.max(np.abs(pulse_spectrum))
    active[0] = False
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise ValueError("pulse spectrum has no positive-frequency bins above cutoff")

    n = elements.shape[0]
    spectra = {
        key: np.zeros((n, n, frequency.size), dtype=complex)
        for key in ("PP", "PS", "SP", "SS")
    }
    maximum_traction = {"P": 0.0, "SV": 0.0}
    maximum_reciprocity = 0.0
    maximum_energy = 0.0
    maximum_condition = 0.0
    maximum_guide = 0
    for frequency_index in active_indices:
        frequency_hz = float(frequency[frequency_index])
        response = elastic_sdh_modal_scattering(
            frequency_hz,
            defect.radius_m,
            material,
            0.0,
            relative_angles.ravel(),
            int(n_max),
        )
        # Matrix rows are outgoing P/SV, columns incident P/SV.
        modal_amplitudes = {
            "PP": response.flux_f_pp.reshape(n, n),
            "PS": response.flux_f_ps.reshape(n, n),
            "SP": response.flux_f_sp.reshape(n, n),
            "SS": response.flux_f_ss.reshape(n, n),
        }
        for key in spectra:
            phase = np.exp(-1j * omega[frequency_index] * (delays[key] - time[0]))
            spectra[key][:, :, frequency_index] = (
                pulse_spectrum[frequency_index]
                * np.conjugate(modal_amplitudes[key])
                * phase
            )
        validation = response.validation_metadata
        for mode in maximum_traction:
            maximum_traction[mode] = max(
                maximum_traction[mode],
                validation["normalized_boundary_traction_residual"][mode],
            )
        maximum_reciprocity = max(
            maximum_reciprocity,
            validation["maximum_partial_wave_reciprocity_error"],
        )
        maximum_energy = max(
            maximum_energy,
            validation["maximum_partial_wave_energy_balance_error"],
        )
        maximum_condition = max(
            maximum_condition, validation["maximum_condition_number"]
        )
        maximum_guide = max(
            maximum_guide,
            recommended_n_max(frequency_hz, defect.radius_m, material),
        )
    if time.size % 2 == 0:
        for spectrum in spectra.values():
            spectrum[:, :, -1] = spectrum[:, :, -1].real
    fmcs = {
        key: np.asarray(np.fft.irfft(value, n=time.size, axis=2), dtype=float)
        for key, value in spectra.items()
    }
    # The explicit support check above is the no-truncation criterion. This
    # endpoint diagnostic samples only the last/first few acquisition samples;
    # using the whole guard interval would incorrectly count a legitimate SS
    # tail that is still several sigma away from the actual record boundary.
    edge_samples = min(8, max(1, time.size // 100))
    edge_ratios = {}
    for key, fmc in fmcs.items():
        peak = max(float(np.max(np.abs(fmc))), np.finfo(float).tiny)
        edge = max(
            float(np.max(np.abs(fmc[:, :, :edge_samples]))),
            float(np.max(np.abs(fmc[:, :, -edge_samples:]))),
        )
        edge_ratios[key] = edge / peak
    return ElasticSDHModalFMCResult(
        fmc_pp=fmcs["PP"],
        fmc_ps=fmcs["PS"],
        fmc_sp=fmcs["SP"],
        fmc_ss=fmcs["SS"],
        frequency_domain_pp=spectra["PP"],
        frequency_domain_ps=spectra["PS"],
        frequency_domain_sp=spectra["SP"],
        frequency_domain_ss=spectra["SS"],
        frequency_hz=np.asarray(frequency, dtype=float),
        time_s=time.copy(),
        travel_time_pp_s=delays["PP"],
        travel_time_ps_s=delays["PS"],
        travel_time_sp_s=delays["SP"],
        travel_time_ss_s=delays["SS"],
        metadata={
            "model_id": "M2-2D_complete_P-SV_ideal_wavefield",
            "modal_matrix_layout": "rows outgoing (P,SV), columns incident (P,SV)",
            "amplitude_normalization": "energy-flux differential cross-section amplitude sqrt(m)",
            "measurement_status": "four separate ideal wavefield responses; not voltage",
            "paper_phasor": "exp(-i omega t)",
            "numpy_irfft_phasor": "exp(+i omega t)",
            "paper_to_fft_conversion": "complex conjugate modal amplitude",
            "propagation": "mode-dependent phase-only point-to-centre; no spreading",
            "H_tx": 1.0,
            "H_rx": 1.0,
            "n_max": int(n_max),
            "maximum_recommended_n_max_over_active_bins": int(maximum_guide),
            "active_frequency_bin_count": int(active_indices.size),
            "active_frequency_range_hz": [
                float(frequency[active_indices[0]]),
                float(frequency[active_indices[-1]]),
            ],
            "maximum_normalized_boundary_residual": maximum_traction,
            "maximum_partial_wave_reciprocity_error": float(maximum_reciprocity),
            "maximum_partial_wave_energy_balance_error": float(maximum_energy),
            "maximum_condition_number": float(maximum_condition),
            "incident_angle_rad": incident_angles,
            "outgoing_angle_rad": outgoing_angles,
            "fft_edge_to_global_peak_ratio": edge_ratios,
            "fft_endpoint_diagnostic_samples": int(edge_samples),
            "omitted": [
                "combined voltage FMC",
                "probe directivity and finite aperture",
                "coupling-layer physics",
                "piezoelectric response",
                "receive polarization sensitivity",
                "geometric spreading and attenuation",
                "general h!=0 3D T-matrix and SH coupling",
            ],
        },
    )
