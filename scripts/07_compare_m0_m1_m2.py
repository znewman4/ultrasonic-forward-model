"""Generate M2-2D kernel validation and M0/M1/M2/experiment comparisons."""
from __future__ import annotations

import json
import time as clock

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from common import (
    CENTRE_FREQUENCY_HZ,
    MAT_PATH,
    REFLECTOR,
    ROOT,
    SIGMA_S,
    WAVE_SPEED_M_S,
    X_GRID_M,
    Z_GRID_M,
    save_report,
)
from src.data_loader import experimental_fmc_array, load_fmc
from src.geometry import array_coordinates
from src.imaging.tfm import tfm_image
from src.models.boundary_circle import (
    CircularReflector,
    circular_boundary_points,
    simulate_circular_reflector_fmc,
)
from src.models.elastic_sdh import ElasticSideDrilledHole, simulate_elastic_sdh_fmc
from src.models.point_reflector import simulate_point_reflector_fmc
from src.scattering.elastic_sdh import (
    ElasticMaterial,
    elastic_sdh_scattering,
    recommended_n_max,
)


MATERIAL = ElasticMaterial(2700.0, WAVE_SPEED_M_S, 3100.0)
RADIUS_M = 0.5e-3
M1_POINTS = 64
M2_N_MAX = 18
LEVEL_6DB = 10.0 ** (-6.0 / 20.0)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.vdot(first, second).real / denominator) if denominator else float("nan")


def _normalise(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values))
    return values / scale if scale else np.zeros_like(values)


def _crossings(coordinate: np.ndarray, values: np.ndarray) -> tuple[float, float] | None:
    magnitude = np.abs(values)
    peak = int(np.argmax(magnitude))
    threshold = LEVEL_6DB * magnitude[peak]
    left = np.flatnonzero(magnitude[:peak] < threshold)
    right = np.flatnonzero(magnitude[peak + 1 :] < threshold)
    if left.size == 0 or right.size == 0:
        return None
    il = int(left[-1]); ir = int(peak + 1 + right[0])
    xl = float(np.interp(threshold, magnitude[il : il + 2], coordinate[il : il + 2]))
    xr = float(
        np.interp(
            threshold,
            magnitude[ir - 1 : ir + 1][::-1],
            coordinate[ir - 1 : ir + 1][::-1],
        )
    )
    return xl, xr


def _slice_metrics(coordinate: np.ndarray, values: np.ndarray) -> tuple[float | None, float | None]:
    crossings = _crossings(coordinate, values)
    if crossings is None:
        return None, None
    left, right = crossings
    magnitude = np.abs(values)
    outside = (coordinate < left) | (coordinate > right)
    if not np.any(outside):
        return (right - left) * 1e3, None
    ratio = max(float(np.max(magnitude[outside]) / np.max(magnitude)), np.finfo(float).tiny)
    return (right - left) * 1e3, 20.0 * np.log10(ratio)


def _image_metrics(image: np.ndarray) -> dict[str, object]:
    iz, ix = np.unravel_index(np.argmax(image), image.shape)
    x = float(X_GRID_M[ix]); z = float(Z_GRID_M[iz])
    lateral_width, lateral_sidelobe = _slice_metrics(X_GRID_M, image[iz])
    axial_width, axial_sidelobe = _slice_metrics(Z_GRID_M, image[:, ix])
    return {
        "peak_m": [x, z],
        "localisation_error_mm": float(np.hypot(x - REFLECTOR[0], z - REFLECTOR[1]) * 1e3),
        "lateral_minus_6_db_width_mm": lateral_width,
        "axial_minus_6_db_width_mm": axial_width,
        "lateral_peak_sidelobe_level_db": lateral_sidelobe,
        "axial_peak_sidelobe_level_db": axial_sidelobe,
    }


def _gated_peak_metrics(time: np.ndarray, fmc: np.ndarray, gate: np.ndarray) -> dict[str, object]:
    envelope = np.abs(hilbert(fmc, axis=2))[:, :, gate]
    gated_time = time[gate]
    indices = np.argmax(envelope, axis=2)
    peak_time = gated_time[indices]
    rms = np.sqrt(np.mean(fmc[:, :, gate] ** 2, axis=2))
    return {
        "peak_time_s": peak_time,
        "rms": rms,
        "central_peak_us": float(peak_time[fmc.shape[0] // 2 - 1, fmc.shape[1] // 2 - 1] * 1e6),
        "peak_time_range_us": [float(np.min(peak_time) * 1e6), float(np.max(peak_time) * 1e6)],
    }


def _plot_kernel(fig_dir, metrics: dict[str, object]) -> None:
    beta = np.linspace(-np.pi, np.pi, 721)
    incident_angles = np.deg2rad([0.0, 30.0, 60.0])
    responses = [
        elastic_sdh_scattering(
            CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, alpha, beta, M2_N_MAX
        )
        for alpha in incident_angles
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for alpha, response in zip(incident_angles, responses, strict=True):
        label = rf"$\alpha={np.rad2deg(alpha):.0f}^\circ$"
        axes[0, 0].plot(np.rad2deg(beta), np.abs(response.f_pp), label=label)
        axes[1, 0].plot(np.rad2deg(beta), np.unwrap(np.angle(response.f_pp)), label=label)
        axes[0, 1].plot(np.rad2deg(beta), np.abs(response.f_ps), label=label)
        axes[1, 1].plot(np.rad2deg(beta), np.unwrap(np.angle(response.f_ps)), label=label)
    axes[0, 0].set(title="P→P magnitude", ylabel="|F_PP|")
    axes[0, 1].set(title="P→SV magnitude", ylabel="|F_PS|")
    axes[1, 0].set(xlabel="Outgoing angle β (degrees)", ylabel="Unwrapped phase (rad)")
    axes[1, 1].set(xlabel="Outgoing angle β (degrees)", ylabel="Unwrapped phase (rad)")
    for ax in axes.ravel(): ax.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "far_field_angles.png", dpi=180); plt.close(fig)

    reference = responses[0]
    orders = reference.orders
    fig, ax = plt.subplots(figsize=(9, 5))
    selected = np.argsort(np.abs(reference.longitudinal_coefficients))[-7:]
    for index in selected:
        contribution = reference.longitudinal_coefficients[index] * np.exp(
            1j * orders[index] * (beta - np.pi / 2.0)
        )
        ax.plot(np.rad2deg(beta), contribution.real, label=f"m={orders[index]}")
    ax.set(
        xlabel="Outgoing angle β (degrees)",
        ylabel="Real P far-field contribution",
        title="Largest individual harmonic contributions",
    )
    ax.grid(True, alpha=0.25); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(fig_dir / "harmonic_contributions.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(orders, np.abs(reference.longitudinal_coefficients), "o-", label="|A_m| P")
    ax.semilogy(orders, np.abs(reference.shear_coefficients), "s-", label="|B_m| SV")
    ax.set(xlabel="Harmonic order m", ylabel="Coefficient magnitude", title="Outgoing potential coefficients")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "coefficient_magnitudes.png", dpi=180); plt.close(fig)

    n_values = np.arange(2, 21, 2)
    convergence_beta = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    reference_n = 24
    reference_response = elastic_sdh_scattering(
        CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.0, convergence_beta, reference_n
    )
    residuals = []; errors = []
    for n_max in n_values:
        response = elastic_sdh_scattering(
            CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.0, convergence_beta, int(n_max)
        )
        residuals.append(response.traction_residual_metadata["maximum_normalized_boundary_residual"])
        errors.append(np.linalg.norm(response.f_pp - reference_response.f_pp) / np.linalg.norm(reference_response.f_pp))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(n_values, residuals, "o-", label="boundary traction residual")
    ax.semilogy(n_values, errors, "s-", label=f"F_PP error vs m_max={reference_n}")
    ax.axvline(recommended_n_max(CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL), color="black", ls="--", label="paper guide")
    ax.set(xlabel="m_max", ylabel="Relative measure", title="Harmonic convergence")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "traction_and_series_convergence.png", dpi=180); plt.close(fig)
    metrics["harmonic_convergence"] = {
        "m_max": n_values.tolist(),
        "normalized_traction_residual": [float(v) for v in residuals],
        "relative_f_pp_error": [float(v) for v in errors],
        "reference_m_max": reference_n,
        "paper_guide_m_max": recommended_n_max(CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL),
    }

    radii = np.array([0.05, 0.1, 0.25, 0.5, 1.0]) * 1e-3
    fig, ax = plt.subplots(figsize=(9, 5))
    radius_metrics = {}
    for radius in radii:
        response = elastic_sdh_scattering(
            CENTRE_FREQUENCY_HZ,
            float(radius),
            MATERIAL,
            0.0,
            beta,
            recommended_n_max(CENTRE_FREQUENCY_HZ, float(radius), MATERIAL),
        )
        ax.plot(np.rad2deg(beta), np.abs(response.f_pp), label=f"a={radius*1e3:g} mm")
        radius_metrics[f"{radius*1e3:g}_mm"] = {
            "maximum_abs_f_pp": float(np.max(np.abs(response.f_pp))),
            "backscatter_abs_f_pp": float(np.abs(response.f_pp[-1])),
        }
    ax.set(xlabel="Outgoing angle β (degrees)", ylabel="|F_PP|", title="Radius changes elastic angular scattering")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "radius_sensitivity.png", dpi=180); plt.close(fig)
    metrics["single_frequency_radius_sensitivity"] = radius_metrics
    metrics["single_frequency_kernel"] = {
        "frequency_hz": CENTRE_FREQUENCY_HZ,
        "radius_m": RADIUS_M,
        "incident_angles_deg": np.rad2deg(incident_angles).tolist(),
        "m_max": M2_N_MAX,
        "maximum_normalized_traction_residual": float(
            max(r.traction_residual_metadata["maximum_normalized_boundary_residual"] for r in responses)
        ),
    }


def _plot_ascans(path, time, fmcs, pairs) -> None:
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 8), sharex=True)
    for ax, (label, i, j) in zip(axes, pairs, strict=True):
        for name, fmc in fmcs.items():
            ax.plot(time * 1e6, _normalise(fmc[i, j]), label=name, alpha=0.85)
        ax.set(ylabel="Normalized amplitude", title=label)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize="small")
    axes[-1].set(xlabel="Time (µs)", xlim=(10.0, 16.5))
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_spectra(path, time, fmcs, pair) -> None:
    dt = time[1] - time[0]
    frequency = np.fft.rfftfreq(time.size, dt)
    mask = (frequency >= 2e6) & (frequency <= 8e6)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    i, j = pair
    for name, fmc in fmcs.items():
        spectrum = np.fft.rfft(fmc[i, j])
        spectrum = spectrum / max(np.max(np.abs(spectrum[mask])), np.finfo(float).tiny)
        axes[0].plot(frequency[mask] * 1e-6, np.abs(spectrum[mask]), label=name)
        axes[1].plot(frequency[mask] * 1e-6, np.unwrap(np.angle(spectrum[mask])), label=name)
    axes[0].set(ylabel="Normalized magnitude", title="Representative complex spectra")
    axes[1].set(xlabel="Frequency (MHz)", ylabel="Unwrapped phase (rad)")
    for ax in axes: ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize="small")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_maps(path, maps, key, label, cmap="viridis") -> None:
    fig, axes = plt.subplots(1, len(maps), figsize=(15, 3.6), sharex=True, sharey=True)
    for ax, (name, values) in zip(axes, maps.items(), strict=True):
        image = ax.imshow(values[key], origin="lower", aspect="auto", cmap=cmap)
        ax.set(title=name, xlabel="Receiver")
        fig.colorbar(image, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Transmitter")
    fig.suptitle(label); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_tfm(path, images) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(15, 4), sharex=True, sharey=True)
    for ax, (name, image) in zip(axes, images.items(), strict=True):
        im = ax.imshow(
            _normalise(image),
            extent=[X_GRID_M[0]*1e3, X_GRID_M[-1]*1e3, Z_GRID_M[-1]*1e3, Z_GRID_M[0]*1e3],
            aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0,
        )
        ax.scatter([0.0], [40.0], marker="x", color="cyan")
        ax.set(title=name, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.14, top=0.88, wspace=0.12)
    color_axis = fig.add_axes((0.92, 0.18, 0.012, 0.64))
    fig.colorbar(im, cax=color_axis, label="Normalized TFM")
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_profiles(path, images) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, image in images.items():
        iz, ix = np.unravel_index(np.argmax(image), image.shape)
        axes[0].plot(X_GRID_M * 1e3, _normalise(image[iz]), label=name)
        axes[1].plot(_normalise(image[:, ix]), Z_GRID_M * 1e3, label=name)
    axes[0].axhline(LEVEL_6DB, color="black", ls="--")
    axes[1].axvline(LEVEL_6DB, color="black", ls="--")
    axes[1].invert_yaxis()
    axes[0].set(xlabel="x (mm)", ylabel="Normalized amplitude", title="Lateral peak profiles")
    axes[1].set(xlabel="Normalized amplitude", ylabel="z (mm)", title="Axial peak profiles")
    for ax in axes: ax.grid(True, alpha=0.25); ax.legend(fontsize="small")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_differences(path, images) -> None:
    normalized = {name: _normalise(image) for name, image in images.items()}
    differences = {
        "M1 − M0": normalized["M1 boundary"] - normalized["M0 point"],
        "M2 − M0": normalized["M2-2D elastic"] - normalized["M0 point"],
        "Experiment − M2": normalized["Experimental"] - normalized["M2-2D elastic"],
    }
    limit = max(np.max(np.abs(value)) for value in differences.values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, (name, image) in zip(axes, differences.items(), strict=True):
        im = ax.imshow(image, extent=[X_GRID_M[0]*1e3, X_GRID_M[-1]*1e3, Z_GRID_M[-1]*1e3, Z_GRID_M[0]*1e3], aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set(title=name, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)"); fig.colorbar(im, ax=axes, shrink=0.8)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.14, top=0.9, wspace=0.12)
    fig.savefig(path, dpi=180); plt.close(fig)


def _angle_bins(elements, spectra, travel_times, frequency_hz) -> dict[str, dict[str, list[float]]]:
    centre = REFLECTOR
    incoming = centre - elements
    alpha = np.arctan2(incoming[:, 1], incoming[:, 0])
    outgoing = elements - centre
    beta = np.arctan2(outgoing[:, 1], outgoing[:, 0])
    relative = np.angle(np.exp(1j * (beta[None, :] - alpha[:, None])))
    phase_remove = np.exp(1j * 2.0 * np.pi * frequency_hz * travel_times)
    edges = np.linspace(-np.pi, np.pi, 13)
    centres = 0.5 * (edges[:-1] + edges[1:])
    output = {}
    for name, spectrum in spectra.items():
        dephased = spectrum * phase_remove
        values = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            selection = (relative >= left) & (relative < right)
            values.append(np.mean(dephased[selection]) if np.any(selection) else np.nan + 1j*np.nan)
        values = np.asarray(values)
        scale = np.nanmax(np.abs(values))
        output[name] = {
            "angle_deg": np.rad2deg(centres).tolist(),
            "normalized_magnitude": (np.abs(values) / scale).tolist(),
            "phase_rad": np.angle(values).tolist(),
        }
    return output


def _plot_angle_bins(path, binned) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, values in binned.items():
        angle = values["angle_deg"]
        axes[0].plot(angle, values["normalized_magnitude"], "o-", label=name)
        axes[1].plot(angle, values["phase_rad"], "o-", label=name)
    axes[0].set(ylabel="Normalized coherent magnitude", title="Angle-binned dephased response")
    axes[1].set(xlabel="Relative outgoing angle β−α (degrees)", ylabel="Phase (rad)")
    for ax in axes: ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=2); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _radius_broadband_study(time, element) -> dict[str, object]:
    reference = None
    values = {}
    for radius in (0.05e-3, 0.1e-3, 0.25e-3, 0.5e-3, 1.0e-3):
        n_max = max(12, recommended_n_max(7.75e6, radius, MATERIAL))
        result = simulate_elastic_sdh_fmc(
            time,
            element[np.newaxis, :],
            ElasticSideDrilledHole(*REFLECTOR, radius),
            MATERIAL,
            CENTRE_FREQUENCY_HZ,
            SIGMA_S,
            n_max,
            spectrum_relative_cutoff=1e-8,
        )
        trace = result.fmc_ll[0, 0]
        if reference is None: reference = trace
        envelope = np.abs(hilbert(trace))
        values[f"{radius*1e3:g}_mm"] = {
            "radius_m": radius,
            "peak_time_us": float(time[np.argmax(envelope)] * 1e6),
            "peak_abs_amplitude_uncalibrated": float(np.max(np.abs(trace))),
            "rms_amplitude_uncalibrated": float(np.sqrt(np.mean(trace**2))),
            "correlation_with_0.05_mm": _correlation(trace, reference),
        }
    return values


def main() -> None:
    start = clock.perf_counter()
    result_dir = ROOT / "results" / "comparisons" / "M0_M1_M2"
    figure_dir = result_dir / "figures"
    kernel_dir = figure_dir / "esm_2d"
    data_dir = ROOT / "data" / "synthetic" / "elastic_sdh"
    for directory in (result_dir, figure_dir, kernel_dir, data_dir): directory.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {}
    _plot_kernel(kernel_dir, metrics)

    loaded = load_fmc(MAT_PATH)
    time = loaded.metadata.time_s
    elements = array_coordinates(MAT_PATH)
    experimental = experimental_fmc_array(loaded)
    m0, m0_times = simulate_point_reflector_fmc(time, elements, REFLECTOR, WAVE_SPEED_M_S, CENTRE_FREQUENCY_HZ, SIGMA_S)
    boundary = circular_boundary_points(CircularReflector(*REFLECTOR, RADIUS_M, M1_POINTS))
    m1, m1_times = simulate_circular_reflector_fmc(time, elements, boundary, WAVE_SPEED_M_S, CENTRE_FREQUENCY_HZ, SIGMA_S)
    m2_start = clock.perf_counter()
    m2_result = simulate_elastic_sdh_fmc(
        time, elements, ElasticSideDrilledHole(*REFLECTOR, RADIUS_M), MATERIAL,
        CENTRE_FREQUENCY_HZ, SIGMA_S, M2_N_MAX, spectrum_relative_cutoff=1e-8,
    )
    m2_runtime = clock.perf_counter() - m2_start
    m2 = m2_result.fmc_ll

    image_start = clock.perf_counter()
    images = {
        "M0 point": tfm_image(m0, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S),
        "M1 boundary": tfm_image(m1, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S),
        "M2-2D elastic": tfm_image(m2, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S),
        "Experimental": tfm_image(experimental, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S),
    }
    tfm_runtime = clock.perf_counter() - image_start

    np.savez_compressed(
        data_dir / "M2_2D_elastic_sdh_fmc.npz",
        fmc_ll=m2,
        frequency_domain_ll=m2_result.frequency_domain_ll,
        scattering_kernel_ps=m2_result.scattering_kernel_ls,
        frequency_hz=m2_result.frequency_hz,
        time_s=time,
        element_coordinates_m=elements,
        centre_m=REFLECTOR,
        radius_m=np.array(RADIUS_M),
        material_density_kg_m3=np.array(MATERIAL.density_kg_m3),
        longitudinal_speed_m_s=np.array(MATERIAL.longitudinal_speed_m_s),
        shear_speed_m_s=np.array(MATERIAL.shear_speed_m_s),
        incident_angle_rad=m2_result.metadata["incident_angle_rad"],
        outgoing_angle_rad=m2_result.metadata["outgoing_angle_rad"],
        travel_time_s=m2_result.metadata["travel_time_s"],
        tfm_image=images["M2-2D elastic"], x_grid_m=X_GRID_M, z_grid_m=Z_GRID_M,
        model_parameters_json=np.array(json.dumps({
            key: value for key, value in m2_result.metadata.items()
            if not isinstance(value, np.ndarray)
        })),
    )

    gate = (time >= 10.0e-6) & (time <= 16.5e-6)
    fmcs = {"M0 point": m0, "M1 boundary": m1, "M2-2D elastic": m2, "Experimental": experimental}
    maps = {name: _gated_peak_metrics(time, fmc, gate) for name, fmc in fmcs.items()}
    centre_index = int(np.argmin(np.abs(elements[:, 0])))
    quarter_left = int(np.argmin(np.abs(elements[:, 0] + 10e-3)))
    quarter_right = int(np.argmin(np.abs(elements[:, 0] - 10e-3)))
    pairs = [("Central pulse-echo", centre_index, centre_index), ("Off-axis pulse-echo", quarter_left, quarter_left), ("Pitch-catch", quarter_left, quarter_right)]
    _plot_ascans(figure_dir / "representative_ascans.png", time, fmcs, pairs)
    _plot_spectra(figure_dir / "representative_complex_spectra.png", time, fmcs, (centre_index, centre_index))
    _plot_maps(figure_dir / "peak_time_maps.png", maps, "peak_time_s", "Envelope peak time in 10–16.5 µs gate (s)")
    _plot_maps(figure_dir / "rms_tx_rx_maps.png", maps, "rms", "Gated RMS amplitude (model units)")
    _plot_tfm(figure_dir / "tfm_images.png", images)
    _plot_profiles(figure_dir / "tfm_profiles.png", images)
    _plot_differences(figure_dir / "tfm_model_differences.png", images)

    dt = time[1] - time[0]
    frequency = np.fft.rfftfreq(time.size, dt)
    centre_bin = int(np.argmin(np.abs(frequency - CENTRE_FREQUENCY_HZ)))
    spectra_at_centre = {name: np.fft.rfft(fmc, axis=2)[:, :, centre_bin] for name, fmc in fmcs.items()}
    binned = _angle_bins(elements, spectra_at_centre, m0_times, float(frequency[centre_bin]))
    _plot_angle_bins(figure_dir / "angle_binned_response.png", binned)

    pair_metrics = {}
    for label, i, j in pairs:
        traces = {name: fmc[i, j, gate] for name, fmc in fmcs.items()}
        envelopes = {name: np.abs(hilbert(trace)) for name, trace in traces.items()}
        pair_metrics[label] = {
            "indices": [i, j],
            "envelope_peak_us": {
                name: float(time[gate][np.argmax(envelope)] * 1e6)
                for name, envelope in envelopes.items()
            },
            "waveform_correlation_with_experiment": {
                name: _correlation(trace, traces["Experimental"])
                for name, trace in traces.items() if name != "Experimental"
            },
            "envelope_correlation_with_experiment": {
                name: _correlation(envelope, envelopes["Experimental"])
                for name, envelope in envelopes.items() if name != "Experimental"
            },
            "model_waveform_correlations": {
                "M0_M1": _correlation(traces["M0 point"], traces["M1 boundary"]),
                "M0_M2": _correlation(traces["M0 point"], traces["M2-2D elastic"]),
                "M1_M2": _correlation(traces["M1 boundary"], traces["M2-2D elastic"]),
            },
        }

    metrics.update({
        "model": "M2-2D exact plane-strain P-SV scattering by a traction-free circular cavity",
        "configuration": {
            "centre_m": REFLECTOR.tolist(), "radius_m": RADIUS_M,
            "material": {"density_kg_m3": MATERIAL.density_kg_m3, "c_p_m_s": MATERIAL.longitudinal_speed_m_s, "c_s_m_s": MATERIAL.shear_speed_m_s},
            "centre_frequency_hz": CENTRE_FREQUENCY_HZ, "sigma_s": SIGMA_S,
            "m1_boundary_points": M1_POINTS, "m2_m_max": M2_N_MAX,
            "array_elements": int(elements.shape[0]), "time_samples": int(time.size),
        },
        "m2_validation": {
            "maximum_normalized_traction_residual_over_active_fft_bins": m2_result.metadata["maximum_normalized_boundary_residual"],
            "maximum_condition_number_over_active_fft_bins": m2_result.metadata["maximum_condition_number"],
            "active_frequency_range_hz": m2_result.metadata["active_frequency_range_hz"],
            "active_frequency_bin_count": m2_result.metadata["active_frequency_bin_count"],
            "selected_m_max": M2_N_MAX,
            "maximum_paper_guide_m_max": m2_result.metadata["maximum_recommended_n_max_over_active_bins"],
            "fft_edge_to_global_peak_ratio": m2_result.metadata["edge_to_global_peak_ratio"],
            "maximum_fmc_reciprocity_error": float(np.max(np.abs(m2 - m2.swapaxes(0, 1)))),
        },
        "runtime_s": {"m2_fmc": m2_runtime, "four_tfm_images": tfm_runtime, "whole_analysis": clock.perf_counter() - start},
        "output_dimensions": {"fmc": list(m2.shape), "positive_frequency_fmc": list(m2_result.frequency_domain_ll.shape), "tfm": list(images["M2-2D elastic"].shape)},
        "representative_ascans": pair_metrics,
        "gated_arrival_maps": {name: {"central_peak_us": value["central_peak_us"], "peak_time_range_us": value["peak_time_range_us"]} for name, value in maps.items()},
        "tfm": {name: _image_metrics(image) for name, image in images.items()},
        "angle_binned_response": binned,
        "broadband_radius_sensitivity": _radius_broadband_study(time, elements[centre_index]),
        "experimental_interpretation": "Comparisons use separate normalization and a common 10–16.5 us gate. Absolute model-to-experiment amplitude is uncalibrated. The experimental ROI maximum is broad and is not treated as confirmed SDH ground truth.",
        "unsupported": ["general h!=0 3D T-matrix", "finite probe plane-wave spectrum", "Auld electromechanical receive model", "SH coupling", "P-to-SV measured trace", "free-surface/multiple scattering", "calibrated voltage amplitude"],
    })
    metrics["runtime_s"]["whole_analysis"] = clock.perf_counter() - start
    save_report(result_dir / "metrics.json", metrics)
    configuration = {
        "model": metrics["model"], "parameters": metrics["configuration"],
        "paper_convention": "exp(-i omega t), H_m^(1) outgoing",
        "fft_conversion": "conjugate paper F_PP for NumPy irfft positive frequencies",
        "propagation": "phase-only point-to-centre, H_tx=H_rx=1",
    }
    save_report(result_dir / "configuration.json", configuration)

    central = pair_metrics["Central pulse-echo"]
    m2_image = metrics["tfm"]["M2-2D elastic"]
    experimental_image = metrics["tfm"]["Experimental"]
    summary = f"""# M0/M1/M2-2D comparison

M2-2D is the exact plane-strain P--SV solution for a traction-free circular
cavity. Its first FMC uses only P-to-P scattering with phase-only propagation
and unit transmit/receive transfer functions. Absolute amplitudes are
uncalibrated.

## Validation

- Selected `m_max`: {M2_N_MAX}; maximum paper guide over active bins: {m2_result.metadata['maximum_recommended_n_max_over_active_bins']}.
- Maximum normalized dense-boundary traction residual: {m2_result.metadata['maximum_normalized_boundary_residual']:.3e}.
- Maximum FMC reciprocity error: {np.max(np.abs(m2-m2.swapaxes(0,1))):.3e}.
- FFT edge/global peak ratio: {m2_result.metadata['edge_to_global_peak_ratio']:.3e}.
- M2 FMC runtime: {m2_runtime:.3f} s; shape: {m2.shape}.

## Comparison

- Central envelope peaks: M0 {central['envelope_peak_us']['M0 point']:.2f} µs,
  M1 {central['envelope_peak_us']['M1 boundary']:.2f} µs, M2-2D
  {central['envelope_peak_us']['M2-2D elastic']:.2f} µs, and experiment
  {central['envelope_peak_us']['Experimental']:.2f} µs.
- Central envelope correlations with experiment: M0
  {central['envelope_correlation_with_experiment']['M0 point']:.3f}, M1
  {central['envelope_correlation_with_experiment']['M1 boundary']:.3f}, and
  M2-2D {central['envelope_correlation_with_experiment']['M2-2D elastic']:.3f}.
- M2 TFM peak: {m2_image['peak_m']} m; localization error: {m2_image['localisation_error_mm']:.3f} mm.
- M2 lateral/axial -6 dB widths: {m2_image['lateral_minus_6_db_width_mm']:.3f}/{m2_image['axial_minus_6_db_width_mm']:.3f} mm.
- Experimental TFM peak: {experimental_image['peak_m']} m. This broad ROI maximum is not confirmed SDH ground truth.

Radius changes both the magnitude and phase/angular structure of `F_PP`, so it
changes A-scan waveform and amplitude rather than merely moving an arrival.
See `metrics.json` and `figures/esm_2d/radius_sensitivity.png` for the
single-frequency study and the broadband central-element table.

## Scope

This is not the complete Boström--Bövik finite-probe 3D measurement model. It
does not include the general `h != 0` T-matrix, finite aperture, Auld reception,
SH coupling, a P-to-SV receive path, surface multiple scattering, or calibrated
probe voltage.
"""
    (result_dir / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
