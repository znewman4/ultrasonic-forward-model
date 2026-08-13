"""Validate and analyse the complete M2-2D P--SV modal pipeline."""
from __future__ import annotations

import json
import time as clock
from pathlib import Path

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
from src.imaging.tfm import tfm_image, tfm_image_mode_pair
from src.models.elastic_sdh import (
    ElasticSideDrilledHole,
    simulate_elastic_sdh_modal_fmc,
)
from src.scattering.elastic_sdh import (
    ElasticMaterial,
    elastic_sdh_modal_scattering,
    normalized_modal_reciprocity_error,
    recommended_n_max,
)


MATERIAL = ElasticMaterial(2700.0, WAVE_SPEED_M_S, 3100.0)
RADIUS_M = 0.5e-3
N_MAX = 26
DT_S = 20.0e-9
TIME_S = np.arange(1600) * DT_S  # 0--31.98 us; contains the full SS pulse.
MODES = ("PP", "PS", "SP", "SS")


def _normalise(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values))
    return values / scale if scale else np.zeros_like(values)


def _image_metrics(image: np.ndarray) -> dict[str, float | list[float]]:
    iz, ix = np.unravel_index(np.argmax(image), image.shape)
    x = float(X_GRID_M[ix]); z = float(Z_GRID_M[iz])
    return {
        "peak_m": [x, z],
        "localisation_error_mm": float(np.hypot(x - REFLECTOR[0], z - REFLECTOR[1]) * 1e3),
        "normalized_peak": 1.0,
    }


def _folders() -> dict[str, Path]:
    base = ROOT / "results" / "comparisons" / "M2_multimode"
    folders = {
        "base": base,
        "scattering": base / "scattering",
        "fmc": base / "fmc",
        "tfm": base / "tfm",
        "validation": base / "validation",
        "radius": base / "radius_sweep",
        "data": ROOT / "data" / "synthetic" / "elastic_sdh" / "multimode",
    }
    for folder in folders.values(): folder.mkdir(parents=True, exist_ok=True)
    return folders


def _scattering_analysis(folders: dict[str, Path], metrics: dict) -> None:
    beta = np.linspace(-np.pi, np.pi, 721)
    result = elastic_sdh_modal_scattering(
        CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.0, beta, N_MAX
    )
    labels = ((0, 0, "PP"), (1, 0, "PS"), (0, 1, "SP"), (1, 1, "SS"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for outgoing, incident, label in labels:
        values = result.flux_far_field_sqrt_m[outgoing, incident]
        axes[0, 0].plot(np.rad2deg(beta), np.abs(values), label=label)
        axes[1, 0].plot(np.rad2deg(beta), np.unwrap(np.angle(values)), label=label)
        fractions = result.angular_scattered_power_fraction[outgoing, incident]
        axes[0, 1].plot(np.rad2deg(beta), fractions, label=label)
        axes[1, 1].plot(
            np.rad2deg(beta), result.differential_cross_section_m[outgoing, incident] * 1e3,
            label=label,
        )
    axes[0, 0].set(title="Flux-normalized modal magnitudes", ylabel=r"$|F^{flux}|$ ($\sqrt{m}$)")
    axes[1, 0].set(xlabel="Scattering angle β (degrees)", ylabel="Unwrapped phase (rad)")
    axes[0, 1].set(title="Outgoing fraction of scattered power", ylabel="Fraction")
    axes[1, 1].set(xlabel="Scattering angle β (degrees)", ylabel=r"Differential cross-section (mm)")
    for ax in axes.ravel(): ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(folders["scattering"] / "full_modal_scattering_matrix.png", dpi=180); plt.close(fig)

    # A separate raw/normalized view preserves debugging visibility.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for outgoing, incident, label in labels:
        axes[0].plot(np.rad2deg(beta), np.abs(result.raw_far_field_potential[outgoing, incident]), label=label)
        axes[1].plot(np.rad2deg(beta), np.abs(result.flux_far_field_sqrt_m[outgoing, incident]), label=label)
    axes[0].set(title="Raw potential far fields", xlabel="β (degrees)", ylabel="|F raw|")
    axes[1].set(title="Energy-flux normalization", xlabel="β (degrees)", ylabel=r"$|F^{flux}|$ ($\sqrt{m}$)")
    for ax in axes: ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(folders["scattering"] / "raw_vs_flux_normalization.png", dpi=180); plt.close(fig)

    # Frequency-angle fingerprints show resonant phase and amplitude structure,
    # which a single-frequency polar pattern cannot reveal.
    frequency_grid = np.linspace(1.0e6, 10.0e6, 73)
    beta_grid = np.linspace(-np.pi, np.pi, 181)
    frequency_angle = np.empty((frequency_grid.size, 2, 2, beta_grid.size), dtype=complex)
    for frequency_index, frequency in enumerate(frequency_grid):
        selected_n_max = max(N_MAX, recommended_n_max(float(frequency), RADIUS_M, MATERIAL) + 8)
        frequency_angle[frequency_index] = elastic_sdh_modal_scattering(
            float(frequency), RADIUS_M, MATERIAL, 0.0, beta_grid, selected_n_max
        ).flux_far_field_sqrt_m
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for column, (outgoing, incident, label) in enumerate(labels):
        values = frequency_angle[:, outgoing, incident]
        magnitude = axes[0, column].pcolormesh(
            np.rad2deg(beta_grid), frequency_grid * 1e-6, np.abs(values), shading="auto"
        )
        phase = axes[1, column].pcolormesh(
            np.rad2deg(beta_grid), frequency_grid * 1e-6, np.angle(values),
            shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi,
        )
        axes[0, column].set_title(f"|F_{label}| (√m)")
        axes[1, column].set_title(f"phase(F_{label}) (rad)")
        axes[1, column].set_xlabel("Scattering angle β (degrees)")
        fig.colorbar(magnitude, ax=axes[0, column], shrink=0.8)
        fig.colorbar(phase, ax=axes[1, column], shrink=0.8)
    axes[0, 0].set_ylabel("Frequency (MHz)"); axes[1, 0].set_ylabel("Frequency (MHz)")
    fig.suptitle("Complete modal scattering versus angle and frequency")
    fig.tight_layout(); fig.savefig(folders["scattering"] / "modal_frequency_angle_maps.png", dpi=180); plt.close(fig)

    # Individual azimuthal orders at backscatter expose which partial waves
    # build the complete response and make the truncation choice auditable.
    backscatter_phase = np.exp(1j * result.orders * (np.pi - np.pi / 2.0))
    incident_k = np.array([
        2.0 * np.pi * CENTRE_FREQUENCY_HZ / MATERIAL.longitudinal_speed_m_s,
        2.0 * np.pi * CENTRE_FREQUENCY_HZ / MATERIAL.shear_speed_m_s,
    ])
    column_scale = np.sqrt(2.0 / (np.pi * incident_k))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    harmonic_metrics = {}
    for ax, (outgoing, incident, label) in zip(axes.ravel(), labels, strict=True):
        contribution = (
            result.plane_wave_scattered_coefficients[:, outgoing, incident]
            * backscatter_phase * column_scale[incident]
        )
        ax.stem(result.orders, np.abs(contribution), basefmt=" ")
        ax.set(title=f"{label} at β=180°", ylabel="Partial |F| (√m)")
        ax.grid(True, alpha=0.25)
        dominant = int(np.argmax(np.abs(contribution)))
        harmonic_metrics[label] = {
            "dominant_order": int(result.orders[dominant]),
            "dominant_magnitude_sqrt_m": float(np.abs(contribution[dominant])),
            "orders": result.orders.tolist(),
            "magnitudes_sqrt_m": np.abs(contribution).tolist(),
        }
    axes[-1, 0].set_xlabel("Harmonic order m"); axes[-1, 1].set_xlabel("Harmonic order m")
    fig.suptitle("Harmonic-order contributions to backscatter")
    fig.tight_layout(); fig.savefig(folders["scattering"] / "harmonic_order_contributions.png", dpi=180); plt.close(fig)

    metrics["frequency_angle_scattering"] = {
        "frequency_range_mhz": [float(frequency_grid[0] * 1e-6), float(frequency_grid[-1] * 1e-6)],
        "frequency_samples": int(frequency_grid.size),
        "angle_samples": int(beta_grid.size),
    }
    metrics["harmonic_order_contributions"] = harmonic_metrics

    # Reversed-ray reciprocity is checked over a useful angular subset.
    reciprocity_errors = []
    for outgoing_angle in np.linspace(-np.pi, np.pi, 37, endpoint=False):
        forward = elastic_sdh_modal_scattering(
            CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.0, np.array([outgoing_angle]), N_MAX
        )
        reversed_ray = elastic_sdh_modal_scattering(
            CENTRE_FREQUENCY_HZ,
            RADIUS_M,
            MATERIAL,
            outgoing_angle + np.pi,
            np.array([np.pi]),
            N_MAX,
        )
        reciprocity_errors.append(normalized_modal_reciprocity_error(forward, reversed_ray, MATERIAL))
    rotation = 0.731
    rotated = elastic_sdh_modal_scattering(
        CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, rotation, beta + rotation, N_MAX
    )
    rotational_symmetry_error = float(
        np.linalg.norm(rotated.flux_far_field_sqrt_m - result.flux_far_field_sqrt_m)
        / np.linalg.norm(result.flux_far_field_sqrt_m)
    )
    metrics["single_frequency_scattering"] = {
        "frequency_hz": CENTRE_FREQUENCY_HZ,
        "radius_m": RADIUS_M,
        "m_max": N_MAX,
        "traction_residual": result.validation_metadata["normalized_boundary_traction_residual"],
        "maximum_partial_wave_reciprocity_error": result.validation_metadata["maximum_partial_wave_reciprocity_error"],
        "maximum_reversed_ray_far_field_reciprocity_error": float(max(reciprocity_errors)),
        "rotational_symmetry_error": rotational_symmetry_error,
        "maximum_partial_wave_energy_balance_error": result.validation_metadata["maximum_partial_wave_energy_balance_error"],
        "maximum_condition_number": result.validation_metadata["maximum_condition_number"],
    }

    n_values = np.arange(8, 29, 2)
    reference = elastic_sdh_modal_scattering(
        CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.0, beta, 30
    )
    traction_p = []; traction_sv = []; series_error = []; energy_error = []; reciprocity = []
    for n_max in n_values:
        current = elastic_sdh_modal_scattering(
            CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.0, beta, int(n_max)
        )
        residual = current.validation_metadata["normalized_boundary_traction_residual"]
        traction_p.append(residual["P"]); traction_sv.append(residual["SV"])
        series_error.append(np.linalg.norm(current.raw_far_field_potential-reference.raw_far_field_potential)/np.linalg.norm(reference.raw_far_field_potential))
        energy_error.append(current.validation_metadata["maximum_partial_wave_energy_balance_error"])
        reciprocity.append(current.validation_metadata["maximum_partial_wave_reciprocity_error"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(n_values, traction_p, "o-", label="P incident traction residual")
    ax.semilogy(n_values, traction_sv, "s-", label="SV incident traction residual")
    ax.semilogy(n_values, series_error, "^-", label="Full far-field error vs m_max=30")
    ax.semilogy(n_values, energy_error, ".-", label="Partial-wave energy error")
    ax.semilogy(n_values, reciprocity, "x-", label="Partial-wave reciprocity error")
    ax.axvline(recommended_n_max(CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL), color="black", ls="--", label="paper guide")
    ax.axvline(N_MAX, color="grey", ls=":", label="production m_max")
    ax.set(xlabel="m_max", ylabel="Relative error", title="Complete modal convergence and validation")
    ax.grid(True, alpha=0.25); ax.legend(fontsize="small"); fig.tight_layout()
    fig.savefig(folders["validation"] / "convergence_traction_reciprocity_energy.png", dpi=180); plt.close(fig)
    metrics["convergence"] = {
        "m_max": n_values.tolist(), "traction_P": traction_p, "traction_SV": traction_sv,
        "far_field_error": series_error, "energy_error": energy_error, "reciprocity_error": reciprocity,
        "reference_m_max": 30, "paper_guide_at_5MHz": recommended_n_max(CENTRE_FREQUENCY_HZ, RADIUS_M, MATERIAL),
    }


def _modal_fmc_and_tfm(folders: dict[str, Path], metrics: dict) -> None:
    loaded = load_fmc(MAT_PATH)
    elements = array_coordinates(MAT_PATH)
    defect = ElasticSideDrilledHole(*REFLECTOR, RADIUS_M)
    start = clock.perf_counter()
    modal = simulate_elastic_sdh_modal_fmc(
        TIME_S, elements, defect, MATERIAL, CENTRE_FREQUENCY_HZ, SIGMA_S,
        N_MAX, spectrum_relative_cutoff=1e-8,
    )
    fmc_runtime = clock.perf_counter() - start
    fmcs = {mode: getattr(modal, f"fmc_{mode.lower()}") for mode in MODES}
    spectra = {mode: getattr(modal, f"frequency_domain_{mode.lower()}") for mode in MODES}
    delays = {mode: getattr(modal, f"travel_time_{mode.lower()}_s") for mode in MODES}

    data_path = folders["data"] / "M2_2D_multimode_fmc.npz"
    np.savez_compressed(
        data_path,
        fmc_pp=fmcs["PP"], fmc_ps=fmcs["PS"], fmc_sp=fmcs["SP"], fmc_ss=fmcs["SS"],
        spectrum_pp=spectra["PP"], spectrum_ps=spectra["PS"], spectrum_sp=spectra["SP"], spectrum_ss=spectra["SS"],
        time_s=TIME_S, frequency_hz=modal.frequency_hz, element_coordinates_m=elements,
        centre_m=REFLECTOR, radius_m=np.array(RADIUS_M),
        travel_time_pp_s=delays["PP"], travel_time_ps_s=delays["PS"], travel_time_sp_s=delays["SP"], travel_time_ss_s=delays["SS"],
        density_kg_m3=np.array(MATERIAL.density_kg_m3), c_p_m_s=np.array(MATERIAL.longitudinal_speed_m_s), c_s_m_s=np.array(MATERIAL.shear_speed_m_s),
        model_parameters_json=np.array(json.dumps({key:value for key,value in modal.metadata.items() if not isinstance(value,np.ndarray)})),
    )

    centre = int(np.argmin(np.abs(elements[:, 0])))
    rms_maps = {mode: np.sqrt(np.mean(fmcs[mode]**2, axis=2)) for mode in MODES}
    representative_pairs = {"PP": (centre, centre), "SS": (centre, centre)}
    for mode in ("PS", "SP"):
        representative_pairs[mode] = tuple(
            int(index) for index in np.unravel_index(np.argmax(rms_maps[mode]), rms_maps[mode].shape)
        )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    arrival_metrics = {}
    for ax, mode in zip(axes.ravel(), MODES, strict=True):
        transmitter, receiver = representative_pairs[mode]
        trace = fmcs[mode][transmitter, receiver]
        envelope = np.abs(hilbert(trace))
        geometric = float(delays[mode][transmitter, receiver])
        peak = float(TIME_S[np.argmax(envelope)])
        arrival_metrics[mode] = {
            "representative_pair_indices": [transmitter, receiver],
            "geometric_us": geometric*1e6,
            "envelope_peak_us": peak*1e6,
        }
        ax.plot(TIME_S*1e6, trace)
        ax.axvline(geometric*1e6, color="black", ls="--", label="centre-path delay")
        ax.set(title=f"{mode} A-scan, pair ({transmitter},{receiver})", ylabel="Flux-normalized wavefield")
        ax.grid(True, alpha=0.25); ax.legend()
    axes[-1,0].set_xlabel("Time (µs)"); axes[-1,1].set_xlabel("Time (µs)")
    fig.tight_layout(); fig.savefig(folders["fmc"] / "representative_modal_ascans.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    rms_metrics = {}
    for ax, mode in zip(axes.ravel(), MODES, strict=True):
        rms = rms_maps[mode]
        rms_metrics[mode] = {"minimum": float(np.min(rms)), "maximum": float(np.max(rms)), "mean": float(np.mean(rms))}
        im = ax.imshow(rms, origin="lower", aspect="auto", cmap="viridis")
        ax.set(title=f"{mode} FMC RMS", xlabel="Receiver", ylabel="Transmitter")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(folders["fmc"] / "modal_fmc_rms_maps.png", dpi=180); plt.close(fig)

    start = clock.perf_counter()
    speed_pairs = {
        "PP": (MATERIAL.longitudinal_speed_m_s, MATERIAL.longitudinal_speed_m_s),
        "PS": (MATERIAL.longitudinal_speed_m_s, MATERIAL.shear_speed_m_s),
        "SP": (MATERIAL.shear_speed_m_s, MATERIAL.longitudinal_speed_m_s),
        "SS": (MATERIAL.shear_speed_m_s, MATERIAL.shear_speed_m_s),
    }
    images = {
        mode: tfm_image_mode_pair(fmcs[mode], TIME_S, elements, X_GRID_M, Z_GRID_M, *speed_pairs[mode])
        for mode in MODES
    }
    wrong_ps = tfm_image(fmcs["PS"], TIME_S, elements, X_GRID_M, Z_GRID_M, MATERIAL.longitudinal_speed_m_s)
    tfm_runtime = clock.perf_counter() - start
    tfm_metrics = {mode: _image_metrics(image) for mode,image in images.items()}
    tfm_metrics["PS_imaged_with_PP_delays"] = _image_metrics(wrong_ps)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, mode in zip(axes.ravel(), MODES, strict=True):
        ax.imshow(_normalise(images[mode]), extent=[X_GRID_M[0]*1e3,X_GRID_M[-1]*1e3,Z_GRID_M[-1]*1e3,Z_GRID_M[0]*1e3], aspect="auto", cmap="inferno", vmin=0,vmax=1)
        ax.scatter([0],[40],marker="x",color="cyan"); ax.set(title=f"TFM {mode}", xlabel="x (mm)", ylabel="z (mm)")
    fig.tight_layout(); fig.savefig(folders["tfm"] / "correct_modal_tfm.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax,image,title in ((axes[0],images["PS"],"PS data, correct PS delays"),(axes[1],wrong_ps,"PS data, wrong PP delays")):
        ax.imshow(_normalise(image), extent=[X_GRID_M[0]*1e3,X_GRID_M[-1]*1e3,Z_GRID_M[-1]*1e3,Z_GRID_M[0]*1e3], aspect="auto", cmap="inferno",vmin=0,vmax=1)
        ax.scatter([0],[40],marker="x",color="cyan"); ax.set(title=title,xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)"); fig.tight_layout(); fig.savefig(folders["tfm"] / "ps_correct_vs_wrong_pp_delays.png", dpi=180); plt.close(fig)

    # PP-only experimental comparison remains on the acquisition's shorter record.
    experimental = experimental_fmc_array(loaded)
    pp_short = fmcs["PP"][:, :, : loaded.metadata.time_s.size]
    central_exp = experimental[centre, centre]
    central_pp = pp_short[centre, centre]
    gate = (loaded.metadata.time_s >= 10e-6) & (loaded.metadata.time_s <= 16.5e-6)
    pp_env = np.abs(hilbert(central_pp[gate])); exp_env=np.abs(hilbert(central_exp[gate]))
    pp_experimental = {
        "synthetic_pp_peak_us": float(loaded.metadata.time_s[gate][np.argmax(pp_env)]*1e6),
        "experimental_peak_us": float(loaded.metadata.time_s[gate][np.argmax(exp_env)]*1e6),
        "envelope_correlation": float(np.vdot(pp_env,exp_env).real/(np.linalg.norm(pp_env)*np.linalg.norm(exp_env))),
        "limitation": "Experimental duration is ~20 us; PS/SS echoes are truncated or absent, so only PP is compared.",
    }
    metrics.update({
        "modal_fmc": {
            "shape": list(fmcs["PP"].shape), "time_range_us": [float(TIME_S[0]*1e6),float(TIME_S[-1]*1e6)],
            "arrival_times": arrival_metrics,
            "arrival_time_differences_from_PP_us": {
                mode: {
                    "geometric": arrival_metrics[mode]["geometric_us"] - arrival_metrics["PP"]["geometric_us"],
                    "envelope_peak": arrival_metrics[mode]["envelope_peak_us"] - arrival_metrics["PP"]["envelope_peak_us"],
                }
                for mode in ("PS", "SP", "SS")
            },
            "rms": rms_metrics,
            "validation": {key:value for key,value in modal.metadata.items() if key in ("n_max","maximum_recommended_n_max_over_active_bins","maximum_normalized_boundary_residual","maximum_partial_wave_reciprocity_error","maximum_partial_wave_energy_balance_error","maximum_condition_number","fft_edge_to_global_peak_ratio")},
        },
        "tfm": tfm_metrics,
        "experimental_pp_only": pp_experimental,
        "runtime_s": {"modal_fmc": fmc_runtime, "five_tfm_images": tfm_runtime},
        "dataset": str(data_path.relative_to(ROOT)),
    })


def _radius_sweep(folders: dict[str, Path], metrics: dict) -> None:
    beta = np.linspace(-np.pi, np.pi, 361)
    radii = np.array([0.1,0.25,0.5,0.75,1.0])*1e-3
    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True)
    labels=((0,0,"PP"),(1,0,"PS"),(0,1,"SP"),(1,1,"SS"))
    values={}
    for radius in radii:
        n_max=max(N_MAX,recommended_n_max(CENTRE_FREQUENCY_HZ,float(radius),MATERIAL)+8)
        result=elastic_sdh_modal_scattering(CENTRE_FREQUENCY_HZ,float(radius),MATERIAL,0,beta,n_max)
        entry={}
        for ax,(outgoing,incident,label) in zip(axes.ravel(),labels,strict=True):
            amplitude=np.abs(result.flux_far_field_sqrt_m[outgoing,incident])
            ax.plot(np.rad2deg(beta),amplitude,label=f"a={radius*1e3:g} mm")
            entry[label]={"maximum_flux_amplitude_sqrt_m":float(np.max(amplitude)),"total_cross_section_m":float(np.trapezoid(result.differential_cross_section_m[outgoing,incident],beta))}
        values[f"{radius*1e3:g}_mm"]={"radius_m":float(radius),"m_max":n_max,"modes":entry}
    for ax,(_,_,label) in zip(axes.ravel(),labels,strict=True):
        ax.set(title=label,xlabel="β (degrees)",ylabel=r"$|F^{flux}|$ ($\sqrt{m}$)");ax.grid(True,alpha=.25);ax.legend(fontsize="small")
    fig.suptitle("Modal scattering fingerprint versus SDH radius");fig.tight_layout();fig.savefig(folders["radius"] / "modal_radius_sweep.png",dpi=180);plt.close(fig)
    metrics["radius_sweep"]=values


def main() -> None:
    start=clock.perf_counter(); folders=_folders(); metrics={
        "model":"M2-2D complete ideal P-SV elastic wavefield",
        "configuration":{"radius_m":RADIUS_M,"centre_m":REFLECTOR.tolist(),"frequency_hz":CENTRE_FREQUENCY_HZ,"m_max":N_MAX,"dt_s":DT_S,"time_samples":int(TIME_S.size),"material":{"density_kg_m3":MATERIAL.density_kg_m3,"c_p_m_s":MATERIAL.longitudinal_speed_m_s,"c_s_m_s":MATERIAL.shear_speed_m_s}},
        "assumptions":["infinite homogeneous isotropic 2D plane-strain medium","traction-free circular cavity","phase-only centre-path propagation","H_tx=H_rx=1 for each separate mode","no combined voltage FMC","no probe directivity/couplant/piezoelectric or receive-polarization model"],
    }
    _scattering_analysis(folders,metrics);_modal_fmc_and_tfm(folders,metrics);_radius_sweep(folders,metrics)
    metrics["runtime_s"]["whole_analysis"]=clock.perf_counter()-start
    save_report(folders["base"] / "metrics.json",metrics)
    validation=metrics["modal_fmc"]["validation"];tfm=metrics["tfm"]
    summary=f"""# M2-2D complete P--SV modal pipeline

The complete idealized plane-strain matrix `[[F_PP,F_SP],[F_PS,F_SS]]` is
implemented in raw potential and energy-flux normalizations. Four modal FMCs
and four correctly focused TFM images remain separate; `H_tx=H_rx=1` and no
probe-voltage interpretation is made.

## Validation

- Production `m_max`: {N_MAX}; maximum paper guide over the pulse band: {validation['maximum_recommended_n_max_over_active_bins']}.
- Broadband traction residuals: {validation['maximum_normalized_boundary_residual']}.
- Maximum partial-wave reciprocity error: {validation['maximum_partial_wave_reciprocity_error']:.3e}.
- Maximum partial-wave energy-balance error: {validation['maximum_partial_wave_energy_balance_error']:.3e}.
- FFT edge/global peak ratios: {validation['fft_edge_to_global_peak_ratio']}.

## Modal propagation and imaging

- FMC shape per mode: {metrics['modal_fmc']['shape']}; time record: 0--{TIME_S[-1]*1e6:.2f} us.
- Representative geometric/peak arrival times: {metrics['modal_fmc']['arrival_times']}.
- Correct-delay TFM localization errors (mm): PP {tfm['PP']['localisation_error_mm']:.3f}, PS {tfm['PS']['localisation_error_mm']:.3f}, SP {tfm['SP']['localisation_error_mm']:.3f}, SS {tfm['SS']['localisation_error_mm']:.3f}.
- PS data focused with PP delays has localization error {tfm['PS_imaged_with_PP_delays']['localisation_error_mm']:.3f} mm.

Only PP is compared with the ~20 us experiment. Converted/shear experimental
validation is not claimed because those echoes reach or exceed the record end.
Full numerical tables are in `metrics.json`.
"""
    (folders["base"] / "summary.md").write_text(summary,encoding="utf-8")


if __name__ == "__main__": main()
