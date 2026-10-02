"""Compare the M0 physics ladder: timing -> spreading/attenuation -> FD -> directivity.

Each rung adds one effect to the previous rung (see
``src/models/point_reflector_physics.py`` and ``docs/propagation_physics.md``).
For every rung this script reports what changes in the FMC (amplitude across the
aperture, waveform, spectrum) and in the TFM image, and whether the change moves
the gated amplitude pattern closer to the experiment.
"""
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
from src.models.point_reflector_physics import (
    LadderStep,
    PropagationPhysics,
    pair_transfer,
    physics_ladder,
    simulate_ladder_step,
    simulate_point_reflector_fmc_fd,
    simulate_point_reflector_fmc_td,
)
from src.propagation import element_directivity, element_ray_geometry


# Illustrative low-loss aluminium value at 5 MHz (0.01 dB/mm). Uncalibrated:
# it should be replaced by a value measured on this specimen.
ATTENUATION_DB_PER_M = 10.0
ATTENUATION_NP_PER_M = ATTENUATION_DB_PER_M / (20.0 / np.log(10.0))
ATTENUATION_EXPONENT = 1.0
SWEEP_DB_PER_M = (0.0, 10.0, 50.0, 200.0, 1000.0)
GATE_S = (10.0e-6, 16.5e-6)  # same gate as scripts/07_compare_m0_m1_m2.py
LEVEL_6DB = 10.0 ** (-6.0 / 20.0)


def _element_width_m() -> float:
    """Measured element width from the MAT-file element corner coordinates."""
    metadata = load_fmc(MAT_PATH).metadata.array
    corners = np.asarray(metadata.element_corners_m, dtype=float)
    centres = np.asarray(metadata.element_centres_m, dtype=float)
    return float(np.median(2.0 * np.abs(corners[:, 0, 0] - centres[:, 0])))


def _normalise(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values))
    return values / scale if scale else np.zeros_like(values)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.ravel(first); b = np.ravel(second)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _db(ratio: float) -> float:
    return float(20.0 * np.log10(max(ratio, np.finfo(float).tiny)))


def _centroid_hz(trace: np.ndarray, dt: float) -> float:
    power = np.abs(np.fft.rfft(trace)) ** 2
    frequency = np.fft.rfftfreq(trace.size, dt)
    return float(np.sum(frequency * power) / np.sum(power))


def _width_mm(coordinate: np.ndarray, values: np.ndarray) -> float | None:
    """Interpolated -6 dB main-lobe width in mm."""
    magnitude = np.abs(values)
    peak = int(np.argmax(magnitude))
    threshold = LEVEL_6DB * magnitude[peak]
    left = np.flatnonzero(magnitude[:peak] < threshold)
    right = np.flatnonzero(magnitude[peak + 1 :] < threshold)
    if left.size == 0 or right.size == 0:
        return None
    il = int(left[-1]); ir = int(peak + 1 + right[0])
    xl = np.interp(threshold, magnitude[il : il + 2], coordinate[il : il + 2])
    xr = np.interp(threshold, magnitude[ir - 1 : ir + 1][::-1], coordinate[ir - 1 : ir + 1][::-1])
    return float((xr - xl) * 1e3)


def _image_metrics(image: np.ndarray) -> dict[str, object]:
    iz, ix = np.unravel_index(np.argmax(image), image.shape)
    x = float(X_GRID_M[ix]); z = float(Z_GRID_M[iz])
    return {
        "peak_m": [x, z],
        "localisation_error_mm": float(np.hypot(x - REFLECTOR[0], z - REFLECTOR[1]) * 1e3),
        "lateral_minus_6_db_width_mm": _width_mm(X_GRID_M, image[iz]),
        "axial_minus_6_db_width_mm": _width_mm(Z_GRID_M, image[:, ix]),
    }


def _gated_amplitude(time: np.ndarray, fmc: np.ndarray) -> np.ndarray:
    """Peak Hilbert envelope per pair inside the common arrival gate."""
    gate = (time >= GATE_S[0]) & (time <= GATE_S[1])
    return np.max(np.abs(hilbert(fmc[:, :, gate], axis=2)), axis=2)


def _plot_factors(path, elements, width_m) -> None:
    _, sine = element_ray_geometry(elements, REFLECTOR)
    max_angle = float(np.rad2deg(np.max(np.abs(np.arcsin(sine)))))
    theta = np.linspace(-90.0, 90.0, 721)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for frequency in (2.5e6, 5.0e6, 7.5e6, 10.0e6):
        axes[0].plot(
            theta,
            element_directivity(np.sin(np.deg2rad(theta)), frequency, width_m, WAVE_SPEED_M_S),
            label=f"{frequency / 1e6:.1f} MHz",
        )
    axes[0].axvspan(-max_angle, max_angle, color="grey", alpha=0.15, label="angles used at 40 mm")
    axes[0].set(
        xlabel="Angle from element normal θ (deg)", ylabel="p(θ, f)",
        title=f"Element directivity, a = {width_m * 1e3:.2f} mm (Holmes eq. 5)",
    )
    frequency = np.linspace(0.5e6, 10.0e6, 200)
    path_m = 2.0 * REFLECTOR[1]
    for db_per_m in SWEEP_DB_PER_M[1:]:
        alpha = db_per_m / (20.0 / np.log(10.0)) * frequency / CENTRE_FREQUENCY_HZ
        axes[1].plot(frequency / 1e6, np.exp(-alpha * path_m), label=f"{db_per_m:g} dB/m at 5 MHz")
    axes[1].set(
        xlabel="Frequency (MHz)", ylabel="exp(-α(f)·80 mm)",
        title="Frequency-dependent attenuation, n = 1, 80 mm path",
    )
    for ax in axes:
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_amplitude_maps(path, maps: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(maps), figsize=(3.2 * len(maps), 3.4))
    for ax, (name, values) in zip(axes, maps.items(), strict=True):
        im = ax.imshow(20.0 * np.log10(_normalise(values) + 1e-12), vmin=-12, vmax=0, cmap="viridis")
        ax.set(title=name, xlabel="Receiver", ylabel="Transmitter")
    fig.colorbar(im, ax=axes, label="Gated peak envelope (dB re max)", shrink=0.8)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_ascans(path, time, fmcs: dict[str, np.ndarray], pairs) -> None:
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 3.2 * len(pairs)))
    gate = (time >= GATE_S[0]) & (time <= GATE_S[1])
    for ax, (label, (i, j)) in zip(axes, pairs.items(), strict=True):
        for name, fmc in fmcs.items():
            ax.plot(time[gate] * 1e6, fmc[i, j, gate] / np.max(np.abs(fmc[i, j])), label=name)
        ax.set(title=f"{label}: tx {i}, rx {j} (each trace normalised to its own peak)",
               xlabel="Time (µs)", ylabel="Amplitude")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    for ax, (i, j) in zip(axes, pairs.values(), strict=True):
        peak = time[np.argmax(np.abs(next(iter(fmcs.values()))[i, j]))]
        ax.set_xlim((peak - 1.5e-6) * 1e6, (peak + 1.5e-6) * 1e6)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_spectra(path, time, fmcs: dict[str, np.ndarray], pair) -> None:
    i, j = pair
    frequency = np.fft.rfftfreq(time.size, time[1] - time[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, fmc in fmcs.items():
        spectrum = np.abs(np.fft.rfft(fmc[i, j]))
        ax.plot(frequency / 1e6, spectrum / np.max(spectrum), label=name)
    ax.set(xlim=(0, 10), xlabel="Frequency (MHz)", ylabel="|S(f)| / max",
           title=f"Edge pulse-echo spectrum (tx {i}, rx {j}), each normalised")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_tfm(path, images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(3.4 * len(images), 3.6), sharey=True)
    extent = [X_GRID_M[0] * 1e3, X_GRID_M[-1] * 1e3, Z_GRID_M[-1] * 1e3, Z_GRID_M[0] * 1e3]
    for ax, (name, image) in zip(axes, images.items(), strict=True):
        im = ax.imshow(20 * np.log10(_normalise(image) + 1e-12), extent=extent, vmin=-30, vmax=0,
                       cmap="inferno", aspect="auto")
        ax.set(title=name, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, label="dB re own max", shrink=0.8)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_profiles(path, images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, image in images.items():
        iz, ix = np.unravel_index(np.argmax(image), image.shape)
        axes[0].plot(X_GRID_M * 1e3, 20 * np.log10(_normalise(image[iz]) + 1e-12), label=name)
        axes[1].plot(Z_GRID_M * 1e3, 20 * np.log10(_normalise(image[:, ix]) + 1e-12), label=name)
    axes[0].set(xlabel="x (mm)", ylabel="dB", title="Lateral profile through peak", ylim=(-40, 1))
    axes[1].set(xlabel="z (mm)", ylabel="dB", title="Axial profile through peak", ylim=(-40, 1), xlim=(38, 42))
    for ax in axes:
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_sweep(path, sweep: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = sweep["attenuation_db_per_m_at_5mhz"]
    axes[0].semilogx(x[1:], sweep["central_centroid_shift_khz"][1:], "o-")
    axes[0].set(xlabel="α at 5 MHz (dB/m)", ylabel="Centroid shift (kHz)", title="Central pulse-echo downshift")
    axes[1].semilogx(x[1:], sweep["edge_to_centre_amplitude_change_db"][1:], "o-")
    axes[1].set(xlabel="α at 5 MHz (dB/m)", ylabel="Δ edge/centre ratio (dB)", title="Change in aperture amplitude taper")
    axes[2].semilogx(x[1:], sweep["tfm_lateral_width_mm"][1:], "o-")
    axes[2].axhline(sweep["tfm_lateral_width_mm"][0], color="black", ls="--", label="α = 0")
    axes[2].set(xlabel="α at 5 MHz (dB/m)", ylabel="mm", title="TFM lateral −6 dB width"); axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    result_dir = ROOT / "results" / "comparisons" / "M0_physics_ladder"
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_fmc(MAT_PATH)
    time = np.asarray(loaded.metadata.time_s, dtype=float)
    dt = float(time[1] - time[0])
    elements = array_coordinates(MAT_PATH)
    width_m = _element_width_m()
    n = elements.shape[0]
    centre = int(np.argmin(np.abs(elements[:, 0])))
    pairs = {"Central pulse-echo": (centre, centre), "Edge pulse-echo": (0, 0), "Edge pitch-catch": (0, n - 1)}

    ladder = physics_ladder(width_m, ATTENUATION_NP_PER_M, ATTENUATION_EXPONENT)
    # Diagnostic rung (not in the ladder): directivity evaluated only at 5 MHz in
    # the time domain. Comparing it with M0-FD+D isolates the frequency
    # dependence of the directivity from its mean amplitude effect.
    directivity_f0 = LadderStep(
        "M0-TD+D(f0)", "time",
        PropagationPhysics(
            geometric_spreading=True, attenuation_np_per_m=ATTENUATION_NP_PER_M,
            element_width_m=width_m, directivity_frequency_dependent=False,
        ),
        "M0-TD+ with directivity frozen at the centre frequency",
    )

    fmcs: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}
    for step in (*ladder, directivity_f0):
        start = clock.perf_counter()
        fmcs[step.label] = simulate_ladder_step(
            step, time, elements, REFLECTOR, WAVE_SPEED_M_S, CENTRE_FREQUENCY_HZ, SIGMA_S
        )
        runtimes[step.label] = clock.perf_counter() - start

    images = {
        name: tfm_image(fmc, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)
        for name, fmc in fmcs.items()
    }
    experimental = experimental_fmc_array(loaded)
    images["Experiment"] = tfm_image(experimental, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)
    amplitude_maps = {name: _gated_amplitude(time, fmc) for name, fmc in fmcs.items()}
    amplitude_maps["Experiment"] = _gated_amplitude(time, experimental)

    # --- Per-rung metrics --------------------------------------------------
    steps: dict[str, dict[str, object]] = {}
    previous = None
    exp_map = _normalise(amplitude_maps["Experiment"])
    # Exclude dead/weak experimental channels (visible as stripes in the gated
    # amplitude map): an element is live if its median gated amplitude, as
    # transmitter, is at least 25 % of the array-wide median.
    channel_level = np.median(exp_map, axis=1)
    live = channel_level >= 0.25 * np.median(channel_level)
    live_pairs = live[:, np.newaxis] & live[np.newaxis, :]
    for step in (*ladder, directivity_f0):
        name = step.label
        fmc = fmcs[name]
        amp = amplitude_maps[name]
        image = images[name]
        record: dict[str, object] = {
            "domain": step.domain,
            "description": step.description,
            "runtime_s": runtimes[name],
            "max_reciprocity_error_relative": float(np.max(np.abs(fmc - fmc.transpose(1, 0, 2))) / np.max(np.abs(fmc))),
            "edge_to_centre_pulse_echo_amplitude_db": _db(amp[0, 0] / amp[centre, centre]),
            "edge_pitch_catch_to_centre_amplitude_db": _db(amp[0, n - 1] / amp[centre, centre]),
            "aperture_amplitude_dynamic_range_db": _db(np.max(amp) / np.min(amp)),
            "central_pulse_echo_centroid_mhz": _centroid_hz(fmc[centre, centre], dt) / 1e6,
            "edge_pulse_echo_centroid_mhz": _centroid_hz(fmc[0, 0], dt) / 1e6,
            "amplitude_map_cosine_vs_experiment_live_pairs": _cosine(
                _normalise(amp)[live_pairs], exp_map[live_pairs]
            ),
            "amplitude_map_rms_db_error_vs_experiment_live_pairs": float(np.sqrt(np.mean(
                (20 * np.log10(_normalise(amp)[live_pairs]) - 20 * np.log10(exp_map[live_pairs] + 1e-12)) ** 2
            ))),
            "tfm": _image_metrics(image),
        }
        if previous is not None and name != directivity_f0.label:
            prev_fmc = fmcs[previous]
            record["change_vs_previous_rung"] = {
                "previous": previous,
                "fmc_cosine_similarity": _cosine(fmc, prev_fmc),
                "fmc_relative_l2_difference_after_normalising": float(
                    np.linalg.norm(_normalise(fmc) - _normalise(prev_fmc)) / np.linalg.norm(_normalise(prev_fmc))
                ),
                "tfm_relative_l2_difference_after_normalising": float(
                    np.linalg.norm(_normalise(image) - _normalise(images[previous]))
                    / np.linalg.norm(_normalise(images[previous]))
                ),
            }
        steps[name] = record
        if name != directivity_f0.label:
            previous = name

    # TD+ vs FD must agree to numerical precision (same physics, two domains).
    td_fd_error = float(np.max(np.abs(fmcs["M0-FD"] - fmcs["M0-TD+"])) / np.max(np.abs(fmcs["M0-TD+"])))
    # Frequency dependence of directivity: FD+D vs directivity frozen at f0.
    freq_dependence = {
        "fmc_relative_l2_difference": float(
            np.linalg.norm(fmcs["M0-FD+D"] - fmcs["M0-TD+D(f0)"]) / np.linalg.norm(fmcs["M0-TD+D(f0)"])
        ),
        "tfm_relative_l2_difference_after_normalising": float(
            np.linalg.norm(_normalise(images["M0-FD+D"]) - _normalise(images["M0-TD+D(f0)"]))
            / np.linalg.norm(_normalise(images["M0-TD+D(f0)"]))
        ),
    }

    # --- Attenuation sensitivity sweep (frequency-dependent, n = 1) ---------
    sweep: dict[str, list[float]] = {
        "attenuation_db_per_m_at_5mhz": list(SWEEP_DB_PER_M),
        "central_centroid_shift_khz": [], "edge_to_centre_amplitude_change_db": [],
        "tfm_lateral_width_mm": [], "tfm_axial_width_mm": [],
    }
    reference_centroid = None; reference_ratio = None
    for db_per_m in SWEEP_DB_PER_M:
        physics = PropagationPhysics(
            geometric_spreading=True, element_width_m=width_m,
            attenuation_np_per_m=db_per_m / (20.0 / np.log(10.0)),
            attenuation_frequency_exponent=ATTENUATION_EXPONENT,
        )
        fmc = simulate_point_reflector_fmc_fd(time, elements, REFLECTOR, WAVE_SPEED_M_S, CENTRE_FREQUENCY_HZ, SIGMA_S, physics).fmc
        amp = _gated_amplitude(time, fmc)
        centroid = _centroid_hz(fmc[centre, centre], dt)
        ratio = _db(amp[0, n - 1] / amp[centre, centre])
        if reference_centroid is None:
            reference_centroid, reference_ratio = centroid, ratio
        image = tfm_image(fmc, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)
        metrics = _image_metrics(image)
        sweep["central_centroid_shift_khz"].append((centroid - reference_centroid) / 1e3)
        sweep["edge_to_centre_amplitude_change_db"].append(ratio - reference_ratio)
        sweep["tfm_lateral_width_mm"].append(metrics["lateral_minus_6_db_width_mm"])
        sweep["tfm_axial_width_mm"].append(metrics["axial_minus_6_db_width_mm"])

    # --- Figures ------------------------------------------------------------
    ladder_names = [step.label for step in ladder]
    _plot_factors(figure_dir / "physics_factors.png", elements, width_m)
    _plot_amplitude_maps(
        figure_dir / "gated_amplitude_maps.png",
        {name: amplitude_maps[name] for name in (*ladder_names[:2], *ladder_names[3:], "Experiment")},
    )
    _plot_ascans(figure_dir / "representative_ascans.png", time, {k: fmcs[k] for k in ladder_names}, pairs)
    _plot_spectra(figure_dir / "edge_pulse_echo_spectra.png", time, {k: fmcs[k] for k in ladder_names}, (0, 0))
    _plot_tfm(figure_dir / "tfm_images.png", {k: images[k] for k in (*ladder_names[:2], *ladder_names[3:], "Experiment")})
    _plot_profiles(figure_dir / "tfm_profiles.png", {k: images[k] for k in (*ladder_names, "Experiment")})
    _plot_sweep(figure_dir / "attenuation_sensitivity.png", sweep)

    _, sine = element_ray_geometry(elements, REFLECTOR)
    configuration = {
        "reflector_m": REFLECTOR.tolist(),
        "wave_speed_m_s": WAVE_SPEED_M_S,
        "centre_frequency_hz": CENTRE_FREQUENCY_HZ,
        "sigma_s": SIGMA_S,
        "element_width_m_from_mat_corners": width_m,
        "element_pitch_m": float(np.median(np.diff(elements[:, 0]))),
        "a_over_lambda_at_f0": width_m * CENTRE_FREQUENCY_HZ / WAVE_SPEED_M_S,
        "maximum_ray_angle_deg": float(np.rad2deg(np.max(np.abs(np.arcsin(sine))))),
        "attenuation_db_per_m_at_f0": ATTENUATION_DB_PER_M,
        "attenuation_np_per_m_at_f0": ATTENUATION_NP_PER_M,
        "attenuation_frequency_exponent": ATTENUATION_EXPONENT,
        "attenuation_status": "illustrative placeholder, not measured on this specimen",
        "spreading_reference_amplitude_m": 1.0,
        "gate_s": list(GATE_S),
        "ladder": [{"label": s.label, "domain": s.domain, "description": s.description} for s in ladder],
    }
    metrics = {
        "configuration": configuration,
        "td_plus_vs_fd_max_relative_error": td_fd_error,
        "directivity_frequency_dependence": freq_dependence,
        "steps": steps,
        "experiment": {
            "dead_or_weak_channels": np.flatnonzero(~live).tolist(),
            "tfm": _image_metrics(images["Experiment"]),
            "edge_to_centre_pulse_echo_amplitude_db": _db(amplitude_maps["Experiment"][0, 0] / amplitude_maps["Experiment"][centre, centre]),
            "edge_pitch_catch_to_centre_amplitude_db": _db(amplitude_maps["Experiment"][0, n - 1] / amplitude_maps["Experiment"][centre, centre]),
        },
        "attenuation_sweep": sweep,
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_report(result_dir / "configuration.json", configuration)
    print(json.dumps({k: metrics[k] for k in ("td_plus_vs_fd_max_relative_error", "directivity_frequency_dependence")}, indent=2))
    for name, record in steps.items():
        print(name, json.dumps({k: v for k, v in record.items() if k not in ("description",)}, indent=1))
    print("experiment", json.dumps(metrics["experiment"], indent=1))
    print("sweep", json.dumps(sweep, indent=1))


if __name__ == "__main__":
    main()
