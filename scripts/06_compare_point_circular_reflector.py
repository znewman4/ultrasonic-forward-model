"""Generate the Model M0 versus Model M1 comparison deliverables."""
from __future__ import annotations

import json

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
from src.data_loader import load_fmc
from src.geometry import array_coordinates
from src.imaging.tfm import tfm_image
from src.models.boundary_circle import (
    CircularReflector,
    circular_boundary_points,
    simulate_circular_reflector_fmc,
)
from src.models.point_reflector import simulate_point_reflector_fmc


MODEL_M0 = "M0_point_reflector"
MODEL_M1 = "M1_boundary_circle"
BASE_RADIUS_M = 0.5e-3
BASE_N_POINTS = 64
WIDTH_LEVEL_DB = -6.0
WIDTH_LEVEL = 10.0 ** (WIDTH_LEVEL_DB / 20.0)
CONVERGENCE_COUNTS = (4, 8, 16, 32, 64, 128)
CONVERGENCE_REFERENCE_COUNT = 256
ZERO_LIMIT_RADII_M = (0.0, 1e-6, 10e-6, 50e-6, 0.1e-3, 0.25e-3, 0.5e-3)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the normalized waveform inner product."""
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0.0:
        return float("nan")
    return float(np.vdot(first, second).real / denominator)


def _threshold_crossings(
    coordinate: np.ndarray,
    values: np.ndarray,
    relative_level: float = WIDTH_LEVEL,
) -> tuple[float, float] | None:
    """Find interpolated crossings around the connected global-peak lobe."""
    magnitude = np.abs(values)
    if magnitude.size < 3 or np.max(magnitude) <= 0.0:
        return None
    peak = int(np.argmax(magnitude))
    threshold = relative_level * magnitude[peak]
    left_below = np.flatnonzero(magnitude[:peak] < threshold)
    right_below = np.flatnonzero(magnitude[peak + 1 :] < threshold)
    if left_below.size == 0 or right_below.size == 0:
        return None
    left_low = int(left_below[-1])
    right_low = int(peak + 1 + right_below[0])
    left = float(
        np.interp(
            threshold,
            magnitude[left_low : left_low + 2],
            coordinate[left_low : left_low + 2],
        )
    )
    right = float(
        np.interp(
            threshold,
            magnitude[right_low - 1 : right_low + 1][::-1],
            coordinate[right_low - 1 : right_low + 1][::-1],
        )
    )
    return left, right


def _slice_metrics(coordinate: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    """Return a -6 dB width and peak sidelobe level for one image slice.

    The sidelobe is the largest sampled value outside the connected main-lobe
    interval bounded by the -6 dB crossings.
    """
    magnitude = np.abs(values)
    crossings = _threshold_crossings(coordinate, magnitude)
    if crossings is None:
        return {"minus_6_db_width_mm": None, "peak_sidelobe_level_db": None}
    left, right = crossings
    outside = (coordinate < left) | (coordinate > right)
    peak = float(np.max(magnitude))
    if not np.any(outside) or peak == 0.0:
        sidelobe_db = None
    else:
        ratio = max(float(np.max(magnitude[outside]) / peak), np.finfo(float).tiny)
        sidelobe_db = 20.0 * np.log10(ratio)
    return {
        "minus_6_db_width_mm": (right - left) * 1e3,
        "peak_sidelobe_level_db": sidelobe_db,
    }


def _image_metrics(image: np.ndarray) -> dict[str, object]:
    peak_z_index, peak_x_index = np.unravel_index(np.argmax(image), image.shape)
    peak_x = float(X_GRID_M[peak_x_index])
    peak_z = float(Z_GRID_M[peak_z_index])
    lateral = _slice_metrics(X_GRID_M, image[peak_z_index, :])
    axial = _slice_metrics(Z_GRID_M, image[:, peak_x_index])
    return {
        "peak_m": [peak_x, peak_z],
        "localisation_error_mm": float(
            np.hypot(peak_x - REFLECTOR[0], peak_z - REFLECTOR[1]) * 1e3
        ),
        "lateral_minus_6_db_width_mm": lateral["minus_6_db_width_mm"],
        "axial_minus_6_db_width_mm": axial["minus_6_db_width_mm"],
        "lateral_peak_sidelobe_level_db": lateral["peak_sidelobe_level_db"],
        "axial_peak_sidelobe_level_db": axial["peak_sidelobe_level_db"],
    }


def _representative_pairs(elements: np.ndarray) -> list[tuple[str, int, int]]:
    centre = int(np.argmin(np.abs(elements[:, 0])))
    left_quarter = int(np.argmin(np.abs(elements[:, 0] + 10e-3)))
    right_quarter = int(np.argmin(np.abs(elements[:, 0] - 10e-3)))
    return [
        ("central_pulse_echo", centre, centre),
        ("off_axis_pulse_echo", left_quarter, left_quarter),
        ("pitch_catch", left_quarter, right_quarter),
    ]


def _ascan_metrics(
    time: np.ndarray,
    m0_fmc: np.ndarray,
    m0_times: np.ndarray,
    m1_fmc: np.ndarray,
    m1_times: np.ndarray,
    pairs: list[tuple[str, int, int]],
) -> dict[str, dict[str, float | list[int]]]:
    envelope_m0 = np.abs(hilbert(m0_fmc, axis=2))
    envelope_m1 = np.abs(hilbert(m1_fmc, axis=2))
    result: dict[str, dict[str, float | list[int]]] = {}
    for name, transmitter, receiver in pairs:
        earliest = float(np.min(m1_times[transmitter, receiver]))
        latest = float(np.max(m1_times[transmitter, receiver]))
        point_time = float(m0_times[transmitter, receiver])
        gate = (time >= min(earliest, point_time) - 4.0 * SIGMA_S) & (
            time <= max(latest, point_time) + 4.0 * SIGMA_S
        )
        indices = np.flatnonzero(gate)
        m0_peak_index = int(indices[np.argmax(envelope_m0[transmitter, receiver, gate])])
        m1_peak_index = int(indices[np.argmax(envelope_m1[transmitter, receiver, gate])])
        m0_trace = m0_fmc[transmitter, receiver, gate]
        m1_trace = m1_fmc[transmitter, receiver, gate]
        result[name] = {
            "element_indices": [transmitter, receiver],
            "waveform_correlation": _normalized_correlation(m0_trace, m1_trace),
            "m0_geometric_arrival_us": point_time * 1e6,
            "m0_envelope_peak_us": float(time[m0_peak_index] * 1e6),
            "m1_earliest_boundary_arrival_us": earliest * 1e6,
            "m1_mean_boundary_arrival_us": float(
                np.mean(m1_times[transmitter, receiver]) * 1e6
            ),
            "m1_latest_boundary_arrival_us": latest * 1e6,
            "m1_envelope_peak_us": float(time[m1_peak_index] * 1e6),
            "envelope_peak_shift_m1_minus_m0_us": float(
                (time[m1_peak_index] - time[m0_peak_index]) * 1e6
            ),
        }
    return result


def _plot_ascans(
    path,
    time: np.ndarray,
    m0_fmc: np.ndarray,
    m1_fmc: np.ndarray,
    m1_times: np.ndarray,
    pairs: list[tuple[str, int, int]],
) -> None:
    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 8), sharex=True)
    for ax, (name, transmitter, receiver) in zip(axes, pairs, strict=True):
        ax.plot(time * 1e6, m0_fmc[transmitter, receiver], label="M0 point")
        ax.plot(time * 1e6, m1_fmc[transmitter, receiver], label="M1 boundary")
        ax.axvspan(
            np.min(m1_times[transmitter, receiver]) * 1e6,
            np.max(m1_times[transmitter, receiver]) * 1e6,
            color="grey",
            alpha=0.15,
            label="M1 boundary arrival range",
        )
        ax.set(ylabel="Amplitude", title=name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize="small")
    axes[-1].set(xlabel="Time (µs)", xlim=(11.0, 15.5))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tfm(path, m0_image: np.ndarray, m1_image: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    display = None
    for ax, image, title in (
        (axes[0], m0_image, "M0: point reflector"),
        (axes[1], m1_image, "M1: circular boundary"),
    ):
        display = ax.imshow(
            image / np.max(image),
            extent=[
                X_GRID_M[0] * 1e3,
                X_GRID_M[-1] * 1e3,
                Z_GRID_M[-1] * 1e3,
                Z_GRID_M[0] * 1e3,
            ],
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax.scatter([REFLECTOR[0] * 1e3], [REFLECTOR[1] * 1e3], color="cyan", marker="x")
        ax.set(title=title, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(display, ax=axes, label="Normalised TFM amplitude", shrink=0.9)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.9, wspace=0.12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tfm_slices(path, m0_image: np.ndarray, m1_image: np.ndarray) -> None:
    m0_z, m0_x = np.unravel_index(np.argmax(m0_image), m0_image.shape)
    m1_z, m1_x = np.unravel_index(np.argmax(m1_image), m1_image.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(X_GRID_M * 1e3, m0_image[m0_z] / np.max(m0_image), label="M0")
    axes[0].plot(X_GRID_M * 1e3, m1_image[m1_z] / np.max(m1_image), label="M1")
    axes[0].axhline(WIDTH_LEVEL, color="black", ls="--", label="−6 dB")
    axes[0].set(xlabel="x (mm)", ylabel="Normalised amplitude", title="Lateral peak slices")
    axes[1].plot(m0_image[:, m0_x] / np.max(m0_image), Z_GRID_M * 1e3, label="M0")
    axes[1].plot(m1_image[:, m1_x] / np.max(m1_image), Z_GRID_M * 1e3, label="M1")
    axes[1].axvline(WIDTH_LEVEL, color="black", ls="--", label="−6 dB")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Normalised amplitude", ylabel="z (mm)", title="Axial peak slices")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _convergence_metrics(
    time: np.ndarray, selected_elements: np.ndarray
) -> dict[str, object]:
    reference_boundary = circular_boundary_points(
        CircularReflector(*REFLECTOR, BASE_RADIUS_M, CONVERGENCE_REFERENCE_COUNT)
    )
    reference, _ = simulate_circular_reflector_fmc(
        time,
        selected_elements,
        reference_boundary,
        WAVE_SPEED_M_S,
        CENTRE_FREQUENCY_HZ,
        SIGMA_S,
    )
    entries: dict[str, dict[str, float]] = {}
    for count in CONVERGENCE_COUNTS:
        boundary = circular_boundary_points(
            CircularReflector(*REFLECTOR, BASE_RADIUS_M, count)
        )
        fmc, _ = simulate_circular_reflector_fmc(
            time,
            selected_elements,
            boundary,
            WAVE_SPEED_M_S,
            CENTRE_FREQUENCY_HZ,
            SIGMA_S,
        )
        entries[str(count)] = {
            "relative_l2_error": float(
                np.linalg.norm(fmc - reference) / np.linalg.norm(reference)
            ),
            "waveform_correlation": _normalized_correlation(fmc.ravel(), reference.ravel()),
        }
    return {
        "radius_mm": BASE_RADIUS_M * 1e3,
        "selected_element_indices": [0, 16, 32, 47, 63],
        "reference_n_points": CONVERGENCE_REFERENCE_COUNT,
        "values": entries,
    }


def _zero_radius_metrics(
    time: np.ndarray, selected_elements: np.ndarray
) -> dict[str, object]:
    m0, _ = simulate_point_reflector_fmc(
        time,
        selected_elements,
        REFLECTOR,
        WAVE_SPEED_M_S,
        CENTRE_FREQUENCY_HZ,
        SIGMA_S,
    )
    entries: dict[str, dict[str, float]] = {}
    for radius in ZERO_LIMIT_RADII_M:
        boundary = circular_boundary_points(
            CircularReflector(*REFLECTOR, radius, BASE_N_POINTS)
        )
        m1, _ = simulate_circular_reflector_fmc(
            time,
            selected_elements,
            boundary,
            WAVE_SPEED_M_S,
            CENTRE_FREQUENCY_HZ,
            SIGMA_S,
        )
        entries[f"{radius * 1e6:g}_um"] = {
            "radius_m": radius,
            "relative_l2_difference_from_m0": float(
                np.linalg.norm(m1 - m0) / np.linalg.norm(m0)
            ),
            "waveform_correlation_with_m0": _normalized_correlation(
                m1.ravel(), m0.ravel()
            ),
        }
    return {
        "n_points": BASE_N_POINTS,
        "selected_element_indices": [0, 16, 32, 47, 63],
        "values": entries,
    }


def _plot_numerical_studies(path, convergence: dict, zero_limit: dict) -> None:
    convergence_values = convergence["values"]
    counts = np.asarray([int(value) for value in convergence_values])
    errors = np.asarray(
        [convergence_values[str(value)]["relative_l2_error"] for value in counts]
    )
    zero_values = zero_limit["values"]
    radii_um = np.asarray([entry["radius_m"] * 1e6 for entry in zero_values.values()])
    differences = np.asarray(
        [entry["relative_l2_difference_from_m0"] for entry in zero_values.values()]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(counts, errors, "o-")
    axes[0].set(
        xlabel="Boundary-point count M",
        ylabel=f"Relative L2 error vs M={CONVERGENCE_REFERENCE_COUNT}",
        title="M1 quadrature convergence",
    )
    axes[1].plot(radii_um, differences, "o-")
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set(
        xlabel="Radius (µm)",
        ylabel="Relative L2 difference from M0",
        title="M1 zero-radius limit",
    )
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(path, metrics: dict[str, object]) -> None:
    m0 = metrics["tfm"]["M0"]
    m1 = metrics["tfm"]["M1"]
    central = metrics["representative_ascans"]["central_pulse_echo"]
    text = f"""# M0 point versus M1 boundary-circle comparison

## Configuration

- M0: original point reflector at x={REFLECTOR[0] * 1e3:g} mm,
  z={REFLECTOR[1] * 1e3:g} mm.
- M1: circular boundary with radius {BASE_RADIUS_M * 1e3:g} mm and
  M={BASE_N_POINTS} equally weighted points.
- Both models use the same pulse, array geometry, wave speed, time vector, and
  TFM implementation.

## Results

- Central A-scan correlation: {central['waveform_correlation']:.6f}.
- Central M1 boundary arrivals span
  {central['m1_earliest_boundary_arrival_us']:.6f}–{central['m1_latest_boundary_arrival_us']:.6f} µs.
- M0 localisation error: {m0['localisation_error_mm']:.3f} mm; M1 localisation
  error: {m1['localisation_error_mm']:.3f} mm.
- M0 lateral/axial −6 dB widths: {m0['lateral_minus_6_db_width_mm']:.3f}/
  {m0['axial_minus_6_db_width_mm']:.3f} mm.
- M1 lateral/axial −6 dB widths: {m1['lateral_minus_6_db_width_mm']:.3f}/
  {m1['axial_minus_6_db_width_mm']:.3f} mm.
- Sidelobe metrics and full convergence/zero-radius tables are in `metrics.json`.

## Interpretation

M1 captures finite circular geometry and coherent interference among its
geometric boundary paths. Equal weights make its radius-zero limit reproduce
M0 and prevent amplitude from increasing with discretisation count.

M1 **does not satisfy the traction-free elastic boundary condition and does not
include mode conversion**. It also omits realistic cylindrical scattering
coefficients, illumination/shadowing, directivity, attenuation, and geometric
spreading. These results describe this geometric boundary quadrature, not a
rigorous elastic side-drilled-hole solution.

The reported −6 dB widths use interpolated amplitude crossings at
`10^(-6/20)` of the image peak. Peak sidelobe level is the largest sampled
amplitude outside the connected −6 dB main lobe of the corresponding axial or
lateral peak slice, expressed relative to that slice peak.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    point_data_dir = ROOT / "data" / "synthetic" / "point_reflector"
    boundary_data_dir = ROOT / "data" / "synthetic" / "boundary_circle"
    result_dir = ROOT / "results" / "comparisons" / "M0_point_vs_M1_boundary"
    figure_dir = result_dir / "figures"
    for directory in (point_data_dir, boundary_data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    time = load_fmc(MAT_PATH).metadata.time_s
    elements = array_coordinates(MAT_PATH)
    circular = CircularReflector(*REFLECTOR, BASE_RADIUS_M, BASE_N_POINTS)
    boundary = circular_boundary_points(circular)

    m0_fmc, m0_times = simulate_point_reflector_fmc(
        time,
        elements,
        REFLECTOR,
        WAVE_SPEED_M_S,
        CENTRE_FREQUENCY_HZ,
        SIGMA_S,
    )
    m1_fmc, m1_times = simulate_circular_reflector_fmc(
        time,
        elements,
        boundary,
        WAVE_SPEED_M_S,
        CENTRE_FREQUENCY_HZ,
        SIGMA_S,
    )
    m0_image = tfm_image(m0_fmc, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)
    m1_image = tfm_image(m1_fmc, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)

    common_parameters = {
        "wave_speed_m_s": WAVE_SPEED_M_S,
        "centre_frequency_hz": CENTRE_FREQUENCY_HZ,
        "sigma_s": SIGMA_S,
        "amplitude": 1.0,
    }
    np.savez_compressed(
        point_data_dir / "M0_point_reflector_fmc.npz",
        fmc=m0_fmc,
        time_s=time,
        element_coordinates_m=elements,
        reflector_coordinate_m=REFLECTOR,
        travel_time_s=m0_times,
        tfm_image=m0_image,
        x_grid_m=X_GRID_M,
        z_grid_m=Z_GRID_M,
        model_id=np.array(MODEL_M0),
        centre_x_m=np.array(REFLECTOR[0]),
        centre_z_m=np.array(REFLECTOR[1]),
        wave_speed_m_s=np.array(WAVE_SPEED_M_S),
        centre_frequency_hz=np.array(CENTRE_FREQUENCY_HZ),
        sigma_s=np.array(SIGMA_S),
        amplitude=np.array(1.0),
        model_parameters_json=np.array(
            json.dumps({**common_parameters, "x_m": REFLECTOR[0], "z_m": REFLECTOR[1]})
        ),
    )
    np.savez_compressed(
        boundary_data_dir / "M1_boundary_circle_fmc.npz",
        fmc=m1_fmc,
        time_s=time,
        element_coordinates_m=elements,
        boundary_coordinates_m=boundary,
        boundary_travel_time_s=m1_times,
        tfm_image=m1_image,
        x_grid_m=X_GRID_M,
        z_grid_m=Z_GRID_M,
        model_id=np.array(MODEL_M1),
        centre_x_m=np.array(circular.x_m),
        centre_z_m=np.array(circular.z_m),
        radius_m=np.array(circular.radius_m),
        n_points=np.array(circular.n_points),
        boundary_weight=np.array(1.0 / circular.n_points),
        wave_speed_m_s=np.array(WAVE_SPEED_M_S),
        centre_frequency_hz=np.array(CENTRE_FREQUENCY_HZ),
        sigma_s=np.array(SIGMA_S),
        amplitude=np.array(1.0),
        model_parameters_json=np.array(
            json.dumps(
                {
                    **common_parameters,
                    "x_m": circular.x_m,
                    "z_m": circular.z_m,
                    "radius_m": circular.radius_m,
                    "n_points": circular.n_points,
                    "boundary_weight": 1.0 / circular.n_points,
                }
            )
        ),
    )

    pairs = _representative_pairs(elements)
    ascan_metrics = _ascan_metrics(time, m0_fmc, m0_times, m1_fmc, m1_times, pairs)
    selected_indices = np.array([0, 16, 32, 47, 63])
    convergence = _convergence_metrics(time, elements[selected_indices])
    zero_limit = _zero_radius_metrics(time, elements[selected_indices])
    metrics: dict[str, object] = {
        "comparison": "M0 point reflector versus M1 circular boundary",
        "configuration": {
            "centre_m": REFLECTOR.tolist(),
            "radius_m": BASE_RADIUS_M,
            "n_boundary_points": BASE_N_POINTS,
            **common_parameters,
        },
        "representative_ascans": ascan_metrics,
        "full_fmc_waveform_correlation": _normalized_correlation(
            m0_fmc.ravel(), m1_fmc.ravel()
        ),
        "full_fmc_relative_l2_difference": float(
            np.linalg.norm(m1_fmc - m0_fmc) / np.linalg.norm(m0_fmc)
        ),
        "tfm": {"M0": _image_metrics(m0_image), "M1": _image_metrics(m1_image)},
        "convergence_vs_boundary_point_count": convergence,
        "radius_tending_to_zero": zero_limit,
        "metric_definitions": {
            "waveform_correlation": "Normalised real waveform inner product (cosine similarity) over the stated samples.",
            "minus_6_db_width": "Interpolated amplitude crossings at 10^(-6/20) around the connected global-peak lobe.",
            "peak_sidelobe_level": "Largest sampled slice amplitude outside the connected -6 dB main lobe, in dB relative to the slice peak.",
            "convergence_and_zero_limit": "Computed on element indices [0, 16, 32, 47, 63] with the full time vector.",
        },
        "M1_scope": "Finite circular geometry and coherent geometric path interference.",
        "M1_exclusions": [
            "traction-free elastic boundary condition",
            "mode conversion",
            "realistic cylindrical scattering coefficients",
            "illumination and shadowing",
            "directivity",
            "attenuation",
            "geometric spreading",
        ],
    }

    _plot_ascans(figure_dir / "representative_ascans.png", time, m0_fmc, m1_fmc, m1_times, pairs)
    _plot_tfm(figure_dir / "tfm_comparison.png", m0_image, m1_image)
    _plot_tfm_slices(figure_dir / "tfm_minus_6_db_slices.png", m0_image, m1_image)
    _plot_numerical_studies(
        figure_dir / "convergence_and_zero_radius.png", convergence, zero_limit
    )
    save_report(result_dir / "metrics.json", metrics)
    _write_summary(result_dir / "summary.md", metrics)


if __name__ == "__main__":
    main()
