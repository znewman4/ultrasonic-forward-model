"""Audit the experimental MAT acquisition for model-to-data accuracy limits.

This is deliberately a measurement audit, not a parameter-fitting step.  It
reports only fields and signal properties that can be established from the MAT
file; filename tokens such as ``h40mm`` and ``g30`` remain unverified hints.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

from common import MAT_PATH, ROOT
from src.data_loader import experimental_fmc_array, load_fmc


OUTPUT = ROOT / "results" / "comparisons" / "M2_multimode" / "validation"


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    aa -= np.mean(aa); bb -= np.mean(bb)
    denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denominator) if denominator else float("nan")


def _gate_metrics(fmc: np.ndarray, time_s: np.ndarray, start_us: float, stop_us: float) -> dict:
    gate = (time_s >= start_us * 1e-6) & (time_s < stop_us * 1e-6)
    values = fmc[..., gate]
    reversed_values = values.transpose(1, 0, 2)
    scale = np.linalg.norm(values)
    return {
        "start_us": start_us,
        "stop_us": stop_us,
        "sample_count_all_traces": int(values.size),
        "minimum_clip_count": int(np.count_nonzero(values == np.min(fmc))),
        "maximum_clip_count": int(np.count_nonzero(values == np.max(fmc))),
        "relative_reciprocity_l2": float(np.linalg.norm(values - reversed_values) / scale),
        "reciprocity_correlation": _correlation(values, reversed_values),
    }


def _mean_gated_spectrum(
    fmc: np.ndarray, time_s: np.ndarray, start_us: float, stop_us: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    gate = (time_s >= start_us * 1e-6) & (time_s < stop_us * 1e-6)
    gated = fmc[..., gate]
    window = tukey(gated.shape[-1], alpha=0.25)
    spectra = np.fft.rfft(gated * window, axis=-1)
    power = np.mean(np.abs(spectra) ** 2, axis=(0, 1))
    frequency_hz = np.fft.rfftfreq(gated.shape[-1], np.mean(np.diff(time_s)))
    positive = power > 0.0
    peak = int(np.argmax(power))
    centroid = float(np.sum(frequency_hz * power) / np.sum(power))
    cumulative = np.cumsum(power) / np.sum(power)
    f05 = float(np.interp(0.05, cumulative, frequency_hz))
    f95 = float(np.interp(0.95, cumulative, frequency_hz))
    return frequency_hz, power, {
        "gate_us": [start_us, stop_us],
        "peak_frequency_mhz": float(frequency_hz[peak] * 1e-6),
        "power_centroid_mhz": centroid * 1e-6,
        "power_5_to_95_percent_band_mhz": [f05 * 1e-6, f95 * 1e-6],
        "nonzero_bins": int(np.count_nonzero(positive)),
    }


def _pair_alignment(fmc: np.ndarray, gate: np.ndarray, maximum_lag: int = 4) -> dict:
    correlations: list[float] = []
    lags: list[int] = []
    n = fmc.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a = fmc[i, j, gate]
            b = fmc[j, i, gate]
            candidates = []
            for lag in range(-maximum_lag, maximum_lag + 1):
                if lag < 0:
                    candidates.append((_correlation(a[-lag:], b[:lag]), lag))
                elif lag > 0:
                    candidates.append((_correlation(a[:-lag], b[lag:]), lag))
                else:
                    candidates.append((_correlation(a, b), lag))
            correlation, lag = max(candidates, key=lambda item: -np.inf if np.isnan(item[0]) else item[0])
            correlations.append(correlation); lags.append(lag)
    lag_values, lag_counts = np.unique(lags, return_counts=True)
    return {
        "maximum_tested_lag_samples": maximum_lag,
        "median_best_aligned_correlation": float(np.nanmedian(correlations)),
        "best_lag_histogram": {str(int(k)): int(v) for k, v in zip(lag_values, lag_counts)},
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = load_fmc(MAT_PATH)
    fmc = experimental_fmc_array(loaded)
    time_s = loaded.metadata.time_s
    dt = np.diff(time_s)
    centres = loaded.metadata.array.element_centres_m
    corners = loaded.metadata.array.element_corners_m
    endpoint_vectors = corners[:, 1] - corners[:, 0]
    endpoint_midpoints = np.mean(corners, axis=1)
    midpoint_offsets = centres - endpoint_midpoints

    levels = np.unique(fmc)
    level_steps = np.diff(levels)
    gates = {
        "early_0_to_2_us": _gate_metrics(fmc, time_s, 0.0, 2.0),
        "quiet_2_to_8_us": _gate_metrics(fmc, time_s, 2.0, 8.0),
        "echo_10_to_16_5_us": _gate_metrics(fmc, time_s, 10.0, 16.5),
        "late_16_5_to_end_us": _gate_metrics(fmc, time_s, 16.5, float(time_s[-1] * 1e6 + 1e-9)),
    }
    echo_gate = (time_s >= 10e-6) & (time_s < 16.5e-6)
    frequency_hz, echo_power, spectrum = _mean_gated_spectrum(fmc, time_s, 10.0, 16.5)
    tx_rms = np.sqrt(np.mean(fmc**2, axis=(1, 2)))
    early = time_s < 2e-6
    baseline_means = np.mean(fmc[..., early], axis=-1)

    report = {
        "source_file": str(MAT_PATH),
        "scope": "Read-only audit; filename tokens are not treated as verified metadata.",
        "record": {
            "fmc_shape": list(fmc.shape),
            "channel_storage_shape": list(loaded.time_data.shape),
            "sample_count": int(time_s.size),
            "sample_interval_ns": float(np.mean(dt) * 1e9),
            "sampling_frequency_mhz": float(1.0 / np.mean(dt) * 1e-6),
            "duration_us": float(time_s[-1] * 1e6),
            "maximum_timestep_nonuniformity_ps": float(np.max(np.abs(dt - np.mean(dt))) * 1e12),
            "complete_tx_rx_grid": bool(fmc.shape[0] == fmc.shape[1] == 64),
        },
        "stored_metadata": {
            "manufacturer": loaded.metadata.array.manufacturer,
            "nominal_centre_frequency_mhz": loaded.metadata.array.centre_frequency_hz * 1e-6,
            "stored_longitudinal_speed_m_s": loaded.metadata.material_wave_speed_m_s,
            "stored_shear_speed_m_s": None,
            "element_centre_x_span_mm": [float(np.min(centres[:, 0]) * 1e3), float(np.max(centres[:, 0]) * 1e3)],
            "centre_pitch_mm": float(np.median(np.diff(centres[:, 0])) * 1e3),
            "endpoint_vector_mm": (np.median(endpoint_vectors, axis=0) * 1e3).tolist(),
            "centre_minus_endpoint_midpoint_mm": (np.median(midpoint_offsets, axis=0) * 1e3).tolist(),
            "endpoint_interpretation_warning": "Stored endpoint midpoints do not equal stored element centres; do not infer active element width/directivity without schema documentation.",
        },
        "digitisation": {
            "unique_amplitude_levels": int(levels.size),
            "minimum": float(np.min(fmc)),
            "maximum": float(np.max(fmc)),
            "median_level_increment": float(np.median(level_steps)),
            "minimum_clip_count": int(np.count_nonzero(fmc == np.min(fmc))),
            "maximum_clip_count": int(np.count_nonzero(fmc == np.max(fmc))),
            "zero_count": int(np.count_nonzero(fmc == 0.0)),
            "interpretation": "Exactly 256 levels and endpoint occupancy are consistent with normalized 8-bit acquisition and clipping.",
        },
        "gates": gates,
        "reciprocal_pair_alignment_echo_gate": _pair_alignment(fmc, echo_gate),
        "channel_variation": {
            "transmitter_mean_rms_minimum": float(np.min(tx_rms)),
            "transmitter_mean_rms_maximum": float(np.max(tx_rms)),
            "maximum_to_minimum_ratio": float(np.max(tx_rms) / np.min(tx_rms)),
            "early_baseline_mean_absolute_median": float(np.median(np.abs(baseline_means))),
            "early_baseline_mean_absolute_maximum": float(np.max(np.abs(baseline_means))),
        },
        "echo_spectrum": spectrum,
        "missing_for_calibrated_prediction": [
            "measured shear speed and density for this block",
            "attenuation/dispersion versus frequency",
            "verified specimen dimensions and SDH centre/radius",
            "wedge/couplant path and time-zero calibration",
            "active element width/elevation and endpoint-field definitions",
            "transmit pulse voltage and per-element transmit response",
            "receive polarization sensitivity and per-element receive response",
            "analogue/digital filters, gain, averaging and digitizer calibration",
            "reference or background acquisition for transfer-function estimation",
        ],
    }

    (OUTPUT / "experimental_mat_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(frequency_hz * 1e-6, echo_power / np.max(echo_power))
    axes[0, 0].axvline(5.0, color="k", ls="--", lw=1, label="nominal 5 MHz")
    axes[0, 0].set(xlim=(0, 15), title="Mean echo-gated power spectrum", xlabel="Frequency (MHz)", ylabel="Normalized power")
    axes[0, 0].legend()
    axes[0, 1].plot(np.arange(64), tx_rms, marker=".")
    axes[0, 1].set(title="Mean RMS by transmitting element", xlabel="Transmit element", ylabel="RMS")
    axes[1, 0].hist(fmc.ravel(), bins=256)
    axes[1, 0].set(title="Digitized amplitude distribution", xlabel="Stored amplitude", ylabel="Count")
    central = fmc[31, 31]
    axes[1, 1].plot(time_s * 1e6, central)
    axes[1, 1].axvspan(10.0, 16.5, color="C1", alpha=0.18, label="PP comparison gate")
    axes[1, 1].set(title="Representative experimental A-scan", xlabel="Time (µs)", ylabel="Amplitude")
    axes[1, 1].legend()
    for ax in axes.ravel(): ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "experimental_mat_audit.png", dpi=180)
    plt.close(fig)

    summary = f"""# Experimental MAT-file audit

This is a read-only audit of `{MAT_PATH.name}`. Filename tokens such as
`h40mm`, `hole1mm`, and `g30` are useful clues, but are not stored as verified
fields inside the acquisition.

## What the file establishes

- Complete 64 x 64 FMC, {time_s.size} samples at {np.mean(dt)*1e9:.3f} ns
  ({1/np.mean(dt)*1e-6:.1f} MHz), ending at {time_s[-1]*1e6:.2f} µs.
- Array centre pitch {np.median(np.diff(centres[:, 0]))*1e3:.3f} mm and centre
  span {(np.max(centres[:, 0])-np.min(centres[:, 0]))*1e3:.3f} mm.
- Stored nominal centre frequency 5 MHz and longitudinal speed
  {loaded.metadata.material_wave_speed_m_s:.0f} m/s. No shear speed is stored.
- Exactly {levels.size} amplitude levels with step {np.median(level_steps):.7f};
  {np.count_nonzero(fmc == np.min(fmc)) + np.count_nonzero(fmc == np.max(fmc)):,}
  samples occupy the two extrema. This is consistent with normalized 8-bit
  acquisition with clipping.
- Echo-gated reciprocity correlation is
  {gates['echo_10_to_16_5_us']['reciprocity_correlation']:.5f}; after allowing
  ±4 samples, the median reciprocal-pair correlation is
  {report['reciprocal_pair_alignment_echo_gate']['median_best_aligned_correlation']:.5f}.
- Transmitter-averaged RMS varies by a factor of
  {report['channel_variation']['maximum_to_minimum_ratio']:.2f}, evidence that
  one shared transmit/receive amplitude is inadequate.
- The 10–16.5 µs echo-gated power spectrum peaks at
  {spectrum['peak_frequency_mhz']:.3f} MHz with a 5–95% power band of
  {spectrum['power_5_to_95_percent_band_mhz'][0]:.3f}–{spectrum['power_5_to_95_percent_band_mhz'][1]:.3f} MHz.

## Accuracy improvements supported by this audit

1. Reacquire with lower analogue gain and at least 12–16 bit digitization;
   clipping and 8-bit quantization cannot be repaired by the scattering model.
2. Estimate a complex source/receiver transfer function from a reference or
   back-wall acquisition instead of using the nominal 5 MHz Gaussian pulse.
3. Calibrate time zero and any wedge/couplant delay before adjusting the cavity
   location to absorb timing bias.
4. Measure `c_p`, `c_s`, density, and preferably attenuation on the actual block.
   The MAT file stores only `c_p=6300 m/s`; M2 currently assumes `c_s=3100 m/s`.
5. Calibrate per-element transmit and receive sensitivities. Reciprocal pairs
   are strongly correlated, but the nearly 3:1 transmitter RMS spread is large.
6. Confirm the element endpoint schema and obtain active width/elevation before
   implementing aperture directivity; their stored midpoints do not coincide
   with the stored centres.
7. Record at least 32 µs (preferably 35 µs with margin). The present 19.98 µs
   record truncates representative PS arrivals and excludes SS arrivals.
8. Supply the specimen drawing/scan log and a background/control scan to
   distinguish the SDH echo from the broad response near 40 mm, plausibly a
   boundary response. The MAT file alone cannot make that assignment.

These changes improve model-to-experiment accuracy without changing the now
validated ideal elastic cavity kernel. Probe voltage prediction still requires
finite aperture, polarization sensitivity, coupling, spreading, attenuation,
and electronics response.
"""
    (OUTPUT / "experimental_mat_audit.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
