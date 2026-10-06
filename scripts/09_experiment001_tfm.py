"""TFM of experiment_001 FMC files (BRAIN1 half-matrix .mat), with and without time-zero offset.

For every defect folder this writes, into ``<defect>/python/``: ``tfm_comparison.png``,
``tfm_raw_time.npz``, ``tfm_offset_corrected.npz`` and ``tfm_report.json``, and adds a
``derived_from_data`` block to ``<defect>/acquisition.json`` (measured values only).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from common import ROOT
from src.experiment_io import load_hmc
from src.imaging.tfm import tfm_image

DEFECTS = Path(r"C:\Users\js23252\OneDrive - University of Bristol\PhD\Experiments\experiment_001\defects")
C_M_S = 6300.0
# Brain1 'Imaging: Contact TFM (v3)' window for SDH_01 (user screenshot): 39.69 x 125.874 mm, 0.16 mm pixels.
BRAIN1_SDH_SETTINGS = {
    "source": "screenshot of BRAIN1 'Imaging: Contact TFM (v3)' window supplied for SDH_01 only",
    "x_size_mm": 39.69, "x_offset_mm": 0, "z_size_mm": 125.874, "z_offset_mm": 0, "pixel_size_mm": 0.16,
    "show_couplant_only": False, "filter_on": True, "filter_freq_MHz": 5, "percent_bandwidth": 200,
    "velocity_m_s": 6300, "angle_dependent_velocity": "None", "attenuation_correction": True,
    "attenuation_dB_per_mm": 0, "angle_limiter": False, "angle_limit_deg": 30,
    "look_elevation_deg": 0, "aperture_window": "Hanning", "aperture_weight_correction": "By weights",
    "interpolation": "Linear", "use_gpu": True, "display_range_dB": 40,
}


def time_zero_offset(fmc, time_s, windows=((14e-6, 18e-6), (30e-6, 34e-6))):
    """Offset = t1 - (t2 - t1) from the first two back-wall echoes in pulse-echo traces."""
    dt = time_s[1] - time_s[0]
    peaks = []
    for i in range(fmc.shape[0]):
        env = np.abs(hilbert(fmc[i, i]))
        row = []
        for lo, hi in windows:
            m = (time_s >= lo) & (time_s <= hi)
            k = int(np.argmax(np.where(m, env, -1)))
            y0, y1, y2 = env[k - 1], env[k], env[k + 1]
            row.append(time_s[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * dt)
        peaks.append(row)
    p = np.array(peaks)
    t1, t2 = p[:, 0], p[:, 1]
    off = t1 - (t2 - t1)
    q = np.percentile(off, [25, 75])
    return float(np.median(off)), float(np.median(t1)), float(np.median(t2)), [float(q[0]), float(q[1])]


def process(defect_dir: Path, x_mm, z_mm, step_mm) -> None:
    mat_path = next(defect_dir.glob("*_FMC.mat"))
    fmc, time_s, elements, filled, traces = load_hmc(mat_path)
    offset, t1, t2, iqr = time_zero_offset(fmc, time_s)
    thickness = (t2 - t1) * C_M_S / 2
    print(f"{defect_dir.name}: FMC {fmc.shape}, offset {offset*1e6:.3f} us, back wall {thickness*1e3:.2f} mm")

    x = np.arange(x_mm[0], x_mm[1] + 1e-9, step_mm) * 1e-3
    z = np.arange(z_mm[0], z_mm[1] + 1e-9, step_mm) * 1e-3
    out = defect_dir / "python"
    out.mkdir(exist_ok=True)

    images, peaks = {}, {}
    for label, shift in (("raw_time", 0.0), ("offset_corrected", offset)):
        img = tfm_image(fmc, time_s - shift, elements, x, z, C_M_S)
        images[label] = img
        iz, ix = np.unravel_index(np.argmax(img), img.shape)
        peaks[label] = [float(x[ix] * 1e3), float(z[iz] * 1e3)]
        np.savez_compressed(out / f"tfm_{label}.npz", image=img, x_m=x, z_m=z, shift_s=shift)

    fig, axes = plt.subplots(1, 2, figsize=(12, 11), sharey=True)
    ref = max(i.max() for i in images.values())
    for ax, (label, img) in zip(axes, images.items()):
        db = 20 * np.log10(np.maximum(img / ref, 1e-6))
        im = ax.imshow(db, extent=[x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3], aspect="equal",
                       cmap="jet", vmin=-40, vmax=0)
        ax.set(title=f"{defect_dir.name}: {label}", xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, label="dB re max", shrink=0.8)
    fig.savefig(out / "tfm_comparison.png", dpi=150)
    plt.close(fig)

    derived = {
        "note": "Measured from the FMC .mat by scripts/09_experiment001_tfm.py; not instrument settings.",
        "capture_type": "half matrix capture (tx <= rx), expanded to full matrix by reciprocity for imaging",
        "recorded_tx_rx_pairs": int(filled.sum()),
        "element_pitch_mm": round(float(np.median(np.diff(elements[:, 0]))) * 1e3, 4),
        "array_aperture_mm_centre_to_centre": round(float(np.ptp(elements[:, 0])) * 1e3, 3),
        "time_zero_offset_us": round(offset * 1e6, 4),
        "time_zero_offset_iqr_us": [round(v * 1e6, 4) for v in iqr],
        "time_zero_offset_method": "t1 - (t2 - t1) from first two back-wall echo envelope peaks in pulse-echo traces",
        "backwall_echo_times_us": [round(t1 * 1e6, 4), round(t2 * 1e6, 4)],
        "apparent_thickness_mm_at_6300_m_s": round(thickness * 1e3, 3),
        "samples_per_period_at_centre_freq": 5.0,
        "fraction_of_samples_at_plus_minus_1": round(float(np.mean(np.abs(traces) >= 0.999)), 5),
        "traces_peaking_at_plus_minus_1": int(np.sum(np.abs(traces).max(axis=0) >= 0.999)),
        "python_tfm": {
            "outputs": ["python/tfm_comparison.png", "python/tfm_raw_time.npz",
                        "python/tfm_offset_corrected.npz", "python/tfm_report.json"],
            "hilbert_on": True, "interpolation": "linear", "velocity_m_s": C_M_S,
            "grid_x_mm": list(x_mm), "grid_z_mm": list(z_mm), "pixel_size_mm": step_mm,
            "image_peak_mm_raw_time": peaks["raw_time"],
            "image_peak_mm_offset_corrected": peaks["offset_corrected"],
        },
    }
    (out / "tfm_report.json").write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")

    acq_path = defect_dir / "acquisition.json"
    acq = json.loads(acq_path.read_text(encoding="utf-8"))
    acq["derived_from_data"] = derived
    if defect_dir.name == "SDH_01":
        acq.setdefault("unrecorded_settings", {})["tfm_settings"] = BRAIN1_SDH_SETTINGS
    acq_path.write_text(json.dumps(acq, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defect", nargs="*", default=None, help="defect folder names (default: all)")
    ap.add_argument("--z", nargs=2, type=float, default=[0.0, 125.874], help="depth range, mm")
    ap.add_argument("--x", nargs=2, type=float, default=[-19.845, 19.845], help="lateral range, mm")
    ap.add_argument("--step", type=float, default=0.16, help="pixel size, mm (BRAIN1 default used)")
    args = ap.parse_args()
    names = args.defect or sorted(p.name for p in DEFECTS.iterdir() if p.is_dir())
    for name in names:
        process(DEFECTS / name, args.x, args.z, args.step)


if __name__ == "__main__":
    main()
