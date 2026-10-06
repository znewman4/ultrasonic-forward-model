"""TFM of experiment_001 FMC (BRAIN1 half-matrix .mat), with and without time-zero offset."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert

from common import ROOT, save_report
from src.imaging.tfm import tfm_image

EXPERIMENT = Path(r"C:\Users\js23252\OneDrive - University of Bristol\PhD\Experiments\experiment_001")
C_M_S = 6300.0


def load_hmc(mat_path: Path):
    """Load BRAIN1 exp_data and expand half-matrix capture (tx <= rx) to a full (N, N, Nt) cube."""
    exp = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["exp_data"]
    tx = np.asarray(exp.tx, dtype=int) - 1
    rx = np.asarray(exp.rx, dtype=int) - 1
    traces = np.asarray(exp.time_data, dtype=float)  # (Nt, Ntraces)
    n = int(max(tx.max(), rx.max()) + 1)
    fmc = np.zeros((n, n, traces.shape[0]))
    filled = np.zeros((n, n), dtype=bool)
    for k, (i, j) in enumerate(zip(tx, rx)):
        fmc[i, j] = traces[:, k]
        filled[i, j] = True
    # reciprocity: fill the missing (rx, tx) from (tx, rx)
    missing = ~filled
    fmc[missing] = fmc.transpose(1, 0, 2)[missing]
    assert np.all(filled | filled.T), "pairs are not a complete half matrix"
    elements = np.column_stack((np.asarray(exp.array.el_xc, float), np.asarray(exp.array.el_zc, float)))
    return fmc, np.asarray(exp.time, float), elements, filled


def time_zero_offset(fmc, time_s, windows=((14e-6, 18e-6), (30e-6, 34e-6))):
    """Offset = t1 - (t2 - t1) from the first two back-wall echoes in pulse-echo traces."""
    dt = time_s[1] - time_s[0]
    n = fmc.shape[0]
    peaks = []
    for i in range(n):
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
    return float(np.median(t1 - (t2 - t1))), float(np.median(t1)), float(np.median(t2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defect", default="SDH_01")
    ap.add_argument("--mat", default="first3sdh_FMC.mat")
    ap.add_argument("--z", nargs=2, type=float, default=[0.0, 120.0], help="depth range, mm (120 mm ~ 2.5 back-wall echoes)")
    ap.add_argument("--x", nargs=2, type=float, default=[-25.0, 25.0], help="lateral range, mm")
    ap.add_argument("--step", type=float, default=0.25, help="pixel size, mm")
    args = ap.parse_args()

    fmc, time_s, elements, filled = load_hmc(EXPERIMENT / "defects" / args.defect / args.mat)
    offset, t1, t2 = time_zero_offset(fmc, time_s)
    thickness = (t2 - t1) * C_M_S / 2
    print(f"FMC {fmc.shape}, offset {offset*1e6:.3f} us, back wall {thickness*1e3:.2f} mm")

    x = np.arange(args.x[0], args.x[1] + 1e-9, args.step) * 1e-3
    z = np.arange(args.z[0], args.z[1] + 1e-9, args.step) * 1e-3
    out = ROOT / "figures" / "experiment001" / args.defect
    out.mkdir(parents=True, exist_ok=True)

    images = {}
    for label, shift in (("raw_time", 0.0), ("offset_corrected", offset)):
        img = tfm_image(fmc, time_s - shift, elements, x, z, C_M_S)
        images[label] = img
        np.savez_compressed(out / f"tfm_{label}.npz", image=img, x_m=x, z_m=z, shift_s=shift)

    fig, axes = plt.subplots(1, 2, figsize=(12, 11), sharey=True)
    ref = max(i.max() for i in images.values())
    for ax, (label, img) in zip(axes, images.items()):
        db = 20 * np.log10(np.maximum(img / ref, 1e-6))
        im = ax.imshow(db, extent=[x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3], aspect="equal",
                       cmap="jet", vmin=-40, vmax=0)
        iz, ix = np.unravel_index(np.argmax(img), img.shape)
        ax.set(title=f"{label} (peak x={x[ix]*1e3:.1f}, z={z[iz]*1e3:.1f} mm)", xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, label="dB re max", shrink=0.8)
    fig.savefig(out / "tfm_comparison.png", dpi=150)
    plt.close(fig)

    save_report(out / "report.json", {
        "fmc_shape": list(fmc.shape), "pairs_recorded": int(filled.sum()),
        "time_zero_offset_us": offset * 1e6, "backwall_echo_times_us": [t1 * 1e6, t2 * 1e6],
        "backwall_depth_mm_at_6300": thickness * 1e3, "grid_step_mm": args.step,
        "peaks_mm": {k: [float(x[np.unravel_index(np.argmax(v), v.shape)[1]]*1e3),
                         float(z[np.unravel_index(np.argmax(v), v.shape)[0]]*1e3)] for k, v in images.items()},
    })


if __name__ == "__main__":
    main()
