"""Print diagnostics for an ultrasonic FMC MATLAB file and save its metadata."""
from pathlib import Path
import sys
import numpy as np
try:
    from .data_loader import load_fmc_mat, save_metadata
except ImportError:  # Supports: python src/analyze_mat.py
    from data_loader import load_fmc_mat, save_metadata


def main(path: Path) -> None:
    data = load_fmc_mat(path)
    y, t = data.time_data, data.metadata.time_s
    tx, rx, a = data.metadata.transmitters, data.metadata.receivers, data.metadata.array
    print(f"FILE: {path.name}")
    print(f"ACQUISITION: {y.shape[1]} A-scans ({len(np.unique(tx))} transmitters x {len(np.unique(rx))} receivers), {y.shape[0]} samples/A-scan")
    dt = float(np.median(np.diff(t)))
    print(f"TIME: {t[0]:.3g} to {t[-1]:.6g} s, dt={dt:.3g} s, sample rate={1/dt/1e6:.3g} MHz")
    print(f"ARRAY: {len(a.element_centres_m)} elements, centre frequency={a.centre_frequency_hz/1e6:.3g} MHz, manufacturer={a.manufacturer}")
    x = a.element_centres_m[:, 0]
    print(f"GEOMETRY: x={x.min():.6g}..{x.max():.6g} m (pitch={np.median(np.diff(x)):.6g} m)")
    if data.metadata.material_wave_speed_m_s is not None:
        print(f"MATERIAL: stored wave speed={data.metadata.material_wave_speed_m_s:.6g} m/s")
    print(f"SIGNAL: range=[{y.min():.6g}, {y.max():.6g}], mean={y.mean():.6g}, rms={np.sqrt(np.mean(y*y)):.6g}, nonzero={np.count_nonzero(y)/y.size:.1%}")
    # A-scan ordering and reciprocity (same transmitter/receiver swapped).
    pairs = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(tx, rx))}
    complete = len(pairs) == len(np.unique(tx)) * len(np.unique(rx))
    errs = []
    for (i, j), k in pairs.items():
        if (j, i) in pairs and (i, j) != (j, i):
            errs.append(np.linalg.norm(y[:, k] - y[:, pairs[(j, i)]]) / (np.linalg.norm(y[:, k]) + 1e-12))
    print(f"FMC ORDER: complete Cartesian grid={complete}; reciprocity median relative error={np.median(errs):.4g}" if errs else f"FMC ORDER: complete Cartesian grid={complete}")
    rms = np.sqrt(np.mean(y*y, axis=0))
    top = np.argsort(rms)[-5:][::-1]
    print("DOMINANT A-SCANS (tx, rx, peak time, peak amplitude, RMS):")
    for k in top:
        q = int(np.argmax(np.abs(y[:, k])))
        print(f"  ({int(tx[k])},{int(rx[k])})  {t[q]:.6g} s  {np.max(np.abs(y[:,k])):.6g}  {rms[k]:.6g}")
    print(f"METADATA: saved to {save_metadata(data)}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat")
    main(p)
