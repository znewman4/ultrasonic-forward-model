"""Compare synthetic and experimental A-scans and TFM images."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from common import ROOT, X_GRID_M, Z_GRID_M, fwhm, peak_location, save_report


def main() -> None:
    output = ROOT / "figures" / "comparison"; output.mkdir(parents=True, exist_ok=True)
    synthetic = np.load(ROOT / "data" / "synthetic" / "point_reflector" / "M0_point_reflector_fmc.npz")
    experimental = np.load(ROOT / "data" / "processed" / "experimental_tfm.npz")
    sfmc, efmc, time = synthetic["fmc"], experimental["fmc"], synthetic["time_s"]
    centre = int(np.argmin(np.abs(synthetic["element_coordinates_m"][:, 0])))
    strace, etrace = sfmc[centre, centre], efmc[centre, centre]
    predicted = float(synthetic["travel_time_s"][centre, centre])
    gate = np.abs(time - predicted) <= 2.0e-6
    gated_indices = np.flatnonzero(gate)
    experimental_echo_index = int(gated_indices[np.argmax(np.abs(hilbert(etrace))[gate])])
    experimental_echo_time = float(time[experimental_echo_index])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time*1e6, strace/np.max(np.abs(strace)), label="synthetic")
    ax.plot(time*1e6, etrace/np.max(np.abs(etrace)), label="experimental", alpha=0.75)
    ax.axvline(predicted*1e6, color="black", ls="--", label="geometric prediction")
    ax.axvline(experimental_echo_time*1e6, color="red", ls=":", label="experimental gated envelope peak")
    ax.set(xlabel="Time (µs)", ylabel="Individually normalized amplitude", title="Central pulse-echo A-scan comparison")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(output / "central_ascans.png", dpi=180); plt.close(fig)

    sim_image, exp_image = synthetic["tfm_image"], experimental["tfm_image"]
    siz, six, sx, sz = peak_location(sim_image); eiz, eix, ex, ez = peak_location(exp_image)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, image, title in ((axes[0], sim_image, "Synthetic"), (axes[1], exp_image, "Experimental (30–50 mm ROI)")):
        ax.imshow(image/np.max(image), extent=[X_GRID_M[0]*1e3, X_GRID_M[-1]*1e3, Z_GRID_M[-1]*1e3, Z_GRID_M[0]*1e3], aspect="auto", cmap="inferno", vmin=0, vmax=1)
        ax.set(title=title, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)"); fig.tight_layout(); fig.savefig(output / "tfm_comparison.png", dpi=180); plt.close(fig)

    def sidelobe_ratio(image: np.ndarray, iz: int, ix: int) -> float:
        mask = np.ones(image.shape, dtype=bool)
        mask[max(0,iz-10):iz+11, max(0,ix-4):ix+5] = False
        return float(np.max(image[mask]) / np.max(image))
    report = {
        "predicted_central_arrival_us": predicted*1e6,
        "experimental_gated_echo_us": experimental_echo_time*1e6,
        "timing_difference_us": (experimental_echo_time-predicted)*1e6,
        "synthetic_tfm_peak_m": [sx, sz], "experimental_tfm_peak_m": [ex, ez],
        "synthetic_lateral_fwhm_mm": (fwhm(X_GRID_M, sim_image[siz]) or 0)*1e3,
        "synthetic_axial_fwhm_mm": (fwhm(Z_GRID_M, sim_image[:,six]) or 0)*1e3,
        "experimental_lateral_fwhm_mm": (fwhm(X_GRID_M, exp_image[eiz]) or 0)*1e3,
        "experimental_axial_fwhm_mm": (fwhm(Z_GRID_M, exp_image[:,eix]) or 0)*1e3,
        "synthetic_relative_outside_peak": sidelobe_ratio(sim_image,siz,six),
        "experimental_relative_outside_peak": sidelobe_ratio(exp_image,eiz,eix),
        "experimental_ambiguity": "The strongest ROI response is a broad lateral band near 40.4 mm, not a clearly isolated point-like indication. It may include a boundary response, and this dataset alone does not support confidently labelling the ROI maximum as the 1 mm hole.",
        "interpretation": {"timing": "Compared using a ±2 µs gate about the geometric prediction.", "location": "Compared on the identical x-z grid and 30–50 mm experimental depth ROI.", "amplitude": "Not a validation target: traces are normalized separately because the model omits realistic amplitude physics."},
        "model_exclusions": ["realistic SDH scattering", "geometric spreading", "array-element directivity", "attenuation", "probe/electronics response", "noise", "mode conversion", "boundary reflections"]
    }
    save_report(output / "report.json", report)


if __name__ == "__main__":
    main()
