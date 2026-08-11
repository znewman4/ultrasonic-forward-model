"""Generate synthetic FMC data and a point-reflector TFM image."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from common import CENTRE_FREQUENCY_HZ, MAT_PATH, REFLECTOR, ROOT, SIGMA_S, WAVE_SPEED_M_S, X_GRID_M, Z_GRID_M, peak_location, save_report
from src.data_loader import load_fmc
from src.forward_model import simulate_point_reflector_ascan, simulate_point_reflector_fmc
from src.geometry import array_coordinates
from src.imaging.tfm import tfm_image


def main() -> None:
    figure_dir = ROOT / "figures" / "synthetic"
    data_dir = ROOT / "data" / "synthetic" / "point_reflector"
    figure_dir.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_fmc(MAT_PATH)
    time = loaded.metadata.time_s
    elements = array_coordinates(MAT_PATH)
    centre = int(np.argmin(np.abs(elements[:, 0])))
    trace, arrival = simulate_point_reflector_ascan(time, elements[centre], elements[centre], REFLECTOR, WAVE_SPEED_M_S, CENTRE_FREQUENCY_HZ, SIGMA_S)
    observed = float(time[np.argmax(np.abs(hilbert(trace)))])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time * 1e6, trace); ax.axvline(arrival * 1e6, color="red", ls="--", label="predicted")
    ax.axvline(observed * 1e6, color="black", ls=":", label="envelope maximum")
    ax.set(xlabel="Time (µs)", ylabel="Amplitude", title="Synthetic central pulse-echo A-scan")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(figure_dir / "central_ascan.png", dpi=180); plt.close(fig)

    fmc, travel_times = simulate_point_reflector_fmc(time, elements, REFLECTOR, WAVE_SPEED_M_S, CENTRE_FREQUENCY_HZ, SIGMA_S)
    image = tfm_image(fmc, time, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)
    iz, ix, peak_x, peak_z = peak_location(image)
    error = float(np.hypot(peak_x - REFLECTOR[0], peak_z - REFLECTOR[1]))
    np.savez_compressed(data_dir / "M0_point_reflector_fmc.npz", fmc=fmc, time_s=time, travel_time_s=travel_times, element_coordinates_m=elements, reflector_m=REFLECTOR, x_grid_m=X_GRID_M, z_grid_m=Z_GRID_M, tfm_image=image, model_id=np.array("M0_point_reflector"))

    fig, ax = plt.subplots(figsize=(8, 5))
    display = image / np.max(image)
    im = ax.imshow(display, extent=[X_GRID_M[0]*1e3, X_GRID_M[-1]*1e3, Z_GRID_M[-1]*1e3, Z_GRID_M[0]*1e3], aspect="auto", cmap="inferno")
    ax.scatter([0], [40], marker="x", color="cyan", label="true reflector")
    ax.scatter([peak_x*1e3], [peak_z*1e3], facecolors="none", edgecolors="white", label="TFM maximum")
    fig.colorbar(im, ax=ax, label="Normalized display amplitude"); ax.set(xlabel="x (mm)", ylabel="z (mm)", title="Synthetic point-reflector TFM"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "tfm_image.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(X_GRID_M*1e3, image[iz]); axes[0].set(xlabel="x (mm)", title="Lateral slice")
    axes[1].plot(image[:, ix], Z_GRID_M*1e3); axes[1].invert_yaxis(); axes[1].set(ylabel="z (mm)", title="Axial slice")
    for ax in axes: ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(figure_dir / "tfm_slices.png", dpi=180); plt.close(fig)
    save_report(figure_dir / "report.json", {"fmc_shape": list(fmc.shape), "predicted_arrival_us": arrival*1e6, "envelope_peak_us": observed*1e6, "tfm_peak_m": [peak_x, peak_z], "localisation_error_mm": error*1e3})


if __name__ == "__main__":
    main()
