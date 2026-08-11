"""Form a depth-gated TFM image from the experimental FMC acquisition."""
import matplotlib.pyplot as plt
import numpy as np

from common import MAT_PATH, ROOT, WAVE_SPEED_M_S, X_GRID_M, Z_GRID_M, peak_location, save_report
from src.data_loader import experimental_fmc_array, load_fmc
from src.geometry import array_coordinates
from src.imaging.tfm import tfm_image


def main() -> None:
    figure_dir = ROOT / "figures" / "experimental"
    processed_dir = ROOT / "data" / "processed"
    figure_dir.mkdir(parents=True, exist_ok=True); processed_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_fmc(MAT_PATH)
    fmc = experimental_fmc_array(loaded)
    elements = array_coordinates(MAT_PATH)
    image = tfm_image(fmc, loaded.metadata.time_s, elements, X_GRID_M, Z_GRID_M, WAVE_SPEED_M_S)
    iz, ix, peak_x, peak_z = peak_location(image)
    nominal_error = float(np.hypot(peak_x, peak_z - 0.04))
    np.savez_compressed(processed_dir / "experimental_tfm.npz", fmc=fmc, time_s=loaded.metadata.time_s, element_coordinates_m=elements, x_grid_m=X_GRID_M, z_grid_m=Z_GRID_M, tfm_image=image)

    fig, ax = plt.subplots(figsize=(8, 5))
    display = image / np.max(image)
    im = ax.imshow(display, extent=[X_GRID_M[0]*1e3, X_GRID_M[-1]*1e3, Z_GRID_M[-1]*1e3, Z_GRID_M[0]*1e3], aspect="auto", cmap="inferno")
    ax.scatter([0], [40], marker="x", color="cyan", label="filename nominal")
    ax.scatter([peak_x*1e3], [peak_z*1e3], facecolors="none", edgecolors="white", label="ROI maximum")
    fig.colorbar(im, ax=ax, label="Normalized display amplitude"); ax.set(xlabel="x (mm)", ylabel="z (mm)", title="Experimental TFM (30–50 mm depth ROI)"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "tfm_image.png", dpi=180); plt.close(fig)
    save_report(figure_dir / "report.json", {"fmc_shape": list(fmc.shape), "depth_roi_mm": [30.0, 50.0], "strongest_roi_indication_m": [peak_x, peak_z], "offset_from_filename_nominal_mm": nominal_error*1e3, "ground_truth_note": "The approximately 40 mm depth and 1 mm diameter are filename metadata, not stored ground truth."})


if __name__ == "__main__":
    main()
