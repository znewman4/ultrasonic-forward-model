"""Generate propagation geometry, path, matrix, and pulse-echo diagnostics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import MAT_PATH, REFLECTOR, ROOT, WAVE_SPEED_M_S, save_report
from src.geometry import array_coordinates
from src.propagation import travel_time_matrix


def main() -> None:
    output = ROOT / "figures" / "propagation"
    output.mkdir(parents=True, exist_ok=True)
    elements = array_coordinates(MAT_PATH)
    times = travel_time_matrix(elements, REFLECTOR, WAVE_SPEED_M_S)
    centre = int(np.argmin(np.abs(elements[:, 0])))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(elements[:, 0] * 1e3, elements[:, 1] * 1e3, s=18, label="elements")
    ax.scatter(* (REFLECTOR * 1e3), marker="x", s=90, label="point reflector")
    ax.set(xlabel="x (mm)", ylabel="z (mm)", title="Array and point-reflector geometry")
    ax.invert_yaxis(); ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "geometry.png", dpi=180); plt.close(fig)

    pairs = [(centre, centre, "central pulse-echo"), (0, 63, "left-right"), (15, 44, "asymmetric")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(elements[:, 0] * 1e3, elements[:, 1] * 1e3, s=10, color="black")
    ax.scatter(*(REFLECTOR * 1e3), marker="x", s=90, color="red")
    for tx, rx, label in pairs:
        ax.plot([elements[tx, 0] * 1e3, 0, elements[rx, 0] * 1e3], [0, 40, 0], label=label)
    ax.set(xlabel="x (mm)", ylabel="z (mm)", title="Representative propagation paths")
    ax.invert_yaxis(); ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "representative_paths.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(times * 1e6, origin="lower", aspect="equal")
    fig.colorbar(image, ax=ax, label="Travel time (µs)")
    ax.set(xlabel="Receiver index", ylabel="Transmitter index", title="Point-reflector travel-time matrix")
    fig.tight_layout(); fig.savefig(output / "travel_time_matrix.png", dpi=180); plt.close(fig)

    diagonal = np.diag(times) * 1e6
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(64), diagonal)
    ax.set(xlabel="Element index", ylabel="Pulse-echo time (µs)", title="Pulse-echo travel time", xlim=(0, 63))
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "pulse_echo_time.png", dpi=180); plt.close(fig)

    save_report(output / "report.json", {
        "reflector_m": REFLECTOR.tolist(),
        "matrix_symmetric": bool(np.allclose(times, times.T)),
        "central_element_index_zero_based": centre,
        "central_pulse_echo_us": float(times[centre, centre] * 1e6),
        "minimum_pulse_echo_us": float(np.min(diagonal)),
        "left_edge_pulse_echo_us": float(diagonal[0]),
        "right_edge_pulse_echo_us": float(diagonal[-1]),
        "central_is_shortest": bool(np.argmin(diagonal) in (31, 32)),
    })


if __name__ == "__main__":
    main()
