"""Shared configuration and helpers for reproducible analysis scripts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAT_PATH = ROOT / "data" / "raw" / "5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat"
WAVE_SPEED_M_S = 6300.0
CENTRE_FREQUENCY_HZ = 5.0e6
SIGMA_S = 0.35e-6
REFLECTOR = np.array([0.0, 0.04])
X_GRID_M = np.linspace(-0.025, 0.025, 101)
Z_GRID_M = np.linspace(0.030, 0.050, 101)


def save_report(path: Path, report: dict) -> None:
    """Write an analysis report as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def peak_location(image: np.ndarray) -> tuple[int, int, float, float]:
    """Return peak indices and x/z coordinates on the common imaging grid."""
    iz, ix = np.unravel_index(np.argmax(image), image.shape)
    return iz, ix, float(X_GRID_M[ix]), float(Z_GRID_M[iz])


def fwhm(coordinate: np.ndarray, values: np.ndarray) -> float | None:
    """Estimate full width at half maximum from samples around the global peak."""
    magnitude = np.abs(values)
    if magnitude.size == 0 or np.max(magnitude) <= 0.0:
        return None
    above = magnitude >= 0.5 * np.max(magnitude)
    peak = int(np.argmax(magnitude))
    left = peak
    right = peak
    while left > 0 and above[left - 1]:
        left -= 1
    while right < magnitude.size - 1 and above[right + 1]:
        right += 1
    return float(coordinate[right] - coordinate[left])
