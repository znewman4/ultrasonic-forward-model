"""Loader and metadata writer for experimental ultrasonic FMC MATLAB files."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class ArrayMetadata:
    manufacturer: str
    centre_frequency_hz: float
    element_centres_m: np.ndarray  # (n_elements, 3), columns x/y/z
    element_corners_m: np.ndarray  # (n_elements, 2, 3), endpoints x1/y1/z1 and x2/y2/z2


@dataclass(frozen=True)
class FMCMetadata:
    source_file: str
    material_wave_speed_m_s: float | None
    time_s: np.ndarray
    transmitters: np.ndarray
    receivers: np.ndarray
    array: ArrayMetadata


@dataclass(frozen=True)
class FMCData:
    """Experimental FMC data; A-scans are indexed as ``time_data[:, channel]``."""

    time_data: np.ndarray
    metadata: FMCMetadata


def load_fmc(path: str | Path) -> FMCData:
    """Load the expected ``exp_data`` MATLAB structure into clean Python objects."""
    path = Path(path)
    contents = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "exp_data" not in contents:
        raise ValueError(f"{path} does not contain an 'exp_data' MATLAB variable")
    exp = contents["exp_data"]
    required = ("time_data", "tx", "rx", "time", "array")
    missing = [name for name in required if not hasattr(exp, name)]
    if missing:
        raise ValueError(f"exp_data is missing fields: {', '.join(missing)}")
    array = exp.array
    centres = np.column_stack((array.el_xc, array.el_yc, array.el_zc)).astype(float)
    corners = np.stack(
        (
            np.column_stack((array.el_x1, array.el_y1, array.el_z1)),
            np.column_stack((array.el_x2, array.el_y2, array.el_z2)),
        ), axis=1,
    ).astype(float)
    speed = None
    if hasattr(exp, "material") and hasattr(exp.material, "vel_spherical_harmonic_coeffs"):
        speed = float(exp.material.vel_spherical_harmonic_coeffs)
    metadata = FMCMetadata(
        source_file=str(path),
        material_wave_speed_m_s=speed,
        time_s=np.asarray(exp.time, dtype=float),
        transmitters=np.asarray(exp.tx, dtype=int),
        receivers=np.asarray(exp.rx, dtype=int),
        array=ArrayMetadata(
            manufacturer=str(array.manufacturer),
            centre_frequency_hz=float(array.centre_freq),
            element_centres_m=centres,
            element_corners_m=corners,
        ),
    )
    return FMCData(time_data=np.asarray(exp.time_data, dtype=float), metadata=metadata)


# Backward-compatible name used by the diagnostics script and any existing callers.
load_fmc_mat = load_fmc


def metadata_dict(data: FMCData) -> dict[str, Any]:
    """Return JSON-ready metadata, intentionally excluding the A-scan matrix."""
    result = asdict(data.metadata)
    result["time_s"] = result["time_s"].tolist()
    result["transmitters"] = result["transmitters"].tolist()
    result["receivers"] = result["receivers"].tolist()
    result["array"]["element_centres_m"] = result["array"]["element_centres_m"].tolist()
    result["array"]["element_corners_m"] = result["array"]["element_corners_m"].tolist()
    result["time_data_shape"] = list(data.time_data.shape)
    return result


def save_metadata(data: FMCData, directory: str | Path = "data/metadata") -> Path:
    """Write metadata as a readable JSON sidecar and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{Path(data.metadata.source_file).stem}.json"
    target.write_text(json.dumps(metadata_dict(data), indent=2) + "\n", encoding="utf-8")
    return target


def experimental_fmc_array(data: FMCData) -> np.ndarray:
    """Order channel A-scans by stored tx/rx indices as ``(N, N, Nt)``."""
    tx = np.asarray(data.metadata.transmitters, dtype=int)
    rx = np.asarray(data.metadata.receivers, dtype=int)
    traces = np.asarray(data.time_data, dtype=float)
    if traces.ndim != 2 or traces.shape[1] != tx.size or tx.shape != rx.shape:
        raise ValueError("Experimental time data and tx/rx channel arrays disagree")
    labels = np.unique(np.concatenate((tx, rx)))
    label_to_index = {int(label): index for index, label in enumerate(labels)}
    n = labels.size
    result = np.empty((n, n, traces.shape[0]), dtype=float)
    filled = np.zeros((n, n), dtype=bool)
    for channel, (transmitter, receiver) in enumerate(zip(tx, rx)):
        i = label_to_index[int(transmitter)]
        j = label_to_index[int(receiver)]
        if filled[i, j]:
            raise ValueError(f"Duplicate tx/rx channel ({transmitter}, {receiver})")
        result[i, j] = traces[:, channel]
        filled[i, j] = True
    if not np.all(filled):
        raise ValueError("Experimental tx/rx channels do not form a complete FMC grid")
    return result
