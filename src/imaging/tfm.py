"""Basic longitudinal-wave Total Focusing Method imaging."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def tfm_image(
    fmc_data: ArrayLike,
    time_s: ArrayLike,
    element_coordinates: ArrayLike,
    x_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
    wave_speed_m_s: float,
    *,
    pixel_chunk_size: int = 128,
) -> FloatArray:
    """Form a TFM image by linearly interpolating and summing signed FMC data.

    For every image pixel, samples are evaluated at the transmitter-pixel-
    receiver travel times. Values outside the recorded time interval contribute
    zero. The coherent signed sum is returned as its absolute magnitude, without
    internal normalization.

    Returns:
        Image with shape ``(len(z_grid_m), len(x_grid_m))``.
    """
    return tfm_image_mode_pair(
        fmc_data,
        time_s,
        element_coordinates,
        x_grid_m,
        z_grid_m,
        wave_speed_m_s,
        wave_speed_m_s,
        pixel_chunk_size=pixel_chunk_size,
    )


def tfm_image_mode_pair(
    fmc_data: ArrayLike,
    time_s: ArrayLike,
    element_coordinates: ArrayLike,
    x_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
    transmit_wave_speed_m_s: float,
    receive_wave_speed_m_s: float,
    *,
    pixel_chunk_size: int = 128,
) -> FloatArray:
    """Form TFM using independent transmit and receive modal speeds.

    For a path labelled ``AB``, pass the speed of incident/transmit mode ``A``
    and outgoing/receive mode ``B``. The delay is
    ``distance_i/c_A + distance_j/c_B``. When both speeds are equal this is
    exactly the legacy :func:`tfm_image` calculation.
    """
    fmc = np.asarray(fmc_data)
    time = np.asarray(time_s, dtype=float)
    elements = np.asarray(element_coordinates, dtype=float)
    x_grid = np.asarray(x_grid_m, dtype=float)
    z_grid = np.asarray(z_grid_m, dtype=float)
    if fmc.ndim != 3 or fmc.shape[0] != fmc.shape[1]:
        raise ValueError("fmc_data must have shape (N, N, Nt)")
    n, _, nt = fmc.shape
    if time.shape != (nt,) or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be finite with shape (Nt,)")
    if nt < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be strictly increasing with at least 2 samples")
    if elements.shape != (n, 2) or not np.all(np.isfinite(elements)):
        raise ValueError("element_coordinates must be finite with shape (N, 2)")
    if x_grid.ndim != 1 or z_grid.ndim != 1 or x_grid.size == 0 or z_grid.size == 0:
        raise ValueError("x_grid_m and z_grid_m must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(x_grid)) or not np.all(np.isfinite(z_grid)):
        raise ValueError("imaging grids must contain finite values")
    try:
        transmit_speed = float(transmit_wave_speed_m_s)
        receive_speed = float(receive_wave_speed_m_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("transmit and receive wave speeds must be positive and finite") from exc
    if (
        not np.isfinite(transmit_speed)
        or transmit_speed <= 0.0
        or not np.isfinite(receive_speed)
        or receive_speed <= 0.0
    ):
        raise ValueError("transmit and receive wave speeds must be positive and finite")
    if not isinstance(pixel_chunk_size, int) or pixel_chunk_size <= 0:
        raise ValueError("pixel_chunk_size must be a positive integer")

    xx, zz = np.meshgrid(x_grid, z_grid)
    pixels = np.column_stack((xx.ravel(), zz.ravel()))
    traces = np.asarray(fmc).reshape(n * n, nt)
    pair_indices = np.arange(n * n)[np.newaxis, :]
    image = np.empty(pixels.shape[0], dtype=float)

    for start in range(0, pixels.shape[0], pixel_chunk_size):
        stop = min(start + pixel_chunk_size, pixels.shape[0])
        distances = np.linalg.norm(
            pixels[start:stop, np.newaxis, :] - elements[np.newaxis, :, :], axis=2
        )
        delays = (
            distances[:, :, np.newaxis] / transmit_speed
            + distances[:, np.newaxis, :] / receive_speed
        ).reshape(stop - start, n * n)
        upper = np.searchsorted(time, delays, side="right")
        valid = (delays >= time[0]) & (delays <= time[-1])
        upper = np.clip(upper, 1, nt - 1)
        lower = upper - 1
        t0 = time[lower]
        weight = (delays - t0) / (time[upper] - t0)
        sampled = (
            traces[pair_indices, lower] * (1.0 - weight)
            + traces[pair_indices, upper] * weight
        )
        sampled[~valid] = 0.0
        image[start:stop] = np.abs(np.sum(sampled, axis=1))
    return image.reshape(z_grid.size, x_grid.size)
