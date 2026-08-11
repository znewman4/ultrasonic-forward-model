"""Tests for channel-indexed experimental FMC reshaping."""
from pathlib import Path

import numpy as np

from src.data_loader import experimental_fmc_array, load_fmc


MAT_PATH = Path(__file__).parents[1] / "data" / "raw" / "5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat"


def test_experimental_channels_are_ordered_using_tx_rx_indices() -> None:
    loaded = load_fmc(MAT_PATH)
    fmc = experimental_fmc_array(loaded)
    assert fmc.shape == (64, 64, 1000)
    for channel in (0, 63, 64, 4095):
        i = loaded.metadata.transmitters[channel] - 1
        j = loaded.metadata.receivers[channel] - 1
        assert np.array_equal(fmc[i, j], loaded.time_data[:, channel])
