"""Array-level tests for the idealized M2-2D LL FMC."""
import numpy as np
import pytest

from src.models.elastic_sdh import (
    ElasticSideDrilledHole,
    modal_travel_times,
    simulate_elastic_sdh_fmc,
    simulate_elastic_sdh_modal_fmc,
)
from src.scattering.elastic_sdh import ElasticMaterial


MATERIAL = ElasticMaterial(2700.0, 6300.0, 3100.0)
DEFECT = ElasticSideDrilledHole(0.0, 20.0e-3, 0.5e-3)
ELEMENTS = np.column_stack((np.linspace(-2.0e-3, 2.0e-3, 3), np.zeros(3)))
TIME = np.arange(1024) * 20.0e-9
MODAL_TIME = np.arange(1600) * 20.0e-9


@pytest.fixture(scope="module")
def result():
    return simulate_elastic_sdh_fmc(
        TIME,
        ELEMENTS,
        DEFECT,
        MATERIAL,
        5.0e6,
        0.35e-6,
        16,
        spectrum_relative_cutoff=1.0e-7,
    )


def test_m2_fmc_output_shapes_and_real_data(result) -> None:
    assert result.fmc_ll.shape == (3, 3, TIME.size)
    assert result.frequency_domain_ll.shape == (3, 3, TIME.size // 2 + 1)
    assert result.scattering_kernel_ls.shape == result.frequency_domain_ll.shape
    assert result.frequency_hz.shape == (TIME.size // 2 + 1,)
    assert np.isrealobj(result.fmc_ll)
    assert np.max(np.abs(result.fmc_ll)) > 0.0


def test_m2_fmc_is_reciprocal(result) -> None:
    assert np.allclose(result.fmc_ll, result.fmc_ll.swapaxes(0, 1), atol=2.0e-13)
    assert np.allclose(
        result.frequency_domain_ll,
        result.frequency_domain_ll.swapaxes(0, 1),
        atol=2.0e-10,
    )


def test_m2_peak_lies_near_longitudinal_round_trip(result) -> None:
    trace = result.fmc_ll[1, 1]
    predicted = 2.0 * 20.0e-3 / MATERIAL.longitudinal_speed_m_s
    peak_time = TIME[int(np.argmax(np.abs(trace)))]
    assert abs(peak_time - predicted) < 1.5e-6


def test_m2_records_convergence_and_wrap_diagnostics(result) -> None:
    assert result.metadata["maximum_normalized_boundary_residual"] < 1.0e-8
    assert result.metadata["n_max"] == 16
    assert result.metadata["edge_to_global_peak_ratio"] < 1.0e-5


def test_short_time_record_is_rejected_before_fft_wraparound() -> None:
    with pytest.raises(ValueError, match="too short for wrap-free"):
        simulate_elastic_sdh_fmc(
            np.arange(300) * 20.0e-9,
            ELEMENTS,
            DEFECT,
            MATERIAL,
            5.0e6,
            0.35e-6,
            16,
        )


@pytest.fixture(scope="module")
def modal_result():
    return simulate_elastic_sdh_modal_fmc(
        MODAL_TIME,
        ELEMENTS,
        DEFECT,
        MATERIAL,
        5.0e6,
        0.35e-6,
        24,
        spectrum_relative_cutoff=1.0e-7,
    )


def test_modal_travel_times_use_the_four_speed_combinations() -> None:
    delays = modal_travel_times(ELEMENTS, DEFECT, MATERIAL)
    distance = np.linalg.norm(ELEMENTS - np.array([DEFECT.x_m, DEFECT.z_m]), axis=1)
    assert np.allclose(delays["PP"], distance[:, None] / 6300.0 + distance[None, :] / 6300.0)
    assert np.allclose(delays["PS"], distance[:, None] / 6300.0 + distance[None, :] / 3100.0)
    assert np.allclose(delays["SP"], distance[:, None] / 3100.0 + distance[None, :] / 6300.0)
    assert np.allclose(delays["SS"], distance[:, None] / 3100.0 + distance[None, :] / 3100.0)


def test_modal_fmc_shapes_and_separate_arrivals(modal_result) -> None:
    fmcs = (
        modal_result.fmc_pp,
        modal_result.fmc_ps,
        modal_result.fmc_sp,
        modal_result.fmc_ss,
    )
    assert all(value.shape == (3, 3, MODAL_TIME.size) for value in fmcs)
    assert all(np.isrealobj(value) for value in fmcs)
    assert all(np.max(np.abs(value)) > 0.0 for value in fmcs)
    centre = 1
    peak_times = [
        MODAL_TIME[np.argmax(np.abs(value[centre, centre]))] for value in fmcs
    ]
    assert peak_times[0] < peak_times[1]
    assert np.isclose(peak_times[1], peak_times[2], atol=0.25e-6)
    assert peak_times[2] < peak_times[3]


def test_cross_mode_fmc_obeys_flux_normalized_reciprocity(modal_result) -> None:
    expected_ratio = np.sqrt(
        MATERIAL.longitudinal_speed_m_s / MATERIAL.shear_speed_m_s
    )
    assert np.allclose(
        modal_result.fmc_ps,
        expected_ratio * modal_result.fmc_sp.swapaxes(0, 1),
        rtol=2.0e-10,
        atol=2.0e-13,
    )


def test_modal_fmc_validation_and_no_wrap(modal_result) -> None:
    assert max(modal_result.metadata["maximum_normalized_boundary_residual"].values()) < 1.0e-9
    assert modal_result.metadata["maximum_partial_wave_reciprocity_error"] < 1.0e-12
    assert modal_result.metadata["maximum_partial_wave_energy_balance_error"] < 1.0e-12
    assert max(modal_result.metadata["fft_edge_to_global_peak_ratio"].values()) < 1.0e-5


def test_twenty_microsecond_record_rejects_full_modal_depth() -> None:
    deep_defect = ElasticSideDrilledHole(0.0, 40.0e-3, 0.5e-3)
    with pytest.raises(ValueError, match="too short for wrap-free multimode"):
        simulate_elastic_sdh_modal_fmc(
            np.arange(1000) * 20.0e-9,
            ELEMENTS,
            deep_defect,
            MATERIAL,
            5.0e6,
            0.35e-6,
            24,
        )
