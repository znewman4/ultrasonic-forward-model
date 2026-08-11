"""Tests for convention-independent M2 material and wavenumber primitives."""
import numpy as np
import pytest

from src.scattering.elastic_sdh import (
    ElasticMaterial,
    angular_frequency,
    longitudinal_wavenumber,
    shear_wavenumber,
)


MATERIAL = ElasticMaterial(2700.0, 6300.0, 3100.0)


def test_lame_material_identities() -> None:
    assert np.isclose(MATERIAL.mu, MATERIAL.density_kg_m3 * MATERIAL.shear_speed_m_s**2)
    assert np.isclose(
        MATERIAL.lambda_ + 2.0 * MATERIAL.mu,
        MATERIAL.density_kg_m3 * MATERIAL.longitudinal_speed_m_s**2,
    )


def test_frequency_and_wavenumber_identities() -> None:
    frequency = np.array([0.0, 5.0e6])
    omega = angular_frequency(frequency)
    assert np.allclose(omega, 2.0 * np.pi * frequency)
    assert np.allclose(
        longitudinal_wavenumber(frequency, MATERIAL),
        omega / MATERIAL.longitudinal_speed_m_s,
    )
    assert np.allclose(
        shear_wavenumber(frequency, MATERIAL), omega / MATERIAL.shear_speed_m_s
    )


@pytest.mark.parametrize(
    "values",
    [(-1.0, 6300.0, 3100.0), (2700.0, 0.0, 3100.0), (2700.0, 3000.0, 3100.0)],
)
def test_invalid_material_raises(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        ElasticMaterial(*values)


def test_unstable_bulk_modulus_raises() -> None:
    with pytest.raises(ValueError):
        ElasticMaterial(2700.0, 1.05, 1.0)
