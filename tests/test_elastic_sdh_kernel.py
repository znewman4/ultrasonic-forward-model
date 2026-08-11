"""Validation tests for the exact M2-2D P--SV cavity kernel."""
import numpy as np
from scipy.special import h1vp, jvp

from src.scattering.elastic_sdh import (
    ElasticMaterial,
    bessel_j_values,
    elastic_sdh_broadband_scattering,
    elastic_sdh_scattering,
    hankel1_values,
    harmonic_orders,
    incident_l_coefficients,
    normalized_boundary_traction_residual,
    recommended_n_max,
    solve_harmonic_coefficients,
    total_boundary_traction,
    traction_rphi_p,
    traction_rphi_sv,
    traction_rr_p,
    traction_rr_sv,
)


MATERIAL = ElasticMaterial(2700.0, 6300.0, 3100.0)
FREQUENCY_HZ = 5.0e6
RADIUS_M = 0.5e-3


def test_harmonic_orders_and_jacobi_anger_coefficients() -> None:
    orders = harmonic_orders(3)
    angle = 0.37
    coefficients = incident_l_coefficients(orders, angle)
    assert np.array_equal(orders, np.arange(-3, 4))
    assert np.allclose(coefficients, (1j) ** orders * np.exp(-1j * orders * angle))


def test_second_derivatives_match_scipy_derivatives() -> None:
    orders = harmonic_orders(5)
    argument = 2.7
    regular = bessel_j_values(orders, argument)
    outgoing = hankel1_values(orders, argument)
    assert np.allclose(regular.second_derivative, jvp(orders, argument, 2))
    assert np.allclose(outgoing.second_derivative, h1vp(orders, argument, 2))


def test_traction_operators_match_documented_equations() -> None:
    m = 3
    radius = 0.7e-3
    value = 0.4 - 0.7j
    first = 1200.0 + 80.0j
    second = -2.1e6 + 0.3e6j
    lam = MATERIAL.lambda_
    mu = MATERIAL.mu
    expected_p_rr = (lam + 2 * mu) * second + lam * (
        first / radius - m**2 * value / radius**2
    )
    expected_p_rphi = 2j * m * mu * (first / radius - value / radius**2)
    expected_sv_rphi = mu * (
        -second + first / radius - m**2 * value / radius**2
    )
    assert np.allclose(
        traction_rr_p(value, first, second, m, radius, MATERIAL), expected_p_rr
    )
    assert np.allclose(
        traction_rphi_p(value, first, second, m, radius, MATERIAL), expected_p_rphi
    )
    assert np.allclose(
        traction_rr_sv(value, first, second, m, radius, MATERIAL), expected_p_rphi
    )
    assert np.allclose(
        traction_rphi_sv(value, first, second, m, radius, MATERIAL), expected_sv_rphi
    )


def test_each_harmonic_satisfies_both_traction_equations() -> None:
    for order in range(-8, 9):
        coefficient = incident_l_coefficients(np.array([order]), 0.31)[0]
        solution = solve_harmonic_coefficients(
            FREQUENCY_HZ, RADIUS_M, MATERIAL, order, coefficient
        )
        assert solution.boundary_traction_residual < 2.0e-13


def test_dense_boundary_traction_residual_decreases_with_truncation() -> None:
    angles = np.linspace(-np.pi, np.pi, 1001, endpoint=False)
    residuals = []
    for n_max in (2, 6, 10, 14):
        result = elastic_sdh_scattering(
            FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.27, angles, n_max
        )
        rr, rphi = total_boundary_traction(angles, result, MATERIAL)
        assert rr.shape == angles.shape
        assert rphi.shape == angles.shape
        residuals.append(normalized_boundary_traction_residual(angles, result, MATERIAL))
    assert all(later < earlier for earlier, later in zip(residuals, residuals[1:]))
    assert residuals[-1] < 1.0e-8


def test_harmonic_far_field_converges_at_paper_guide() -> None:
    beta = np.linspace(-np.pi, np.pi, 181, endpoint=False)
    guide = recommended_n_max(FREQUENCY_HZ, RADIUS_M, MATERIAL)
    low = elastic_sdh_scattering(
        FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.1, beta, guide - 4
    )
    guided = elastic_sdh_scattering(
        FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.1, beta, guide
    )
    reference = elastic_sdh_scattering(
        FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.1, beta, guide + 4
    )
    low_error = np.linalg.norm(low.f_pp - reference.f_pp) / np.linalg.norm(reference.f_pp)
    guide_error = np.linalg.norm(guided.f_pp - reference.f_pp) / np.linalg.norm(
        reference.f_pp
    )
    assert guide_error < low_error
    assert guide_error < 1.0e-8


def test_circular_symmetry_under_simultaneous_rotation() -> None:
    beta = np.array([-1.2, -0.1, 0.8, 2.2])
    rotation = 0.73
    first = elastic_sdh_scattering(
        FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.24, beta, 15
    )
    second = elastic_sdh_scattering(
        FREQUENCY_HZ, RADIUS_M, MATERIAL, 0.24 + rotation, beta + rotation, 15
    )
    assert np.allclose(first.f_pp, second.f_pp, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(first.f_ps, second.f_ps, rtol=1.0e-12, atol=1.0e-12)


def test_pp_reciprocity_for_reversed_rays() -> None:
    alpha = -0.42
    beta = 1.13
    forward = elastic_sdh_scattering(
        FREQUENCY_HZ, RADIUS_M, MATERIAL, alpha, np.array([beta]), 15
    )
    reversed_ray = elastic_sdh_scattering(
        FREQUENCY_HZ,
        RADIUS_M,
        MATERIAL,
        beta + np.pi,
        np.array([alpha + np.pi]),
        15,
    )
    assert np.allclose(forward.f_pp, reversed_ray.f_pp, rtol=1.0e-12, atol=1.0e-12)


def test_radius_changes_response_and_small_radius_scattering_tends_to_zero() -> None:
    beta = np.array([0.4, 1.7, np.pi])
    radii = (1.0e-6, 2.0e-6, 0.5e-3)
    responses = [
        elastic_sdh_scattering(
            FREQUENCY_HZ,
            radius,
            MATERIAL,
            0.0,
            beta,
            recommended_n_max(FREQUENCY_HZ, radius, MATERIAL),
        ).f_pp
        for radius in radii
    ]
    assert np.linalg.norm(responses[0]) < np.linalg.norm(responses[1])
    assert np.linalg.norm(responses[1]) < np.linalg.norm(responses[2])
    assert not np.allclose(responses[0], responses[2])


def test_broadband_response_shape_and_zero_frequency() -> None:
    result = elastic_sdh_broadband_scattering(
        np.array([0.0, 4.5e6, 5.0e6]),
        RADIUS_M,
        MATERIAL,
        np.array([0.0, 0.4]),
        np.array([-1.0, 0.0, 1.0]),
        15,
    )
    assert result.f_pp.shape == (3, 2, 3)
    assert result.f_ps.shape == (3, 2, 3)
    assert np.all(result.f_pp[0] == 0.0)
    assert np.all(result.f_ps[0] == 0.0)
    assert np.any(np.abs(result.f_pp[1:]) > 0.0)
