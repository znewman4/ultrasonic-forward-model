r"""M2-2D: exact plane-strain P-SV scattering by a circular cavity.

Conventions
-----------
Physical phasors use the Bostrom--Bovik (2003) convention
``exp(-i*omega*t)``. Outgoing cylindrical components are therefore SciPy's
``H_m^(1)``. The SDH cross-section is the x-y plane, its cylinder axis is +z,
and polar angle ``phi`` increases counter-clockwise from +x toward +y. In the
repository's x-z inspection arrays, paper-coordinate y is repository-coordinate
z. Incident angle ``alpha`` and observation angle ``beta`` use the same angular
orientation.

The plane-strain potentials are

``u = grad(Phi) + curl(Psi*e_z)``,

so ``u_r = Phi_,r + Psi_,phi/r`` and
``u_phi = Phi_,phi/r - Psi_,r``. Potential coefficients in this module are
relative to a unit incident P potential; they are not displacement-normalised
or energy-flux-normalised amplitudes.

The unit incident P potential is

``Phi_inc = exp(i*k_p*r*cos(phi-alpha))``

and has regular coefficients ``C_m=i**m*exp(-i*m*alpha)``. Scattered potentials
are expanded in ``H_m^(1)``. The returned far-field functions are defined only
by

``Phi_sc ~ sqrt(2/(pi*k_p*r))*exp(i*(k_p*r-pi/4))*F_PP(beta)``

and the analogous expression with ``Psi_sc``, ``k_s`` and ``F_PS``. Thus
``F_PP=sum A_m exp(i*m*(beta-pi/2))`` and similarly for ``F_PS``. They are
potential-amplitude angular functions, not the full three-dimensional or
finite-probe T-matrix amplitudes of Bostrom and Bovik (2003).

The FFT-coupled array model uses NumPy's inverse-transform convention
``x(t)=sum X(omega)*exp(+i*omega*t)`` and causal phase ``exp(-i*omega*tau)``.
It therefore conjugates the paper-convention scattering amplitude at the FFT
interface. This is a representation conversion, not a different physical
solution.
Earlier M0/M1 code constructs time shifts directly and does not expose a phasor
or Hankel radiation convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import h1vp, hankel1, jv, jvp


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
ILL_CONDITIONED_THRESHOLD = 1.0e12


class ReferenceEquationsUnavailableError(NotImplementedError):
    """Reserved for unsupported full-3D or finite-probe reference operations."""


@dataclass(frozen=True)
class ElasticMaterial:
    """Homogeneous, stable, isotropic elastic material."""

    density_kg_m3: float
    longitudinal_speed_m_s: float
    shear_speed_m_s: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.density_kg_m3, self.longitudinal_speed_m_s, self.shear_speed_m_s),
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("density and wave speeds must be positive finite values")
        if self.longitudinal_speed_m_s <= self.shear_speed_m_s:
            raise ValueError("longitudinal_speed_m_s must exceed shear_speed_m_s")
        if self.mu <= 0.0 or self.bulk_modulus <= 0.0:
            raise ValueError("wave speeds do not define a stable isotropic material")

    @property
    def mu(self) -> float:
        """Shear Lamé modulus ``mu=rho*c_s**2`` in pascals."""
        return self.density_kg_m3 * self.shear_speed_m_s**2

    @property
    def lambda_(self) -> float:
        """First Lamé modulus ``lambda=rho*c_p**2-2*mu`` in pascals."""
        return self.density_kg_m3 * self.longitudinal_speed_m_s**2 - 2.0 * self.mu

    @property
    def lame_lambda(self) -> float:
        """Readable alias for :attr:`lambda_`."""
        return self.lambda_

    @property
    def lambda_pa(self) -> float:
        """Unit-explicit alias for :attr:`lambda_`."""
        return self.lambda_

    @property
    def mu_pa(self) -> float:
        """Unit-explicit alias for :attr:`mu`."""
        return self.mu

    @property
    def bulk_modulus(self) -> float:
        """Three-dimensional bulk modulus ``lambda+2*mu/3`` in pascals."""
        return self.lambda_ + 2.0 * self.mu / 3.0


@dataclass(frozen=True)
class HarmonicRadialValues:
    """Cylindrical function and derivatives with respect to its argument."""

    value: ComplexArray
    first_derivative: ComplexArray
    second_derivative: ComplexArray


@dataclass(frozen=True)
class HarmonicSolution:
    """Outgoing P/SV coefficients and diagnostics for one harmonic."""

    longitudinal_coefficient: complex
    shear_coefficient: complex
    condition_number: float
    boundary_traction_residual: float


@dataclass(frozen=True)
class ElasticSDHScatteringResult:
    """Scalar-frequency M2-2D P-incidence scattering result."""

    s_ll: ComplexArray
    s_ls: ComplexArray
    longitudinal_coefficients: ComplexArray
    shear_coefficients: ComplexArray
    orders: NDArray[np.int64]
    convergence_metadata: dict[str, Any]
    traction_residual_metadata: dict[str, Any]
    frequency_hz: float
    radius_m: float
    incident_angle_rad: float
    scatter_angle_rad: FloatArray

    @property
    def f_pp(self) -> ComplexArray:
        """Far-field P-potential angular function defined in the module docstring."""
        return self.s_ll

    @property
    def f_ps(self) -> ComplexArray:
        """Far-field SV-potential angular function for incident P potential."""
        return self.s_ls


@dataclass(frozen=True)
class ElasticSDHBroadbandResult:
    """Frequency/incident-angle/outgoing-angle M2-2D response arrays."""

    frequency_hz: FloatArray
    incident_angle_rad: FloatArray
    scatter_angle_rad: FloatArray
    f_pp: ComplexArray
    f_ps: ComplexArray
    n_max_by_frequency: NDArray[np.int64]
    maximum_normalized_boundary_residual: float


def _nonnegative_frequencies(frequency_hz: ArrayLike) -> FloatArray:
    frequencies = np.asarray(frequency_hz, dtype=float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequency_hz must contain finite non-negative values")
    return frequencies


def angular_frequency(frequency_hz: ArrayLike) -> FloatArray:
    """Return ``omega=2*pi*f`` in radians per second."""
    return 2.0 * np.pi * _nonnegative_frequencies(frequency_hz)


def longitudinal_wavenumber(
    frequency_hz: ArrayLike, material: ElasticMaterial
) -> FloatArray:
    """Return P wavenumber ``k_p=omega/c_p`` in rad/m."""
    if not isinstance(material, ElasticMaterial):
        raise TypeError("material must be an ElasticMaterial")
    return angular_frequency(frequency_hz) / material.longitudinal_speed_m_s


def shear_wavenumber(
    frequency_hz: ArrayLike, material: ElasticMaterial
) -> FloatArray:
    """Return SV wavenumber ``k_s=omega/c_s`` in rad/m."""
    if not isinstance(material, ElasticMaterial):
        raise TypeError("material must be an ElasticMaterial")
    return angular_frequency(frequency_hz) / material.shear_speed_m_s


def recommended_n_max(
    frequency_hz: float, radius_m: float, material: ElasticMaterial
) -> int:
    """Return the paper's practical guide ``floor(k_s*a)+10``."""
    frequency = float(frequency_hz)
    radius = float(radius_m)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    return int(np.floor(float(shear_wavenumber(frequency, material)) * radius) + 10)


def harmonic_orders(n_max: int) -> NDArray[np.int64]:
    """Return integer harmonic orders ``[-n_max,...,n_max]``."""
    if (
        not isinstance(n_max, (int, np.integer))
        or isinstance(n_max, (bool, np.bool_))
        or n_max < 0
    ):
        raise ValueError("n_max must be a non-negative integer")
    return np.arange(-n_max, n_max + 1, dtype=np.int64)


def incident_l_coefficients(
    harmonic_order: ArrayLike, incident_angle_rad: float
) -> ComplexArray:
    """Return ``C_m=i**m exp(-i*m*alpha)`` for the unit P potential."""
    orders = np.asarray(harmonic_order)
    if not np.issubdtype(orders.dtype, np.integer):
        raise TypeError("harmonic_order must contain integers")
    angle = float(incident_angle_rad)
    if not np.isfinite(angle):
        raise ValueError("incident_angle_rad must be finite")
    return np.asarray((1j) ** orders * np.exp(-1j * orders * angle), dtype=complex)


def _second_derivative_from_bessel_equation(
    orders: NDArray[np.int64], argument: float, value: ComplexArray, first: ComplexArray
) -> ComplexArray:
    if not np.isfinite(argument) or argument <= 0.0:
        raise ValueError("argument must be a positive finite scalar")
    return -(first / argument) - (1.0 - (orders / argument) ** 2) * value


def bessel_j_values(harmonic_order: ArrayLike, argument: float) -> HarmonicRadialValues:
    """Return ``J_m``, ``J_m'`` and ``J_m''`` (argument derivatives)."""
    orders = np.asarray(harmonic_order)
    if not np.issubdtype(orders.dtype, np.integer):
        raise TypeError("harmonic_order must contain integers")
    value = np.asarray(jv(orders, argument), dtype=complex)
    first = np.asarray(jvp(orders, argument, 1), dtype=complex)
    second = _second_derivative_from_bessel_equation(orders, argument, value, first)
    return HarmonicRadialValues(value, first, second)


def hankel1_values(harmonic_order: ArrayLike, argument: float) -> HarmonicRadialValues:
    """Return ``H_m^(1)`` and its first two argument derivatives."""
    orders = np.asarray(harmonic_order)
    if not np.issubdtype(orders.dtype, np.integer):
        raise TypeError("harmonic_order must contain integers")
    value = np.asarray(hankel1(orders, argument), dtype=complex)
    first = np.asarray(h1vp(orders, argument, 1), dtype=complex)
    second = _second_derivative_from_bessel_equation(orders, argument, value, first)
    return HarmonicRadialValues(value, first, second)


def _validate_traction_inputs(
    harmonic_order: int, radius_m: float, material: ElasticMaterial
) -> tuple[int, float]:
    if not isinstance(material, ElasticMaterial):
        raise TypeError("material must be an ElasticMaterial")
    if not isinstance(harmonic_order, (int, np.integer)):
        raise TypeError("harmonic_order must be an integer")
    radius = float(radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    return int(harmonic_order), radius


def traction_rr_l(
    radial_value: complex,
    radial_first_derivative_m: complex,
    radial_second_derivative_m2: complex,
    harmonic_order: int,
    radius_m: float,
    material: ElasticMaterial,
) -> complex:
    r"""Return P-harmonic radial traction ``sigma_rr``.

    For ``Phi=F(r)exp(i*m*phi)``, isotropic stress gives
    ``sigma_rr=(lambda+2mu)F''+lambda(F'/r-m^2 F/r^2)``.
    ``F`` may be ``J_m(k_p r)`` or ``H_m^(1)(k_p r)``; the supplied
    derivatives must be with respect to radius.
    """
    m, radius = _validate_traction_inputs(harmonic_order, radius_m, material)
    return complex(
        (material.lambda_ + 2.0 * material.mu) * radial_second_derivative_m2
        + material.lambda_
        * (radial_first_derivative_m / radius - m**2 * radial_value / radius**2)
    )


def traction_rtheta_l(
    radial_value: complex,
    radial_first_derivative_m: complex,
    radial_second_derivative_m2: complex,
    harmonic_order: int,
    radius_m: float,
    material: ElasticMaterial,
) -> complex:
    r"""Return P-harmonic tangential traction ``sigma_rphi``.

    ``sigma_rphi=2*i*m*mu*(F'/r-F/r^2)`` for
    ``Phi=F(r)exp(i*m*phi)``. The second derivative is accepted so all four
    traction operators share one explicit interface.
    """
    del radial_second_derivative_m2
    m, radius = _validate_traction_inputs(harmonic_order, radius_m, material)
    return complex(
        2j * m * material.mu * (radial_first_derivative_m / radius - radial_value / radius**2)
    )


def traction_rr_s(
    radial_value: complex,
    radial_first_derivative_m: complex,
    radial_second_derivative_m2: complex,
    harmonic_order: int,
    radius_m: float,
    material: ElasticMaterial,
) -> complex:
    r"""Return SV-harmonic radial traction ``sigma_rr``.

    For ``Psi=F(r)exp(i*m*phi)`` and the documented curl convention,
    ``sigma_rr=2*i*m*mu*(F'/r-F/r^2)``.
    """
    del radial_second_derivative_m2
    m, radius = _validate_traction_inputs(harmonic_order, radius_m, material)
    return complex(
        2j * m * material.mu * (radial_first_derivative_m / radius - radial_value / radius**2)
    )


def traction_rtheta_s(
    radial_value: complex,
    radial_first_derivative_m: complex,
    radial_second_derivative_m2: complex,
    harmonic_order: int,
    radius_m: float,
    material: ElasticMaterial,
) -> complex:
    r"""Return SV-harmonic tangential traction ``sigma_rphi``.

    ``sigma_rphi=mu*(-F''+F'/r-m^2 F/r^2)`` for
    ``Psi=F(r)exp(i*m*phi)``.
    """
    m, radius = _validate_traction_inputs(harmonic_order, radius_m, material)
    return complex(
        material.mu
        * (
            -radial_second_derivative_m2
            + radial_first_derivative_m / radius
            - m**2 * radial_value / radius**2
        )
    )


# P/SV terminology aliases used by the M2 convention document.
traction_rr_p = traction_rr_l
traction_rphi_p = traction_rtheta_l
traction_rr_sv = traction_rr_s
traction_rphi_sv = traction_rtheta_s


def _radial_derivatives(values: HarmonicRadialValues, wavenumber: float, index: int = 0):
    return (
        complex(np.ravel(values.value)[index]),
        complex(wavenumber * np.ravel(values.first_derivative)[index]),
        complex(wavenumber**2 * np.ravel(values.second_derivative)[index]),
    )


def solve_harmonic_coefficients(
    frequency_hz: float,
    radius_m: float,
    material: ElasticMaterial,
    harmonic_order: int,
    incident_coefficient: complex,
) -> HarmonicSolution:
    """Solve the traction-free 2x2 system for one incident-P harmonic."""
    frequency = float(frequency_hz)
    radius = float(radius_m)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    _validate_traction_inputs(harmonic_order, radius, material)
    coefficient = complex(incident_coefficient)
    if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
        raise ValueError("incident_coefficient must be finite")
    m = int(harmonic_order)
    kp = float(longitudinal_wavenumber(frequency, material))
    ks = float(shear_wavenumber(frequency, material))
    regular_p = _radial_derivatives(bessel_j_values(np.array([m]), kp * radius), kp)
    outgoing_p = _radial_derivatives(hankel1_values(np.array([m]), kp * radius), kp)
    outgoing_sv = _radial_derivatives(hankel1_values(np.array([m]), ks * radius), ks)

    incident_traction = coefficient * np.array(
        [
            traction_rr_p(*regular_p, m, radius, material),
            traction_rphi_p(*regular_p, m, radius, material),
        ],
        dtype=complex,
    )
    matrix = np.array(
        [
            [
                traction_rr_p(*outgoing_p, m, radius, material),
                traction_rr_sv(*outgoing_sv, m, radius, material),
            ],
            [
                traction_rphi_p(*outgoing_p, m, radius, material),
                traction_rphi_sv(*outgoing_sv, m, radius, material),
            ],
        ],
        dtype=complex,
    )
    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise np.linalg.LinAlgError(f"non-finite harmonic matrix condition number for m={m}")
    if condition_number > ILL_CONDITIONED_THRESHOLD:
        warnings.warn(
            f"M2 harmonic m={m} is ill-conditioned (cond={condition_number:.3e})",
            RuntimeWarning,
            stacklevel=2,
        )
    solution = np.linalg.solve(matrix, -incident_traction)
    residual_vector = matrix @ solution + incident_traction
    residual = float(
        np.linalg.norm(residual_vector)
        / max(np.linalg.norm(incident_traction), np.finfo(float).tiny)
    )
    return HarmonicSolution(
        longitudinal_coefficient=complex(solution[0]),
        shear_coefficient=complex(solution[1]),
        condition_number=condition_number,
        boundary_traction_residual=residual,
    )


def _far_field_function(
    coefficients: ComplexArray, orders: NDArray[np.int64], scatter_angles: FloatArray
) -> ComplexArray:
    phase = np.exp(
        1j * orders[:, np.newaxis] * (scatter_angles.ravel()[np.newaxis, :] - np.pi / 2.0)
    )
    values = coefficients @ phase
    return values.reshape(scatter_angles.shape)


def incident_p_potential(
    radius_m: ArrayLike,
    angle_rad: ArrayLike,
    frequency_hz: float,
    material: ElasticMaterial,
    incident_angle_rad: float,
) -> ComplexArray:
    """Evaluate the analytic unit incident P potential."""
    radius = np.asarray(radius_m, dtype=float)
    angle = np.asarray(angle_rad, dtype=float)
    kp = float(longitudinal_wavenumber(frequency_hz, material))
    return np.asarray(
        np.exp(1j * kp * radius * np.cos(angle - incident_angle_rad)), dtype=complex
    )


def scattered_p_potential(
    radius_m: ArrayLike, angle_rad: ArrayLike, result: ElasticSDHScatteringResult, material: ElasticMaterial
) -> ComplexArray:
    """Reconstruct the scattered P potential outside the cavity."""
    radius, angle = np.broadcast_arrays(
        np.asarray(radius_m, dtype=float), np.asarray(angle_rad, dtype=float)
    )
    if np.any(radius < result.radius_m):
        raise ValueError("scattered field is defined only for r >= cavity radius")
    kp = float(longitudinal_wavenumber(result.frequency_hz, material))
    terms = result.longitudinal_coefficients[:, None] * hankel1(
        result.orders[:, None], kp * radius.ravel()[None, :]
    ) * np.exp(1j * result.orders[:, None] * angle.ravel()[None, :])
    return np.sum(terms, axis=0).reshape(radius.shape)


def scattered_sv_potential(
    radius_m: ArrayLike, angle_rad: ArrayLike, result: ElasticSDHScatteringResult, material: ElasticMaterial
) -> ComplexArray:
    """Reconstruct the scattered SV potential outside the cavity."""
    radius, angle = np.broadcast_arrays(
        np.asarray(radius_m, dtype=float), np.asarray(angle_rad, dtype=float)
    )
    if np.any(radius < result.radius_m):
        raise ValueError("scattered field is defined only for r >= cavity radius")
    ks = float(shear_wavenumber(result.frequency_hz, material))
    terms = result.shear_coefficients[:, None] * hankel1(
        result.orders[:, None], ks * radius.ravel()[None, :]
    ) * np.exp(1j * result.orders[:, None] * angle.ravel()[None, :])
    return np.sum(terms, axis=0).reshape(radius.shape)


def total_displacement(
    radius_m: ArrayLike, angle_rad: ArrayLike, result: ElasticSDHScatteringResult, material: ElasticMaterial
) -> tuple[ComplexArray, ComplexArray]:
    """Return total radial and azimuthal displacement components."""
    radius, angle = np.broadcast_arrays(
        np.asarray(radius_m, dtype=float), np.asarray(angle_rad, dtype=float)
    )
    if np.any(radius < result.radius_m) or np.any(radius <= 0.0):
        raise ValueError("total displacement requires r >= cavity radius > 0")
    flat_r = radius.ravel()
    flat_angle = angle.ravel()
    kp = float(longitudinal_wavenumber(result.frequency_hz, material))
    ks = float(shear_wavenumber(result.frequency_hz, material))
    delta = flat_angle - result.incident_angle_rad
    incident = np.exp(1j * kp * flat_r * np.cos(delta))
    ur = 1j * kp * np.cos(delta) * incident
    uphi = -1j * kp * np.sin(delta) * incident
    exp_m = np.exp(1j * result.orders[:, None] * flat_angle[None, :])
    hp = hankel1(result.orders[:, None], kp * flat_r[None, :])
    hp_prime = h1vp(result.orders[:, None], kp * flat_r[None, :], 1)
    hs = hankel1(result.orders[:, None], ks * flat_r[None, :])
    hs_prime = h1vp(result.orders[:, None], ks * flat_r[None, :], 1)
    ur += np.sum(result.longitudinal_coefficients[:, None] * kp * hp_prime * exp_m, axis=0)
    uphi += np.sum(
        result.longitudinal_coefficients[:, None]
        * (1j * result.orders[:, None] / flat_r[None, :])
        * hp
        * exp_m,
        axis=0,
    )
    ur += np.sum(
        result.shear_coefficients[:, None]
        * (1j * result.orders[:, None] / flat_r[None, :])
        * hs
        * exp_m,
        axis=0,
    )
    uphi -= np.sum(result.shear_coefficients[:, None] * ks * hs_prime * exp_m, axis=0)
    return ur.reshape(radius.shape), uphi.reshape(radius.shape)


def total_boundary_traction(
    angle_rad: ArrayLike,
    result: ElasticSDHScatteringResult,
    material: ElasticMaterial,
) -> tuple[ComplexArray, ComplexArray]:
    """Reconstruct exact incident plus truncated scattered traction at ``r=a``."""
    angle = np.asarray(angle_rad, dtype=float)
    if not np.all(np.isfinite(angle)):
        raise ValueError("angle_rad must be finite")
    frequency = result.frequency_hz
    radius = result.radius_m
    kp = float(longitudinal_wavenumber(frequency, material))
    ks = float(shear_wavenumber(frequency, material))
    delta = angle.ravel() - result.incident_angle_rad
    incident = np.exp(1j * kp * radius * np.cos(delta))
    rr = -kp**2 * (
        material.lambda_ + 2.0 * material.mu * np.cos(delta) ** 2
    ) * incident
    rphi = 2.0 * material.mu * kp**2 * np.sin(delta) * np.cos(delta) * incident
    exp_m = np.exp(1j * result.orders[:, None] * angle.ravel()[None, :])
    for index, m in enumerate(result.orders):
        p = _radial_derivatives(hankel1_values(np.array([m]), kp * radius), kp)
        sv = _radial_derivatives(hankel1_values(np.array([m]), ks * radius), ks)
        rr += (
            result.longitudinal_coefficients[index]
            * traction_rr_p(*p, int(m), radius, material)
            + result.shear_coefficients[index]
            * traction_rr_sv(*sv, int(m), radius, material)
        ) * exp_m[index]
        rphi += (
            result.longitudinal_coefficients[index]
            * traction_rphi_p(*p, int(m), radius, material)
            + result.shear_coefficients[index]
            * traction_rphi_sv(*sv, int(m), radius, material)
        ) * exp_m[index]
    return rr.reshape(angle.shape), rphi.reshape(angle.shape)


def normalized_boundary_traction_residual(
    angle_rad: ArrayLike, result: ElasticSDHScatteringResult, material: ElasticMaterial
) -> float:
    """Return maximum total traction norm divided by maximum incident norm."""
    angle = np.asarray(angle_rad, dtype=float)
    rr, rphi = total_boundary_traction(angle, result, material)
    kp = float(longitudinal_wavenumber(result.frequency_hz, material))
    delta = angle - result.incident_angle_rad
    incident = np.exp(1j * kp * result.radius_m * np.cos(delta))
    incident_rr = -kp**2 * (
        material.lambda_ + 2.0 * material.mu * np.cos(delta) ** 2
    ) * incident
    incident_rphi = (
        2.0 * material.mu * kp**2 * np.sin(delta) * np.cos(delta) * incident
    )
    numerator = np.max(np.sqrt(np.abs(rr) ** 2 + np.abs(rphi) ** 2))
    denominator = np.max(
        np.sqrt(np.abs(incident_rr) ** 2 + np.abs(incident_rphi) ** 2)
    )
    return float(numerator / denominator)


def elastic_sdh_scattering(
    frequency_hz: float,
    radius_m: float,
    material: ElasticMaterial,
    incident_angle_rad: float,
    scatter_angle_rad: ArrayLike,
    n_max: int,
) -> ElasticSDHScatteringResult:
    """Return exact truncated M2-2D P-to-P and P-to-SV potential scattering."""
    frequency = float(frequency_hz)
    radius = float(radius_m)
    angle = float(incident_angle_rad)
    scatter_angles = np.asarray(scatter_angle_rad, dtype=float)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    if not np.isfinite(angle) or not np.all(np.isfinite(scatter_angles)):
        raise ValueError("incident and scatter angles must be finite")
    orders = harmonic_orders(n_max)
    incident = incident_l_coefficients(orders, angle)
    solutions = [
        solve_harmonic_coefficients(frequency, radius, material, int(m), coefficient)
        for m, coefficient in zip(orders, incident, strict=True)
    ]
    coefficients_p = np.asarray(
        [solution.longitudinal_coefficient for solution in solutions], dtype=complex
    )
    coefficients_sv = np.asarray(
        [solution.shear_coefficient for solution in solutions], dtype=complex
    )
    f_pp = _far_field_function(coefficients_p, orders, scatter_angles)
    f_ps = _far_field_function(coefficients_sv, orders, scatter_angles)
    phi_grid = np.linspace(-np.pi, np.pi, 721, endpoint=False)
    provisional = ElasticSDHScatteringResult(
        s_ll=f_pp,
        s_ls=f_ps,
        longitudinal_coefficients=coefficients_p,
        shear_coefficients=coefficients_sv,
        orders=orders,
        convergence_metadata={},
        traction_residual_metadata={},
        frequency_hz=frequency,
        radius_m=radius,
        incident_angle_rad=angle,
        scatter_angle_rad=scatter_angles,
    )
    boundary_residual = normalized_boundary_traction_residual(phi_grid, provisional, material)
    kp_a = float(longitudinal_wavenumber(frequency, material) * radius)
    ks_a = float(shear_wavenumber(frequency, material) * radius)
    max_coefficient = max(
        float(np.max(np.abs(coefficients_p))),
        float(np.max(np.abs(coefficients_sv))),
        np.finfo(float).tiny,
    )
    edge_ratio = max(
        abs(coefficients_p[0]),
        abs(coefficients_p[-1]),
        abs(coefficients_sv[0]),
        abs(coefficients_sv[-1]),
    ) / max_coefficient
    return ElasticSDHScatteringResult(
        s_ll=f_pp,
        s_ls=f_ps,
        longitudinal_coefficients=coefficients_p,
        shear_coefficients=coefficients_sv,
        orders=orders,
        convergence_metadata={
            "n_max": int(n_max),
            "guide_n_max_floor_ks_a_plus_10": int(np.floor(ks_a) + 10),
            "k_p_a": kp_a,
            "k_s_a": ks_a,
            "edge_coefficient_ratio": float(edge_ratio),
        },
        traction_residual_metadata={
            "maximum_individual_harmonic_residual": float(
                max(solution.boundary_traction_residual for solution in solutions)
            ),
            "maximum_normalized_boundary_residual": boundary_residual,
            "boundary_angle_samples": int(phi_grid.size),
            "maximum_condition_number": float(
                max(solution.condition_number for solution in solutions)
            ),
        },
        frequency_hz=frequency,
        radius_m=radius,
        incident_angle_rad=angle,
        scatter_angle_rad=scatter_angles,
    )


def elastic_sdh_broadband_scattering(
    frequency_hz: ArrayLike,
    radius_m: float,
    material: ElasticMaterial,
    incident_angle_rad: ArrayLike,
    scatter_angle_rad: ArrayLike,
    n_max: int | None = None,
) -> ElasticSDHBroadbandResult:
    """Evaluate M2-2D on a frequency/incident/outgoing angular grid.

    The output arrays have shape ``(Nf, Nalpha, Nbeta)``. Circular symmetry is
    used explicitly: for each positive frequency one alpha=0 solution is
    evaluated at all unique requested relative directions ``beta-alpha``.
    Zero frequency is defined as zero scattering. If ``n_max`` is omitted, the
    Bostrom--Bovik guide ``floor(k_s*a)+10`` is used independently per bin.
    """
    frequencies = _nonnegative_frequencies(frequency_hz)
    incident_angles = np.atleast_1d(np.asarray(incident_angle_rad, dtype=float))
    scatter_angles = np.atleast_1d(np.asarray(scatter_angle_rad, dtype=float))
    if frequencies.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    if incident_angles.ndim != 1 or scatter_angles.ndim != 1:
        raise ValueError("incident and scatter angle arrays must be one-dimensional")
    if not np.all(np.isfinite(incident_angles)) or not np.all(np.isfinite(scatter_angles)):
        raise ValueError("incident and scatter angles must be finite")
    if n_max is not None:
        harmonic_orders(n_max)

    relative = scatter_angles[np.newaxis, :] - incident_angles[:, np.newaxis]
    f_pp = np.zeros((frequencies.size, *relative.shape), dtype=complex)
    f_ps = np.zeros_like(f_pp)
    orders_used = np.zeros(frequencies.size, dtype=np.int64)
    maximum_residual = 0.0
    for index, frequency in enumerate(frequencies):
        if frequency == 0.0:
            continue
        selected_n_max = (
            int(n_max)
            if n_max is not None
            else recommended_n_max(float(frequency), radius_m, material)
        )
        result = elastic_sdh_scattering(
            float(frequency),
            radius_m,
            material,
            0.0,
            relative.ravel(),
            selected_n_max,
        )
        f_pp[index] = result.f_pp.reshape(relative.shape)
        f_ps[index] = result.f_ps.reshape(relative.shape)
        orders_used[index] = selected_n_max
        maximum_residual = max(
            maximum_residual,
            float(result.traction_residual_metadata["maximum_normalized_boundary_residual"]),
        )
    return ElasticSDHBroadbandResult(
        frequency_hz=np.asarray(frequencies, dtype=float),
        incident_angle_rad=incident_angles,
        scatter_angle_rad=scatter_angles,
        f_pp=f_pp,
        f_ps=f_ps,
        n_max_by_frequency=orders_used,
        maximum_normalized_boundary_residual=maximum_residual,
    )


def unsupported_full_3d_t_matrix(*args, **kwargs):
    """Guard the unimplemented general ``h != 0`` Boström–Bövik T-matrix."""
    del args, kwargs
    raise ReferenceEquationsUnavailableError(
        "the general 3D h!=0 T-matrix requires the Olsson (1994) entries and is not M2-2D"
    )
