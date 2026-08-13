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
from typing import Any, Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import h1vp, hankel1, jv, jvp


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
ILL_CONDITIONED_THRESHOLD = 1.0e12
ModalName = Literal["P", "SV"]
MODES: tuple[ModalName, ModalName] = ("P", "SV")


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


@dataclass(frozen=True)
class HarmonicScatteringMatrix:
    """One order's raw transition matrix and flux-conservation diagnostics.

    Rows are outgoing ``(P, SV)`` modes and columns are regular incident
    ``(P, SV)`` modes. ``transition_matrix_potential`` maps a regular Bessel
    potential coefficient to outgoing Hankel potential coefficients. Since one
    P or SV cylindrical potential coefficient carries the same radial power in
    this potential convention, ``I + 2*T`` is also the flux-normalized partial
    wave scattering matrix.
    """

    transition_matrix_potential: ComplexArray
    partial_wave_scattering_matrix: ComplexArray
    condition_number: float
    traction_residual_by_incident_mode: FloatArray
    reciprocity_error: float
    energy_balance_error: float


@dataclass(frozen=True)
class ElasticSDHModalScatteringResult:
    """Complete scalar-frequency M2-2D P--SV scattering matrix.

    Matrix rows are outgoing ``(P, SV)`` and columns are incident ``(P, SV)``:
    ``[[F_PP, F_SP], [F_PS, F_SS]]``. ``raw_far_field_potential`` follows the
    potential/Hankel normalization in this module. ``flux_far_field_sqrt_m`` is
    scaled so its squared magnitude is differential scattering cross-section
    in metres for a unit-flux incident plane wave.
    """

    raw_far_field_potential: ComplexArray
    flux_far_field_sqrt_m: ComplexArray
    differential_cross_section_m: FloatArray
    angular_scattered_power_fraction: FloatArray
    transition_matrices_potential: ComplexArray
    plane_wave_scattered_coefficients: ComplexArray
    partial_wave_scattering_matrices: ComplexArray
    orders: NDArray[np.int64]
    plane_wave_coefficients: ComplexArray
    frequency_hz: float
    radius_m: float
    incident_angle_rad: float
    scatter_angle_rad: FloatArray
    convergence_metadata: dict[str, Any]
    validation_metadata: dict[str, Any]

    @property
    def f_pp(self) -> ComplexArray:
        return self.raw_far_field_potential[0, 0]

    @property
    def f_sp(self) -> ComplexArray:
        return self.raw_far_field_potential[0, 1]

    @property
    def f_ps(self) -> ComplexArray:
        return self.raw_far_field_potential[1, 0]

    @property
    def f_ss(self) -> ComplexArray:
        return self.raw_far_field_potential[1, 1]

    @property
    def flux_f_pp(self) -> ComplexArray:
        return self.flux_far_field_sqrt_m[0, 0]

    @property
    def flux_f_sp(self) -> ComplexArray:
        return self.flux_far_field_sqrt_m[0, 1]

    @property
    def flux_f_ps(self) -> ComplexArray:
        return self.flux_far_field_sqrt_m[1, 0]

    @property
    def flux_f_ss(self) -> ComplexArray:
        return self.flux_far_field_sqrt_m[1, 1]


@dataclass(frozen=True)
class ElasticSDHModalBroadbandResult:
    """Full modal response on frequency/incident/outgoing angular grids."""

    frequency_hz: FloatArray
    incident_angle_rad: FloatArray
    scatter_angle_rad: FloatArray
    raw_far_field_potential: ComplexArray
    flux_far_field_sqrt_m: ComplexArray
    n_max_by_frequency: NDArray[np.int64]
    maximum_normalized_boundary_residual: float
    maximum_reciprocity_error: float
    maximum_energy_balance_error: float


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


def incident_sv_coefficients(
    harmonic_order: ArrayLike, incident_angle_rad: float
) -> ComplexArray:
    """Return regular coefficients for ``Psi_inc=exp(i*k_s*d.x)``.

    The scalar SV potential obeys the same Jacobi--Anger expansion as the P
    potential. Its clockwise transverse displacement polarization follows from
    the existing convention ``u=curl(Psi*e_z)``; no extra harmonic phase is
    inserted here.
    """
    return incident_l_coefficients(harmonic_order, incident_angle_rad)


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


def solve_harmonic_scattering_matrix(
    frequency_hz: float,
    radius_m: float,
    material: ElasticMaterial,
    harmonic_order: int,
) -> HarmonicScatteringMatrix:
    """Solve one order's complete P--SV traction-free transition matrix.

    The boundary system is ``M_m T_m = -R_m``. Columns of ``R_m`` are the
    tractions from unit regular P and SV Bessel potentials; columns of ``M_m``
    are the tractions from unit outgoing P and SV Hankel potentials. The solve
    is performed for both right-hand sides together and never uses an explicit
    inverse.

    Since ``J_m=(H_m^(1)+H_m^(2))/2``, the incoming coefficient is one half of
    the regular coefficient and the total outgoing coefficient is ``T+I/2``.
    Hence the partial-wave scattering matrix is ``S_m=I+2*T_m``. With the
    present potentials, equal P and SV Hankel coefficients carry equal radial
    energy flux, so losslessness requires ``S_m^H S_m=I``.
    """
    frequency = float(frequency_hz)
    radius = float(radius_m)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    m, radius = _validate_traction_inputs(harmonic_order, radius, material)
    kp = float(longitudinal_wavenumber(frequency, material))
    ks = float(shear_wavenumber(frequency, material))
    regular_p = _radial_derivatives(bessel_j_values(np.array([m]), kp * radius), kp)
    regular_sv = _radial_derivatives(bessel_j_values(np.array([m]), ks * radius), ks)
    outgoing_p = _radial_derivatives(hankel1_values(np.array([m]), kp * radius), kp)
    outgoing_sv = _radial_derivatives(hankel1_values(np.array([m]), ks * radius), ks)

    regular_traction = np.array(
        [
            [
                traction_rr_p(*regular_p, m, radius, material),
                traction_rr_sv(*regular_sv, m, radius, material),
            ],
            [
                traction_rphi_p(*regular_p, m, radius, material),
                traction_rphi_sv(*regular_sv, m, radius, material),
            ],
        ],
        dtype=complex,
    )
    outgoing_traction = np.array(
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
    condition_number = float(np.linalg.cond(outgoing_traction))
    if not np.isfinite(condition_number):
        raise np.linalg.LinAlgError(f"non-finite harmonic matrix condition number for m={m}")
    if condition_number > ILL_CONDITIONED_THRESHOLD:
        warnings.warn(
            f"M2 harmonic m={m} is ill-conditioned (cond={condition_number:.3e})",
            RuntimeWarning,
            stacklevel=2,
        )
    transition = np.linalg.solve(outgoing_traction, -regular_traction)
    residual_vectors = outgoing_traction @ transition + regular_traction
    denominators = np.maximum(
        np.linalg.norm(regular_traction, axis=0), np.finfo(float).tiny
    )
    residuals = np.linalg.norm(residual_vectors, axis=0) / denominators
    scattering = np.eye(2, dtype=complex) + 2.0 * transition
    identity = np.eye(2, dtype=complex)
    # The sign matrix records the clockwise SV polarization implied by
    # u=curl(Psi*e_z); reciprocity is S=D S^T D in this potential basis.
    parity = np.diag([1.0, -1.0])
    reciprocity_error = float(
        np.linalg.norm(scattering - parity @ scattering.T @ parity, ord=2)
    )
    energy_error = float(
        np.linalg.norm(scattering.conj().T @ scattering - identity, ord=2)
    )
    return HarmonicScatteringMatrix(
        transition_matrix_potential=np.asarray(transition, dtype=complex),
        partial_wave_scattering_matrix=np.asarray(scattering, dtype=complex),
        condition_number=condition_number,
        traction_residual_by_incident_mode=np.asarray(residuals, dtype=float),
        reciprocity_error=reciprocity_error,
        energy_balance_error=energy_error,
    )


def solve_harmonic_coefficients(
    frequency_hz: float,
    radius_m: float,
    material: ElasticMaterial,
    harmonic_order: int,
    incident_coefficient: complex,
    *,
    incident_mode: ModalName = "P",
) -> HarmonicSolution:
    """Solve one harmonic for P or SV incidence; P remains the default."""
    frequency = float(frequency_hz)
    radius = float(radius_m)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    _validate_traction_inputs(harmonic_order, radius, material)
    coefficient = complex(incident_coefficient)
    if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
        raise ValueError("incident_coefficient must be finite")
    if incident_mode not in MODES:
        raise ValueError(f"incident_mode must be one of {MODES}")
    modal_index = MODES.index(incident_mode)
    harmonic = solve_harmonic_scattering_matrix(
        frequency, radius, material, int(harmonic_order)
    )
    solution = harmonic.transition_matrix_potential[:, modal_index] * coefficient
    residual = float(harmonic.traction_residual_by_incident_mode[modal_index])
    return HarmonicSolution(
        longitudinal_coefficient=complex(solution[0]),
        shear_coefficient=complex(solution[1]),
        condition_number=harmonic.condition_number,
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


def _incident_plane_wave_traction(
    angle_rad: FloatArray,
    incident_angle_rad: float,
    incident_mode: ModalName,
    frequency_hz: float,
    radius_m: float,
    material: ElasticMaterial,
) -> tuple[ComplexArray, ComplexArray]:
    """Return exact incident ``(sigma_rr, sigma_rphi)`` on ``r=radius``."""
    delta = np.asarray(angle_rad, dtype=float) - float(incident_angle_rad)
    if incident_mode == "P":
        k = float(longitudinal_wavenumber(frequency_hz, material))
        potential = np.exp(1j * k * radius_m * np.cos(delta))
        rr = -k**2 * (
            material.lambda_ + 2.0 * material.mu * np.cos(delta) ** 2
        ) * potential
        rphi = (
            2.0
            * material.mu
            * k**2
            * np.sin(delta)
            * np.cos(delta)
            * potential
        )
    elif incident_mode == "SV":
        k = float(shear_wavenumber(frequency_hz, material))
        potential = np.exp(1j * k * radius_m * np.cos(delta))
        rr = (
            2.0
            * material.mu
            * k**2
            * np.sin(delta)
            * np.cos(delta)
            * potential
        )
        rphi = material.mu * k**2 * np.cos(2.0 * delta) * potential
    else:
        raise ValueError(f"incident_mode must be one of {MODES}")
    return np.asarray(rr, dtype=complex), np.asarray(rphi, dtype=complex)


def total_boundary_traction_modal(
    angle_rad: ArrayLike,
    result: ElasticSDHModalScatteringResult,
    material: ElasticMaterial,
    incident_mode: ModalName,
) -> tuple[ComplexArray, ComplexArray]:
    """Reconstruct both boundary tractions for P or SV plane-wave incidence."""
    angle = np.asarray(angle_rad, dtype=float)
    if not np.all(np.isfinite(angle)):
        raise ValueError("angle_rad must be finite")
    if incident_mode not in MODES:
        raise ValueError(f"incident_mode must be one of {MODES}")
    incident_index = MODES.index(incident_mode)
    rr, rphi = _incident_plane_wave_traction(
        angle.ravel(),
        result.incident_angle_rad,
        incident_mode,
        result.frequency_hz,
        result.radius_m,
        material,
    )
    kp = float(longitudinal_wavenumber(result.frequency_hz, material))
    ks = float(shear_wavenumber(result.frequency_hz, material))
    exp_m = np.exp(1j * result.orders[:, None] * angle.ravel()[None, :])
    for index, m in enumerate(result.orders):
        p = _radial_derivatives(
            hankel1_values(np.array([m]), kp * result.radius_m), kp
        )
        sv = _radial_derivatives(
            hankel1_values(np.array([m]), ks * result.radius_m), ks
        )
        p_coefficient = result.plane_wave_scattered_coefficients[
            index, 0, incident_index
        ]
        sv_coefficient = result.plane_wave_scattered_coefficients[
            index, 1, incident_index
        ]
        rr += (
            p_coefficient
            * traction_rr_p(*p, int(m), result.radius_m, material)
            + sv_coefficient
            * traction_rr_sv(*sv, int(m), result.radius_m, material)
        ) * exp_m[index]
        rphi += (
            p_coefficient
            * traction_rphi_p(*p, int(m), result.radius_m, material)
            + sv_coefficient
            * traction_rphi_sv(*sv, int(m), result.radius_m, material)
        ) * exp_m[index]
    return rr.reshape(angle.shape), rphi.reshape(angle.shape)


def normalized_boundary_traction_residual_modal(
    angle_rad: ArrayLike,
    result: ElasticSDHModalScatteringResult,
    material: ElasticMaterial,
    incident_mode: ModalName,
) -> float:
    """Return the maximum total traction relative to incident traction."""
    angle = np.asarray(angle_rad, dtype=float)
    rr, rphi = total_boundary_traction_modal(angle, result, material, incident_mode)
    incident_rr, incident_rphi = _incident_plane_wave_traction(
        angle,
        result.incident_angle_rad,
        incident_mode,
        result.frequency_hz,
        result.radius_m,
        material,
    )
    numerator = np.max(np.hypot(np.abs(rr), np.abs(rphi)))
    denominator = np.max(np.hypot(np.abs(incident_rr), np.abs(incident_rphi)))
    return float(numerator / denominator)


def elastic_sdh_modal_scattering(
    frequency_hz: float,
    radius_m: float,
    material: ElasticMaterial,
    incident_angle_rad: float,
    scatter_angle_rad: ArrayLike,
    n_max: int,
) -> ElasticSDHModalScatteringResult:
    """Return the complete raw and flux-normalized M2-2D scattering matrix.

    The matrix layout is ``[[F_PP,F_SP],[F_PS,F_SS]]`` (outgoing row,
    incident column). For incident mode ``a``, the flux-normalized amplitude is

    ``F_flux[b,a] = sqrt(2/(pi*k_a))*F_raw[b,a]``.

    It has units ``sqrt(m)`` and obeys
    ``d sigma_(b<-a)/d beta = abs(F_flux[b,a])**2``. The angular scattered
    power fraction divides this differential cross-section by its sum over the
    two outgoing modes. It is a fraction of *scattered* power, not of the
    infinite incident plane wave.
    """
    frequency = float(frequency_hz)
    radius = float(radius_m)
    alpha = float(incident_angle_rad)
    beta = np.asarray(scatter_angle_rad, dtype=float)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    if not np.isfinite(alpha) or not np.all(np.isfinite(beta)):
        raise ValueError("incident and scatter angles must be finite")
    orders = harmonic_orders(n_max)
    plane_coefficients = incident_l_coefficients(orders, alpha)
    harmonic_results = [
        solve_harmonic_scattering_matrix(frequency, radius, material, int(m))
        for m in orders
    ]
    transition = np.stack(
        [item.transition_matrix_potential for item in harmonic_results], axis=0
    )
    partial_s = np.stack(
        [item.partial_wave_scattering_matrix for item in harmonic_results], axis=0
    )
    scattered_coefficients = transition * plane_coefficients[:, None, None]
    raw = np.empty((2, 2, *beta.shape), dtype=complex)
    for outgoing_index in range(2):
        for incident_index in range(2):
            raw[outgoing_index, incident_index] = _far_field_function(
                scattered_coefficients[:, outgoing_index, incident_index],
                orders,
                beta,
            )
    incident_wavenumbers = np.array(
        [
            float(longitudinal_wavenumber(frequency, material)),
            float(shear_wavenumber(frequency, material)),
        ]
    )
    column_scale = np.sqrt(2.0 / (np.pi * incident_wavenumbers))
    scale_shape = (1, 2) + (1,) * beta.ndim
    flux = raw * column_scale.reshape(scale_shape)
    differential_cross_section = np.abs(flux) ** 2
    outgoing_total = np.sum(differential_cross_section, axis=0, keepdims=True)
    fractions = np.divide(
        differential_cross_section,
        outgoing_total,
        out=np.zeros_like(differential_cross_section),
        where=outgoing_total > np.finfo(float).tiny,
    )
    provisional = ElasticSDHModalScatteringResult(
        raw_far_field_potential=raw,
        flux_far_field_sqrt_m=flux,
        differential_cross_section_m=differential_cross_section,
        angular_scattered_power_fraction=fractions,
        transition_matrices_potential=transition,
        plane_wave_scattered_coefficients=scattered_coefficients,
        partial_wave_scattering_matrices=partial_s,
        orders=orders,
        plane_wave_coefficients=plane_coefficients,
        frequency_hz=frequency,
        radius_m=radius,
        incident_angle_rad=alpha,
        scatter_angle_rad=beta,
        convergence_metadata={},
        validation_metadata={},
    )
    phi_grid = np.linspace(-np.pi, np.pi, 721, endpoint=False)
    traction_residuals = {
        mode: normalized_boundary_traction_residual_modal(
            phi_grid, provisional, material, mode
        )
        for mode in MODES
    }
    max_coefficient = max(float(np.max(np.abs(scattered_coefficients))), np.finfo(float).tiny)
    edge_ratio = float(
        max(
            np.max(np.abs(scattered_coefficients[0])),
            np.max(np.abs(scattered_coefficients[-1])),
        )
        / max_coefficient
    )
    kp_a = float(longitudinal_wavenumber(frequency, material) * radius)
    ks_a = float(shear_wavenumber(frequency, material) * radius)
    return ElasticSDHModalScatteringResult(
        raw_far_field_potential=raw,
        flux_far_field_sqrt_m=flux,
        differential_cross_section_m=differential_cross_section,
        angular_scattered_power_fraction=fractions,
        transition_matrices_potential=transition,
        plane_wave_scattered_coefficients=scattered_coefficients,
        partial_wave_scattering_matrices=partial_s,
        orders=orders,
        plane_wave_coefficients=plane_coefficients,
        frequency_hz=frequency,
        radius_m=radius,
        incident_angle_rad=alpha,
        scatter_angle_rad=beta,
        convergence_metadata={
            "n_max": int(n_max),
            "guide_n_max_floor_ks_a_plus_10": int(np.floor(ks_a) + 10),
            "k_p_a": kp_a,
            "k_s_a": ks_a,
            "edge_coefficient_ratio": edge_ratio,
        },
        validation_metadata={
            "normalized_boundary_traction_residual": traction_residuals,
            "maximum_individual_harmonic_traction_residual": float(
                max(
                    np.max(item.traction_residual_by_incident_mode)
                    for item in harmonic_results
                )
            ),
            "maximum_condition_number": float(
                max(item.condition_number for item in harmonic_results)
            ),
            "maximum_partial_wave_reciprocity_error": float(
                max(item.reciprocity_error for item in harmonic_results)
            ),
            "maximum_partial_wave_energy_balance_error": float(
                max(item.energy_balance_error for item in harmonic_results)
            ),
            "boundary_angle_samples": int(phi_grid.size),
        },
    )


def normalized_modal_reciprocity_error(
    forward: ElasticSDHModalScatteringResult,
    reversed_rays: ElasticSDHModalScatteringResult,
    material: ElasticMaterial,
) -> float:
    """Return the relative error in the flux-amplitude reciprocity relations.

    ``reversed_rays`` must be evaluated with incident direction ``beta+pi`` and
    observation direction ``alpha+pi`` corresponding elementwise to
    ``forward``. For cross modes the selected normalization obeys

    ``sqrt(k_p) F_flux_PS = sqrt(k_s) F_flux_SP_reversed``.

    Same-mode amplitudes are equal directly. The wavenumber weighting is the
    detailed-balance factor associated with differential cross-section
    normalization; raw potential far fields are equal under ray reversal.
    """
    if not np.isclose(forward.frequency_hz, reversed_rays.frequency_hz):
        raise ValueError("forward and reversed results must use the same frequency")
    kp = float(longitudinal_wavenumber(forward.frequency_hz, material))
    ks = float(shear_wavenumber(forward.frequency_hz, material))
    differences = np.concatenate(
        [
            np.ravel(forward.flux_f_pp - reversed_rays.flux_f_pp),
            np.ravel(forward.flux_f_ss - reversed_rays.flux_f_ss),
            np.ravel(np.sqrt(kp) * forward.flux_f_ps - np.sqrt(ks) * reversed_rays.flux_f_sp),
            np.ravel(np.sqrt(ks) * forward.flux_f_sp - np.sqrt(kp) * reversed_rays.flux_f_ps),
        ]
    )
    references = np.concatenate(
        [
            np.ravel(forward.flux_f_pp),
            np.ravel(forward.flux_f_ss),
            np.ravel(np.sqrt(kp) * forward.flux_f_ps),
            np.ravel(np.sqrt(ks) * forward.flux_f_sp),
        ]
    )
    return float(
        np.max(np.abs(differences))
        / max(float(np.max(np.abs(references))), np.finfo(float).tiny)
    )


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


def elastic_sdh_modal_broadband_scattering(
    frequency_hz: ArrayLike,
    radius_m: float,
    material: ElasticMaterial,
    incident_angle_rad: ArrayLike,
    scatter_angle_rad: ArrayLike,
    n_max: int | None = None,
) -> ElasticSDHModalBroadbandResult:
    """Evaluate the complete modal matrix on ``(f,alpha,beta)`` grids.

    Far-field arrays have shape ``(Nf,2,2,Nalpha,Nbeta)`` with outgoing and
    incident modal axes in that order. Circular symmetry is used so a single
    alpha-zero solve serves all requested angle pairs at each frequency.
    """
    frequencies = _nonnegative_frequencies(frequency_hz)
    alphas = np.atleast_1d(np.asarray(incident_angle_rad, dtype=float))
    betas = np.atleast_1d(np.asarray(scatter_angle_rad, dtype=float))
    if frequencies.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    if alphas.ndim != 1 or betas.ndim != 1:
        raise ValueError("incident and scatter angle arrays must be one-dimensional")
    if not np.all(np.isfinite(alphas)) or not np.all(np.isfinite(betas)):
        raise ValueError("incident and scatter angles must be finite")
    if n_max is not None:
        harmonic_orders(n_max)
    relative = betas[None, :] - alphas[:, None]
    output_shape = (frequencies.size, 2, 2, alphas.size, betas.size)
    raw = np.zeros(output_shape, dtype=complex)
    flux = np.zeros_like(raw)
    orders_used = np.zeros(frequencies.size, dtype=np.int64)
    maximum_traction = 0.0
    maximum_reciprocity = 0.0
    maximum_energy = 0.0
    for index, frequency in enumerate(frequencies):
        if frequency == 0.0:
            continue
        selected = (
            int(n_max)
            if n_max is not None
            else recommended_n_max(float(frequency), radius_m, material)
        )
        result = elastic_sdh_modal_scattering(
            float(frequency), radius_m, material, 0.0, relative.ravel(), selected
        )
        raw[index] = result.raw_far_field_potential.reshape(
            2, 2, alphas.size, betas.size
        )
        flux[index] = result.flux_far_field_sqrt_m.reshape(
            2, 2, alphas.size, betas.size
        )
        orders_used[index] = selected
        validation = result.validation_metadata
        maximum_traction = max(
            maximum_traction,
            max(validation["normalized_boundary_traction_residual"].values()),
        )
        maximum_reciprocity = max(
            maximum_reciprocity,
            validation["maximum_partial_wave_reciprocity_error"],
        )
        maximum_energy = max(
            maximum_energy,
            validation["maximum_partial_wave_energy_balance_error"],
        )
    return ElasticSDHModalBroadbandResult(
        frequency_hz=np.asarray(frequencies, dtype=float),
        incident_angle_rad=alphas,
        scatter_angle_rad=betas,
        raw_far_field_potential=raw,
        flux_far_field_sqrt_m=flux,
        n_max_by_frequency=orders_used,
        maximum_normalized_boundary_residual=float(maximum_traction),
        maximum_reciprocity_error=float(maximum_reciprocity),
        maximum_energy_balance_error=float(maximum_energy),
    )


def unsupported_full_3d_t_matrix(*args, **kwargs):
    """Guard the unimplemented general ``h != 0`` Boström–Bövik T-matrix."""
    del args, kwargs
    raise ReferenceEquationsUnavailableError(
        "the general 3D h!=0 T-matrix requires the Olsson (1994) entries and is not M2-2D"
    )
