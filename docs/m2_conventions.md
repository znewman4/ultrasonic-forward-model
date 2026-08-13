# M2-2D conventions and derivation

## Scope and source

`M2-2D` is the exact two-dimensional, plane-strain P--SV solution for a
traction-free circular cylindrical cavity in an infinite homogeneous isotropic
solid. It is obtained by separation of variables and is consistent with the
physical conventions in Boström and Bövik (2003), *Ultrasonic scattering by a
side-drilled hole*, especially their equations (3), (4), and (6)--(9).

It is not their complete three-dimensional `h != 0` T-matrix/probe model. It
omits finite probe aperture, the double wavenumber integrations, Auld
electromechanical reception, SH coupling away from `h=0`, the planar free
surface, and multiple scattering between the surface and hole.

## Phasor and coordinates

The paper explicitly adopts the physical time factor

\[
    \exp(-i\omega t).
\]

Consequently, `H_m^(1)(kr) ~ exp(+ikr)/sqrt(r)` is outward travelling. The
earlier pasted M2 specification stated `exp(+i omega t)` and `H^(1)` outgoing;
those two statements are incompatible. The implementation follows the supplied
paper, which is the requested master source.

The SDH cross-section is the paper's x-y plane, the cylinder axis is z, and the
boundary is `r=a`. Polar angle `phi` is measured counter-clockwise from +x
toward +y. `alpha` is the incident P propagation direction and `beta` is the
outgoing observation direction, measured in the same sense. Repository arrays
use an x-z inspection plane; repository z is mapped to paper y, while the
cylinder axis is suppressed.

The two-dimensional reduction sets axial wavenumber `h=0` and all axial
derivatives to zero. P and SV couple at the traction-free boundary. SH
decouples and is outside the two-dimensional P--SV model.

## Material

For density `rho`, first Lamé constant `lambda`, and shear modulus `mu`,

\[
 c_p=\sqrt{(\lambda+2\mu)/\rho},\qquad
 c_s=\sqrt{\mu/\rho},\qquad
 k_p=\omega/c_p,\qquad k_s=\omega/c_s.
\]

Stress is positive in tension and

\[
 \boldsymbol\sigma=\lambda\,\mathrm{tr}(\boldsymbol\epsilon)\mathbf I
 +2\mu\boldsymbol\epsilon,
 \qquad
 \boldsymbol\epsilon=\tfrac12(\nabla\mathbf u+\nabla\mathbf u^T).
\]

The traction on a surface whose normal points in `+e_r` is
`t = sigma . e_r`, so its polar components are `t_r=sigma_rr` and
`t_phi=sigma_rphi`. A cavity is traction-free regardless of whether the cavity
normal is described with the opposite sign, because the required vector is
zero.

## Potentials, displacement, and strain

The complex displacement amplitude is

\[
 \mathbf u=\nabla\Phi+\nabla\times(\Psi\mathbf e_z).
\]

For one harmonic `Phi=F(r) exp(i m phi)` or
`Psi=G(r) exp(i m phi)`, this gives

\[
 u_r=\Phi_{,r}+\frac1r\Psi_{,\phi},\qquad
 u_\phi=\frac1r\Phi_{,\phi}-\Psi_{,r}.
\]

The polar strains used to derive the code are

\[
 \epsilon_{rr}=u_{r,r},\quad
 \epsilon_{\phi\phi}=\frac1r u_{\phi,\phi}+\frac{u_r}{r},\quad
 \epsilon_{r\phi}=\frac12\left(\frac1r u_{r,\phi}+u_{\phi,r}
 -\frac{u_\phi}{r}\right).
\]

Substitution gives the boundary traction amplitudes, excluding the common
factor `exp(i m phi)`:

\[
 \begin{aligned}
 T_{rr}^{P}[F]&=(\lambda+2\mu)F''
 +\lambda\left(\frac{F'}r-\frac{m^2F}{r^2}\right),\\
 T_{r\phi}^{P}[F]&=2i m\mu\left(\frac{F'}r-\frac{F}{r^2}\right),\\
 T_{rr}^{SV}[G]&=2i m\mu\left(\frac{G'}r-\frac{G}{r^2}\right),\\
 T_{r\phi}^{SV}[G]&=\mu\left(-G''+\frac{G'}r-\frac{m^2G}{r^2}\right).
 \end{aligned}
\]

Primes here are derivatives with respect to radius. The implementation obtains
argument derivatives from SciPy and multiplies them by `k` and `k^2`.

## Incident and scattered fields

The unit incident P-potential plane wave is

\[
 \Phi^{inc}=\exp\{i k_p r\cos(\phi-\alpha)\}
 =\sum_{m=-\infty}^{\infty}C_mJ_m(k_pr)e^{im\phi},
 \qquad C_m=i^m e^{-im\alpha}.
\]

The unit incident SV-potential plane wave has the same scalar Jacobi--Anger
coefficient sequence, with `k_s` replacing `k_p`:

\[
 \Psi^{inc}=\exp\{i k_s r\cos(\phi-\alpha)\}
 =\sum_m C_mJ_m(k_sr)e^{im\phi}.
\]

The polarization is not added by hand: it follows from
`u=curl(Psi e_z)`. With the stated polar basis this is the clockwise transverse
polarization, which fixes the signs in cross-mode reciprocity.

The scattered potentials are

\[
 \Phi^{sc}=\sum_m A_mH_m^{(1)}(k_pr)e^{im\phi},\qquad
 \Psi^{sc}=\sum_m B_mH_m^{(1)}(k_sr)e^{im\phi}.
\]

`A_m` and `B_m` have the same potential units as the unit incident-potential
normalization; numerically they are dimensionless relative coefficients. They
are not displacement-, power-, or probe-voltage-normalized coefficients.
The paper uses real even/odd cosine and sine parities. The implementation's
orders `m=-m_max,...,+m_max` in the complex `exp(i m phi)` basis span the same
angular space and avoid separate parity bookkeeping at `h=0`.

At `r=a`, each harmonic is independent. Defining the two-column regular
traction matrix `R_m` from unit regular P and SV potentials and the outgoing
traction matrix `M_m` from outgoing P and SV potentials, the code solves

\[
 M_m T_m=-R_m
\]

with both right-hand sides at once. Thus every harmonic provides the complete
transition matrix, not just the incident-P column. For example, the P column is

\[
 \begin{bmatrix}
 T_{rr}^{P}[H_m^{(1)}(k_pr)] & T_{rr}^{SV}[H_m^{(1)}(k_sr)]\\
 T_{r\phi}^{P}[H_m^{(1)}(k_pr)] & T_{r\phi}^{SV}[H_m^{(1)}(k_sr)]
 \end{bmatrix}_{r=a}
 \begin{bmatrix}A_m\\B_m\end{bmatrix}
 =-C_m
 \begin{bmatrix}
 T_{rr}^{P}[J_m(k_pr)]\\T_{r\phi}^{P}[J_m(k_pr)]
 \end{bmatrix}_{r=a}
\]

with `numpy.linalg.solve`; it never forms an explicit inverse. In the regular
plus outgoing basis, `J_m=(H_m^{(1)}+H_m^{(2)})/2`, so the incoming/outgoing
partial-wave matrix is

\[
 S_m=I+2T_m.
\]

For a lossless traction-free cavity this matrix is unitary in the potential
basis used here. The SV polarization convention gives the signed reciprocity
identity

\[
 S_m=D S_m^T D,\qquad D=\operatorname{diag}(1,-1).
\]

## Far-field functions

Using

\[
 H_m^{(1)}(kr)\sim\sqrt{\frac{2}{\pi kr}}
 \exp\left[i\left(kr-\frac{m\pi}{2}-\frac\pi4\right)\right],
\]

the angular functions are defined by

\[
 \Phi^{sc}\sim\sqrt{\frac{2}{\pi k_pr}}e^{i(k_pr-\pi/4)}F_{PP}(\beta,\alpha),
 \quad
 F_{PP}=\sum_m A_m e^{im(\beta-\pi/2)},
\]

and

\[
 \Psi^{sc}\sim\sqrt{\frac{2}{\pi k_sr}}e^{i(k_sr-\pi/4)}F_{PS}(\beta,\alpha),
 \quad
 F_{PS}=\sum_m B_m e^{im(\beta-\pi/2)}.
\]

Repeating the construction for incident SV gives `F_SP` and `F_SS`. The matrix
layout throughout the code has outgoing modes as rows and incident modes as
columns:

\[
 \mathbf F^{raw}=\begin{bmatrix}
 F_{PP}&F_{SP}\\F_{PS}&F_{SS}
 \end{bmatrix}.
\]

These are potential-amplitude angular functions, not the paper's full
finite-probe electrical response. Circular symmetry makes them functions of
`beta-alpha`.

## Energy-flux normalization and physical comparisons

For a plane potential of modal type `a` and amplitude `C`, the time-averaged
incident intensity is

\[
 I_a=\frac12\rho\omega^2c_a|\mathbf u|^2
     =\frac12\rho\frac{\omega^4}{c_a}|C|^2,
 \qquad |\mathbf u|=k_a|C|.
\]

Combining this with the Hankel asymptotic shows that the differential scattered
power divided by incident intensity is

\[
 \frac{d\sigma_{ba}}{d\beta}
 =\frac{2}{\pi k_a}|F^{raw}_{ba}|^2.
\]

The code therefore defines

\[
 F^{flux}_{ba}=\sqrt{\frac{2}{\pi k_a}}F^{raw}_{ba},
 \qquad \frac{d\sigma_{ba}}{d\beta}=|F^{flux}_{ba}|^2.
\]

It has units `sqrt(m)` and permits P/SV cross-sections and outgoing fractions
of scattered power to be compared. A fraction of the total power in an
infinite plane wave is not finite; losslessness is instead tested exactly by
`S_m^H S_m=I` for every partial wave.

Normalized reversed-ray reciprocity is direct for PP and SS. For conversion,
the speed-dependent detailed-balance relation is

\[
 \sqrt{k_p}F^{flux}_{PS}(\beta,\alpha)
 =\sqrt{k_s}F^{flux}_{SP}(\alpha+\pi,\beta+\pi),
\]

equivalent to equality of the corresponding raw potential far fields under
ray reversal with the established polarization signs.

## Truncation and checks

The paper reports the practical azimuthal guide

\[
 m_{max}=\lfloor k_s a\rfloor+10.
\]

The implementation exposes this guide but also tests explicit convergence,
both traction components on a dense boundary grid for P and SV incidence,
simultaneous angular rotation, normalized full-matrix reciprocity, and exact
partial-wave energy balance. The production broadband study uses `m_max=26`,
above the guide over the active pulse bins, because dense SV traction residuals
converge more slowly. Small radius is checked only for convergence toward zero
elastic scattering, not equality with M0, whose amplitude is arbitrary.

## FFT and modal array coupling

NumPy `irfft` reconstructs its positive-frequency coefficients using
`exp(+i omega t)`, opposite to the paper's phasor. A paper-convention amplitude
`Q(omega)` is therefore represented as `conj(Q)` in the positive-frequency FFT
bins. The backward-compatible LL array model uses

\[
 X_{ij}(\omega)=P(\omega)F_{PP}(\beta_j,\alpha_i)^*
 \exp[-i\omega(r_i+r_j)/c_p].
\]

The complete ideal-wavefield model instead builds four spectra using

\[
\tau_{PP}=r_i/c_p+r_j/c_p,\quad
\tau_{PS}=r_i/c_p+r_j/c_s,\quad
\tau_{SP}=r_i/c_s+r_j/c_p,\quad
\tau_{SS}=r_i/c_s+r_j/c_s.
\]

It returns `fmc_pp`, `fmc_ps`, `fmc_sp`, and `fmc_ss` separately. This is
phase-only propagation with `H_tx=H_rx=1`; the real pulse spectrum is
constructed from exactly the same Gaussian-windowed cosine used by M0/M1. The
time record is rejected if a five-sigma pulse window would cross either FFT
edge. No modal sum is interpreted as transducer voltage because finite aperture,
transmit coupling, receive polarization, and electromechanical sensitivity are
not yet present.

Mode-aware TFM uses the same transmit/receive speed pair as each FMC. Focusing
PS data with PP delays is an intentional validation counterexample: it produces
strong defocusing and a roughly 26 mm localization error in the reference
study, compared with sub-millimetre error using PS delays.

## Experimental boundary

The current MAT record ends at 19.98 µs. For the reference 40 mm-deep,
1 mm-diameter configuration, representative synthetic PP, PS/SP, and SS peaks
occur at approximately 12.56, 21.28, and 25.50 µs. Only PP is therefore compared
with experiment. No converted/shear experimental validation is claimed.

The file stores only longitudinal speed and nominal frequency; it does not
provide shear speed, density, attenuation, time zero, verified defect geometry,
element aperture definitions, per-channel transfer functions, or acquisition
gain/filter settings. Its 256 amplitude levels and occupied endpoints are
consistent with clipped 8-bit data. These measurement limitations remain
separate from validation of the analytical cavity boundary problem.
