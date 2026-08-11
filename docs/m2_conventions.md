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
decouples and is omitted from the first FMC model.

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

At `r=a`, each harmonic is independent. The code solves

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

with `numpy.linalg.solve`; it never forms an explicit inverse.

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

These are potential-amplitude angular functions, not the paper's full
finite-probe electrical response. Circular symmetry makes them functions of
`beta-alpha`. For this normalization, P--P reversal is
`F_PP(beta,alpha)=F_PP(alpha+pi,beta+pi)`.

## Truncation and checks

The paper reports the practical azimuthal guide

\[
 m_{max}=\lfloor k_s a\rfloor+10.
\]

The implementation exposes this guide but also tests explicit convergence,
both traction components on a dense boundary grid, simultaneous angular
rotation, and the derived P--P reversal relation. Small radius is checked only
for convergence toward zero elastic scattering, not equality with M0, whose
amplitude is arbitrary.

## FFT and first array coupling

NumPy `irfft` reconstructs its positive-frequency coefficients using
`exp(+i omega t)`, opposite to the paper's phasor. A paper-convention amplitude
`Q(omega)` is therefore represented as `conj(Q)` in the positive-frequency FFT
bins. The first LL array model uses

\[
 X_{ij}(\omega)=P(\omega)F_{PP}(\beta_j,\alpha_i)^*
 \exp[-i\omega(r_i+r_j)/c_p].
\]

This is phase-only propagation with `H_tx=H_rx=1`. The real pulse spectrum is
constructed from exactly the same Gaussian-windowed cosine used by M0/M1. The
time record is rejected if a five-sigma pulse window would cross either FFT
edge. `F_PS` is stored but does not enter the trace because an SV propagation
leg and polarization-sensitive receiver have not yet been defined.
