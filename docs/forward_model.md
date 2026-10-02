# Forward models M0 and M1

The canonical implementations are organised as:

- `src/models/point_reflector.py`: Model M0;
- `src/models/boundary_circle.py`: Model M1.

The earlier `src.forward_model` and `src.geometry` public imports are retained
as compatibility aliases.

Model M2-2D is the exact plane-strain P--SV solution for a traction-free circular
cavity. Its cylindrical-wave kernel is in `src/scattering/elastic_sdh.py`; its
first idealized LL array coupling is in `src/models/elastic_sdh.py`. The full
derivation and convention boundary are documented in
[`m2_conventions.md`](m2_conventions.md).

## Propagation assumptions

The first propagation model represents the defect as a single point reflector in
the two-dimensional $x$-$z$ inspection plane. The 64 array elements lie at
$z=0$, and their measured centre coordinates are read from the experimental
MAT file. Propagation takes place through homogeneous aluminium with a constant
longitudinal-wave speed $c_L$. Paths are straight, and the same wave speed is
used on the transmit and receive legs.

For transmitter element $i$, receiver element $j$, and reflector position
\(\mathbf{x}_d\):

- $r_i$ is the Euclidean distance from transmitter $i$ to the reflector;
- $r_j$ is the Euclidean distance from the reflector to receiver $j$;
- $L_{ij} = r_i + r_j$ is the complete transmitter-reflector-receiver path
  length;
- $t_{ij}$ is the predicted echo arrival time for that element pair.

The travel time is

\[
t_{ij} = \frac{r_i + r_j}{c_L} = \frac{L_{ij}}{c_L}.
\]

Evaluating this expression for every transmitter-receiver combination produces
a symmetric 64 x 64 travel-time matrix. This propagation calculation determines
where an echo should arrive in each FMC A-scan.

## Pulse model

The source pulse is a Gaussian-windowed sinusoid:

\[
p(t) = \exp\left(-\frac{t^2}{2\sigma^2}\right)\cos(2\pi f_0 t),
\]

where $f_0$ is the centre frequency and $\sigma$ controls the temporal width of
the Gaussian envelope. The unshifted pulse is centred at $t=0$. For an arrival
time $t_{ij}$, the model evaluates $p(t-t_{ij})$, centring the pulse at the
predicted echo arrival.

This is a deterministic, real-valued, symmetric pulse with a single prescribed
centre frequency and width. It supplies waveform shape and timing only. The
current model uses it to assemble synthetic FMC data as described below.

## Synthetic FMC assembly

For every transmitter-receiver pair, the first forward model places the pulse at
the calculated point-reflector arrival time:

\[
s_{ij}(t) = A\,p(t-t_{ij}),
\]

where $A$ is one configurable constant shared by all element pairs. Evaluating
this expression for all 64 transmitters, 64 receivers, and $N_t$ time samples
produces an array with shape $(64, 64, N_t)$. Because $t_{ij}=t_{ji}$ and the
same amplitude is used in both directions, the synthetic FMC data are
reciprocal.

This stage predicts only echo timing and an idealized waveform. It does not yet
include propagation attenuation, element directivity, realistic side-drilled
hole scattering, noise, elastic mode conversion, or multiple reflectors.

## Model M1: geometric circular-boundary extension

The optional circular reflector samples a circle at evenly spaced boundary
points and coherently averages the existing shifted point pulse over those
points. It provides a simple finite-extent path-length approximation while
leaving the original point-reflector model unchanged. Its formulation,
comparison analysis, and interpretation are documented in
[`circular_reflector.md`](circular_reflector.md).

This extension is not a rigorous elastic cylinder-scattering model. It adds no
traction-free boundary conditions, mode conversion, directivity, attenuation,
or realistic cylindrical scattering coefficients.

For boundary point `m`, transmitter `i`, and receiver `j`, M1 uses full
two-dimensional Euclidean distances:

\[
t_{ijm}=\frac{\|\mathbf{x}_i-\mathbf{x}_m\|_2+
\|\mathbf{x}_m-\mathbf{x}_j\|_2}{c_L}.
\]

It then forms

\[
s_{ij}(t)=\sum_{m=0}^{M-1}\frac{1}{M}p(t-t_{ijm}).
\]

The model therefore captures finite boundary geometry and coherent path
interference. It does not satisfy the traction-free elastic boundary condition
and does not include mode conversion.

## Total Focusing Method

For a candidate image point $(x,z)$, the longitudinal-wave focusing delay is

\[
\tau_{ij}(x,z) = \frac{|\mathbf{x}_i-\mathbf{x}| +
|\mathbf{x}_j-\mathbf{x}|}{c_L}.
\]

The basic TFM image coherently sums the signed A-scan values evaluated at these
delays and then takes the magnitude:

\[
I(x,z) = \left|\sum_{i,j}s_{ij}\!\left(\tau_{ij}(x,z)\right)\right|.
\]

Samples at non-integer time indices are evaluated by linear interpolation.
Delays outside the recorded time interval contribute zero. The implementation
does not normalize internally; normalization is applied only when displaying an
image.

## Model M2-2D: analytical elastic cavity

Boström and Bövik (2003) use `exp(-i omega t)`, so outward cylindrical waves
use `H_n^(1)`. The cylinder axis is perpendicular to the repository x-z plane.
Incident angle `alpha` and outgoing angle `beta` are measured from +x toward
+z. The unit longitudinal scalar potential is

\[
\Phi^{inc}=\exp[i k_Lr\cos(\theta-\alpha)]
\]

with coefficients

\[
a_n=i^n\exp(-in\alpha).
\]

Scattered P and SV potentials use coefficients `A_m` and `B_m` multiplying
`H_m^(1)(k_p r)` and `H_m^(1)(k_s r)`. At every harmonic, the two unknowns are
found from the two traction-free equations `sigma_rr=0` and `sigma_rphi=0`.
Far-field functions `F_PP` and `F_PS` are defined directly from the Hankel
asymptotic; they are potential amplitudes, not probe voltage or energy-flux
coefficients.

The first array model evaluates `F_PP(beta_j,alpha_i,omega)`, multiplies by the
existing pulse spectrum, and applies phase-only centre-path propagation. NumPy
`irfft` uses `exp(+i omega t)` synthesis, so the paper-convention response is
conjugated at the FFT boundary. `H_tx=H_rx=1`. `F_PS` is retained but not added
to the trace because no SV propagation and receive-polarization model has been
defined.

The unsupported general `h != 0` T-matrix remains guarded because the paper
refers its explicit entries to Olsson (1994). M2-2D also omits finite aperture,
Auld electromechanical reception, the planar free surface and its multiple
scattering, attenuation, directivity, spreading, and calibrated voltage.

## Shared assumptions and limitations

- one M0 point, M1 circular boundary, or M2-2D circular cavity in an x-z plane;
- measured element-centre geometry with all elements at $z=0$;
- straight longitudinal-wave paths through homogeneous aluminium at 6300 m/s;
- one Gaussian-windowed 5 MHz sinusoid for every element pair;
- constant amplitude $A=1$ for M0/M1; uncalibrated potential-amplitude
  scattering for M2-2D;
- no geometric spreading, directivity, attenuation, probe/electronics response,
  noise, or boundary reflections in the first array couplings (the M0 physics
  ladder adds spreading, directivity and attenuation to M0 only; see
  [`propagation_physics.md`](propagation_physics.md));
- experimental comparison targets timing and localization, not amplitude
  fidelity.
