# Roadmap

## Model M0: point-reflector FMC model

- [x] Load and validate the measured 64-element array geometry
- [x] Define a configurable point reflector
- [x] Compute and test point-reflector path lengths and travel times
- [x] Generate and test a Gaussian-windowed ultrasonic pulse
- [x] Generate and validate one delayed point-reflector A-scan
- [x] Generate and test synthetic 64 x 64 point-reflector FMC signals
- [x] Implement and validate synthetic Total Focusing Method (TFM) imaging
- [x] Reshape the indexed experimental channels and form an experimental TFM image
- [x] Complete an initial synthetic-versus-experimental timing and localization comparison

## Model M1: boundary-point circular reflector

- [x] Organise M0 and M1 under `src/models/` with backward-compatible imports
- [x] Generate evenly spaced circular boundary coordinates
- [x] Compute vectorised full-Euclidean `(N, N, M)` travel times
- [x] Coherently sum the existing pulse using equal weights `1/M`
- [x] Save complete M0 and M1 compressed NPZ datasets in model-specific folders
- [x] Run the same TFM implementation on M0 and M1
- [x] Compare A-scans, correlation, timing, localisation, −6 dB widths, and sidelobes
- [x] Demonstrate convergence with boundary-point count
- [x] Demonstrate that radius tending to zero approaches M0
- [x] Save figures, numerical metrics, and a concise comparison summary

## Later propagation and measurement improvements

- [x] Geometric spreading and element directivity (M0 physics ladder)
- [x] Material attenuation (power law; value still to be measured)
- [x] Frequency-domain M0 synthesis validated against the time domain
- [ ] Apply spreading/directivity to M1 boundary points and M2-2D centre paths
- [ ] Measure specimen attenuation (e.g. from back-wall echoes)
- [x] Frequency-dependent elastic SDH scattering in M2-2D
- [x] Compute P-to-SV mode conversion in the M2-2D kernel
- [ ] Add an explicit SV propagation and receive-polarization model

## Model M2: analytical elastic SDH

- [x] Verify the supplied Boström–Bövik paper and adopt its `exp(-iωt)` convention
- [x] Define and validate isotropic elastic material and wavenumber primitives
- [x] Document time, radiation, coordinate, potential, stress, and FFT conventions
- [x] Implement harmonic orders and longitudinal Jacobi–Anger coefficients
- [x] Implement and test regular/outgoing cylindrical functions and derivatives
- [x] Derive and separately test P and SV boundary traction operators
- [x] Solve each traction-free 2 x 2 harmonic system without explicit inversion
- [x] Define asymptotic potential-amplitude `F_PP` and `F_PS`
- [x] Validate both boundary tractions, convergence, rotation, P--P reversal, and radius sensitivity
- [x] Couple the verified P--P kernel to the existing pulse with an explicit FFT conversion
- [x] Retain the P-to-SV kernel for a later polarization-aware receive model
- [x] Generate M0/M1/M2 synthetic, TFM, radius, and experimental comparisons
- [ ] Implement the general `h != 0` 3D T-matrix using the Olsson (1994) entries
- [ ] Add the paper's finite-probe spectrum and Auld electromechanical receive model
