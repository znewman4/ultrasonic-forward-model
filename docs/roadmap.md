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

- [ ] Geometric spreading and element directivity
- [ ] Material attenuation
- [x] Frequency-dependent elastic SDH scattering in M2-2D
- [x] Complete the energy-normalized P--SV scattering matrix
- [x] Add separate P/SV propagation and mode-aware TFM
- [ ] Add finite-aperture transmit and receive-polarization transfer functions

## Model M2: analytical elastic SDH

- [x] Verify the supplied Boström–Bövik paper and adopt its `exp(-iωt)` convention
- [x] Define and validate isotropic elastic material and wavenumber primitives
- [x] Document time, radiation, coordinate, potential, stress, and FFT conventions
- [x] Implement harmonic orders and longitudinal Jacobi–Anger coefficients
- [x] Implement and test regular/outgoing cylindrical functions and derivatives
- [x] Derive and separately test P and SV boundary traction operators
- [x] Solve each traction-free 2 x 2 harmonic system without explicit inversion
- [x] Define all raw potential amplitudes `F_PP`, `F_PS`, `F_SP`, and `F_SS`
- [x] Add incident SV harmonics through the same traction-free systems
- [x] Add energy-flux amplitudes, differential cross-sections, and conversion fractions
- [x] Validate both boundary tractions for P/SV incidence, convergence, rotation, normalized reciprocity, and partial-wave energy conservation
- [x] Couple the verified P--P kernel to the existing pulse with an explicit FFT conversion
- [x] Generate separate PP, PS, SP, and SS FMCs with mode-dependent delays
- [x] Generate mode-aware TFM images and demonstrate PS defocusing with PP delays
- [x] Extend the synthetic record to 31.98 µs and limit experimental comparison to PP
- [x] Run and document a read-only experimental MAT-file accuracy audit
- [x] Generate M0/M1/M2 synthetic, TFM, radius, and experimental comparisons
- [ ] Reacquire a ≥32–35 µs, lower-gain, higher-bit-depth experimental record
- [ ] Measure actual-block `c_s`, density, attenuation, and time zero
- [ ] Calibrate complex pulse/electronics and per-element transfer functions
- [ ] Implement the general `h != 0` 3D T-matrix using the Olsson (1994) entries
- [ ] Add the paper's finite-probe spectrum and Auld electromechanical receive model
