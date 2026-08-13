# Ultrasonic forward model

## Aim
Develop a Python-based synthetic ultrasonic array pipeline for:

- 5 MHz linear ultrasonic array
- 64 elements
- Aluminium specimen
- 1 mm diameter side-drilled hole

The wider aim is to understand:

defect parameters
→ predicted FMC data
→ TFM/scattering information
→ probabilistic defect characterisation
→ uncertainty propagation into remaining useful life

## Project stages

### Model M0 — Point-reflector FMC model

Represent the hole initially as a point reflector.

Tasks:
- [x] Define and validate the measured 64-element linear-array geometry.
- [x] Define a configurable point-reflector position.
- [x] Calculate transmitter-defect-receiver distances.
- [x] Calculate arrival times:
  t_ij = (r_i + r_j) / c_L
- [x] Generate a Gaussian-windowed 5 MHz pulse.
- [x] Generate all 64 x 64 synthetic point-reflector A-scans.
- [x] Test synthetic FMC reciprocity.

## Current models

- **M0 — point reflector:** the unchanged original model places one existing
  Gaussian-windowed pulse at each point-reflector transmitter/receiver time.
- **M1 — boundary circle:** samples a circular boundary at `M` evenly spaced
  points, places the same pulse at every boundary travel time, and coherently
  sums the contributions with equal weights `1/M`.
- **M2-2D — analytical elastic SDH:** exact plane-strain P--SV scattering by a
  traction-free circular cavity. Its complete ideal-wavefield matrix contains
  P→P, P→SV, SV→P, and SV→SV responses, with energy-flux normalization,
  mode-dependent propagation, and separate modal TFM images. These modes are
  deliberately not combined into a predicted probe-voltage signal.

The canonical M0/M1 implementations are in `src/models/point_reflector.py` and
`src/models/boundary_circle.py`. The M2-2D array interface is in
`src/models/elastic_sdh.py`, with the analytical kernel in
`src/scattering/elastic_sdh.py`. Existing imports from `src.forward_model` and
`src.geometry` remain supported as compatibility aliases for M0 and M1.

M1 captures finite geometry and coherent path interference, but is a
**geometric finite-extent approximation, not a rigorous elastic
cylinder-scattering model**. It does not satisfy the traction-free elastic
boundary condition and does not include mode conversion. Directivity,
attenuation, geometric spreading, illumination/shadowing, and realistic
cylindrical scattering coefficients are also absent. See
[`docs/circular_reflector.md`](docs/circular_reflector.md).

## Generated outputs

- `figures/propagation/`: geometry, representative paths, travel-time heatmap,
  and pulse-echo timing;
- `figures/synthetic/`: example A-scan, point-reflector TFM image, slices, and
  measured localization report;
- `figures/experimental/`: experimental 30–50 mm depth-ROI TFM image and report;
- `figures/comparison/`: normalized A-scan and matched-grid TFM comparisons;
- `data/synthetic/point_reflector/M0_point_reflector_fmc.npz`: M0 FMC, timing,
  geometry, parameters, and TFM data;
- `data/synthetic/boundary_circle/M1_boundary_circle_fmc.npz`: M1 FMC, time
  vector, array and boundary coordinates, boundary travel times, parameters,
  and TFM data;
- `results/comparisons/M0_point_vs_M1_boundary/`: representative A-scans,
  waveform/timing metrics, matched TFM results, −6 dB widths, sidelobes,
  boundary-count convergence, and the zero-radius comparison;
- `data/synthetic/elastic_sdh/M2_2D_elastic_sdh_fmc.npz`: M2-2D LL FMC,
  complex spectrum, retained P-to-SV kernel, geometry, material parameters, and
  TFM image;
- `data/synthetic/elastic_sdh/multimode/M2_2D_multimode_fmc.npz`: separate PP,
  PS, SP, and SS ideal-wavefield FMC arrays and spectra on a 31.98 µs record;
- `results/comparisons/M0_M1_M2/`: single-frequency scattering patterns,
  coefficient and convergence diagnostics, M0/M1/M2/experimental A-scans and
  spectra, arrival/RMS maps, TFM images, width/localization metrics, and radius
  studies;
- `results/comparisons/M2_multimode/`: full modal scattering patterns,
  flux/cross-section diagnostics, traction/reciprocity/energy validation,
  modal FMC and TFM figures, wrong-delay imaging, radius sweep, and a
  reproducible audit of the experimental MAT file;
- `data/processed/experimental_tfm.npz`: tx/rx-indexed experimental FMC and TFM
  result.

The experimental 40 mm and 1 mm values are filename metadata, not stored ground
truth. The strongest 30–50 mm ROI response is broad laterally, so its assignment
to the side-drilled hole is treated as ambiguous rather than assumed.

## Reproducing the analyses

From the project root:

```powershell
python -m pytest -q
python scripts/02_validate_propagation.py
python scripts/03_synthetic_point_reflector_tfm.py
python scripts/04_experimental_tfm.py
python scripts/05_compare_synthetic_experimental.py
python scripts/06_compare_point_circular_reflector.py
python scripts/07_compare_m0_m1_m2.py
python scripts/08_m2_multimode.py
python scripts/09_audit_experimental_mat.py
```

The analytical conventions and derivation are in
[`docs/m2_conventions.md`](docs/m2_conventions.md). Boström and Bövik use
`exp(-iωt)` with `H_m^(1)` outgoing. NumPy's inverse FFT uses the opposite
positive-frequency synthesis sign, so the array model performs an explicit
conjugation at that interface.

M2-2D is a complete **two-dimensional P--SV cavity scattering and modal
propagation model**, but not the complete Boström--Bövik finite-probe,
three-dimensional measurement model. It omits the general `h != 0` T-matrix,
finite-aperture probe spectrum, Auld electromechanical reception, SH coupling,
surface multiple scattering, and calibrated voltage amplitude. Its four FMCs
assume `H_tx=H_rx=1` and are ideal elastic wavefield responses.

The experimental MAT file contains a complete 64×64, 50 MHz FMC but only a
19.98 µs record. It stores `c_p=6300 m/s` but no `c_s`, has exactly 256
amplitude levels with occupied endpoints (consistent with normalized 8-bit
clipped data), and has substantial per-transmitter RMS variation. Its echo-gated
spectrum peaks near 5.85 MHz rather than exactly at the nominal 5 MHz. These
facts motivate pulse/transfer-function calibration, lower-gain higher-bit-depth
reacquisition, time-zero calibration, measured material properties, and a
record of at least 32–35 µs before experimental converted-mode validation. See
[`experimental_mat_audit.md`](results/comparisons/M2_multimode/validation/experimental_mat_audit.md).

Deliverables:
- Array/defect geometry plot.
- Example A-scans.
- Complete synthetic FMC dataset.
- Travel-time validation tests.

### Stage 2 — Total Focusing Method

Tasks:
- [x] Define an imaging grid.
- [x] Calculate travel times from every element pair to every image point.
- [x] Implement delay-and-sum TFM with linear time interpolation.
- [x] Test point-reflector localization and location changes.
- [x] Form synthetic and experimental TFM images on a matched grid.
- [ ] Test localization under different noise levels.

Deliverables:
- TFM image.
- Localisation-error measurement.
- Study of noise, aperture and element count.

### Stage 3 — Improved propagation model

Add:
- Geometric spreading.
- Element directivity.
- Attenuation.
- Frequency-domain propagation.
- Complex phase.

General model:
D_ij(ω) = P(ω) G_i(ω) S(ω) G_j(ω)

Deliverables:
- Time- and frequency-domain FMC data.
- Comparison with the simple point-reflector model.

### Stage 4 — Finite-sized defect scattering

Replace the constant reflector amplitude with:

S(ω, α_i, β_j; a)

where:
- a is hole radius;
- α_i is incident angle;
- β_j is scattering angle.

Possible approaches:
- Parametric angular scattering model.
- Analytical cylindrical-scattering model.
- Scattering coefficients imported from FE or published data.
- Scalar Helmholtz model as an optional validation exercise.

Deliverables:
- Scattering patterns for different hole sizes.
- Library indexed by size, frequency and angle.
- Comparison between point and finite-hole models.

### Stage 5 — Probabilistic inversion

Infer:
θ = (x, z, a)

from synthetic FMC or extracted scattering data.

Tasks:
- Define priors.
- Define a complex Gaussian likelihood.
- Evaluate a grid posterior.
- Compare amplitude-only and complex inversion.
- Add noise and model mismatch.

Deliverables:
- Posterior distributions for position and diameter.
- Credible intervals.
- Bias, error and uncertainty-calibration results.

### Stage 6 — Optional extensions

Possible extensions:
- Hole-versus-crack classification.
- 2D scalar Helmholtz simulation.
- Elastic FE-generated scattering library.
- Reduced-order or surrogate forward model.
- Crack-growth and remaining-life uncertainty propagation.
- Array-design optimisation.

## Next milestone

Calibrate the measured pulse, time zero, per-element response, and actual block
properties, then add finite-aperture transmit/receive transfer functions and
validated propagation spreading. Keep the four modal wavefields separate until
receive polarization and electromechanical voltage conversion are defined.

## Suggested project structure

ultrasonic-forward-model/
├── src/
│   ├── models/
│   │   ├── point_reflector.py
│   │   ├── boundary_circle.py
│   │   └── elastic_sdh.py
│   ├── scattering/
│   │   └── elastic_sdh.py
│   ├── geometry.py
│   ├── pulse.py
│   └── imaging/tfm.py
├── tests/
├── data/synthetic/
│   ├── point_reflector/
│   ├── boundary_circle/
│   └── elastic_sdh/
├── results/comparisons/M0_point_vs_M1_boundary/
├── results/comparisons/M0_M1_M2/
├── results/comparisons/M2_multimode/
└── README.md

