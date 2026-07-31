#README.md

## Aim
"""
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

### Stage 1 — Point-reflector FMC model

Represent the hole initially as a point reflector.

Tasks:
- Define a 64-element linear array.
- Generate a band-limited 5 MHz pulse.
- Define the defect position.
- Calculate transmitter–defect–receiver distances.
- Calculate arrival times:
  t_ij = (r_i + r_j) / c_L
- Generate all 64 × 64 synthetic A-scans.
- Check reciprocity: s_ij ≈ s_ji.

Deliverables:
- Array/defect geometry plot.
- Example A-scans.
- Complete synthetic FMC dataset.
- Travel-time validation tests.

### Stage 2 — Total Focusing Method

Tasks:
- Define an imaging grid.
- Calculate travel times from every element pair to every image point.
- Implement delay-and-sum TFM.
- Test localisation under different noise levels.

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

## Immediate milestone

Generate one synthetic A-scan for:

- one transmitter;
- one receiver;
- one defect;
- one delayed 5 MHz pulse.

Then expand the same calculation to the full 64 × 64 FMC dataset.

## Suggested project structure

ultrasonic-forward-model/
├── src/
│   ├── geometry.py
│   ├── pulse.py
│   ├── fmc.py
│   ├── tfm.py
│   └── plotting.py
├── notebooks/
├── tests/
├── figures/
├── data/
├── requirements.txt
└── README.md

"""