# Model M1: geometric circular-reflector approximation

## Scope

The circular-reflector model is a geometric finite-extent approximation for a
side-drilled hole in the two-dimensional x-z plane. A circle of radius `a` is
represented by `M` evenly spaced boundary points,

\[
\mathbf{x}_m = \mathbf{x}_c + a
  [\cos(2\pi m/M),\,\sin(2\pi m/M)], \qquad m=0,\ldots,M-1.
\]

For array elements `i` and `j`, the travel time through boundary point `m` is

\[
t_{ij}^{(m)} =
\frac{|\mathbf{x}_i-\mathbf{x}_m|+|\mathbf{x}_j-\mathbf{x}_m|}{c_L}.
\]

The existing point-reflector pulse is placed at each of these times and summed
coherently with equal quadrature weights:

\[
s_{ij}(t) = \frac{A}{M}\sum_{m=0}^{M-1}
p\!\left(t-t_{ij}^{(m)}\right).
\]

The `1/M` weighting makes the zero-radius limit equal to the existing point
reflector of amplitude `A`, and keeps the scale from growing merely because the
boundary discretisation is refined. Increasing `M` converges the numerical
quadrature of this particular geometric model.

## Interpretation and limitations

This model is **not a rigorous elastic cylinder-scattering model**. Each
boundary point radiates the same pulse with the same positive weight. The model
only introduces a distribution of geometric path lengths and their coherent
phase interference. It does not enforce elastic traction-free boundary
conditions and does not include mode conversion, element directivity,
attenuation, geometric spreading, surface visibility or shadowing, or realistic
angle- and frequency-dependent cylindrical scattering coefficients.

Consequently, changes in A-scan shape, TFM peak, localisation, and image width
should be interpreted as the behaviour of an equal-weight boundary quadrature,
not as quantitative predictions for a physical side-drilled hole. In
particular, contributions from the back of the circle are included even though
a physical cylinder has a more complicated illuminated/shadowed response.

## Comparison analysis

Run the analysis from the repository root:

```powershell
python scripts/06_compare_point_circular_reflector.py
```

It writes the following reproducible outputs to
`results/comparisons/M0_point_vs_M1_boundary/`:

- representative pulse-echo and pitch-catch M0/M1 A-scans;
- matched-grid point and circular-reflector TFM images;
- waveform correlation and boundary-arrival timing metrics;
- localisation error relative to the specified circle centre;
- peak-centred lateral and axial −6 dB widths and sidelobe levels;
- boundary-point convergence and the radius-to-zero comparison;
- `metrics.json` and `summary.md` containing definitions and model exclusions.

The baseline circle has radius 0.5 mm (matching a nominal 1 mm diameter hole)
and uses `M=64`. Convergence is measured against `M=256`, and the zero-radius
study uses `M=64`. These numerical studies use element indices
`[0, 16, 32, 47, 63]` and the complete time vector; the baseline FMC and TFM
comparison uses all 64 elements. All comparisons retain the same pulse, array,
wave speed, time axis, and TFM implementation so only the reflector
approximation changes.

The compressed baseline datasets are saved under
`data/synthetic/point_reflector/` and `data/synthetic/boundary_circle/`.
