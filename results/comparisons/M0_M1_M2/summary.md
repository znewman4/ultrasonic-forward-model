# M0/M1/M2-2D comparison

M2-2D is the exact plane-strain P--SV solution for a traction-free circular
cavity. Its first FMC uses only P-to-P scattering with phase-only propagation
and unit transmit/receive transfer functions. Absolute amplitudes are
uncalibrated.

## Validation

- Selected `m_max`: 18; maximum paper guide over active bins: 17.
- Maximum normalized dense-boundary traction residual: 4.100e-11.
- Maximum FMC reciprocity error: 1.221e-15.
- FFT edge/global peak ratio: 4.582e-10.
- M2 FMC runtime: 2.723 s; shape: (64, 64, 1000).

## Comparison

- Central envelope peaks: M0 12.70 µs, M1 12.72 µs, M2-2D 12.56 µs,
  and experiment 12.82 µs.
- Central envelope correlations with experiment: M0 0.659, M1 0.615, and
  M2-2D 0.522.
- M2 TFM peak: [0.0, 0.039599999999999996] m; localization error: 0.400 mm.
- M2 lateral/axial -6 dB widths: 1.211/0.457 mm.
- Experimental TFM peak: [0.0, 0.0404] m. This broad ROI maximum is not confirmed SDH ground truth.

Radius changes both the magnitude and phase/angular structure of `F_PP`, so it
changes A-scan waveform and amplitude rather than merely moving an arrival.
See `metrics.json` and `figures/esm_2d/radius_sensitivity.png` for the
single-frequency study and the broadband central-element table.

## Scope

This is not the complete Boström--Bövik finite-probe 3D measurement model. It
does not include the general `h != 0` T-matrix, finite aperture, Auld reception,
SH coupling, a P-to-SV receive path, surface multiple scattering, or calibrated
probe voltage.
