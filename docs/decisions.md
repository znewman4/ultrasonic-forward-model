# Modelling decisions

## Begin with a point reflector

The initial defect model is a point reflector. This deliberately removes defect
shape and angular scattering behaviour from the first implementation so that the
array geometry, transmitter-reflector-receiver paths, reciprocity, and echo
arrival times can be isolated and validated before introducing realistic
scattering physics.

This staged approach follows the full-matrix-capture and focusing methodology of
Holmes, Drinkwater and Wilcox (2005), which uses element-pair propagation times
as the basis for processing ultrasonic array data.

Reference: Holmes, C., Drinkwater, B. W., and Wilcox, P. D. (2005),
"Post-processing of the full matrix of ultrasonic transmit-receive array data for
non-destructive evaluation," *NDT & E International*, 38(8), 701-711.

## Use a Gaussian-windowed sinusoidal pulse

The first waveform model is a real Gaussian-windowed sinusoid,
$p(t)=\exp[-t^2/(2\sigma^2)]\cos(2\pi f_0t)$. It gives explicit control of centre
frequency and duration, is straightforward to shift to each predicted arrival
time, and keeps pulse validation separate from amplitude and scattering physics.

At this stage the pulse has unit peak amplitude and no attenuation, directivity,
scattering amplitude, noise, dispersion, or mode conversion. These effects are
deferred until synthetic FMC assembly is introduced.

## Assemble the first FMC model with constant amplitude

The first synthetic FMC model shifts the same Gaussian-windowed pulse to each
predicted arrival time and multiplies every trace by one configurable constant
amplitude. This keeps the integration of geometry, propagation, and pulse timing
directly testable and guarantees reciprocity when the travel-time matrix is
symmetric.

The constant amplitude is a modelling placeholder, not a realistic defect
response. Attenuation, array-element directivity, side-drilled-hole scattering,
noise, and elastic mode conversion remain explicitly deferred.

## Use experimental comparison for timing and localization first

The initial experimental comparison is intended to assess geometric arrival
times and image localization, not amplitude fidelity. Synthetic and experimental
traces may therefore be normalized separately for visualization. Amplitude or
waveform disagreement does not by itself invalidate the propagation model,
because the current model omits probe/electronics response, spreading,
directivity, attenuation, realistic side-drilled-hole scattering, noise, mode
conversion, and boundary reflections.

Experimental peak selection must be explicit. The current analysis searches a
documented 30–50 mm depth region around the filename-based nominal depth. The
strongest response is reported even when its morphology is broad or otherwise
ambiguous; it is not silently relabelled as the hole.

## Name the original point model M0 and the geometric boundary model M1

The unchanged point-reflector implementation is Model M0. Model M1 represents a
circle using evenly spaced boundary points and coherently averages the existing
shifted pulse with weights `1/M`. Canonical implementations live in
`src/models/point_reflector.py` and `src/models/boundary_circle.py`; legacy
imports are compatibility aliases so existing callers retain their behaviour.

Equal weighting was selected as the initial periodic boundary quadrature. It
keeps amplitude independent of discretisation count and makes the zero-radius
limit reproduce M0. M1 captures finite geometry and coherent path interference,
but it does not satisfy the traction-free elastic boundary condition and does
not include mode conversion. It is deliberately not assigned realistic
cylindrical scattering, illumination/shadowing, directivity, attenuation, or
geometric spreading.

The M0/M1 comparison uses the same TFM implementation and reports interpolated
amplitude −6 dB widths. Its peak sidelobe level is operationally defined as the
largest sampled axial or lateral slice response outside the connected −6 dB
main lobe. Waveform correlation is the normalized real inner product (cosine
similarity) over the stated samples. Boundary-count convergence and the
zero-radius limit use a documented five-element subset with the complete time
vector; the baseline M0/M1 FMC and TFM comparison uses all 64 elements.

## Implement the exact h=0 P--SV solution as M2-2D

The supplied Boström and Bövik (2003) paper establishes `exp(-iωt)`,
`H_m^(1)` outgoing waves, the traction-free cavity, and the cylindrical-wave
framework. It does not print the general three-dimensional T-matrix entries,
which it attributes to Olsson (1994). The implemented scope is therefore the
exact `h=0` plane-strain P--SV separation-of-variables solution, labelled
M2-2D. Unsupported `h != 0` and finite-probe operations remain guarded.

The pasted request's combination `exp(+iωt)` plus outgoing `H^(1)` conflicts
with the source paper and with outward phase propagation. The physical kernel
uses the paper's `exp(-iωt)`. At the separate NumPy `irfft` boundary the complex
response is conjugated, because NumPy synthesizes positive frequencies using
`exp(+iωt)`.

The raw far-field outputs are explicitly defined potential-amplitude functions
from the Hankel asymptotic. The completed matrix uses rows for the outgoing mode
and columns for the incident mode,
`[[F_PP,F_SP],[F_PS,F_SS]]`. Regular incident SV harmonics use the same
traction-free systems as incident P. Raw coefficients remain available for
debugging, but comparisons use the flux amplitude
`F_flux_ba=sqrt(2/(pi*k_a))*F_raw_ba`, whose squared magnitude is the
differential scattering cross-section. The exact partial-wave scattering
matrix is unitary; this, rather than a finite “fraction” of an infinite plane
wave, is the lossless energy-balance test.

Separate PP, PS, SP, and SS ideal-wavefield FMCs use the appropriate P or SV
speed on each leg and `H_tx=H_rx=1`. They are not summed because no
polarization-dependent probe transfer function has yet been defined. Separate
modal TFM delay laws are used, and imaging PS with PP delays is retained as an
intentional negative control.

Completion is based on numerical boundary-value evidence: both tractions
decrease with harmonic order for both incident modes, the far-field series
converges, circular rotation and normalized modal reversed-ray reciprocity hold,
the partial-wave scattering matrix conserves energy, and the FFT record has an
explicit no-wrap guard. The comparison remains separately normalized against
experiment because aperture, spreading, receiver/electronics transfer, and
absolute calibration are not yet present.

## Treat the experimental MAT file as an acquisition needing calibration

The acquisition is used as evidence, not as automatically trustworthy ground
truth. A reproducible audit finds 256 amplitude levels, occupied extrema,
19.98 µs duration, substantial transmitter RMS variation, and an echo-gated
power peak near 5.85 MHz. Only `c_p=6300 m/s` is stored; `c_s`, density,
attenuation, time zero, verified hole geometry, gain/filter settings, probe
aperture definitions, and per-channel transfer functions are absent.

Therefore only PP is compared experimentally in the multimode milestone. PS
and SS validation is not claimed because their predicted arrivals reach or
exceed the record end. The next accuracy work is measurement calibration and a
longer, lower-gain, higher-bit-depth acquisition, not tuning ideal cavity
coefficients to compensate for unmodelled probe/electronics physics.
