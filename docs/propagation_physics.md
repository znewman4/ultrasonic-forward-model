# M0 physics ladder: spreading, directivity and attenuation

This extends the point-reflector model M0 with the ray-amplitude physics of
Holmes, Drinkwater and Wilcox (2005), one effect at a time, so the influence of
each effect on FMC and TFM data can be measured separately.

Code: `src/propagation.py` (physics factors), `src/pulse.py`
(`gaussian_pulse_spectrum`), `src/models/point_reflector_physics.py` (ladder),
`tests/test_propagation_physics.py`, `scripts/08_compare_physics_ladder.py`.
Results: `results/comparisons/M0_physics_ladder/`.

## Received pulse

For transmitter $i$, receiver $j$ and a point reflector, the received pulse in
the frequency domain is

\[
H_{ij}(\omega) = P(\omega)\,
\underbrace{\frac{A_0}{\sqrt{d_i d_j}}}_{\text{spreading}}\,
\underbrace{p(\theta_i,f)\,p(\theta_j,f)}_{\text{directivity}}\,
\underbrace{e^{-\alpha(f)(d_i+d_j)}}_{\text{attenuation}}\,
\underbrace{e^{-i\omega(d_i+d_j)/c_L}}_{\text{delay}}
\]

and the A-scan is $s_{ij}(t) = \mathrm{irfft}[H_{ij}(\omega)]$.

| Factor | Formula | Function |
|---|---|---|
| Pulse | $P(\omega)$ = DFT of the zero-centred Gaussian-windowed sinusoid | `pulse.gaussian_pulse_spectrum` |
| Ray geometry | $d_i=\lVert\mathbf x_d-\mathbf x_i\rVert$, $\sin\theta_i=(x_d-x_i)/d_i$ (angle from element normal) | `propagation.element_ray_geometry` |
| Spreading | $A_0/\sqrt{d_{tx}d_{rx}}$ (2D cylindrical, Holmes 2005) | `propagation.beam_spread_amplitude` |
| Directivity | $p(\theta,f)=\mathrm{sinc}(\pi a\sin\theta/\lambda)$, $\lambda=c_L/f$ (Holmes eq. 5) | `propagation.element_directivity` |
| Attenuation | $\alpha(f)=\alpha_{ref}(f/f_{ref})^n$ Np/m; $n=0$ is frequency-independent | `propagation.attenuation_coefficient` |

The directivity is the reduced form of McNab and Stumpf's rectangular-element
function (Holmes eq. 4): the elevation term is dropped because element length
$L \gg a$. For this array $L$ = 15 mm and $a$ = 0.53 mm (read from the MAT-file
element corners), so the reduction holds. NumPy's `np.sinc(x)` is
$\sin(\pi x)/(\pi x)$, so the code passes $a\sin\theta\,f/c_L$.

**Sign convention.** The delay $e^{-i\omega\tau}$ belongs to the
$e^{+i\omega t}$ synthesis convention used by NumPy's `irfft`, so no
conjugation is needed here. (M2-2D is different: its kernel uses the
Boström–Bövik $e^{-i\omega t}$ convention and is conjugated at the FFT.)

**Scatterer.** The reflector is an isotropic point with scattering amplitude 1.
Per `TODO.md`, an angle- and frequency-dependent scatterer response belongs only
in the side-drilled-hole model (M2), not in this pulse model.

## The ladder

Each rung changes exactly one thing relative to the one above it.

| Rung | Domain | Physics |
|---|---|---|
| **M0** | time | timing only: $s_{ij}=p(t-t_{ij})$, unchanged original model |
| **M0-TD+** | time | + spreading + frequency-independent attenuation, both evaluated at $f_0$ |
| **M0-FD** | frequency | same physics as M0-TD+, built in the frequency domain |
| **M0-FD+D** | frequency | + frequency-dependent directivity |
| **M0-FD+D+A** | frequency | + frequency-dependent attenuation ($n=1$) |

There is also a diagnostic rung, **M0-TD+D(f0)**: directivity is evaluated only
at 5 MHz in the time domain. Comparing it with M0-FD+D isolates what the
*frequency dependence* of directivity contributes, as opposed to its mean
amplitude effect.

A single function, `pair_transfer`, computes the amplitude factors for both the
time-domain and frequency-domain synthesizers. That guarantees M0-TD+ and M0-FD
use identical physics. The time-domain synthesizer refuses frequency-dependent
physics with an error, so it never silently approximates.

## Validation (tests)

- Default (all-off) physics reproduces M0 **bit-for-bit**.
- M0-FD reproduces M0-TD+ to a relative error of 8.5 × 10⁻¹⁴.
- The FFT round trip reproduces an off-sample shifted pulse to 10⁻⁹.
- The directivity matches Holmes eq. (5) and is 1 at normal incidence. A null
  falls at $\sin\theta=\lambda/a$, and the function narrows as frequency rises.
- Every rung is reciprocal, and every pulse peaks at $t_{ij}$.
- Directivity lowers the amplitude and spectral centroid of oblique pairs while
  leaving the central pair unchanged. Frequency-dependent attenuation lowers the
  centroid.
- The frequency-domain synthesizer refuses records too short for wrap-free FFT
  synthesis.

## Results (5 MHz, 64 elements, reflector at (0, 40) mm)

At this depth the largest ray angle is 26.4°, $a/\lambda=0.42$ at 5 MHz, and
the attenuation placeholder is 10 dB/m at 5 MHz.

| Rung | Edge/centre pulse-echo amplitude | Edge centroid | TFM lateral −6 dB | RMS dB error vs experiment* |
|---|---|---|---|---|
| M0 | 0.00 dB | 5.000 MHz | 1.222 mm | 6.13 dB |
| M0-TD+ | −1.05 dB | 5.000 MHz | 1.233 mm | 5.89 dB |
| M0-FD | −1.05 dB | 5.000 MHz | 1.233 mm | 5.89 dB |
| M0-FD+D | −2.07 dB | 4.990 MHz | 1.243 mm | 5.67 dB |
| M0-FD+D+A | −2.06 dB | 4.986 MHz | 1.244 mm | 5.67 dB |

\*Gated (10–16.5 µs) peak-envelope amplitude over all 64 × 64 pairs, each map
normalised to its own maximum. Pairs involving dead or weak channels 4, 40, 46
and 52 are excluded.

All rungs place the TFM peak exactly at the reflector, and the axial width stays
at 0.30 mm.

**What makes a meaningful difference?**

1. **Spreading and directivity** each roughly double the amplitude taper across
   the aperture: about −1 dB each at the array edge, about −2 dB together. Each
   lowers the RMS amplitude error against experiment by about 0.2 dB. They
   change the normalised FMC by about 5 % (L2) but the TFM image by only about
   1 %. The TFM lateral width grows by about 2 %, because the edge elements
   (which set the effective aperture) are weighted down.
2. **Time domain vs frequency domain** makes no difference by itself (10⁻¹⁴).
   The frequency domain only matters once a factor depends on frequency.
3. **The frequency dependence of directivity** is a small effect at this
   geometry. Freezing the directivity at 5 MHz changes the FMC by 0.6 % and the
   TFM by 0.5 %. The edge centroid shifts by only 10 kHz, because the rays reach
   just 26° and $a<\lambda/2$. It would matter more for shallow defects, wide
   elements, or high-angle (e.g. sectorial or shear) paths.
4. **Frequency-dependent attenuation** at a realistic aluminium level is
   negligible: a 4 kHz downshift, 0.1 dB, and a 0.6 % TFM change. In the
   sensitivity sweep, it begins to matter only above about 200 dB/m. At 1000 dB/m
   the centroid falls by 380 kHz, the edge taper grows by 8 dB, and the TFM
   lateral width grows from 1.24 to 1.43 mm. That level is typical of attenuative
   materials (e.g. austenitic welds, composites), not aluminium.

**Experimental caveat.** The experimental TFM maximum in the 30–50 mm ROI is
26.6 mm wide laterally, so the gated experimental amplitude is not dominated by
a compact hole echo. The small improvement against experiment is consistent in
direction, but it is not proof that the model is right. Directivity and
spreading are well grounded physically; the attenuation value still needs to be
measured, for example from back-wall echoes.

## Limitations

- The sinc directivity is the 2D fluid-like line-element model (Holmes eq. 5).
  A contact array on a solid would be better described by the Miller–Pursey
  solid directivity, which also predicts shear lobes.
- Attenuation is a power law without the velocity dispersion required by
  causality (Kramers–Kronig). Dispersion is negligible for aluminium.
- $A_0$ is uncalibrated. No probe or electronics transfer function, noise,
  shear-wave paths or surface reflections are included.
- Only M0 is extended. The factor functions are generic, so M1 and M2-2D can
  reuse them (with $(d_i, \theta_i)$ per boundary point, or per centre path).

Reference: Holmes, C., Drinkwater, B. W., and Wilcox, P. D. (2005),
"Post-processing of the full matrix of ultrasonic transmit-receive array data
for non-destructive evaluation", *NDT & E International*, 38(8), 701–711.
