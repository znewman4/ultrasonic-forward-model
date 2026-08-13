# Experimental MAT-file audit

This is a read-only audit of `5MHz_64els_h40mm_hole1mm_Al_g30_20240220.mat`. Filename tokens such as
`h40mm`, `hole1mm`, and `g30` are useful clues, but are not stored as verified
fields inside the acquisition.

## What the file establishes

- Complete 64 x 64 FMC, 1000 samples at 20.000 ns
  (50.0 MHz), ending at 19.98 µs.
- Array centre pitch 0.630 mm and centre
  span 39.690 mm.
- Stored nominal centre frequency 5 MHz and longitudinal speed
  6300 m/s. No shear speed is stored.
- Exactly 256 amplitude levels with step 0.0078125;
  20,782
  samples occupy the two extrema. This is consistent with normalized 8-bit
  acquisition with clipping.
- Echo-gated reciprocity correlation is
  0.99084; after allowing
  ±4 samples, the median reciprocal-pair correlation is
  0.99774.
- Transmitter-averaged RMS varies by a factor of
  2.64, evidence that
  one shared transmit/receive amplitude is inadequate.
- The 10–16.5 µs echo-gated power spectrum peaks at
  5.846 MHz with a 5–95% power band of
  3.893–9.425 MHz.

## Accuracy improvements supported by this audit

1. Reacquire with lower analogue gain and at least 12–16 bit digitization;
   clipping and 8-bit quantization cannot be repaired by the scattering model.
2. Estimate a complex source/receiver transfer function from a reference or
   back-wall acquisition instead of using the nominal 5 MHz Gaussian pulse.
3. Calibrate time zero and any wedge/couplant delay before adjusting the cavity
   location to absorb timing bias.
4. Measure `c_p`, `c_s`, density, and preferably attenuation on the actual block.
   The MAT file stores only `c_p=6300 m/s`; M2 currently assumes `c_s=3100 m/s`.
5. Calibrate per-element transmit and receive sensitivities. Reciprocal pairs
   are strongly correlated, but the nearly 3:1 transmitter RMS spread is large.
6. Confirm the element endpoint schema and obtain active width/elevation before
   implementing aperture directivity; their stored midpoints do not coincide
   with the stored centres.
7. Record at least 32 µs (preferably 35 µs with margin). The present 19.98 µs
   record truncates representative PS arrivals and excludes SS arrivals.
8. Supply the specimen drawing/scan log and a background/control scan to
   distinguish the SDH echo from the broad response near 40 mm, plausibly a
   boundary response. The MAT file alone cannot make that assignment.

These changes improve model-to-experiment accuracy without changing the now
validated ideal elastic cavity kernel. Probe voltage prediction still requires
finite aperture, polarization sensitivity, coupling, spreading, attenuation,
and electronics response.
