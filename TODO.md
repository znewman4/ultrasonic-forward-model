# NOTES

## TO DO 02/10/2026

- Double check point reflector and boundary circle were implemented correctly
- Compare point reflector to boundary circle (A-scans, B-scans ie TFM)
- Compare basic ray response to scatter response that takes into account attentuation, geometric spreading, directivity etc. 
- Consider what modelling the following would do: Array as individual point sources; complex scatterer response; shear wave response as well; reflections and different angles. 
- Compare time domain measurement / images to frequency domain images. 
- Only implement scatterer model in pulse if we are using SDH

### Questions 
- Q: why use a gaussian windowed pulse?  A: smooth decay to zero either side, frequency domain pulse that essentially decays. 
- Q: What additional feature in the above amendments actually makes a meaningful difference to the results
- Q: Why use frequency domain to model pulse? A: Frequency domain allows analysis of complex scattering effects that couldnt  be analysed from time domain. Multiplication of these different functions (attentuation, directivity) becomes convolution in time, as opposed to multiplication in frequency domain. This is because directivity function contains frequency, so different frequencies experience different direction. 

### CLAUDE TASKS 
- Implement a directivity function in the pulse function using one of the following methods: Following the approach of McNab and Stumpf [10], the directivity p(q, f) function of a single rectangular element was defined as pðq; fÞ Z sin c pa sin q cos f  l sin c pL sin q sin f  l (4)  Fig. 1. Phased array geometry.  Table 1 Simulated and experimental array parameters Array parameter Value  Number of elements 64 Element width 0.53 mm Element pitch 0.63 mm Centre frequency 5 MHz Bandwidth (K6 dB) 50% where a is the element width, L is the element length and l is the wavelength of the ultrasonic wave. q and f are the angles from the element normal in the steering and elevation planes, respectively. In the two-dimensional model used here it is implicitly assumed that L[a, which is typical of industrial NDE arrays. Therefore, the directivity functions for the transmitting and receiving elements in this case is reduce to the following expressions:  ptx Z sin c pa sin qtx  l and prx Z sin c pa sin qrx  l (5)
- Implement an amplitude function (due to beam spread) of the following form if appropriate: The amplitude Atx,rx, of the signal after propagating a transmission distance dtx and reflected distance drx in the medium was calculated using  Atx;rx Z ffiffiffiAffiffi0ffiffiffiffiffiffi  dtxdrx
- Implement a frequency domain pulse of the form \[
\exp\left(
-i\omega\frac{d_{\mathrm{tx}}+d_{\mathrm{rx}}}{c_L}
\right) 
- so we can explore other frequency dependant physics.  

- So to clarify, what we want in total is:
1. M0: timing only  
2. M0-TD+: timing + spreading + simple attenuation  
3. M0-FD: same model in frequency domain  
4. Add frequency-dependent directivity  
5. Add frequency-dependent attenuation

#### Status (Claude, 02/10/2026): done
- All 5 rungs are implemented in `src/models/point_reflector_physics.py`. The
  factors are in `src/propagation.py` and the FD pulse spectrum is in
  `src/pulse.py`. Tests are in `tests/test_propagation_physics.py`.
- Comparison: `python scripts/08_compare_physics_ladder.py` writes to
  `results/comparisons/M0_physics_ladder/`. The write-up is in
  `docs/propagation_physics.md`.
- Short answer to "what makes a meaningful difference": spreading and
  directivity (about −1 dB each at the array edge) change the amplitude taper,
  but the TFM changes by only about 1 %. TD vs FD alone changes nothing. The
  frequency dependence of directivity and realistic Al attenuation are each
  below 1 %. Attenuation only matters above about 200 dB/m.
