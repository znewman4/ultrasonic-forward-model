# M0 point versus M1 boundary-circle comparison

## Configuration

- M0: original point reflector at x=0 mm,
  z=40 mm.
- M1: circular boundary with radius 0.5 mm and
  M=64 equally weighted points.
- Both models use the same pulse, array geometry, wave speed, time vector, and
  TFM implementation.

## Results

- Central A-scan correlation: -0.870333.
- Central M1 boundary arrivals span
  12.540081–12.857532 µs.
- M0 localisation error: 0.000 mm; M1 localisation
  error: 0.000 mm.
- M0 lateral/axial −6 dB widths: 1.222/
  0.300 mm.
- M1 lateral/axial −6 dB widths: 1.371/
  0.259 mm.
- Sidelobe metrics and full convergence/zero-radius tables are in `metrics.json`.

## Interpretation

M1 captures finite circular geometry and coherent interference among its
geometric boundary paths. Equal weights make its radius-zero limit reproduce
M0 and prevent amplitude from increasing with discretisation count.

M1 **does not satisfy the traction-free elastic boundary condition and does not
include mode conversion**. It also omits realistic cylindrical scattering
coefficients, illumination/shadowing, directivity, attenuation, and geometric
spreading. These results describe this geometric boundary quadrature, not a
rigorous elastic side-drilled-hole solution.

The reported −6 dB widths use interpolated amplitude crossings at
`10^(-6/20)` of the image peak. Peak sidelobe level is the largest sampled
amplitude outside the connected −6 dB main lobe of the corresponding axial or
lateral peak slice, expressed relative to that slice peak.
