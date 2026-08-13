# M2-2D complete P--SV modal pipeline

The complete idealized plane-strain matrix `[[F_PP,F_SP],[F_PS,F_SS]]` is
implemented in raw potential and energy-flux normalizations. Four modal FMCs
and four correctly focused TFM images remain separate; `H_tx=H_rx=1` and no
probe-voltage interpretation is made.

## Validation

- Production `m_max`: 26; maximum paper guide over the pulse band: 17.
- Broadband traction residuals: {'P': 1.5277899377610309e-15, 'SV': 2.603957306022447e-11}.
- Maximum partial-wave reciprocity error: 1.670e-15.
- Maximum partial-wave energy-balance error: 1.055e-14.
- FFT edge/global peak ratios: {'PP': 2.5221710353495737e-10, 'PS': 2.7701104502149885e-10, 'SP': 2.7701103265374295e-10, 'SS': 1.2532245001072322e-05}.

## Modal propagation and imaging

- FMC shape per mode: [64, 64, 1600]; time record: 0--31.98 us.
- Representative geometric/peak arrival times: {'PP': {'representative_pair_indices': [31, 31], 'geometric_us': 12.698806442308223, 'envelope_peak_us': 12.56}, 'PS': {'representative_pair_indices': [0, 63], 'geometric_us': 21.491613549505363, 'envelope_peak_us': 21.28}, 'SP': {'representative_pair_indices': [0, 63], 'geometric_us': 21.491613549505363, 'envelope_peak_us': 21.28}, 'SS': {'representative_pair_indices': [31, 31], 'geometric_us': 25.80725180211026, 'envelope_peak_us': 25.5}}.
- Correct-delay TFM localization errors (mm): PP 0.400, PS 0.781, SP 0.781, SS 0.800.
- PS data focused with PP delays has localization error 25.924 mm.

Only PP is compared with the ~20 us experiment. Converted/shear experimental
validation is not claimed because those echoes reach or exceed the record end.
Full numerical tables are in `metrics.json`.
