---
title: TFM – Brain1 comparison and full code path
tags: [ultrasonics, TFM, FMC, Hilbert, brain1]
source: src/imaging/tfm.py
related: "[[tfm_algorithm]]"
---

# TFM – Brain1 comparison and full code path

Companion to [[tfm_algorithm]]. Part A compares our `tfm_image` with Bristol's [BRAIN v1](https://github.com/ndtatbristol/brain1). Part B follows one call of ours from input to output.

Brain1 files I read:
- `array processing/fn_calc_tfm_focal_law3.m` (builds the "focal law": delays, weights, flags)
- `array processing/fast DAS3 files/fn_fast_DAS3.m` (applies the focal law to the data)
- `signal-processing/fn_hilbert.m`
- `Imaging/fn_1contact_tfm_wrapper.m` (GUI wrapper that calls the two above)

---

# Part A – Differences from Brain1

## A.1 Same core

Both compute, for every image point $\mathbf r$,

$$
I(\mathbf r)=\Big|\sum_{tx}\sum_{rx} a_{tx,rx}(\mathbf r)\; \tilde s_{tx,rx}\big(\tau_{tx}(\mathbf r)+\tau_{rx}(\mathbf r)\big)\Big|,
\qquad \tau_i(\mathbf r)=\frac{\lVert \mathbf r-\mathbf e_i\rVert}{c}
$$

where $\tilde s$ is the analytic signal of the A-scan, evaluated by linear interpolation, and $a$ is a weight. Brain1's linear-interpolation branch is `fn_fast_linear_interp` inside `fn_fast_DAS3`; its CPU loop is exactly this sum over time traces.

## A.2 Side-by-side

| Aspect | Ours (`tfm_image`) | Brain1 (`fn_calc_tfm_focal_law3` + `fn_fast_DAS3`) |
|---|---|---|
| **Structure** | One function. Distances, delays, interpolation and sum are done in one pass. | Two stages. The *focal law* (lookup tables of time, index, amplitude, size $N_{pix}\times N$) is computed once, then `fn_fast_DAS3` applies it to any FMC data. |
| **Delay tables** | Recomputed per call, per pixel chunk. Pair delay built on the fly as $(d_{tx}+d_{rx})/c$. | `lookup_time(:,i)` is stored per element. The pair time is `lookup_time(:,tx) + lookup_time(:,rx)` inside the loop. Same maths, but reusable across frames. |
| **Loop order** | Chunk of pixels; vectorised over all $N^2$ pairs at once. | Loop over time traces (`for ii = 1:length(focal_law.tt_ind)`); vectorised over all pixels. |
| **Hilbert** | `analytic_signal()` over the whole `(N,N,Nt)` array before the loop. | Same FFT-mask method, applied to `time_data` before the loop (`focal_law.hilbert_on = 1`). |
| **Interpolation** | Linear only. | `'nearest'` (index lookup via `round`), `'linear'`, and on GPU `'lanczos2'/'lanczos3'`. Default `'linear'`. |
| **Out-of-range samples** | Masked to $0$ with a `valid` mask. | Nearest: out-of-range indices are redirected to an appended zero row. Linear: `fn_fast_linear_interp` returns $0$. |
| **Velocity** | One scalar $c$. | Scalar from `fn_get_nominal_velocity`, or angle-dependent $v(\theta,\phi)$ from spherical harmonics / ellipse / polynomial (`angle_dep_vel`). |
| **Element-pixel angle** | Not computed. | $\theta=\operatorname{atan2}(x-x_i,\,z-z_i)$ is used for velocity and weighting. |
| **Aperture limit** | None; all elements contribute to every pixel. | `angle_limit` with `'rectangular'` or `'hanning'` window; `aperture_weight_correction` renormalises the sum of weights per pixel. |
| **Amplitude weighting** | $a\equiv1$. | `lookup_amp`: aperture window, optional `'resolution'` ($\sqrt d/\cos\theta$) or `'detection'` ($\cos\theta/\sqrt d$) weighting, and optional attenuation correction in the wrapper (`fn_apply_atten`). |
| **Half matrix capture** | Requires the full $N\times N$ matrix (the loader errors otherwise). | If only one of $(tx,rx)$, $(rx,tx)$ is present, `tt_weight = 2` doubles it to stand in for the missing reciprocal. |
| **Trace selection** | All $N^2$ traces always summed. | `tt_ind` from `fn_optimise_focal_law2` skips traces that can't contribute. |
| **Separate tx/rx laws** | Not supported. | Supported (`lookup_time_tx/_rx`, e.g. `'CSM'` transmit mode). |
| **Dimensions** | 2-D (x–z). | 2-D or 3-D meshes. |
| **Hardware** | NumPy on CPU. | GPU (CUDA `gpu_tfm.cu`) with CPU fallback. |
| **Time axis offset** | Uses the real `time_s` array in the interpolation, so a nonzero first sample time is handled exactly. | Linear mode uses `x0 = time(1)` too; the nearest-neighbour table uses `round((t - t0/2)/dt)`. |
| **Output** | `np.abs` of the complex sum, no normalisation. | `fn_fast_DAS3` returns the complex sum (`result`); I did not find where magnitude is taken (presumably the display layer), so check if you need to match dB scaling exactly. |

## A.3 The practically important gaps

1. **No aperture/angle limit or directivity weighting.** Large-angle contributions from elements far from a pixel are given equal weight. For the 64-element, 39.69 mm aperture at 30–50 mm depth this mostly shows as lower contrast at the edges of the ROI.
2. **Single velocity.** Fine for isotropic aluminium; not for anisotropic or layered parts.
3. **No weighting for HMC data.** Not needed while you have the full matrix.
4. **Linear interpolation only.** At 50 samples per period (20 ns sampling, 5 MHz) linear is accurate, and applying it to the analytic signal (which is smooth) is better than to RF.
5. **Slight Hilbert difference.** Brain1's mask drops the last positive bin below Nyquist as well as Nyquist itself, and doubles DC. See [[#B.3 The Hilbert transform step]]. This does not matter for band-limited zero-mean data.

---

# Part B – Full path of our implementation

Running example (from `scripts/04_experimental_tfm.py` and `scripts/common.py`):

- $N=64$ elements, pitch such that aperture $=39.69$ mm, array at $z=0$
- $c=6300$ m/s, centre frequency $5$ MHz, so $\lambda=1.26$ mm
- grid: $x\in[-25,25]$ mm (101 points), $z\in[30,50]$ mm (101 points)

```mermaid
flowchart TD
    A["load_fmc → experimental_fmc_array<br/>(N,N,Nt)"] --> B[validate inputs]
    B --> C["analytic_signal (FFT mask)<br/>real → complex"]
    C --> D[flatten: traces (N², Nt)]
    B --> E["meshgrid → pixels (Npix, 2)"]
    E --> F{{"for each chunk of pixels"}}
    D --> F
    F --> G["distances (chunk, N)"]
    G --> H["delays (chunk, N²) = (d_tx+d_rx)/c"]
    H --> I["searchsorted → lower/upper index + weight"]
    I --> J["linear interpolation of complex traces"]
    J --> K["zero outside time window"]
    K --> L["sum over N² pairs"]
    L --> M["np.abs → image pixels"]
    M --> N["reshape (Nz, Nx)"]
```

## B.1 Getting the FMC into shape

`experimental_fmc_array` (`src/data_loader.py`) turns the MAT file's list of channels (each with a transmitter index and a receiver index) into a 3-D array:

```python
result[i, j] = traces[:, channel]     # i = tx index, j = rx index
```

giving `fmc[tx, rx, t]`, shape `(64, 64, Nt)`. It raises if any tx/rx pair is missing or duplicated, so by the time `tfm_image` runs, every one of the $N^2$ pairs exists.

Element coordinates come from `array_coordinates(MAT_PATH)`: an `(64, 2)` array of $(x_i,\,0)$.

## B.2 Input validation

```python
if fmc.ndim != 3 or fmc.shape[0] != fmc.shape[1]:
    raise ValueError("fmc_data must have shape (N, N, Nt)")
n, _, nt = fmc.shape
if nt < 2 or not np.all(np.diff(time) > 0.0):
    raise ValueError("time_s must be strictly increasing with at least 2 samples")
```

The interpolation code later needs `time` strictly increasing (so `searchsorted` is valid and the denominator $t_{k+1}-t_k$ is never zero), and shapes that line up for broadcasting. Everything else (finite values, $c>0$, `pixel_chunk_size>0`) is checked here so the loop can't fail halfway.

## B.3 The Hilbert transform step

**Goal.** A real RF A-scan $s(t)=A(t)\cos(\omega_0 t+\varphi)$ oscillates through zero, so summing RF values from different pairs, and then taking $|\cdot|$, leaves carrier fringes. The **analytic signal**

$$
\tilde s(t)=s(t)+i\,\mathcal H[s](t)=A(t)\,e^{i(\omega_0 t+\varphi)}
$$

has magnitude equal to the envelope $A(t)$, and a phase that rotates smoothly with delay, so coherent sums remain phase-correct.

**Method (frequency domain).** With $S[k]=\text{FFT}\{s\}[k]$, $k=0,\dots,N_t-1$, the analytic signal keeps positive frequencies (doubled) and removes negative ones:

$$
\tilde S[k]=2\,S[k]\,M[k],\qquad
M[k]=\begin{cases}1 & k+1 < N_t/2\\ 0 & \text{otherwise}\end{cases},
\qquad
\tilde s=\text{IFFT}\{\tilde S\}.
$$

`k+1` is because MATLAB bins are 1-based, and the code copies that:

```python
def analytic_signal(data):
    arr = np.asarray(data, dtype=float)
    nt = arr.shape[-1]
    keep = (np.arange(1, nt + 1) < nt / 2).astype(float)       # M[k], 1-based
    return np.fft.ifft(np.fft.fft(arr, axis=-1) * keep, axis=-1) * 2.0
```

The transform runs along the last axis (`axis=-1`, the time axis) for all $64\times64$ traces at once.

Compared with the textbook (and `scipy.signal.hilbert`) mask — DC $\times1$, bins $1..N_t/2-1$ $\times2$, Nyquist $\times1$, negatives $\times0$ — this one doubles DC, drops Nyquist, and also drops bin $N_t/2-1$. Real transducer data has no energy at DC or near Nyquist, so the images agree; the test `test_analytic_signal_matches_scipy_and_gives_envelope_image` checks the imaginary part against scipy to $10^{-6}$ on a 5 MHz pulse.

Called once, before the pixel loop:

```python
if hilbert_on and not np.iscomplexobj(fmc):
    fmc = analytic_signal(fmc)
traces = fmc.reshape(n * n, nt)            # complex128 from here on
```

Caveat shared with Brain1: the FFT is circular and unpadded, so energy at the very start or end of the record can leak to the other end. Gate or window if the record doesn't start and end near zero.

## B.4 Pixel grid and trace flattening

```python
xx, zz = np.meshgrid(x_grid, z_grid)               # both (Nz, Nx)
pixels = np.column_stack((xx.ravel(), zz.ravel())) # (Npix, 2), row-major
traces = fmc.reshape(n * n, nt)                    # row m = tx*N + rx
pair_indices = np.arange(n * n)[np.newaxis, :]     # (1, N²)
```

- $N_{pix}=101\times101=10201$, $N^2=4096$.
- `ravel()` is row-major, so pixel index $p = i_z N_x + i_x$ and the final `reshape(Nz, Nx)` inverts it exactly.
- Trace row $m=tx\cdot N+rx$ (C-order reshape). The delay matrix below is flattened with the same convention, which is what keeps delays and data matched.

## B.5 Pixel-element distances (per chunk)

The pixel list is processed `pixel_chunk_size = 128` at a time to bound memory.

$$
d_{p,i}=\sqrt{(x_p-x_i)^2+(z_p-z_i)^2}
$$

```python
distances = np.linalg.norm(
    pixels[start:stop, np.newaxis, :] - elements[np.newaxis, :, :], axis=2
)    # shapes: (chunk,1,2) - (1,N,2) -> (chunk,N,2) -> norm -> (chunk,N)
```

Example: the pixel at $(0,\,40\text{ mm})$ and the element at $x=0$ give $d=40$ mm. An element at $x=19.8$ mm gives $\sqrt{19.8^2+40^2}=44.6$ mm.

## B.6 Delay for every transmit/receive pair

$$
\tau_{p,tx,rx}=\frac{d_{p,tx}+d_{p,rx}}{c}
$$

```python
delays = (
    distances[:, :, np.newaxis] + distances[:, np.newaxis, :]
).reshape(stop - start, n * n) / speed
```

`distances[:, :, None]` has shape `(chunk,N,1)` and varies with tx; `distances[:, None, :]` has shape `(chunk,1,N)` and varies with rx. Adding broadcasts to `(chunk,N,N)` with entry $[p,tx,rx]$; the reshape flattens tx-major to `(chunk,N²)`, matching `traces`.

Example: tx = rx = element at $x=0$, pixel $(0,40\text{ mm})$:

$$
\tau=\frac{2\times0.040}{6300}=12.7\ \mu\text{s}
$$

(635 samples at 20 ns). This is the pulse-echo arrival from a reflector at 40 mm.

## B.7 Locating the neighbouring samples

For each delay $\tau$ find $k$ with $t_k\le\tau<t_{k+1}$:

```python
upper = np.searchsorted(time, delays, side="right")   # first index with time > tau  (= k+1)
valid = (delays >= time[0]) & (delays <= time[-1])    # tau inside the recorded window
upper = np.clip(upper, 1, nt - 1)                     # keep indices legal
lower = upper - 1
```

- `side="right"` returns $k+1$ for in-range $\tau$.
- For $\tau<t_0$ it returns 0, and for $\tau>t_{N_t-1}$ it returns $N_t$; `clip` forces these into $[1,N_t-1]$ so the array lookups don't go out of bounds. Their values are garbage but are discarded by `valid` below.
- $\tau=t_{N_t-1}$ exactly gives `upper = Nt`, clipped to `Nt-1`, and the weight comes out as 1, which is correct.

## B.8 Linear interpolation of the (complex) traces

$$
w=\frac{\tau-t_k}{t_{k+1}-t_k},\qquad
\tilde s(\tau)\approx(1-w)\,\tilde s(t_k)+w\,\tilde s(t_{k+1})
$$

```python
t0 = time[lower]
weight = (delays - t0) / (time[upper] - t0)
sampled = (
    traces[pair_indices, lower] * (1.0 - weight)
    + traces[pair_indices, upper] * weight
)
sampled[~valid] = 0.0
```

`traces[pair_indices, lower]` uses NumPy advanced indexing: `pair_indices` is `(1,N²)` (trace row) and `lower` is `(chunk,N²)` (time column); they broadcast to `(chunk,N²)`, so element $[p,m]$ is `traces[m, lower[p,m]]`. `weight` is real, `traces` is complex, so `sampled` is complex `(chunk, N²)`.

Linear interpolation of a complex envelope-modulated carrier is accurate when the carrier has many samples per period (50 here); the error scales like $(\omega_0\Delta t)^2/8$ relative amplitude, about $0.6\%$ worst case at $\omega_0\Delta t = 2\pi\cdot5\text{ MHz}\cdot20\text{ ns}=0.63$ rad, which is negligible for localisation.

`sampled[~valid] = 0.0` implements "outside the record contributes nothing".

## B.9 Coherent sum and magnitude

$$
I(\mathbf r_p)=\Big|\sum_{m=1}^{N^2}\tilde s_m(\tau_{p,m})\Big|
$$

```python
image[start:stop] = np.abs(np.sum(sampled, axis=1))   # complex sum, then |.|
...
return image.reshape(z_grid.size, x_grid.size)
```

Sum first, magnitude second: that is the coherent combination. At a true reflector all 4096 terms have (almost) the same phase, so $|\sum|\approx N^2\bar A$. Elsewhere phases differ and the terms largely cancel, with random-phase sum $\sim N\bar A$ (64 times lower).

## B.10 What the image means

- Value = envelope amplitude of the focused signal at that point, in the same units as the FMC data. No weighting or normalisation is applied. Scripts normalise by `np.max` for display.
- Focal-spot size: laterally about $\lambda\,z/A\approx1.26\times40/39.7\approx1.3$ mm at 40 mm depth, and axially about the pulse length. With `hilbert_on=True` that spot is one smooth blob; with `False` it breaks into $\lambda/2$ fringes.
- The peak is the sum of up to 4096 terms, but note that **our image is not divided by $N^2$**, so absolute levels are not comparable between different arrays.

## B.11 Cost and memory

Per chunk, the large arrays are `delays`, `upper`, `lower`, `weight`, `sampled` at `(128, 4096)`, with `sampled` complex128 (16 bytes). That is about $128\times4096\times(8\cdot4+16)\approx25$ MB. The full 101×101 grid requires about $4.2\times10^7$ complex interpolations. The Hilbert step allocates the complex copy of the FMC once: $64\times64\times N_t\times16$ bytes.

Faster options (not implemented): use reciprocity to halve the pairs, precompute per-element tables like Brain1, or use a GPU.

## B.12 Verification

- `tests/test_tfm.py::test_tfm_localises_and_moves_with_reflector` simulates a point reflector and checks the peak is within 0.5 mm of the truth, at two positions.
- `test_analytic_signal_matches_scipy_and_gives_envelope_image` checks the Hilbert implementation and that the envelope image peaks at the right place and is no smaller than the RF-magnitude image.

Both use the same straight-ray model as the forward simulator, so they test the implementation, not the physics. See [[tfm_algorithm#5. Assumptions and limitations]].

---

# References

- [BRAIN v1 repository](https://github.com/ndtatbristol/brain1), files listed at the top.
- C. Holmes, B. W. Drinkwater, P. D. Wilcox, "Post-processing of the full matrix of ultrasonic transmit–receive array data for non-destructive evaluation", *NDT & E Int.* 38(8), 701–711, 2005.
- See [[tfm_algorithm#References]] for the rest.
