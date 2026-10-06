"""Ray-based (M0-FD+D) model versus experiment_001 SDH_01: FMC, TFM and per-pair amplitudes.

Calibration chain (nothing is tuned against the image being compared):
  1. geometry   : 64 elements at 0.63 mm pitch (element WIDTH is a separate, unconfirmed input)
  2. timing     : acquisition time-zero offset fitted from the back-wall echoes of all pairs,
                  and only applied if it confirms the provisional 0.44 us
  3. positions  : each SDH located from pulse-echo/pitch-catch arrival times
  4. pulse      : Gaussian-cosine fitted to a stack of clean SDH echoes (reference SDH)
  5. amplitude  : ONE global A0 fitted from the reference echo, then fixed for every channel
  6. attenuation: disabled (aluminium baseline)
"""
import json
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import hilbert

from common import ROOT, save_report
from src.calibration import fit_gaussian_cosine, fit_global_amplitude, gaussian_cosine, stack_echoes
from src.experiment_io import (
    ELEMENT_COUNT, ELEMENT_PITCH_M, fit_time_zero_offset, load_hmc,
    nominal_array_coordinates, pair_envelope_peaks,
)
from src.imaging.tfm import tfm_image
from src.models.point_reflector_physics import (
    PropagationPhysics, pair_transfer, simulate_point_reflector_fmc_fd,
)
from src.propagation import element_ray_geometry

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

DEFECTS = import_module("09_experiment001_tfm").DEFECTS

# --- inputs -----------------------------------------------------------------------------
WAVE_SPEED_M_S = 6300.0                 # stored BRAIN material value, not an independent calibration
PROVISIONAL_OFFSET_S = 0.44e-6          # to be confirmed below
OFFSET_TOLERANCE_S = 0.02e-6            # half a sample is 20 ns
ELEMENT_WIDTH_M = 0.53e-3               # UNCONFIRMED: from MAT corner coordinates only
ELEMENT_WIDTH_SENSITIVITY_M = (0.30e-3, 0.53e-3, 0.63e-3)
ATTENUATION_NP_PER_M = 0.0              # disabled for the baseline
REFERENCE_SDH = 1                       # index into the sorted SDH list (central, x ~ 0.5 mm)
REFERENCE_HALF_APERTURE_M = 5e-3        # pulse-echo elements used for the reference echo
HEALTHY_FRACTION = 0.5                  # element is "healthy" if back-wall echo >= this * median
X_MM, Z_MM, STEP_MM = (-19.845, 19.845), (0.0, 125.874), 0.16
OUT = ROOT / "results" / "comparisons" / "ray_model_vs_experiment001"


def width_mm(coord_m, values):
    """-6 dB main-lobe width in mm of a 1-D cut through a peak."""
    mag = np.abs(values)
    p = int(np.argmax(mag))
    thr = mag[p] * 10 ** (-6 / 20)
    below_l = np.flatnonzero(mag[:p] < thr)
    below_r = np.flatnonzero(mag[p + 1:] < thr)
    if below_l.size == 0 or below_r.size == 0:
        return float("nan")
    il, ir = int(below_l[-1]), int(p + 1 + below_r[0])
    xl = np.interp(thr, mag[il:il + 2], coord_m[il:il + 2])
    xr = np.interp(thr, mag[ir - 1:ir + 1][::-1], coord_m[ir - 1:ir + 1][::-1])
    return float((xr - xl) * 1e3)


def to_db(a, ref):
    return 20 * np.log10(np.maximum(a / ref, 1e-6))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    mat = next((DEFECTS / "SDH_01").glob("*_FMC.mat"))
    fmc, time_s, file_elements, recorded, _ = load_hmc(mat)
    elements = nominal_array_coordinates()
    dt = time_s[1] - time_s[0]
    assert fmc.shape[0] == ELEMENT_COUNT
    geometry_error = float(np.max(np.abs(file_elements[:, 0] - elements[:, 0])))
    report["geometry"] = {"elements": ELEMENT_COUNT, "pitch_mm": ELEMENT_PITCH_M * 1e3,
                          "max_difference_to_mat_file_um": geometry_error * 1e6,
                          "element_width_mm": ELEMENT_WIDTH_M * 1e3, "element_width_confirmed": False}
    print(f"geometry: 64 x {ELEMENT_PITCH_M*1e3:.2f} mm, max diff to MAT {geometry_error*1e6:.2f} um")

    # ---- 2. timing: confirm the provisional offset on every file, then apply --------------
    fits = {}
    for name in sorted(p.name for p in DEFECTS.iterdir() if p.is_dir()):
        f, t, el, _, _ = load_hmc(next((DEFECTS / name).glob("*_FMC.mat")))
        fits[name] = fit_time_zero_offset(f, t, nominal_array_coordinates()[:, 0], WAVE_SPEED_M_S)
    offsets = {k: v["offset_s"] for k, v in fits.items()}
    median_offset = float(np.median(list(offsets.values())))
    sdh_offset = offsets["SDH_01"]
    confirmed = abs(sdh_offset - PROVISIONAL_OFFSET_S) <= OFFSET_TOLERANCE_S and \
        abs(median_offset - PROVISIONAL_OFFSET_S) <= OFFSET_TOLERANCE_S
    print("time-zero fits (us):", {k: round(v * 1e6, 4) for k, v in offsets.items()})
    print(f"SDH_01 {sdh_offset*1e6:.4f}, median {median_offset*1e6:.4f}, provisional "
          f"{PROVISIONAL_OFFSET_S*1e6:.2f}: confirmed={confirmed}")
    if not confirmed:
        raise SystemExit("provisional 0.44 us NOT confirmed; stop and review")
    offset = sdh_offset
    report["time_zero"] = {"provisional_us": PROVISIONAL_OFFSET_S * 1e6, "confirmed": confirmed,
                           "fits_us": {k: v * 1e6 for k, v in offsets.items()},
                           "applied_us": offset * 1e6,
                           "sdh_fit": fits["SDH_01"]}

    # ---- element sensitivity (diagnostic) -------------------------------------------------
    env = np.abs(hilbert(fmc, axis=2))
    gate = (time_s > 15.5e-6) & (time_s < 17.0e-6)
    backwall = np.array([env[i, i, gate].max() for i in range(ELEMENT_COUNT)])
    sensitivity = backwall / np.median(backwall)
    healthy = sensitivity >= HEALTHY_FRACTION
    report["element_sensitivity"] = {"values": np.round(sensitivity, 3).tolist(),
                                     "unhealthy_elements_1based": (np.flatnonzero(~healthy) + 1).tolist()}
    print("unhealthy elements:", (np.flatnonzero(~healthy) + 1).tolist())

    # ---- experimental TFM in the offset-corrected frame -----------------------------------
    x = np.arange(X_MM[0], X_MM[1] + 1e-9, STEP_MM) * 1e-3
    z = np.arange(Z_MM[0], Z_MM[1] + 1e-9, STEP_MM) * 1e-3
    exp_img = tfm_image(fmc, time_s - offset, elements, x, z, WAVE_SPEED_M_S)

    # ---- 3. SDH positions from arrival times ----------------------------------------------
    sel = (z > 10e-3) & (z < 45e-3)
    sub = exp_img[sel]
    peaks = np.argwhere((sub == maximum_filter(sub, size=25)) & (sub > 0.3 * sub.max()))
    init = sorted((x[ix], z[sel][iz]) for iz, ix in peaks)
    iu = np.triu_indices(ELEMENT_COUNT)
    pair_ok = healthy[iu[0]] & healthy[iu[1]]
    from scipy.optimize import least_squares
    positions, timing = [], []
    for (x0, z0) in init:
        def travel(p):          # p in mm
            d = np.linalg.norm(elements - p * 1e-3, axis=1)
            return (d[iu[0]] + d[iu[1]]) / WAVE_SPEED_M_S + offset
        guess = travel(np.array([x0, z0]) * 1e3)
        picks = pair_envelope_peaks(env, time_s, guess, 15, iu)
        ok = pair_ok & np.isfinite(picks)
        sol = least_squares(lambda p: (travel(p) - picks)[ok] * 1e9, np.array([x0, z0]) * 1e3,
                            loss="soft_l1", f_scale=20.0, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        sol.x = sol.x * 1e-3
        r = np.abs((travel(sol.x * 1e3) - picks)[ok])
        positions.append(sol.x)
        timing.append({"x_mm": sol.x[0] * 1e3, "z_mm": sol.x[1] * 1e3, "pairs": int(ok.sum()),
                       "median_abs_residual_ns": float(np.median(r) * 1e9),
                       "residual_p90_ns": float(np.percentile(r, 90) * 1e9),
                       "inlier_fraction_100ns": float(np.mean(r < 100e-9))})
    positions = np.array(positions)
    report["sdh_positions_from_arrival_times"] = timing
    for k, t_ in enumerate(timing):
        print(f"SDH{k+1}: ({t_['x_mm']:.2f}, {t_['z_mm']:.2f}) mm  median |timing residual| "
              f"{t_['median_abs_residual_ns']:.1f} ns, p90 {t_['residual_p90_ns']:.1f} ns")

    # ---- 4. pulse from a clean experimental echo; 5. global A0 ----------------------------
    ref = positions[REFERENCE_SDH]
    d_ref = np.linalg.norm(elements - ref, axis=1)
    near = (np.abs(elements[:, 0] - ref[0]) <= REFERENCE_HALF_APERTURE_M) & healthy
    idx = np.flatnonzero(near)
    traces = fmc[idx, idx]
    guess = 2 * d_ref[idx] / WAVE_SPEED_M_S + offset
    lag, stacked = stack_echoes(traces, time_s, guess)
    pulse = fit_gaussian_cosine(lag, stacked)
    print(f"pulse fit ({idx.size} echoes): f0 {pulse.centre_frequency_hz/1e6:.3f} MHz, sigma "
          f"{pulse.sigma_s*1e9:.1f} ns, phase {np.degrees(pulse.phase_rad):.0f} deg, "
          f"relative residual {pulse.relative_residual:.2f}")
    sin_ref = element_ray_geometry(elements, ref)[1]
    base_physics = PropagationPhysics(
        geometric_spreading=True, spreading_reference_amplitude_m=1.0,
        attenuation_np_per_m=ATTENUATION_NP_PER_M, element_width_m=ELEMENT_WIDTH_M,
        directivity_frequency_dependent=True)
    f0, sigma = pulse.centre_frequency_hz, pulse.sigma_s
    measured = np.array([env[i, i, int((g - time_s[0]) / dt) - 15:int((g - time_s[0]) / dt) + 16].max()
                         for i, g in zip(idx, guess)])

    def fit_a0(width):
        phys = replace(base_physics, element_width_m=width)
        amp, _ = pair_transfer(elements, ref, WAVE_SPEED_M_S, f0, phys)
        return fit_global_amplitude(np.diag(amp)[idx], measured)

    a0, a0_spread = fit_a0(ELEMENT_WIDTH_M)
    print(f"global A0 = {a0*1e3:.3f} mm (spread across {idx.size} reference traces {a0_spread*100:.0f}%)")
    report["pulse_fit"] = {"echoes_stacked": int(idx.size), "centre_frequency_MHz": f0 / 1e6,
                           "sigma_ns": sigma * 1e9, "phase_deg": float(np.degrees(pulse.phase_rad)),
                           "relative_residual": pulse.relative_residual,
                           "assumed_before_M0": {"centre_frequency_MHz": 5.0, "sigma_ns": 350.0}}
    report["amplitude"] = {"A0_mm": a0 * 1e3, "reference_sdh": REFERENCE_SDH + 1,
                           "reference_traces": int(idx.size), "relative_spread": a0_spread,
                           "attenuation_np_per_m": ATTENUATION_NP_PER_M}

    # ---- simulate every SDH separately (frequency domain), then sum ----------------------
    physics = replace(base_physics, spreading_reference_amplitude_m=a0)
    sim_time = time_s - offset      # sample k is at time_s[k]; arrival = d/c + offset
    singles = [simulate_point_reflector_fmc_fd(sim_time, elements, p, WAVE_SPEED_M_S, f0, sigma,
                                               physics).fmc for p in positions]
    sim_fmc = np.sum(singles, axis=0)
    sim_img = tfm_image(sim_fmc, time_s - offset, elements, x, z, WAVE_SPEED_M_S)
    np.savez_compressed(DEFECTS / "SDH_01" / "python" / "tfm_simulated_ray_model.npz", image=sim_img,
                        x_m=x, z_m=z, positions_m=positions, A0_m=a0, f0_hz=f0, sigma_s=sigma,
                        offset_s=offset)

    # diagnostics: what if the simulation had the experiment's uneven element sensitivity?
    variants = {}
    for label, w in (("healthy_elements_only", healthy.astype(float)), ("sensitivity_weighted", sensitivity)):
        variants[label] = tfm_image(sim_fmc * (w[:, None] * w[None, :])[:, :, None], time_s - offset,
                                    elements, x, z, WAVE_SPEED_M_S)

    # ---- image metrics --------------------------------------------------------------------
    img_metrics = []
    for k, p in enumerate(positions):
        win = (np.abs(x[None, :] - p[0]) < 2e-3) & (np.abs(z[:, None] - p[1]) < 2e-3)
        row = {"sdh": k + 1}
        for label, img in (("experiment", exp_img), ("simulation", sim_img)):
            masked = np.where(win, img, 0)
            iz, ix = np.unravel_index(np.argmax(masked), img.shape)
            row[label] = {"peak_amplitude": float(img[iz, ix]), "peak_mm": [x[ix] * 1e3, z[iz] * 1e3],
                          "lateral_6dB_mm": width_mm(x, img[iz]), "axial_6dB_mm": width_mm(z, img[:, ix])}
        row["peak_ratio_sim_over_exp_dB"] = float(
            20 * np.log10(row["simulation"]["peak_amplitude"] / row["experiment"]["peak_amplitude"]))
        row["variant_peak_ratio_sim_over_exp_dB"] = {
            lab: float(20 * np.log10(np.max(np.where(win, img, 0)) / row["experiment"]["peak_amplitude"]))
            for lab, img in variants.items()}
        img_metrics.append(row)
    roi = (z[:, None] > 8e-3) & (z[:, None] < 45e-3)
    away = roi & np.all([(np.abs(x[None, :] - p[0]) > 5e-3) | (np.abs(z[:, None] - p[1]) > 5e-3)
                         for p in positions], axis=0)
    ref_peak_exp = img_metrics[REFERENCE_SDH]["experiment"]["peak_amplitude"]
    floor = {lab: {"median_dB_re_ref_peak": float(to_db(np.median(img[away]), ref_peak_exp)),
                   "p99_dB_re_ref_peak": float(to_db(np.percentile(img[away], 99), ref_peak_exp))}
             for lab, img in (("experiment", exp_img), ("simulation", sim_img))}
    r2 = (z > 15e-3) & (z < 40e-3)
    a_, b_ = exp_img[r2].ravel(), sim_img[r2].ravel()
    similarity = float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_)))
    report["image"] = {"per_sdh": img_metrics, "background_outside_5mm_of_any_sdh_z8_45mm": floor,
                       "roi_cosine_similarity_z15_40mm": similarity,
                       "full_depth_note": "the point model contains no back wall, multiples or mode-converted arrivals"}
    for m in img_metrics:
        e, s = m["experiment"], m["simulation"]
        print(f"SDH{m['sdh']}: peak exp {e['peak_amplitude']:.1f} sim {s['peak_amplitude']:.1f} "
              f"({m['peak_ratio_sim_over_exp_dB']:+.1f} dB); lat -6dB exp {e['lateral_6dB_mm']:.2f} "
              f"sim {s['lateral_6dB_mm']:.2f} mm; ax exp {e['axial_6dB_mm']:.2f} sim {s['axial_6dB_mm']:.2f} mm; "
              f"variants {m['variant_peak_ratio_sim_over_exp_dB']}")
    print("background (median / p99, dB re ref SDH peak):", floor)

    # ---- per-pair amplitude comparison ----------------------------------------------------
    win_half = int(0.35e-6 / dt)
    rows = []
    arrivals = [(np.linalg.norm(elements - p, axis=1)[:, None] +
                 np.linalg.norm(elements - p, axis=1)[None, :]) / WAVE_SPEED_M_S + offset for p in positions]
    for k, p in enumerate(positions):
        sim_env = np.abs(hilbert(singles[k], axis=2))
        _, sin_t = element_ray_geometry(elements, p)
        angle = np.degrees(np.arcsin(sin_t))
        for i, j in zip(*iu):
            t_arr = arrivals[k][i, j]
            others = [abs(arrivals[m][i, j] - t_arr) for m in range(len(positions)) if m != k]
            if min(others) < 0.8e-6:
                continue   # another SDH echo overlaps this window
            c = int((t_arr - time_s[0]) / dt)
            if c - win_half < 0 or c + win_half >= time_s.size:
                continue
            rows.append((k, i, j, env[i, j, c - win_half:c + win_half + 1].max(),
                         sim_env[i, j, c - win_half:c + win_half + 1].max(),
                         max(abs(angle[i]), abs(angle[j])), healthy[i] and healthy[j]))
    arr = np.array(rows, dtype=float)
    k_, e_, m_, ang, ok = arr[:, 0], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6].astype(bool)
    ratio_db = 20 * np.log10(e_ / m_)
    pair = {"pairs_total": int(len(arr)), "pairs_healthy": int(ok.sum())}
    for lab, mask in (("all_pairs", np.ones_like(ok)), ("healthy_pairs", ok)):
        pair[lab] = {
            "median_exp_over_model_dB": float(np.median(ratio_db[mask])),
            "robust_std_dB": float(1.4826 * np.median(np.abs(ratio_db[mask] - np.median(ratio_db[mask])))),
            "log_amplitude_correlation": float(np.corrcoef(np.log(e_[mask]), np.log(m_[mask]))[0, 1]),
        }
    pair["by_sdh_healthy_median_dB"] = {int(s + 1): float(np.median(ratio_db[ok & (k_ == s)]))
                                         for s in range(len(positions))}
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60)]
    pair["by_angle_healthy"] = [
        {"angle_deg": [lo, hi], "n": int((ok & (ang >= lo) & (ang < hi)).sum()),
         "median_exp_over_model_dB": float(np.median(ratio_db[ok & (ang >= lo) & (ang < hi)]))}
        for lo, hi in bins if (ok & (ang >= lo) & (ang < hi)).sum() > 5]
    # element-width sensitivity: refit A0 per width, then the pair-amplitude scatter
    sens = {}
    for width in ELEMENT_WIDTH_SENSITIVITY_M:
        a0w, _ = fit_a0(width)
        phys = replace(base_physics, element_width_m=width, spreading_reference_amplitude_m=a0w)
        preds = []
        for k, p in enumerate(positions):
            amp, _ = pair_transfer(elements, p, WAVE_SPEED_M_S, f0, phys)
            for kk, i, j in zip(k_[ok & (k_ == k)], arr[ok & (k_ == k), 1].astype(int), arr[ok & (k_ == k), 2].astype(int)):
                preds.append(amp[i, j])
        preds = np.array(preds)
        r = 20 * np.log10(e_[ok] / preds)
        sens[f"{width*1e3:.2f}mm"] = {"A0_mm": a0w * 1e3, "median_dB": float(np.median(r)),
                                      "robust_std_dB": float(1.4826 * np.median(np.abs(r - np.median(r)))),
                                      "log_corr": float(np.corrcoef(np.log(e_[ok]), np.log(preds))[0, 1])}
    pair["element_width_sensitivity"] = sens
    report["pair_amplitudes"] = pair
    print("pair amplitudes:", json.dumps({k: pair[k] for k in ("pairs_total", "pairs_healthy", "all_pairs", "healthy_pairs")}))
    print("by angle:", pair["by_angle_healthy"])
    print("width sensitivity:", sens)

    save_report(OUT / "metrics.json", report)

    # ---- figures ---------------------------------------------------------------------------
    ref_db = exp_img.max()
    ext = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 11), sharey=True)
    for ax, img, title in ((axes[0], exp_img, "Experiment (SDH_01)"), (axes[1], sim_img, "Ray model (M0-FD+D, A0 fixed)")):
        im = ax.imshow(to_db(img, ref_db), extent=ext, aspect="equal", cmap="jet", vmin=-40, vmax=0)
        ax.set(title=title, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="dB re experimental maximum")
    fig.savefig(OUT / "tfm_full_depth.png", dpi=140); plt.close(fig)

    zs = (z > 12e-3) & (z < 40e-3)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    for c_, (img, title) in enumerate(((exp_img, "Experiment"), (sim_img, "Ray model"))):
        ax = fig.add_subplot(gs[0, c_])
        im = ax.imshow(to_db(img[zs], ref_db), extent=[ext[0], ext[1], z[zs][-1] * 1e3, z[zs][0] * 1e3],
                       aspect="equal", cmap="jet", vmin=-40, vmax=0)
        ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.5, label="dB re experimental maximum")
    px, pz = positions[REFERENCE_SDH]
    ax = fig.add_subplot(gs[1, 0:2])
    ix = int(np.argmin(np.abs(x - px))); iz = int(np.argmin(np.abs(z - pz)))
    ax.plot(x * 1e3, to_db(exp_img[iz], ref_db), label="experiment (lateral cut)")
    ax.plot(x * 1e3, to_db(sim_img[iz], ref_db), "--", label="ray model")
    ax.set(xlabel="x (mm)", ylabel="dB re experimental max", ylim=(-50, 3),
           title=f"Lateral cut through SDH {REFERENCE_SDH+1} at z = {pz*1e3:.1f} mm"); ax.legend(); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(to_db(exp_img[:, ix], ref_db), z * 1e3, label="experiment")
    ax.plot(to_db(sim_img[:, ix], ref_db), z * 1e3, "--", label="ray model")
    ax.set(xlabel="dB", ylabel="z (mm)", ylim=(pz * 1e3 + 8, pz * 1e3 - 8), xlim=(-50, 3), title="Axial cut"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "tfm_roi_and_cuts.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for s in range(len(positions)):
        mk = (k_ == s) & ok
        axes[0].scatter(m_[mk], e_[mk], s=4, alpha=.5, label=f"SDH {s+1}")
    lim = [min(m_.min(), e_.min()), max(m_.max(), e_.max())]
    axes[0].plot(lim, lim, "k--", lw=1); axes[0].set(xscale="log", yscale="log", xlabel="model echo envelope", ylabel="experimental echo envelope", title="Per-pair echo amplitude (healthy elements)"); axes[0].legend(markerscale=3)
    axes[1].scatter(ang[ok], ratio_db[ok], s=4, alpha=.4); axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="max(|theta_tx|,|theta_rx|) (deg)", ylabel="experiment / model (dB)", title="Amplitude error vs ray angle"); axes[1].grid(alpha=.3)
    axes[2].bar(np.arange(1, 65), sensitivity, color=np.where(healthy, "C0", "C3")); axes[2].axhline(HEALTHY_FRACTION, color="k", lw=1)
    axes[2].set(xlabel="element", ylabel="back-wall echo / median", title="Element sensitivity (red = excluded)")
    fig.tight_layout(); fig.savefig(OUT / "pair_amplitudes.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(lag * 1e6, stacked, label=f"stack of {idx.size} SDH echoes")
    axes[0].plot(lag * 1e6, gaussian_cosine(lag, pulse.amplitude, f0, sigma, pulse.phase_rad), "--", label="Gaussian-cosine fit")
    axes[0].plot(lag * 1e6, gaussian_cosine(lag, pulse.amplitude, f0, sigma, 0.0), ":", label="same, phase = 0 (as in model)")
    axes[0].set(xlabel="time from envelope peak (us)", title="Reference echo and fitted pulse"); axes[0].legend(); axes[0].grid(alpha=.3)
    freq = np.fft.rfftfreq(lag.size * 4, lag[1] - lag[0])
    axes[1].plot(freq / 1e6, np.abs(np.fft.rfft(stacked, lag.size * 4)) / np.abs(np.fft.rfft(stacked, lag.size * 4)).max(), label="stack")
    axes[1].plot(freq / 1e6, np.abs(np.fft.rfft(gaussian_cosine(lag, pulse.amplitude, f0, sigma, 0.0), lag.size * 4)) / np.abs(np.fft.rfft(stacked, lag.size * 4)).max(), "--", label="model pulse")
    axes[1].set(xlim=(0, 12), xlabel="frequency (MHz)", title="Spectrum"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "pulse_fit.png", dpi=140); plt.close(fig)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
