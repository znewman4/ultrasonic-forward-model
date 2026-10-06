"""Plate ray model (back wall + multiples + scatterer paths via the back wall) vs experiment_001 SDH_01.

Builds on the calibration of ``10_ray_model_vs_experiment001.py`` (read from its metrics.json):
geometry, applied time-zero offset, SDH positions, SDH pulse and SDH A0.  New here:
  * plate thickness T from the all-pair back-wall fit (c = 6300 m/s)
  * back-wall pulse fitted to unclipped first back-wall echoes
  * ONE global back-wall amplitude A_bw from the first back-wall echo (median over healthy
    pulse-echo traces, which is unaffected by clipping while < 50 % of traces are clipped)
  * attenuation off, round-trip factor 1 (lossless baseline); the measured loss is reported
"""
import json
import sys
from dataclasses import replace
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from common import ROOT, save_report
from src.calibration import fit_gaussian_cosine, stack_echoes
from src.experiment_io import load_hmc, nominal_array_coordinates
from src.imaging.tfm import tfm_image
from src.models.plate_ray import back_wall_paths, scatterer_paths, synthesize
from src.models.point_reflector_physics import PropagationPhysics

sys.path.insert(0, str(ROOT / "scripts"))
DEFECTS = import_module("09_experiment001_tfm").DEFECTS

C = 6300.0
ELEMENT_WIDTH_M = 0.53e-3          # UNCONFIRMED (as in script 10)
X_MM, Z_MM, STEP_MM = (-19.845, 19.845), (0.0, 125.874), 0.16
PREV = ROOT / "results" / "comparisons" / "ray_model_vs_experiment001"
OUT = ROOT / "results" / "comparisons" / "plate_ray_model_vs_experiment001"
CLIP = 0.99


def to_db(a, ref):
    return 20 * np.log10(np.maximum(a / ref, 1e-6))


def window_peak(img, x, z, x0, z0, hx=2.5e-3, hz=2.5e-3):
    win = (np.abs(x[None, :] - x0) < hx) & (np.abs(z[:, None] - z0) < hz)
    masked = np.where(win, img, 0.0)
    iz, ix = np.unravel_index(np.argmax(masked), img.shape)
    return float(img[iz, ix]), float(x[ix] * 1e3), float(z[iz] * 1e3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev = json.loads((PREV / "metrics.json").read_text())
    offset = prev["time_zero"]["applied_us"] * 1e-6
    thickness = prev["time_zero"]["sdh_fit"]["thickness_m"]
    healthy = np.array(prev["element_sensitivity"]["values"]) >= 0.5
    positions = np.array([[p["x_mm"], p["z_mm"]] for p in prev["sdh_positions_from_arrival_times"]]) * 1e-3
    f0_sdh = prev["pulse_fit"]["centre_frequency_MHz"] * 1e6
    sigma_sdh = prev["pulse_fit"]["sigma_ns"] * 1e-9
    a0 = prev["amplitude"]["A0_mm"] * 1e-3

    fmc, time_s, _, _, _ = load_hmc(next((DEFECTS / "SDH_01").glob("*_FMC.mat")))
    elements = nominal_array_coordinates()
    n = elements.shape[0]
    dt = time_s[1] - time_s[0]
    env = np.abs(hilbert(fmc, axis=2))
    sim_time = time_s - offset
    freq = np.fft.rfftfreq(time_s.size, dt)
    print(f"T = {thickness*1e3:.3f} mm, offset {offset*1e6:.4f} us, healthy {healthy.sum()}")

    # ---- back-wall pulse (unclipped, healthy, first echo) and A_bw ------------------------
    guess = 2 * thickness / C + offset
    g = (time_s > guess - 0.8e-6) & (time_s < guess + 0.8e-6)
    raw_peak = np.array([np.abs(fmc[i, i, g]).max() for i in range(n)])
    unclipped = healthy & (raw_peak < CLIP)
    idx = np.flatnonzero(unclipped)
    lag, stacked = stack_echoes(fmc[idx, idx], time_s, np.full(idx.size, guess))
    pulse_bw = fit_gaussian_cosine(lag, stacked)
    f0_bw, sigma_bw = pulse_bw.centre_frequency_hz, pulse_bw.sigma_s
    print(f"BW pulse ({idx.size} unclipped echoes): f0 {f0_bw/1e6:.2f} MHz sigma {sigma_bw*1e9:.0f} ns "
          f"phase {np.degrees(pulse_bw.phase_rad):.0f} deg resid {pulse_bw.relative_residual:.2f}")

    base = PropagationPhysics(geometric_spreading=True, spreading_reference_amplitude_m=1.0,
                              element_width_m=ELEMENT_WIDTH_M, directivity_frequency_dependent=True)
    unit = np.diag(back_wall_paths(elements, thickness, C, f0_bw, base, 1.0, 1)[0].amplitude)
    hs = np.flatnonzero(healthy)
    measured = np.array([env[i, i, g].max() for i in hs])
    a_bw = float(np.median(measured / unit[hs]))
    print(f"A_bw = {a_bw*1e3:.3f} mm^0.5*..  (clipped healthy PE traces: {int((healthy & (raw_peak >= CLIP)).sum())}"
          f" of {healthy.sum()})")

    # ---- simulate -------------------------------------------------------------------------
    bw_sets = back_wall_paths(elements, thickness, C, f0_bw, base, a_bw, max_round_trips=3, frequency_hz=freq)
    fmc_bw, dropped_bw = synthesize(sim_time, bw_sets, f0_bw, sigma_bw)
    phys_sdh = replace(base, spreading_reference_amplitude_m=a0)
    fam = {k: [] for k in ("DD", "DB", "BD", "BB")}
    for p in positions:
        for ps in scatterer_paths(elements, p, thickness, C, f0_sdh, phys_sdh, frequency_hz=freq):
            fam[ps.label].append(ps)
    fmc_f = {k: synthesize(sim_time, v, f0_sdh, sigma_sdh)[0] for k, v in fam.items()}
    sim = {
        "direct_only": fmc_f["DD"],
        "no_mixed_paths": fmc_bw + fmc_f["DD"] + fmc_f["BB"],
        "full": fmc_bw + fmc_f["DD"] + fmc_f["BB"] + fmc_f["DB"] + fmc_f["BD"],
    }
    print("back-wall arrivals dropped (beyond record):", dropped_bw, "pair-paths")

    # ---- images ---------------------------------------------------------------------------
    x = np.arange(X_MM[0], X_MM[1] + 1e-9, STEP_MM) * 1e-3
    z = np.arange(Z_MM[0], Z_MM[1] + 1e-9, STEP_MM) * 1e-3
    imgs = {"experiment": tfm_image(fmc, sim_time, elements, x, z, C)}
    for k in ("no_mixed_paths", "full"):
        imgs[k] = tfm_image(sim[k], sim_time, elements, x, z, C)
    imgs["direct_only"] = np.load(DEFECTS / "SDH_01" / "python" / "tfm_simulated_ray_model.npz")["image"]
    np.savez_compressed(DEFECTS / "SDH_01" / "python" / "tfm_simulated_plate_ray_model.npz",
                        image=imgs["full"], x_m=x, z_m=z, thickness_m=thickness, A_bw=a_bw,
                        A0_m=a0, f0_bw_hz=f0_bw, sigma_bw_s=sigma_bw, offset_s=offset)
    ref_peak = window_peak(imgs["experiment"], x, z, positions[1, 0], positions[1, 1])[0]

    report = {"inputs": {"thickness_mm": thickness * 1e3, "A_bw": a_bw, "A0_mm": a0 * 1e3,
                         "bw_pulse": {"f0_MHz": f0_bw / 1e6, "sigma_ns": sigma_bw * 1e9,
                                      "phase_deg": float(np.degrees(pulse_bw.phase_rad)),
                                      "relative_residual": pulse_bw.relative_residual,
                                      "echoes": int(idx.size)},
                         "clipped_healthy_pulse_echo_traces": int((healthy & (raw_peak >= CLIP)).sum()),
                         "attenuation": "off", "round_trip_factor": 1.0}}

    # back-wall ridge and ghost peaks
    ridge = {}
    for label, z_nom in (("BW1", thickness), ("BW2", 2 * thickness)):
        row = {}
        for k in ("experiment", "direct_only", "no_mixed_paths", "full"):
            peak, xm, zm = window_peak(imgs[k], x, z, 0.0, z_nom, hx=20e-3, hz=3e-3)
            row[k] = {"peak": peak, "z_mm": zm}
        row["full_over_exp_dB"] = 20 * np.log10(row["full"]["peak"] / row["experiment"]["peak"])
        ridge[label] = row
    ghosts = []
    for k, p in enumerate(positions):
        row = {"sdh": k + 1, "expected_mm": [p[0] * 1e3, (2 * thickness - p[1]) * 1e3]}
        for lab in ("experiment", "no_mixed_paths", "full"):
            peak, xm, zm = window_peak(imgs[lab], x, z, p[0], 2 * thickness - p[1])
            row[lab] = {"peak": peak, "at_mm": [xm, zm]}
        row["full_over_exp_dB"] = 20 * np.log10(row["full"]["peak"] / row["experiment"]["peak"])
        ghosts.append(row)
    report["back_wall_ridges"] = ridge
    report["sdh_ghost_images"] = ghosts

    # background and global similarity
    def background(img, z_lo, z_hi):
        m = (z[:, None] > z_lo) & (z[:, None] < z_hi)
        for p in positions:
            m = m & ((np.abs(x[None, :] - p[0]) > 5e-3) | (np.abs(z[:, None] - p[1]) > 5e-3))
        return {"median_dB": float(to_db(np.median(img[m]), ref_peak)),
                "p99_dB": float(to_db(np.percentile(img[m], 99), ref_peak))}
    report["background_arcs_region_z36_47mm"] = {k: background(v, 36e-3, 47e-3) for k, v in imgs.items()}
    report["background_z52_95mm"] = {k: background(v, 52e-3, 95e-3) for k, v in imgs.items()}
    roi = (z > 8e-3)
    ref_img = np.clip(to_db(imgs["experiment"][roi], imgs["experiment"].max()), -40, 0).ravel()
    report["log_image_correlation_z8_126mm"] = {
        k: float(np.corrcoef(ref_img, np.clip(to_db(imgs[k][roi], imgs["experiment"].max()), -40, 0).ravel())[0, 1])
        for k in ("direct_only", "no_mixed_paths", "full")}
    for key in ("back_wall_ridges", "background_arcs_region_z36_47mm", "log_image_correlation_z8_126mm"):
        print(key, json.dumps(report[key], default=float))
    for gh in ghosts:
        print(f"ghost SDH{gh['sdh']} at {np.round(gh['expected_mm'],1)} mm: exp {gh['experiment']['peak']:.1f} "
              f"full {gh['full']['peak']:.1f} ({gh['full_over_exp_dB']:+.1f} dB)")

    # ---- per-pair back-wall amplitudes ----------------------------------------------------
    sim_env = np.abs(hilbert(fmc_bw, axis=2))
    iu = np.triu_indices(n)
    dxm = np.abs(elements[iu[0], 0] - elements[iu[1], 0])
    half = int(0.35e-6 / dt)
    rows = {"BW1": [], "BW2": []}
    for name, mult in (("BW1", 2.0), ("BW2", 4.0)):
        arrival = np.sqrt(dxm**2 + (mult * thickness) ** 2) / C + offset
        for k, (i, j) in enumerate(zip(*iu)):
            if not (healthy[i] and healthy[j]):
                continue
            c = int((arrival[k] - time_s[0]) / dt)
            if c - half < 0 or c + half >= time_s.size:
                continue
            raw = np.abs(fmc[i, j, c - half:c + half + 1]).max()
            rows[name].append((dxm[k], env[i, j, c - half:c + half + 1].max(),
                               sim_env[i, j, c - half:c + half + 1].max(), raw >= CLIP))
    pair = {}
    bins = [(0, 10), (10, 20), (20, 30), (30, 45)]
    for name, r in rows.items():
        a = np.array(r, dtype=float)
        use = a[:, 3] == 0
        ratio = 20 * np.log10(a[use, 1] / a[use, 2])
        pair[name] = {"pairs_unclipped": int(use.sum()), "pairs_clipped_excluded": int((~use).sum()),
                      "median_exp_over_model_dB": float(np.median(ratio)),
                      "robust_std_dB": float(1.4826 * np.median(np.abs(ratio - np.median(ratio)))),
                      "by_dx_mm": [{"dx_mm": [lo * 1, hi * 1],
                                    "median_dB": float(np.median(ratio[(a[use, 0] * 1e3 >= lo) & (a[use, 0] * 1e3 < hi)]))}
                                   for lo, hi in bins if ((a[use, 0] * 1e3 >= lo) & (a[use, 0] * 1e3 < hi)).sum() > 5]}
    # decay between echoes along pulse-echo traces (clipping-robust: medians)
    g2 = (time_s > 4 * thickness / C + offset - 0.8e-6) & (time_s < 4 * thickness / C + offset + 0.8e-6)
    e1 = np.array([env[i, i, g].max() for i in hs]); e2 = np.array([env[i, i, g2].max() for i in hs])
    m1 = np.array([sim_env[i, i, g].max() for i in hs]); m2 = np.array([sim_env[i, i, g2].max() for i in hs])
    observed = float(np.median(e2) / np.median(e1)); predicted = float(np.median(m2) / np.median(m1))
    d2 = 2 * thickness * 2   # extra path between echo 1 and 2 (round trip)
    pair["echo2_over_echo1_pulse_echo"] = {
        "observed": observed, "lossless_2D_model": predicted, "observed_over_model_dB": 20 * np.log10(observed / predicted),
        "implied_round_trip_factor": observed / predicted,
        "equivalent_attenuation_dB_per_m_if_all_loss_is_attenuation":
            float(-20 * np.log10(observed / predicted) / (2 * thickness)),
        "note": "echo 1 is partly clipped, so observed ratio is, if anything, an overestimate"}
    report["back_wall_pair_amplitudes"] = pair
    print("pair BW:", json.dumps(pair, default=float))
    save_report(OUT / "metrics.json", report)

    # ---- figures ----------------------------------------------------------------------------
    ref_db = imgs["experiment"].max()
    ext = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
    fig, axes = plt.subplots(1, 4, figsize=(20, 11), sharey=True)
    for ax, k, title in zip(axes, ("experiment", "direct_only", "no_mixed_paths", "full"),
                            ("Experiment", "Point scatterers only\n(previous model)",
                             "+ back wall echoes + hole ghosts\n(no mixed paths)", "+ mixed direct/back-wall paths\n(full plate model)")):
        im = ax.imshow(to_db(imgs[k], ref_db), extent=ext, aspect="equal", cmap="jet", vmin=-40, vmax=0)
        ax.set(title=title, xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, shrink=0.5, label="dB re experimental maximum")
    fig.savefig(OUT / "tfm_full_depth.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (i, j, label) in zip(axes, ((19, 19, "pulse-echo, element 20"), (31, 31, "pulse-echo, element 32"),
                                          (10, 50, "pitch-catch, elements 11 -> 51"))):
        ax.plot(time_s * 1e6, fmc[i, j], lw=.6, label="experiment (RF)")
        ax.plot(time_s * 1e6, sim["full"][i, j], lw=.6, alpha=.8, label="plate ray model (RF)")
        ax.plot(time_s * 1e6, env[i, j], "C0", lw=1.2, ls=":")
        ax.plot(time_s * 1e6, np.abs(hilbert(sim["full"][i, j])), "C1", lw=1.2, ls=":")
        ax.set(title=label, ylabel="amplitude", ylim=(-1.1, 1.1)); ax.grid(alpha=.3)
    axes[0].legend(ncol=2); axes[-1].set_xlabel("time (us)")
    fig.tight_layout(); fig.savefig(OUT / "ascans.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ("BW1", "BW2")):
        a = np.array(rows[name], dtype=float)
        ax.scatter(a[a[:, 3] == 0, 0] * 1e3, 20 * np.log10(a[a[:, 3] == 0, 1] / a[a[:, 3] == 0, 2]), s=5, alpha=.4,
                   label="unclipped")
        ax.scatter(a[a[:, 3] == 1, 0] * 1e3, 20 * np.log10(a[a[:, 3] == 1, 1] / a[a[:, 3] == 1, 2]), s=5, alpha=.4,
                   color="C3", label="clipped (lower bound)")
        ax.axhline(0, color="k", lw=1); ax.legend(); ax.grid(alpha=.3)
        ax.set(xlabel="|x_tx - x_rx| (mm)", ylabel="experiment / model (dB)", title=f"{name} amplitude, healthy elements")
    fig.tight_layout(); fig.savefig(OUT / "back_wall_pair_amplitudes.png", dpi=130); plt.close(fig)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
