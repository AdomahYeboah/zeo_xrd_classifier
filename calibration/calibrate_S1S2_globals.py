"""
S1/S2 global calibration (class-blind outputs only).

S1: per-class comparison of SELECT idealized IZA references (FAU/FER/LTA/MFI .xy)
    against class anchors; globals pooled by MEDIAN over the 4 comparisons.
S2: Oviedo-style statistics measured on the 7 anchors (FWHM quantiles, peak shifts
    vs IZA tops, exp/sim height ratios, residual noise). No per-class outputs used
    downstream: both dicts are single global sets, no framework branching.

Usage:
    python calibrate_S1S2_globals.py --base "D:\\IT\\manuscript_IMMI - Copy (3) - Copy\\project_exp" --out-dir .../calibration_S1S2
"""

import argparse
import json
import os
from glob import glob

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks, savgol_filter

TTH_MIN, TTH_MAX, STEP = 5.0, 50.0, 0.05
TTH_GRID = np.arange(TTH_MIN, TTH_MAX + STEP, STEP)
N_GRID = len(TTH_GRID)
TARGET_FRAMEWORKS = ["FAU", "FER", "LTA", "MFI"]


def load_xy(path):
    d = np.loadtxt(path)
    tth, y = d[:, 0], d[:, 1]
    yi = np.interp(TTH_GRID, tth, y, left=0, right=0)
    yi = savgol_filter(yi, 21, 3)
    yi = yi - yi.min()
    yi = np.clip(yi, 0, None)
    m = yi.max()
    return (yi / m).astype(np.float32) if m > 0 else yi


def label_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0].split("(")[0].strip().upper()


def measure_fwhm(y, height=0.05, distance=5):
    peaks, _ = find_peaks(y, height=height, distance=distance)
    out = []
    for pk in peaks:
        left = pk
        while left > 0 and y[left] > y[pk] / 2:
            left -= 1
        right = pk
        while right < len(y) - 1 and y[right] > y[pk] / 2:
            right += 1
        f = (right - left) * STEP
        if 0.02 < f < 8.0:
            out.append(f)
    return out, peaks


def bg_stats(y):
    bl = minimum_filter1d(y, size=40)
    return float(bl.mean()), float(bl.max()), float(TTH_GRID[int(np.argmax(bl))])


def tail_ratios(y):
    _, peaks = measure_fwhm(y)
    ratios = []
    for pk in peaks:
        if y[pk] < 0.3:
            continue
        half = y[pk] / 2
        right = pk
        while right < N_GRID - 1 and y[right] > half:
            right += 1
        hwhm_pts = right - pk
        if hwhm_pts < 2:
            continue
        far = pk + 2 * hwhm_pts
        if far >= N_GRID:
            continue
        i_meas = float(y[far])
        i_gauss = float(y[pk]) * np.exp(-np.log(2) * 4)
        if i_gauss > 1e-4:
            ratios.append(i_meas / i_gauss)
    return ratios


def height_at(y, two_theta):
    return float(y[int(np.argmin(np.abs(TTH_GRID - two_theta)))])


def main():
    ap = argparse.ArgumentParser(description="S1/S2 class-blind global calibration")
    ap.add_argument("--base", default=".")
    ap.add_argument("--iza-xy", default=None)
    ap.add_argument("--exp-train", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    iza_xy = a.iza_xy or os.path.join(a.base, "iza_xy")
    exp_train = a.exp_train or os.path.join(a.base, "exp_label_copy")
    out_dir = a.out_dir or os.path.join(a.base, "calibration_S1S2")
    os.makedirs(out_dir, exist_ok=True)

    iza = {}
    for fw in TARGET_FRAMEWORKS:
        p = os.path.join(iza_xy, f"{fw}.xy")
        assert os.path.exists(p), f"missing IZA reference {p}"
        iza[fw] = load_xy(p)
    print(f"IZA references: {sorted(iza)}")

    exp_paths = sorted(glob(os.path.join(exp_train, "*.xy")))
    by_fw = {fw: [] for fw in TARGET_FRAMEWORKS}
    for p in exp_paths:
        lbl = label_from_path(p)
        if lbl in by_fw:
            by_fw[lbl].append(load_xy(p))
    for fw in TARGET_FRAMEWORKS:
        print(f"  anchors {fw}: {len(by_fw[fw])}")

    per_class, all_fwhm, all_shifts, all_ratios, all_noise = {}, [], [], [], []
    for fw in TARGET_FRAMEWORKS:
        ys = by_fw[fw]
        ref = iza[fw]
        pk, _ = find_peaks(ref, height=0.1, distance=2)
        tops = sorted(sorted(pk, key=lambda j: -ref[j])[:5])
        tops = [round(float(TTH_GRID[i]), 2) for i in tops]
        fmeans = []
        for y in ys:
            f, pks = measure_fwhm(y)
            all_fwhm.extend(f)
            if f:
                fmeans.append(float(np.mean(f)))
            r = savgol_filter(y, 51, 3)
            all_noise.append(float((y - r).std()))
            for t in tops:
                e_h = height_at(y, t)
                s_h = height_at(ref, t)
                all_ratios.append(e_h / (s_h + 1e-9))
                near = pks[np.argmin(np.abs(TTH_GRID[pks] - t))] if len(pks) else None
                if near is not None and abs(float(TTH_GRID[near]) - t) < 1.0:
                    all_shifts.append(abs(float(TTH_GRID[near]) - t))
        tails = [t for y in ys for t in tail_ratios(y)]
        bgs = [bg_stats(y) for y in ys]
        per_class[fw] = {
            "n_exp": len(ys),
            "exp_fwhm_mean": round(float(np.median(fmeans)), 3) if fmeans else None,
            "bg": [{"mean": round(b[0], 4), "max": round(b[1], 4), "at": b[2]} for b in bgs],
            "iza_tops": tops,
            "tail_mean": round(float(np.mean(tails)), 3) if tails else None,
        }
        print(f"  {fw}: fwhm_mean={per_class[fw]['exp_fwhm_mean']} tail={per_class[fw]['tail_mean']}")

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    class_means = [per_class[fw]["exp_fwhm_mean"] for fw in TARGET_FRAMEWORKS
                   if per_class[fw]["exp_fwhm_mean"] is not None]
    med_fw = float(np.median(class_means))
    s1_broad = [round(clamp(0.6 * med_fw, 0.30, 3.00), 2),
                round(clamp(1.5 * med_fw, 0.30, 3.00), 2)]
    s1_eta = [0.30, 0.80] if float(np.mean(
        [t for t in [per_class[fw]["tail_mean"] for fw in TARGET_FRAMEWORKS] if t is not None])) > 1.2 else [0.00, 0.50]
    bg_pos, bg_means, bg_maxes, bg_hits = [], [], [], 0
    for fw in TARGET_FRAMEWORKS:
        for b in per_class[fw]["bg"]:
            bg_means.append(b["mean"])
            bg_maxes.append(b["max"])
            if b["max"] > 0.15:
                bg_hits += 1
                bg_pos.append(b["at"])
    n_anchors = sum(per_class[fw]["n_exp"] for fw in TARGET_FRAMEWORKS)
    q25, q75 = (round(float(np.percentile(bg_pos, 25)), 1), round(float(np.percentile(bg_pos, 75)), 1)) if bg_pos else (20.0, 20.0)
    S1 = {
        "name": "S1 comparison-calibrated global (median over 4 IZA-vs-anchor comparisons)",
        "broadening": {"fwhm_min": s1_broad[0], "fwhm_max": s1_broad[1],
                       "eta_min": s1_eta[0], "eta_max": s1_eta[1]},
        "orientation": {"r_min": 1.0, "r_max": 1.0},
        "strain_eps": 0.002,
        "amorphous": {"center_min": q25,
                      "center_max": q75,
                      "center_note": "interquartile interval of anchor bg positions; single hump",
                      "center_std": 3.0,
                      "amp_min": round(min(bg_means) * 0.5, 2),
                      "amp_max": round(max(bg_maxes), 2),
                      "sig_min": 1.5, "sig_max": 8.0,
                      "apply_prob": round(bg_hits / max(1, n_anchors), 2)},
        "low_angle_hump": None,
        "boost_centers": None, "hi_amplify": None, "low_suppress": None,
        "dropout_prob": 0.0,
    }

    f10, f90 = float(np.percentile(all_fwhm, 10)), float(np.percentile(all_fwhm, 90))
    r10, r90 = float(np.percentile(all_ratios, 10)), float(np.percentile(all_ratios, 90))
    sh90 = float(np.percentile(all_shifts, 90)) if all_shifts else 0.08
    n10, n90 = float(np.percentile(all_noise, 10)), float(np.percentile(all_noise, 90))
    S2 = {
        "name": "S2 Oviedo-style measured global (anchor quantiles)",
        "broadening": {"fwhm_min": round(clamp(f10, 0.20, 3.00), 2),
                       "fwhm_max": round(clamp(f90, 0.20, 3.00), 2),
                       "eta_min": 0.00, "eta_max": 0.50},
        "orientation": {"r_min": 1.0, "r_max": 1.0},
        "strain_eps": 0.002,
        "amorphous": {"apply_prob": 0.0},
        "low_angle_hump": None,
        "boost_centers": None, "hi_amplify": None, "low_suppress": None,
        "dropout_prob": 0.0,
        "rescale": {"factor_min": round(clamp(r10, 0.30, 3.00), 2),
                    "factor_max": round(clamp(r90, 0.30, 3.00), 2),
                    "apply_prob": 0.85},
        "shift_max": round(clamp(sh90, 0.02, 0.30), 2),
        "noise_scale": [round(clamp(n10, 0.005, 0.20), 3),
                        round(clamp(n90, 0.005, 0.20), 3)],
    }

    stats = {
        "n_anchor_fwhm": len(all_fwhm),
        "anchor_fwhm_p10_p50_p90": [round(f10, 3), round(float(np.median(all_fwhm)), 3), round(f90, 3)],
        "n_shifts": len(all_shifts),
        "shift_abs_median_p90": [round(float(np.median(all_shifts)), 3), round(sh90, 3)] if all_shifts else None,
        "n_ratios": len(all_ratios),
        "ratio_p10_p50_p90": [round(r10, 3), round(float(np.median(all_ratios)), 3), round(r90, 3)],
        "noise_p10_p90": [round(n10, 4), round(n90, 4)],
        "per_class": per_class,
    }
    with open(os.path.join(out_dir, "globals_S1S2.json"), "w") as f:
        json.dump({"S1": S1, "S2": S2, "stats": stats,
                   "provenance": "S1 medians over 4 IZA-vs-anchor comparisons; "
                   "S2 quantiles over pooled anchor measurements; orientation identity, "
                   "strain 0.002, S1 widths, S2 eta are uncalibrated carryovers"},
                  f, indent=2)
    print("\nS1 broadening:", S1["broadening"], "eta:", S1["broadening"]["eta_min"], S1["broadening"]["eta_max"])
    print("S1 amorphous:", {k: S1["amorphous"][k] for k in ("center_min", "amp_min", "amp_max", "apply_prob")})
    print("S2 broadening:", S2["broadening"])
    print("S2 rescale:", S2["rescale"], "shift_max:", S2["shift_max"], "noise:", S2["noise_scale"])
    print("stats:", json.dumps({k: v for k, v in stats.items() if k != "per_class"}, indent=2))
    print(f"\nSaved globals_S1S2.json to {out_dir}")


if __name__ == "__main__":
    main()
