"""Constrained Caglioti fit: binned medians + NNLS + gates.

Fits angle-dependent broadening FWHM^2 = U*tan^2 + V*tan + W from
unlabelled patterns, with experimental training patterns as a check set.

Usage:
    python fit_caglioti.py --unlab-dir <unlabelled_xy> --exp-train-dir <exp_label_copy>
"""

import argparse
import json
import os
from glob import glob

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from zeolite_preproc import N_GRID, STEP, TTH_GRID, preprocess_exp


def peak_widths(y, hmin=0.15):
    out = []
    if y.max() <= 0:
        return out
    y_sm = savgol_filter(y, 11, 3)
    pks, _ = find_peaks(y_sm, height=max(hmin, y_sm.max() * 0.15),
                        distance=max(2, int(0.25 / STEP)))
    for pk in pks:
        left = pk
        while left > 0 and y[left] > y[pk] / 2:
            left -= 1
        right = pk
        while right < len(y) - 1 and y[right] > y[pk] / 2:
            right += 1
        f = (right - left) * STEP
        if 0.05 < f < 5.0:
            out.append((round(float(TTH_GRID[pk]), 2), f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlab-dir", required=True)
    ap.add_argument("--exp-train-dir", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--nbins", type=int, default=8)
    args = ap.parse_args()

    pts = []
    for p in sorted(glob(os.path.join(args.unlab_dir, "*.xy"))):
        for pos, fwh in peak_widths(preprocess_exp(p)):
            th = np.deg2rad(pos / 2.0)
            pts.append((np.tan(th), fwh ** 2))
    pts = np.array(pts)
    print("raw peak widths:", len(pts))

    from scipy.optimize import nnls
    edges = np.quantile(pts[:, 0], np.linspace(0, 1, args.nbins + 1))
    bx, by, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = pts[(pts[:, 0] >= lo) & (pts[:, 0] <= hi)][:, 1]
        if len(m) >= 5:
            bx.append(float(np.median(pts[(pts[:, 0] >= lo) & (pts[:, 0] <= hi)][:, 0])))
            by.append(float(np.median(m)))
            bn.append(len(m))
    A = np.column_stack([np.array(bx) ** 2, np.array(bx), np.ones(len(bx))])
    coef, _ = nnls(A, np.array(by))
    U, V, W = [round(float(c), 4) for c in coef]
    pred = A @ coef
    r2 = 1 - float(((np.array(by) - pred) ** 2).sum()) / float(((np.array(by) - np.array(by).mean()) ** 2).sum())
    print("bins:", len(bx), "counts:", bn)
    print("U,V,W:", U, V, W, "R2:", round(r2, 3))

    ares = []
    for p in sorted(glob(os.path.join(args.exp_train_dir, "*.xy"))):
        y = preprocess_exp(p)
        th = np.deg2rad(TTH_GRID / 2.0)
        fw = np.sqrt(np.clip(U * np.tan(th) ** 2 + V * np.tan(th) + W, 0.01, 9.0))
        ds = [abs(fwh - float(fw[int(np.argmin(np.abs(TTH_GRID - pos)))])) for pos, fwh in peak_widths(y)]
        if ds:
            ares.append(round(float(np.median(ds)), 3))
    print("training-pattern median |meas-pred|:", ares)
    ok = bool(r2 > 0.5 and all(np.isfinite([U, V, W])))
    print("GATE R2>0.5:", r2 > 0.5, "| all params finite:", all(np.isfinite([U, V, W])))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"U": U, "V": V, "W": W, "R2": round(float(r2), 4),
                       "fit_ok": ok, "train_check": ares}, f, indent=2)
    if not ok:
        raise SystemExit("FIT gate failed")


if __name__ == "__main__":
    main()
