"""
ZEO-XRD PREPROCESSING AND AUGMENTATION

Module dedicated to XRD pattern preprocessing and physics-informed data
augmentation for zeolite framework identification.  This mirrors the role of
``autoXRD.py`` in the autoXRD package layout.

Features implemented here
Anti-aliased downsampling (Lanczos-2 + box pre-filter) resamples
         experimental patterns and stored .xy files whose step
         is finer than STEP * 0.75.
phi_resample_jitter() teaches the CNN what aliasing looks like.
         Applied only in the simulated augmentation branch.
load_xy() also anti-aliases finer-than-grid simulated .xy files.

Also contains: SNIP background subtraction, Savitzky-Golay smoothing,
the CIF-to-XY simulation worker (pymatgen), noise / envelope pools and the
full suite of physics-informed ``phi_*`` transforms.
"""

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from glob import glob

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from sklearn.preprocessing import LabelEncoder

from config import (
    COD_XY_DRIVE,
    COD_XY_TMP,
    ENV_SG_WIN,
    FWHM,
    IZA_XY_DRIVE,
    IZA_XY_TMP,
    N_GRID,
    NOISE_SCALE,
    NOISE_SG_WIN,
    SG_ORDER,
    SG_WINDOW,
    SHIFT_MAX,
    SIGMA_DEG,
    SNIP_ITER,
    STEP,
    TARGET_FRAMEWORKS,
    TTH_GRID,
    TTH_MAX,
    TTH_MIN,
    WORKER_CIF,
    FRAMEWORK_PHYSICS,
)

log = logging.getLogger("zeolite_preproc")


# ANTI-ALIASED RESAMPLING

def _lanczos2_kernel(x):
    """Lanczos-2 window: sinc(x) * sinc(x/2) for |x|<2, else 0."""
    x   = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = np.abs(x) < 2.0
    xm = x[mask]
    with np.errstate(invalid="ignore", divide="ignore"):
        sx  = np.where(xm == 0.0, 1.0, np.sin(np.pi * xm)      / (np.pi * xm))
        sx2 = np.where(xm == 0.0, 1.0, np.sin(np.pi * xm / 2.0) / (np.pi * xm / 2.0))
    out[mask] = sx * sx2
    return out


def _antialias_resample(tth_raw, y_raw, tth_out):
    """
    Anti-aliased resample from a fine grid onto tth_out (TTH_GRID).

    Steps:
      1. Box-filter pre-blur (width = ceil(decimation_ratio)) to suppress
         frequencies above the new Nyquist.
      2. Lanczos-2 kernel resampling (4-point support) for sub-pixel accuracy.

    Falls back to np.interp when raw_step >= STEP * 0.75 (no decimation needed).
    """
    if len(tth_raw) < 8:
        return np.interp(tth_out, tth_raw, y_raw, left=0.0, right=0.0).astype(np.float32)
    raw_step = float(np.median(np.diff(tth_raw)))
    if raw_step >= STEP * 0.75:
        return np.interp(tth_out, tth_raw, y_raw, left=0.0, right=0.0).astype(np.float32)
    decimation = STEP / raw_step
    box_width  = max(3, int(np.ceil(decimation)))
    y_blur     = uniform_filter1d(y_raw.astype(np.float64), size=box_width, mode="nearest")
    y_out      = np.zeros(len(tth_out), dtype=np.float64)
    for i, t in enumerate(tth_out):
        frac = (t - tth_raw[0]) / raw_step
        j0   = int(np.floor(frac))
        ws = wv = 0.0
        for dj in range(-1, 3):
            j = j0 + dj
            if 0 <= j < len(y_blur):
                w   = _lanczos2_kernel(np.array([frac - j]))[0]
                wv += w * y_blur[j]; ws += abs(w)
        y_out[i] = wv / ws if ws > 1e-12 else 0.0
    return np.clip(y_out, 0.0, None).astype(np.float32)


# PREPROCESSING

def _snip_background(y, iterations=SNIP_ITER):
    """SNIP iterative background estimator (Statistics-sensitive Non-linear
    Iterative Peak-clipping)."""
    y_bg = np.log(np.log(np.sqrt(np.abs(y) + 1) + 1) + 1)
    n = len(y_bg)
    for i in range(1, iterations + 1):
        y_new = y_bg.copy()
        for j in range(i, n - i):
            y_new[j] = min(y_bg[j], (y_bg[j - i] + y_bg[j + i]) / 2)
        y_bg = y_new
    y_bg = (np.exp(np.exp(y_bg) - 1) - 1) ** 2 - 1
    return np.clip(y_bg, 0, None).astype(np.float32)


def preprocess_exp(path):
    """
    Load and preprocess an experimental .xy file onto TTH_GRID.

    Uses _antialias_resample when the
    raw 2θ step is finer than STEP * 0.75, preventing aliasing-driven
    peak-height corruption and false doublet artefacts.

    Pipeline: AA-resample, SNIP background subtraction, SG smooth, max-normalise
    """
    data = np.loadtxt(path)
    if data.ndim > 1:
        y = _antialias_resample(data[:, 0].astype(np.float64),
                                 data[:, 1].astype(np.float64), TTH_GRID)
    else:
        y = data.astype(np.float32)
    if len(y) != N_GRID:
        y = np.interp(TTH_GRID, np.linspace(TTH_MIN, TTH_MAX, len(y)), y,
                      left=0.0, right=0.0).astype(np.float32)
    is_const = float(np.ptp(y)) < 0.02 * max(1.0, float(y.max()))
    y = np.clip(y - _snip_background(y), 0, None)
    y = np.clip(savgol_filter(y, SG_WINDOW, SG_ORDER), 0, None)
    if is_const:
        s = float(y.std())
        if s > 1e-9:
            y = np.clip((y - y.mean()) / s * 0.10, 0, None)
        return y.astype(np.float32)
    m = y.max()
    return (y / m).astype(np.float32) if m > 0 else y.astype(np.float32)


def smooth_sim(y):
    """SG smooth + min-clip + max-normalise a simulated pattern."""
    y = savgol_filter(y.copy(), SG_WINDOW, SG_ORDER)
    y = np.clip(y - y.min(), 0, None)
    m = y.max()
    return (y / m).astype(np.float32) if m > 0 else y


# CIF SIMULATION WORKER

def _write_worker(path):
    with open(path, "w") as f:
        f.write(textwrap.dedent(f"""\
            import sys, os, numpy as np
            from pymatgen.io.cif import CifParser
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            TTH_GRID  = np.arange({TTH_MIN}, {TTH_MAX} + {STEP}, {STEP})
            SIGMA_DEG = {FWHM} / 2.3548
            calc = XRDCalculator(wavelength="CuKa")
            def simulate(src):
                try:
                    structure = CifParser(src).parse_structures(primitive=True)[0]
                except Exception:
                    return None
                pat = calc.get_pattern(structure, two_theta_range=({TTH_MIN}, {TTH_MAX}))
                if len(pat.x) == 0: return None
                y = np.zeros(len(TTH_GRID), dtype=np.float32)
                for t, i in zip(pat.x, pat.y):
                    y += i * np.exp(-0.5*((TTH_GRID - t)/SIGMA_DEG)**2)
                m = y.max()
                return (y/m).astype(np.float32) if m > 0 else y
            done = skipped = 0
            for line in sys.stdin:
                line = line.strip()
                if not line: continue
                src, out = line.split("|")
                if os.path.exists(out): done += 1; continue
                y = simulate(src)
                if y is not None:
                    np.savetxt(out, np.column_stack([TTH_GRID, y])); done += 1
                else: skipped += 1
            print(f"done={{done}} skipped={{skipped}}", flush=True)
        """))


def _simulate_batch(jobs, worker, desc, batch_size=50):
    pending = [(s, o) for s, o in jobs if not os.path.exists(o)]
    log.info(f"{desc}: {len(pending)} to simulate, {len(jobs)-len(pending)} cached")
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        subprocess.run([sys.executable, worker],
                       input="\n".join(f"{s}|{o}" for s, o in chunk),
                       capture_output=True, text=True)


def ensure_xy(src_dir, tmp_dir, drive_dir, desc, worker,
              src_ext=".cif", out_suffix=".xy", batch_size=50):
    os.makedirs(drive_dir, exist_ok=True)
    srcs = sorted(glob(f"{src_dir}/*{src_ext}"))
    assert srcs, f"No {src_ext} in {src_dir}"
    drive_jobs = [(s, os.path.join(drive_dir,
                   os.path.basename(s).replace(src_ext, out_suffix))) for s in srcs]
    missing = [(s, o) for s, o in drive_jobs if not os.path.exists(o)]
    if missing:
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_jobs = [(s, os.path.join(tmp_dir,
                     os.path.basename(s).replace(src_ext, out_suffix))) for s, _ in missing]
        _simulate_batch(tmp_jobs, worker, desc, batch_size)
        for (_, d), (_, t) in zip(missing, tmp_jobs):
            if os.path.exists(t): shutil.move(t, d)
    found = sorted(glob(f"{drive_dir}/*{out_suffix}"))
    log.info(f"{desc}: {len(found)} .xy files ready")
    assert found
    return found


def load_xy(path):
    """Load .xy, applying anti-alias resample if step < STEP * 0.75."""
    data = np.loadtxt(path)
    if data.ndim < 2:
        return data.astype(np.float32)
    tth_raw, y_raw = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    if len(tth_raw) == N_GRID:
        return y_raw.astype(np.float32)
    return _antialias_resample(tth_raw, y_raw, TTH_GRID)


def label_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0].split("(")[0].strip().upper()


def load_target_patterns(xy_files):
    x, y = [], []
    for p in sorted(xy_files):
        lbl = label_from_path(p)
        if lbl in TARGET_FRAMEWORKS:
            x.append(load_xy(p)); y.append(lbl)
    return np.array(x, dtype=np.float32), np.array(y)


def load_labeled_exp(folder):
    x, y, paths = [], [], []
    for p in sorted(glob(f"{folder}/*.xy")):
        lbl = label_from_path(p)
        if lbl in TARGET_FRAMEWORKS:
            x.append(preprocess_exp(p)); y.append(lbl); paths.append(p)
    return np.array(x, dtype=np.float32), np.array(y), np.array(paths)


# NOISE & ENVELOPE POOLS

noise_pool: list = []
mult_envs:  list = []


def _extract_noise_residuals(patterns, n_segs=15):
    pool, seg_len = [], N_GRID // 4
    for pat in patterns:
        smooth   = savgol_filter(pat, NOISE_SG_WIN, SG_ORDER)
        residual = pat - smooth - (pat - smooth).mean()
        for _ in range(n_segs):
            start = np.random.randint(0, max(1, N_GRID - seg_len))
            seg   = residual[start:start + seg_len].copy()
            seg  -= seg.mean()
            pool.append(seg.astype(np.float32))
    return pool


def _extract_mult_envelopes(patterns):
    envelopes = []
    for pat in patterns:
        win = min(ENV_SG_WIN, N_GRID - 2 if N_GRID % 2 == 0 else N_GRID - 1)
        env = np.clip(savgol_filter(pat, win, SG_ORDER), 0, None)
        mu  = env.mean()
        if mu > 1e-6:
            envelopes.append((env / mu).astype(np.float32))
    return envelopes


def _sample_noise(n):
    if not noise_pool:
        return np.random.normal(0, 0.01, n).astype(np.float32)
    seg = noise_pool[np.random.randint(len(noise_pool))]
    if len(seg) >= n:
        return seg[np.random.randint(0, len(seg) - n + 1):][:n]
    return np.tile(seg, int(np.ceil(n / len(seg))))[:n]


def build_pools(unlabeled_patterns):
    """
    Populate the module-level noise and multi-scaled-envelope pools from a
    list of preprocessed unlabelled experimental patterns.  The pools are
    consumed by augment_sim() / augment_experimental_set().
    """
    global noise_pool, mult_envs
    noise_pool = _extract_noise_residuals(unlabeled_patterns)
    mult_envs  = _extract_mult_envelopes(unlabeled_patterns)


# AUGMENTATION PHYSICS FUNCTIONS

def _pseudo_voigt_kernel(fwhm_deg, eta):
    half_win = max(3, int(4 * fwhm_deg / STEP))
    x        = np.arange(-half_win, half_win + 1) * STEP
    sigma, gamma = fwhm_deg / 2.3548, fwhm_deg / 2.0
    kernel = eta / (1.0 + (x / gamma)**2) + (1 - eta) * np.exp(-0.5 * (x / sigma)**2)
    return (kernel / kernel.sum()).astype(np.float32)


def phi_broad(y, fw):
    bp = FRAMEWORK_PHYSICS[fw]["broadening"]
    return np.clip(np.convolve(y,
        _pseudo_voigt_kernel(np.random.uniform(bp["fwhm_min"], bp["fwhm_max"]),
                              np.random.uniform(bp["eta_min"],  bp["eta_max"])),
        mode="same"), 0, None).astype(np.float32)


def phi_orient(y, fw):
    op = FRAMEWORK_PHYSICS[fw]["orientation"]
    r  = np.random.uniform(op["r_min"], op["r_max"])
    y_sm = savgol_filter(y, 11, 3)
    peaks = [i for i in range(1, len(y_sm)-1)
             if y_sm[i] > y_sm[i-1] and y_sm[i] > y_sm[i+1] and y_sm[i] > 0.05]
    if not peaks: return y
    phi_a = np.random.uniform(0, np.pi/2, len(peaks))
    w = (r**2 * np.cos(phi_a)**2 + np.sin(phi_a)**2 / r)**(-1.5)
    w /= w.mean()
    sp    = max(1, int(0.5 / STEP))
    field = np.ones(N_GRID, dtype=np.float32)
    for pk, ww in zip(peaks, w):
        field += (ww - 1.0) * np.exp(-0.5 * ((np.arange(N_GRID) - pk) / sp)**2)
    return np.clip(y * field, 0, None).astype(np.float32)


def phi_lattice_strain(y, fw):
    if np.random.random() > 0.65: return y
    eps_scale = {"FAU": 0.0015, "FER": 0.0025, "LTA": 0.0035, "MFI": 0.0030}[fw]
    x   = (TTH_GRID - TTH_MIN) / (TTH_MAX - TTH_MIN)
    eps = np.random.normal(0.0, eps_scale) + np.random.normal(0.0, eps_scale*0.5)*(x-0.5)
    theta   = np.deg2rad(TTH_GRID / 2.0)
    shifted = 2.0 * np.rad2deg(np.arcsin(np.clip(np.sin(theta) / (1.0+eps), -0.9999, 0.9999)))
    return np.interp(TTH_GRID, shifted, y, left=0.0, right=0.0).astype(np.float32)


def phi_zero_displacement(y):
    if np.random.random() > 0.50: return y
    theta = np.deg2rad(TTH_GRID / 2.0)
    return np.interp(TTH_GRID,
        TTH_GRID + np.random.uniform(-0.04, 0.04) + np.random.uniform(-0.035, 0.035)*np.cos(theta),
        y, left=0.0, right=0.0).astype(np.float32)


def _add_gaussian_hump(y, params, high_limit=TTH_MAX-2):
    c = np.clip(np.random.normal(params["center_mean"], params["center_std"]),
                TTH_MIN + 0.5, high_limit)
    a = np.random.uniform(params["amp_min"],          params["amp_max"])
    s = np.random.uniform(params["width_sigma_min"],  params["width_sigma_max"])
    return np.clip(y + (a * np.exp(-0.5 * ((TTH_GRID - c) / s)**2)).astype(np.float32), 0, None)


def phi_amorph(y, fw):
    ap = FRAMEWORK_PHYSICS[fw]["amorphous"]
    if np.random.random() <= ap["apply_prob"]: y = _add_gaussian_hump(y, ap)
    ap2 = FRAMEWORK_PHYSICS[fw].get("amorphous2")
    if ap2 is not None and np.random.random() <= ap2["apply_prob"]: y = _add_gaussian_hump(y, ap2)
    return y.astype(np.float32)


def _cosine_taper(si, ei, sv):
    t = np.ones(N_GRID, dtype=np.float32)
    rl = ei - si
    if rl > 0:
        t[:si] = sv
        t[si:ei] = sv + (1.0-sv)*(0.5 - 0.5*np.cos(np.pi*np.arange(rl)/rl))
    return t


def phi_suppress(y, fw):
    sp = FRAMEWORK_PHYSICS[fw].get("low_suppress")
    if sp and np.random.random() < sp["apply_prob"]:
        sv = np.random.uniform(sp["suppress_min"], sp["suppress_max"])
        y  = y * _cosine_taper(int((sp["suppress_start_deg"]-TTH_MIN)/STEP),
                                int((sp["suppress_end_deg"]-TTH_MIN)/STEP), sv)
    return y.astype(np.float32)


def phi_hi_amplify(y, fw):
    ha = FRAMEWORK_PHYSICS[fw].get("hi_amplify")
    if ha is None or np.random.random() > ha["apply_prob"]: return y
    factor = np.random.uniform(ha["factor_min"], ha["factor_max"])
    return np.clip(y * (1.0 + (factor-1.0)*np.exp(-0.5*((TTH_GRID-ha["center"])/ha["sigma"])**2)).astype(np.float32), 0, None)


def phi_mfi_triplet_boost(y, fw):
    mb = FRAMEWORK_PHYSICS[fw].get("mfi_triplet_boost")
    if mb is None or np.random.random() > mb["apply_prob"]: return y
    field = np.ones(N_GRID, dtype=np.float32)
    for c in mb["centers"]:
        field += (np.random.uniform(mb["factor_min"], mb["factor_max"])-1.0) * \
                 np.exp(-0.5*((TTH_GRID-c)/mb["sigma"])**2).astype(np.float32)
    return np.clip(y * field, 0, None).astype(np.float32)


def phi_fer_diagnostic_boost(y, fw):
    fb = FRAMEWORK_PHYSICS[fw].get("fer_diagnostic_boost")
    if fb is None or np.random.random() > fb["apply_prob"]: return y
    field = np.ones(N_GRID, dtype=np.float32)
    for c in fb["centers"]:
        field += (np.random.uniform(fb["factor_min"], fb["factor_max"])-1.0) * \
                 np.exp(-0.5*((TTH_GRID-c)/fb["sigma"])**2).astype(np.float32)
    return np.clip(y * field, 0, None).astype(np.float32)


def phi_peak_dropout(y, fw):
    if np.random.random() > {"FAU": 0.20, "FER": 0.25, "LTA": 0.25, "MFI": 0.42}[fw]: return y
    y_sm = savgol_filter(y, 11, 3)
    peaks, _ = find_peaks(y_sm, height=max(0.04, y_sm.max()*0.08),
                          distance=max(2, int(0.25/STEP)))
    if not len(peaks): return y
    mask = np.ones(N_GRID, dtype=np.float32)
    for pk in np.random.choice(peaks, size=np.random.randint(1, min(4, len(peaks))+1), replace=False):
        w = np.random.uniform(0.18, 0.55) / STEP
        d = np.random.uniform(0.25, 0.85)
        mask *= (1.0 - d * np.exp(-0.5*((np.arange(N_GRID)-pk)/w)**2)).astype(np.float32)
    return np.clip(y * mask, 0, None).astype(np.float32)


def phi_impurity_peaks(y):
    if np.random.random() > 0.28: return y
    out = y.copy()
    for _ in range(np.random.randint(1, 4)):
        c, a, s = (np.random.uniform(TTH_MIN+1, TTH_MAX-1),
                   np.random.uniform(0.02, 0.16),
                   np.random.uniform(0.06, 0.22))
        out += (a * np.exp(-0.5*((TTH_GRID-c)/s)**2)).astype(np.float32)
    return np.clip(out, 0, None).astype(np.float32)


def phi_slope_drift(y):
    if np.random.random() > 0.40: return y
    t = TTH_GRID - TTH_MIN; amp = np.random.uniform(0.05, 0.65)
    mode = np.random.choice(["linear", "quadratic", "concave"])
    drift = amp*(t/t.max()) if mode=="linear" else (amp*(t/t.max())**2 if mode=="quadratic"
            else amp*(1 - 4*((t/t.max())-0.5)**2))
    out = y + drift.astype(np.float32)
    return (out / out.max()).astype(np.float32) if out.max() > 0 else out.astype(np.float32)


def phi_low_angle_hump(y, fw):
    la = FRAMEWORK_PHYSICS[fw].get("low_angle_hump")
    if la is None or np.random.random() > la["apply_prob"]: return y
    return _add_gaussian_hump(y, la, 14.0)


def phi_resample_jitter(y):
    """
    Aliasing augmentation, simulating naive decimation artefacts.
    Applied only to simulated patterns (NOT experimental, which are already
    properly anti-alias resampled by preprocess_exp).
    Probability: 0.40.
    """
    if np.random.random() > 0.40: return y
    uf     = np.random.choice([3, 5, 7, 10])
    y_fine = np.interp(np.linspace(TTH_MIN, TTH_MAX, N_GRID*uf), TTH_GRID, y)
    offset = np.random.randint(0, uf)
    yc     = y_fine[np.arange(offset, N_GRID*uf, uf)[:N_GRID]]
    if len(yc) < N_GRID:
        yc = np.pad(yc, (0, N_GRID - len(yc)), mode="edge")
    m = yc.max()
    return (yc / m).astype(np.float32) if m > 0 else yc.astype(np.float32)


# AUGMENTATION PIPELINES

def augment_sim(x, fw):
    """Full physics-informed augmentation for a simulated pattern.
    Includes phi_resample_jitter just before noise injection."""
    y = x.copy()
    y = phi_lattice_strain(y, fw)
    y = phi_zero_displacement(y)
    y = phi_broad(y, fw)
    y = phi_orient(y, fw)
    y = phi_fer_diagnostic_boost(y, fw)
    y = phi_mfi_triplet_boost(y, fw)
    y = phi_peak_dropout(y, fw)
    y = phi_suppress(y, fw)
    y = phi_hi_amplify(y, fw)
    shift = np.random.uniform(-SHIFT_MAX, SHIFT_MAX)
    y = np.interp(TTH_GRID, TTH_GRID + shift, y, left=0.0, right=0.0).astype(np.float32)
    y = phi_amorph(y, fw)
    y = phi_low_angle_hump(y, fw)
    y = phi_impurity_peaks(y)
    if mult_envs and np.random.random() < 0.45:
        env   = mult_envs[np.random.randint(len(mult_envs))]
        alpha = np.random.uniform(0.2, 0.75)
        y     = y * (alpha * env + (1.0 - alpha))
    y = phi_resample_jitter(y)
    y = y + np.random.uniform(*NOISE_SCALE) * _sample_noise(N_GRID)
    y = smooth_sim(np.clip(y, 0, None))
    y = phi_slope_drift(y)
    return y.astype(np.float32)


def augment_experimental_set(X_exp, y_exp, encoder):
    """
    Augment labelled experimental patterns.
    phi_resample_jitter is intentionally omitted because experimental patterns
    are already anti-alias resampled by preprocess_exp().
    """
    xs, ys = [], []
    for x, ye in zip(X_exp, y_exp):
        fw = TARGET_FRAMEWORKS[int(ye)]
        xs.append(x); ys.append(int(ye))
        for _ in range(N_AUG_EXP_BY_FW.get(fw, 20)):
            yy = x.copy()
            yy = phi_lattice_strain(yy, fw)
            yy = phi_zero_displacement(yy)
            yy = phi_broad(yy, fw)
            yy = phi_orient(yy, fw)
            yy = phi_fer_diagnostic_boost(yy, fw)
            yy = phi_mfi_triplet_boost(yy, fw)
            yy = phi_peak_dropout(yy, fw)
            yy = phi_suppress(yy, fw)
            yy = phi_hi_amplify(yy, fw)
            shift = np.random.uniform(-SHIFT_MAX, SHIFT_MAX)
            yy = np.interp(TTH_GRID, TTH_GRID+shift, yy, left=0.0, right=0.0).astype(np.float32)
            yy = phi_amorph(yy, fw)
            yy = phi_low_angle_hump(yy, fw)
            yy = phi_impurity_peaks(yy)
            if mult_envs and np.random.random() < 0.45:
                env   = mult_envs[np.random.randint(len(mult_envs))]
                alpha = np.random.uniform(0.2, 0.75)
                yy    = yy * (alpha * env + (1.0 - alpha))
            # NOTE: phi_resample_jitter intentionally omitted
            yy = yy + np.random.uniform(*NOISE_SCALE) * _sample_noise(N_GRID)
            yy = smooth_sim(np.clip(yy, 0, None))
            yy = phi_slope_drift(yy)
            xs.append(yy.astype(np.float32)); ys.append(int(ye))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def augment_simulated_set(X_base, y_base, encoder):
    xs, ys = [], []
    for x, ye in zip(X_base, y_base):
        fw = TARGET_FRAMEWORKS[int(ye)]
        xs.append(smooth_sim(x)); ys.append(int(ye))
        for _ in range(N_AUG_SIM_BY_FW.get(fw, 20)):
            xs.append(augment_sim(x, fw)); ys.append(int(ye))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)
