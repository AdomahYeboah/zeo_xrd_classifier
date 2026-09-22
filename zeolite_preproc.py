"""
Preprocessing and physics-informed data augmentation for zeolite XRD patterns.
"""

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from glob import glob

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from sklearn.preprocessing import LabelEncoder

from config import (
    COD_XY_DRIVE,
    COD_XY_TMP,
    EFF,
    ENV_SG_WIN,
    FWHM,
    IZA_XY_DRIVE,
    IZA_XY_TMP,
    N_AUG_EXP,
    N_AUG_SIM,
    N_GRID,
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
)

log = logging.getLogger("zeolite_preproc")


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
    Anti-aliased resample onto tth_out: box pre-filter (width =
    ceil(decimation_ratio)) followed by Lanczos-2 kernel resampling.
    Falls back to np.interp when raw_step >= STEP * 0.75.
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
    Load and preprocess an experimental .xy file onto TTH_GRID:
    AA-resample, SNIP background subtraction, SG smooth, max-normalise.
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
    """Populate the module-level noise and envelope pools from preprocessed
    unlabelled patterns (consumed by augment_pattern)."""
    global noise_pool, mult_envs
    noise_pool = _extract_noise_residuals(unlabeled_patterns)
    mult_envs  = _extract_mult_envelopes(unlabeled_patterns)


def _pseudo_voigt_kernel(fwhm_deg, eta):
    half_win = max(3, int(4 * fwhm_deg / STEP))
    x        = np.arange(-half_win, half_win + 1) * STEP
    sigma, gamma = fwhm_deg / 2.3548, fwhm_deg / 2.0
    kernel = eta / (1.0 + (x / gamma)**2) + (1 - eta) * np.exp(-0.5 * (x / sigma)**2)
    return (kernel / kernel.sum()).astype(np.float32)


def phi_zero_displacement(y):
    if np.random.random() > 0.50: return y
    theta = np.deg2rad(TTH_GRID / 2.0)
    return np.interp(TTH_GRID,
        TTH_GRID + np.random.uniform(-0.04, 0.04) + np.random.uniform(-0.035, 0.035)*np.cos(theta),
        y, left=0.0, right=0.0).astype(np.float32)


"""Class-blind augmentation pipeline (S1S3S4S5-effective).

Ported semantics from the verified six-condition notebook. Takes no class
label: every pattern sees the same operators and ranges. Preferred
orientation is handled structure-side (March-Dollase re-simulation into
iza_tex_xy/cod_tex_xy); there is intentionally no per-peak operator here.
"""


def _gaussian_bump(y, center, amp, sig):
    return np.clip(y + (amp * np.exp(-0.5 * ((TTH_GRID - center) / sig) ** 2)).astype(np.float32), 0, None)


def caglioti_broaden(y, uvw):
    U, V, W = uvw
    y_sm = savgol_filter(y, 11, 3)
    pks, _ = find_peaks(y_sm, height=max(0.04, y_sm.max() * 0.08),
                        distance=max(2, int(0.25 / STEP)))
    if not len(pks):
        return y
    bl = minimum_filter1d(y, size=40).astype(np.float32)
    pkcomp = np.clip(y - bl, 0, None)
    out = bl.copy()
    th = np.deg2rad(TTH_GRID / 2.0)
    fwh = np.sqrt(np.clip(U * np.tan(th) ** 2 + V * np.tan(th) + W, 0.01, 9.0))
    for pk in pks:
        sig = max(float(fwh[pk]) / 2.3548, STEP / 2)
        win = np.abs(TTH_GRID - TTH_GRID[pk]) < 5 * sig
        out[win] += (pkcomp[pk] * np.exp(-0.5 * ((TTH_GRID[win] - TTH_GRID[pk]) / sig) ** 2)).astype(np.float32)
    return np.clip(out, 0, None).astype(np.float32)


def augment_pattern(y, G, is_sim):
    """Full class-blind augmentation for one pattern (no label argument)."""
    y = y.copy()
    eps = G["strain_eps"]
    if np.random.random() <= 0.65:
        x = (TTH_GRID - TTH_MIN) / (TTH_MAX - TTH_MIN)
        e = np.random.normal(0.0, eps) + np.random.normal(0.0, eps * 0.5) * (x - 0.5)
        theta = np.deg2rad(TTH_GRID / 2.0)
        shifted = 2.0 * np.rad2deg(np.arcsin(np.clip(np.sin(theta) / (1.0 + e), -0.9999, 0.9999)))
        y = np.interp(TTH_GRID, shifted, y, left=0.0, right=0.0).astype(np.float32)
    y = phi_zero_displacement(y)
    if G.get("caglioti") is not None:
        y = caglioti_broaden(y, G["caglioti"])
    else:
        bp = G["broadening"]
        y = np.clip(np.convolve(y, _pseudo_voigt_kernel(
            np.random.uniform(bp["fwhm_min"], bp["fwhm_max"]),
            np.random.uniform(bp["eta_min"], bp["eta_max"])), mode="same"), 0, None).astype(np.float32)
    rs = G.get("rescale")
    if rs is not None and np.random.random() < rs["apply_prob"]:
        y_sm = savgol_filter(y, 11, 3)
        rpeaks, _ = find_peaks(y_sm, height=max(0.04, y_sm.max() * 0.08),
                               distance=max(2, int(0.25 / STEP)))
        if len(rpeaks):
            if G.get("rescale_topk"):
                k = min(int(G["rescale_topk"]), len(rpeaks))
                sel = rpeaks[np.argsort(y_sm[rpeaks])[-k:]]
            else:
                sel = np.random.choice(rpeaks, size=np.random.randint(1, min(4, len(rpeaks)) + 1), replace=False)
            for pk in sel:
                f = np.random.uniform(rs["factor_min"], rs["factor_max"])
                w = np.random.uniform(0.18, 0.55) / STEP
                y = y * (1.0 + (f - 1.0) * np.exp(-0.5 * ((np.arange(N_GRID) - pk) / w) ** 2)).astype(np.float32)
            y = np.clip(y, 0, None).astype(np.float32)
    shift = np.random.uniform(-SHIFT_MAX, SHIFT_MAX)
    y = np.interp(TTH_GRID, TTH_GRID + shift, y, left=0.0, right=0.0).astype(np.float32)
    ap = G["amorphous"]
    if np.random.random() < ap["apply_prob"]:
        y = _gaussian_bump(y, np.random.uniform(ap["center_min"], ap["center_max"]),
                           np.random.uniform(ap["amp_min"], ap["amp_max"]),
                           np.random.uniform(ap["sig_min"], ap["sig_max"]))
    y = phi_impurity_peaks(y)
    if mult_envs and np.random.random() < 0.45:
        env = mult_envs[np.random.randint(len(mult_envs))]
        alpha = np.random.uniform(0.2, 0.75)
        y = y * (alpha * env + (1.0 - alpha))
    if is_sim:
        y = phi_resample_jitter(y)
    y = y + np.random.uniform(*NOISE_SCALE) * _sample_noise(N_GRID)
    y = smooth_sim(np.clip(y, 0, None))
    y = phi_slope_drift(y)
    return y.astype(np.float32)


def minimal_augment(y):
    """Minimal level: S1-range broadening plus grid shift at identical counts."""
    bp = EFF["broadening"]
    y = np.clip(np.convolve(y.copy(), _pseudo_voigt_kernel(
        np.random.uniform(bp["fwhm_min"], bp["fwhm_max"]),
        np.random.uniform(bp["eta_min"], bp["eta_max"])), mode="same"), 0, None).astype(np.float32)
    sh = np.random.uniform(-SHIFT_MAX, SHIFT_MAX)
    y = np.interp(TTH_GRID, TTH_GRID + sh, y, left=0.0, right=0.0).astype(np.float32)
    return smooth_sim(np.clip(y, 0, None))


def augment_simulated_set(X_base, y_base, encoder):
    xs, ys = [], []
    for x, ye in zip(X_base, y_base):
        xs.append(smooth_sim(x))
        ys.append(int(ye))
        for _ in range(N_AUG_SIM):
            xs.append(augment_pattern(x, EFF, True))
            ys.append(int(ye))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def augment_experimental_set(X_exp, y_exp, encoder):
    """Augmented experimental copies (resample jitter omitted: inputs are
    already anti-alias resampled by preprocess_exp)."""
    xs, ys = [], []
    for x, ye in zip(X_exp, y_exp):
        xs.append(x)
        ys.append(int(ye))
        for _ in range(N_AUG_EXP):
            xs.append(augment_pattern(x, EFF, False))
            ys.append(int(ye))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)


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


def phi_resample_jitter(y):
    """Simulate naive decimation artefacts (simulated patterns only;
    experimental patterns are already anti-alias resampled)."""
    if np.random.random() > 0.40: return y
    uf     = np.random.choice([3, 5, 7, 10])
    y_fine = np.interp(np.linspace(TTH_MIN, TTH_MAX, N_GRID*uf), TTH_GRID, y)
    offset = np.random.randint(0, uf)
    yc     = y_fine[np.arange(offset, N_GRID*uf, uf)[:N_GRID]]
    if len(yc) < N_GRID:
        yc = np.pad(yc, (0, N_GRID - len(yc)), mode="edge")
    m = yc.max()
    return (yc / m).astype(np.float32) if m > 0 else yc.astype(np.float32)


