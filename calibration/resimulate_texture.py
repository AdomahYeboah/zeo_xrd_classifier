"""Re-simulate patterns with structure-side preferred orientation.

For each .xy pattern, finds the source CIF, re-simulates with pymatgen,
and applies per-reflection March-Dollase texture with a random zone axis
per structure. Originals are never modified; output goes to separate
tex dirs. Structures without a same-label CIF are skipped with a warning.

Usage:
    python resimulate_texture.py --xy-dirs iza_xy cod_xy --cif-dirs IZA_Frameworks cod_cifs --tex-dirs iza_tex_xy cod_tex_xy
"""

import argparse
import logging
import os
from glob import glob

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser

from config import N_GRID, STEP, TARGET_FRAMEWORKS, TTH_MAX, TTH_MIN
from zeolite_preproc import label_from_path

log = logging.getLogger("resimulate_texture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy-dirs", nargs="+", required=True)
    ap.add_argument("--cif-dirs", nargs="+", required=True)
    ap.add_argument("--tex-dirs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--r-min", type=float, default=0.5)
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--base-fwhm", type=float, default=0.2)
    args = ap.parse_args()
    assert len(args.xy_dirs) == len(args.cif_dirs) == len(args.tex_dirs)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    tth_grid = np.arange(TTH_MIN, TTH_MAX + STEP, STEP)
    assert len(tth_grid) == N_GRID
    calc = XRDCalculator(wavelength="CuKa")
    sigma = args.base_fwhm / 2.3548
    rng = np.random.default_rng(args.seed)
    done, missing, fallback = 0, [], 0
    for xy_dir, tex_dir, cif_dir in zip(args.xy_dirs, args.tex_dirs, args.cif_dirs):
        os.makedirs(tex_dir, exist_ok=True)
        for xp in sorted(glob(os.path.join(xy_dir, "*.xy"))):
            lbl = label_from_path(xp)
            if lbl not in TARGET_FRAMEWORKS:
                continue
            out = os.path.join(tex_dir, os.path.basename(xp))
            if os.path.exists(out):
                done += 1
                continue
            cif = os.path.join(cif_dir, os.path.splitext(os.path.basename(xp))[0] + ".cif")
            if not os.path.exists(cif):
                alts = sorted([q for q in glob(os.path.join(cif_dir, "*.cif"))
                               if label_from_path(q) == lbl])
                if alts:
                    log.warning("CIF fallback %s -> %s", os.path.basename(xp), os.path.basename(alts[0]))
                    cif = alts[0]
                else:
                    missing.append(xp)
                    continue
            struct = CifParser(cif).parse_structures(primitive=True)[0]
            pat = calc.get_pattern(struct, two_theta_range=(TTH_MIN, TTH_MAX))
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            r = float(rng.uniform(args.r_min, args.r_max))
            rec = struct.lattice.reciprocal_lattice
            y = np.zeros(N_GRID, dtype=np.float32)
            for t, inten, h in zip(pat.x, pat.y, pat.hkls):
                lst = h if isinstance(h, (list, tuple)) else [h]
                d0 = lst[0]
                hkl = d0["hkl"] if isinstance(d0, dict) else d0
                try:
                    n = np.array(rec.get_cartesian_coords(hkl), float)
                    ca = abs(float(n @ u) / (np.linalg.norm(n) * np.linalg.norm(u)))
                    P = (r ** 2 * ca ** 2 + (1 - ca ** 2) / r) ** (-1.5)
                except Exception:
                    P = 1.0
                    fallback += 1
                y += (inten * P * np.exp(-0.5 * ((tth_grid - t) / sigma) ** 2)).astype(np.float32)
            m = y.max()
            y = (y / m).astype(np.float32) if m > 0 else y
            np.savetxt(out, np.column_stack([tth_grid, y]))
            done += 1
    log.info("textured sims ready: %d, missing CIFs: %d, reflection fallbacks: %d",
             done, len(missing), fallback)
    for m in missing[:10]:
        log.warning("no CIF for %s", m)


if __name__ == "__main__":
    main()
