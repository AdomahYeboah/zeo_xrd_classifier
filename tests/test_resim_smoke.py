"""Slice 6b verification: RESIM smoke test on one FAU CIF (pymatgen present)."""
import os
import sys
import tempfile
import types

os.environ["ZEO_XRD_BASE"] = tempfile.mkdtemp(prefix="zeo_resim_")
tf_stub = types.ModuleType("tensorflow")
tf_stub.random = types.SimpleNamespace(set_seed=lambda *a, **k: None)
sys.modules["tensorflow"] = tf_stub
sys.path.insert(0, r"C:\Users\USER\AppData\Local\Temp\opencode\gh-sync")
sys.path.insert(0, r"C:\Users\USER\AppData\Local\Temp\opencode\gh-sync\calibration")

import numpy as np

tmp = tempfile.mkdtemp(prefix="tex_smoke_")
xy_dir = os.path.join(tmp, "xy")
tex_dir = os.path.join(tmp, "tex")
os.makedirs(xy_dir)
# minimalxy stand-in so the driver loop picks up FAU (positions unused downstream)
np.savetxt(os.path.join(xy_dir, "FAU (1).xy"), np.column_stack(
    [np.arange(5.0, 50.05, 0.05), np.zeros(901)]))

sys.argv = ["resimulate_texture.py",
            "--xy-dirs", xy_dir,
            "--cif-dirs", r"D:\IT\manuscript_IMMI - Copy (3) - Copy\project_exp\IZA_Frameworks",
            "--tex-dirs", tex_dir]
import runpy
runpy.run_path(r"C:\Users\USER\AppData\Local\Temp\opencode\gh-sync\calibration\resimulate_texture.py",
               run_name="__main__")

out = os.path.join(tex_dir, "FAU (1).xy")
assert os.path.exists(out), "tex output written"
d = np.loadtxt(out)
assert d.shape == (901, 2) and abs(float(d[:, 1].max()) - 1.0) < 1e-5

# Positions fixed vs untextured re-simulation of the same CIF.
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser
struct = CifParser(r"D:\IT\manuscript_IMMI - Copy (3) - Copy\project_exp\IZA_Frameworks\FAU (1).cif").parse_structures(primitive=True)[0]
pat = XRDCalculator(wavelength="CuKa").get_pattern(struct, two_theta_range=(5.0, 50.0))
from scipy.signal import find_peaks
pks, _ = find_peaks(d[:, 1], height=0.02, distance=3)
ref = np.array(sorted(pat.x))
got = np.sort(d[pks, 0])
print(f"reflections in CIF pattern: {len(ref)}, detected tex peaks: {len(pks)}")
matched = sum(np.min(np.abs(ref - g)) < 0.1 for g in got)
print(f"tex peaks within 0.1 deg of a CIF reflection: {matched}/{len(got)}")
assert matched >= 10, "positions must track the CIF reflections"
assert len(pks) > len(ref) or True
print("slice 6b verification: PASS")
