"""RESIM smoke test on the vendored FAU CIF (pymatgen present)."""
import os
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"

os.environ["ZEO_XRD_BASE"] = tempfile.mkdtemp(prefix="zeo_resim_")
tf_stub = types.ModuleType("tensorflow")
tf_stub.random = types.SimpleNamespace(set_seed=lambda *a, **k: None)
sys.modules["tensorflow"] = tf_stub
sys.path.insert(0, str(ROOT))

import numpy as np

tmp = tempfile.mkdtemp(prefix="tex_smoke_")
xy_dir = os.path.join(tmp, "xy")
tex_dir = os.path.join(tmp, "tex")
os.makedirs(xy_dir)
# minimal xy stand-in so the driver loop picks up FAU (positions unused downstream)
np.savetxt(os.path.join(xy_dir, "FAU (1).xy"), np.column_stack(
    [np.arange(5.0, 50.05, 0.05), np.zeros(901)]))

cif_dir = tempfile.mkdtemp(prefix="cif_smoke_")
cif_src = DATA / "FAU.cif"
cif_dst = os.path.join(cif_dir, "FAU (1).cif")
cif_dst_bytes = cif_src.read_bytes()
pathlib.Path(cif_dst).write_bytes(cif_dst_bytes)

sys.argv = ["resimulate_texture.py",
            "--xy-dirs", xy_dir,
            "--cif-dirs", cif_dir,
            "--tex-dirs", tex_dir]
import runpy
runpy.run_path(str(ROOT / "calibration" / "resimulate_texture.py"),
               run_name="__main__")

out = os.path.join(tex_dir, "FAU (1).xy")
assert os.path.exists(out), "tex output written"
d = np.loadtxt(out)
assert d.shape == (901, 2) and abs(float(d[:, 1].max()) - 1.0) < 1e-5

# Positions fixed vs untextured re-simulation of the same CIF.
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser
struct = CifParser(str(DATA / "FAU.cif")).parse_structures(primitive=True)[0]
pat = XRDCalculator(wavelength="CuKa").get_pattern(struct, two_theta_range=(5.0, 50.0))
from scipy.signal import find_peaks
pks, _ = find_peaks(d[:, 1], height=0.02, distance=3)
ref = np.array(sorted(pat.x))
got = np.sort(d[pks, 0])
print(f"reflections in CIF pattern: {len(ref)}, detected tex peaks: {len(pks)}")
matched = sum(np.min(np.abs(ref - g)) < 0.1 for g in got)
print(f"tex peaks within 0.1 deg of a CIF reflection: {matched}/{len(got)}")
assert matched >= 10, "positions must track the CIF reflections"
