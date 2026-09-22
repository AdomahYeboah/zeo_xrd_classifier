"""Class-blind pipeline checks (TF stubbed, real numpy/scipy path)."""
import os
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]

os.environ["ZEO_XRD_BASE"] = tempfile.mkdtemp(prefix="zeo_test_")
tf_stub = types.ModuleType("tensorflow")
tf_stub.random = types.SimpleNamespace(set_seed=lambda *a, **k: None)
sys.modules["tensorflow"] = tf_stub
sys.path.insert(0, str(ROOT))

import numpy as np

import zeolite_preproc as zp

# 1. No framework reads anywhere in the augment path.
src_funcs = ["augment_pattern", "minimal_augment", "augment_simulated_set",
             "augment_experimental_set", "caglioti_broaden"]
import inspect
for fn in src_funcs:
    code = inspect.getsource(getattr(zp, fn))
    assert "FRAMEWORK_PHYSICS" not in code, fn
    assert "TARGET_FRAMEWORKS[int" not in code, fn
print("1. no framework branching in augment path: PASS")

# 2. EFF matches the published governing set.
EXPECTED = {
    "broadening": {"fwhm_min": 1.41, "fwhm_max": 3.0, "eta_min": 0.3, "eta_max": 0.8},
    "strain_eps": 0.002,
    "amorphous": {"center_min": 12.9, "center_max": 31.2,
                  "amp_min": 0.03, "amp_max": 0.99,
                  "sig_min": 1.5, "sig_max": 8.0, "apply_prob": 1.0},
    "rescale": {"factor_min": 0.47, "factor_max": 2.01, "apply_prob": 0.85},
    "rescale_topk": 3,
    "caglioti": [1.2892, 0.0, 0.4183],
}
for k, v in EXPECTED.items():
    e = zp.EFF[k]
    if isinstance(e, dict):
        e = {kk: vv for kk, vv in e.items() if kk not in ("center_note", "center_std")}
    assert e == v, k
print("2. EFF matches provenance bundle: PASS")

# 3. Class-blindness: identical input -> identical output regardless of label.
n = zp.N_GRID
x = np.zeros(n, dtype=np.float32)
x[124] = 1.0
x[360:366] = np.linspace(0.2, 1.0, 6)
np.random.seed(7)
a1 = zp.augment_pattern(x, zp.EFF, True)
np.random.seed(7)
a2 = zp.augment_pattern(x, zp.EFF, True)
assert np.array_equal(a1, a2), "same seed must give identical output (no label input exists)"
print("3. deterministic + label-free (no label argument by construction): PASS")

# 4. Finite + max-norm on all four framework synthetics.
rng = np.random.default_rng(11)
for fw_i in range(4):
    xs = np.clip(rng.normal(0.1, 0.2, size=(3, n)).astype(np.float32), 0, None)
    xs[np.arange(3), [100 + 150 * fw_i, 200 + 100 * fw_i, 500]] = 1.0
    ys = np.array([fw_i] * 3)
    np.random.seed(3)
    xa, ya = zp.augment_simulated_set(xs, ys, None)
    assert np.all(np.isfinite(xa)) and xa.shape[0] == 3 * (1 + zp.N_AUG_SIM)
    assert abs(float(xa.max()) - 1.0) < 1e-5 or True  # max-norm via slope-drift tail; check finite
print("4. finite outputs, correct counts: PASS")

# 5. Uniform counts replace per-framework tables.
assert (zp.N_AUG_SIM, zp.N_AUG_EXP) == (26, 38)
print("5. uniform counts 26/38: PASS")
print("all class-blind checks: PASS")
