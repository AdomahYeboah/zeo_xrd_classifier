"""CAM channel-weight broadcasting check (numpy mirror, TF absent locally).

Old code `weights[:, :, None]` expands (1, C) -> (1, C, 1), which only
multiplies against conv_out (1, T, C) when T == C and otherwise crashes or
silently misaligns. Fixed code `weights[:, None, :]` expands to (1, 1, C),
correct for any T.
"""
import numpy as np

rng = np.random.default_rng(0)
T, C = 901, 192  # production shapes: time steps != channels
weights = rng.normal(size=(1, C)).astype(np.float32)
conv_out = rng.normal(size=(1, T, C)).astype(np.float32)

try:
    bad = (weights[:, :, None] * conv_out).sum(axis=-1)
    bad_ok = bad.shape == (1, T)
    print("old code ran, shape:", bad.shape, "(correct only if T == C)")
except ValueError as e:
    bad_ok = False
    print("old code raises:", e)

good = (weights[:, None, :] * conv_out).sum(axis=-1)
assert good.shape == (1, T), good.shape
# channel alignment: manual dot at one timestep
t = 7
assert np.isclose(good[0, t], (weights[0] * conv_out[0, t]).sum())
print("fixed code shape (1, T) with correct channel alignment: PASS")
assert not bad_ok, "old code should not be correct for T != C"
print("all broadcast checks: PASS")
