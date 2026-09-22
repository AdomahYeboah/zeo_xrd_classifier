"""
Central configuration for the zeolite XRD framework classifier.

All paths can be overridden with environment variables prefixed ZEO_XRD_,
e.g. set ZEO_XRD_BASE=D:\data\project_exp before running.
"""

import logging
import os
import random

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _env(name, default):
    return os.environ.get(name, default)


TARGET_FRAMEWORKS = ["FAU", "FER", "LTA", "MFI"]

TTH_MIN, TTH_MAX, STEP = 5.0, 50.0, 0.05
FWHM        = 0.2
RANDOM_SEED = 42

N_AUG_SIM = 26
N_AUG_EXP = 38

SHIFT_MAX    = 0.08
NOISE_SCALE  = (0.02, 0.075)
SG_WINDOW    = 21
SG_ORDER     = 3
ENV_SG_WIN   = 101
NOISE_SG_WIN = 51
SNIP_ITER    = 20

EPOCHS       = 100
PATIENCE     = 25
BATCH        = 64
L2_REG       = 1e-4
CLASS_WEIGHT = {0: 0.95, 1: 3.00, 2: 1.85, 3: 1.75}

BASE          = _env("ZEO_XRD_BASE", "/content/drive/MyDrive/project_exp")
IZA_CIF_DIR   = _env("ZEO_XRD_IZA_CIF_DIR",   f"{BASE}/IZA_Frameworks")
COD_CIF_DIR   = _env("ZEO_XRD_COD_CIF_DIR",   f"{BASE}/cod_cifs")
UNLABELED_DIR = _env("ZEO_XRD_UNLABELED_DIR", f"{BASE}/unlabelled_xy")
EXP_TEST_DIR  = _env("ZEO_XRD_EXP_TEST_DIR",  f"{BASE}/exp_test_xy")
EXP_TRAIN_DIR = _env("ZEO_XRD_EXP_TRAIN_DIR", f"{BASE}/exp_label_copy")
IZA_XY_DRIVE  = _env("ZEO_XRD_IZA_XY_DRIVE",  f"{BASE}/iza_xy")
COD_XY_DRIVE  = _env("ZEO_XRD_COD_XY_DRIVE",  f"{BASE}/cod_xy")
IZA_XY_TMP    = _env("ZEO_XRD_IZA_XY_TMP",    "/content/iza_xy_tmp")
COD_XY_TMP    = _env("ZEO_XRD_COD_XY_TMP",    "/content/cod_xy_tmp")
OUT_DIR       = _env("ZEO_XRD_OUT_DIR",       f"{BASE}/multiscale_output")
MODEL_DIR     = f"{OUT_DIR}/models"
CAM_DIR       = f"{OUT_DIR}/cam_plots"

WORKER_CIF    = _env("ZEO_XRD_WORKER_CIF",    "/content/simulate_cif_worker.py")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CAM_DIR,   exist_ok=True)

TTH_GRID  = np.arange(TTH_MIN, TTH_MAX + STEP, STEP)
SIGMA_DEG = FWHM / 2.3548
N_GRID    = len(TTH_GRID)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# Class-blind augmentation governing set (S1S3S4S5-effective).
# Calibrated against experimental training patterns vs idealized IZA
# references (median over 4 comparisons); see calibrate_S1S2_globals.py.
EFF = {
    "broadening": {"fwhm_min": 1.41, "fwhm_max": 3.0,
                   "eta_min": 0.3, "eta_max": 0.8},
    "strain_eps": 0.002,
    "amorphous": {"center_min": 12.9, "center_max": 31.2,
                  "center_note": "interquartile interval of training-pattern bg positions; single hump",
                  "center_std": 3.0,
                  "amp_min": 0.03, "amp_max": 0.99,
                  "sig_min": 1.5, "sig_max": 8.0, "apply_prob": 1.0},
    "rescale": {"factor_min": 0.47, "factor_max": 2.01, "apply_prob": 0.85},
    "rescale_topk": 3,
    "caglioti": [1.2892, 0.0, 0.4183],
}
