"""
ZEO-XRD CONFIGURATION

Centralised configuration for the zeolite XRD framework classifier
(FAU / FER / LTA / MFI).  This is the single edit point for every path and
hyper-parameter, mirroring the layout of the autoXRD package.

All data paths default to the original Google Colab / Google Drive layout
used by ``zeolite_cnn_multiscale.py`` but can be overridden with environment
variables so the pipeline runs locally without code edits:

    BASE          -> ZEO_XRD_BASE
    IZA_CIF_DIR   -> ZEO_XRD_IZA_CIF_DIR
    COD_CIF_DIR   -> ZEO_XRD_COD_CIF_DIR
    UNLABELED_DIR -> ZEO_XRD_UNLABELED_DIR
    EXP_TEST_DIR  -> ZEO_XRD_EXP_TEST_DIR
    EXP_TRAIN_DIR -> ZEO_XRD_EXP_TRAIN_DIR
    IZA_XY_DRIVE  -> ZEO_XRD_IZA_XY_DRIVE
    COD_XY_DRIVE  -> ZEO_XRD_COD_XY_DRIVE
    IZA_XY_TMP    -> ZEO_XRD_IZA_XY_TMP
    COD_XY_TMP    -> ZEO_XRD_COD_XY_TMP
    OUT_DIR       -> ZEO_XRD_OUT_DIR
    WORKER_CIF    -> ZEO_XRD_WORKER_CIF

Example (local run on Windows):

    set ZEO_XRD_BASE=D:\\IT\\project_exp
    set ZEO_XRD_IZA_XY_TMP=D:\\IT\\tmp\\iza_xy
    set ZEO_XRD_COD_XY_TMP=D:\\IT\\tmp\\cod_xy
    set ZEO_XRD_WORKER_CIF=D:\\IT\\simulate_cif_worker.py
    python zeolite_cnn_multiscale.py
"""

import logging
import os
import random

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _env(name, default):
    return os.environ.get(name, default)


# TARGET FRAMEWORKS & 2θ GRID

TARGET_FRAMEWORKS = ["FAU", "FER", "LTA", "MFI"]

TTH_MIN, TTH_MAX, STEP = 5.0, 50.0, 0.05
FWHM        = 0.2
RANDOM_SEED = 42

# AUGMENTATION COUNTS

N_AUG_SIM_BY_FW = {"FAU": 18, "FER": 34, "LTA": 30, "MFI": 24}
N_AUG_EXP_BY_FW = {"FAU": 24, "FER": 54, "LTA": 42, "MFI": 30}

# SIGNAL PROCESSING

SHIFT_MAX    = 0.08
NOISE_SCALE  = (0.02, 0.075)
SG_WINDOW    = 21
SG_ORDER     = 3
ENV_SG_WIN   = 101
NOISE_SG_WIN = 51
SNIP_ITER    = 20

# CNN TRAINING

EPOCHS       = 100
PATIENCE     = 25
BATCH        = 64
L2_REG       = 1e-4
CLASS_WEIGHT = {0: 0.95, 1: 3.00, 2: 1.85, 3: 1.75}

# FINE-TUNING  (set True to adapt on experimental anchors after main training)

FINE_TUNE_ON_EXPERIMENTAL_ANCHORS = False
FINE_TUNE_EPOCHS        = 12
FINE_TUNE_LR            = 2e-5
FINE_TUNE_EXP_PER_CLASS = 120
SIM_REPLAY_PER_CLASS    = 240
FINE_TUNE_CLASS_WEIGHT  = {0: 1.0, 1: 1.15, 2: 1.10, 3: 1.0}

# PATHS - configurable via environment variables (see module docstring)

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

# Simulation worker script (subprocess), configurable for local runs
WORKER_CIF    = _env("ZEO_XRD_WORKER_CIF",    "/content/simulate_cif_worker.py")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CAM_DIR,   exist_ok=True)

# DERIVED QUANTITIES & SEEDING

TTH_GRID  = np.arange(TTH_MIN, TTH_MAX + STEP, STEP)
SIGMA_DEG = FWHM / 2.3548
N_GRID    = len(TTH_GRID)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# FRAMEWORK PHYSICS - physics-informed augmentation priors per framework

FRAMEWORK_PHYSICS = {
    "FAU": {
        "broadening":    {"fwhm_min": 0.42, "fwhm_max": 0.75, "eta_min": 0.30, "eta_max": 0.80},
        "orientation":   {"r_min": 0.78, "r_max": 0.97},
        "amorphous":     {"center_mean": 30.6, "center_std": 3.8, "amp_min": 0.05, "amp_max": 0.60,
                          "width_sigma_min": 4.0, "width_sigma_max": 8.0, "apply_prob": 0.82},
        "amorphous2":    None, "low_suppress": None, "hi_amplify": None,
        "low_angle_hump": {"center_mean": 8.0, "center_std": 1.5, "amp_min": 0.05, "amp_max": 0.40,
                           "width_sigma_min": 2.0, "width_sigma_max": 4.5, "apply_prob": 0.35},
        "mfi_triplet_boost": None, "fer_diagnostic_boost": None,
    },
    "FER": {
        "broadening":    {"fwhm_min": 0.76, "fwhm_max": 1.91, "eta_min": 0.00, "eta_max": 0.40},
        "orientation":   {"r_min": 0.47, "r_max": 0.97},
        "amorphous":     {"center_mean": 22.6, "center_std": 2.0, "amp_min": 0.10, "amp_max": 0.37,
                          "width_sigma_min": 2.5, "width_sigma_max": 4.5, "apply_prob": 0.75},
        "amorphous2": None, "low_suppress": None,
        "hi_amplify":    {"center": 25.0, "sigma": 4.0, "factor_min": 1.5, "factor_max": 3.5, "apply_prob": 0.55},
        "low_angle_hump": None, "mfi_triplet_boost": None,
        "fer_diagnostic_boost": {"centers": [9.3, 22.1, 25.2], "sigma": 0.34,
                                  "factor_min": 1.15, "factor_max": 2.10, "apply_prob": 0.65},
    },
    "LTA": {
        "broadening":    {"fwhm_min": 0.40, "fwhm_max": 2.40, "eta_min": 0.00, "eta_max": 0.50},
        "orientation":   {"r_min": 0.20, "r_max": 0.97},
        "amorphous":     {"center_mean": 13.0, "center_std": 1.0, "amp_min": 0.15, "amp_max": 0.80,
                          "width_sigma_min": 1.8, "width_sigma_max": 2.0, "apply_prob": 0.85},
        "amorphous2":    {"center_mean": 15.0, "center_std": 1.5, "amp_min": 0.05, "amp_max": 0.30,
                          "width_sigma_min": 1.5, "width_sigma_max": 2.5, "apply_prob": 0.50},
        "low_suppress": None, "hi_amplify": None,
        "low_angle_hump": {"center_mean": 9.0, "center_std": 1.2, "amp_min": 0.10, "amp_max": 0.70,
                           "width_sigma_min": 1.5, "width_sigma_max": 3.5, "apply_prob": 0.50},
        "mfi_triplet_boost": None, "fer_diagnostic_boost": None,
    },
    "MFI": {
        "broadening":    {"fwhm_min": 0.35, "fwhm_max": 1.55, "eta_min": 0.00, "eta_max": 0.55},
        "orientation":   {"r_min": 0.04, "r_max": 0.97},
        "amorphous":     {"center_mean": 22.5, "center_std": 3.2, "amp_min": 0.00, "amp_max": 0.18,
                          "width_sigma_min": 3.0, "width_sigma_max": 7.5, "apply_prob": 0.55},
        "amorphous2": None,
        "low_suppress":  {"suppress_start_deg": 5.0, "suppress_end_deg": 16.0,
                          "suppress_min": 0.02, "suppress_max": 0.35, "apply_prob": 0.78},
        "hi_amplify": None,
        "low_angle_hump": {"center_mean": 8.4, "center_std": 0.9, "amp_min": 0.02, "amp_max": 0.22,
                           "width_sigma_min": 1.2, "width_sigma_max": 3.0, "apply_prob": 0.28},
        "mfi_triplet_boost": {"centers": [7.9, 8.9, 23.1, 23.9, 24.4], "sigma": 0.42,
                               "factor_min": 1.15, "factor_max": 2.20, "apply_prob": 0.55},
        "fer_diagnostic_boost": None,
    },
}
