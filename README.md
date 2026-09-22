# Zeolite XRD framework classifier

Classifies powder XRD patterns of zeolites into four framework types (FAU, FER, LTA, MFI) with a multiscale CNN. Training data are simulated patterns computed from CIF files; seven labelled experimental training patterns (2 FAU, 2 FER, 1 LTA, 2 MFI) calibrate the augmentation and supervise training alongside the simulations.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All paths, hyperparameters, and the class-blind `EFF` augmentation set. Paths can be overridden with `ZEO_XRD_*` environment variables. |
| `zeolite_preproc.py` | CIF-to-pattern simulation worker, preprocessing, class-blind physics-informed augmentation. |
| `zeolite_vis.py` | Grad-CAM computation and plotting. |
| `zeolite_cnn_multiscale.py` | Model definition, training loops, evaluation. Run this file. |
| `calibration/calibrate_S1S2_globals.py` | Derives S1 broadening/amorphous ranges and rescale quantiles from training patterns vs IZA references. |
| `calibration/fit_caglioti.py` | Constrained Caglioti fit (binned medians + NNLS + FIT_OK gate) from unlabelled patterns. |
| `tests/` | Regression tests runnable without TensorFlow. |

## Requirements

Python 3 with `tensorflow`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` and `pymatgen`.

## Data layout

Directories default to a Google Drive layout used in Colab and can be redirected locally through environment variables (see Configuration).

```
BASE/
  IZA_Frameworks/     IZA CIF files (simulation source)
  cod_cifs/           COD CIF files (simulation source)
  iza_xy/             simulated .xy cache for IZA CIFs
  cod_xy/             simulated .xy cache for COD CIFs
  iza_tex_xy/         re-simulated .xy with structure-side texture
  cod_tex_xy/         re-simulated .xy with structure-side texture
  unlabelled_xy/      unlabelled experimental .xy files (20 patterns)
  exp_label_copy/     labelled experimental training patterns, 7 files (2/2/1/2)
  exp_test_xy/        labelled experimental test set, 40 files (held out)
  multiscale_output/  created automatically (models/, cam_plots/)
```

Labels come from file names: the part of the stem before the first underscore, uppercased (`FAU_bulk.xy` gives FAU). Files whose label is not one of the four target frameworks are skipped.

Before the first run, generate the simulation worker script once:

```python
from zeolite_preproc import _write_worker
from config import WORKER_CIF
_write_worker(WORKER_CIF)
```

## Usage

```
python zeolite_cnn_multiscale.py            # train the final model and produce Grad-CAM plots
```

Training logs per-class pattern counts and accuracy after the run. Checkpoints are selected on the simulated validation set alone; there is no cross-validation on the seven training patterns.

## Calibration (run before training)

```
python calibration/calibrate_S1S2_globals.py --base <BASE> --out-dir <OUT>
python calibration/fit_caglioti.py --unlab-dir <BASE>/unlabelled_xy --exp-train-dir <BASE>/exp_label_copy
```

The first script compares four idealized IZA references against the same-framework training patterns and pools the comparisons by median (broadening FWHM 1.41-3.0, rescale factors 0.47-2.01). The second fits angle-dependent broadening (U,V,W = 1.2892, 0.0, 0.4183, R2 0.693) with a FIT_OK gate. Both reproduce the published parameters exactly.

## Pipeline

1. Simulation: CIF files are converted to .xy patterns by a subprocess worker using pymatgen (CuKa radiation, 5 to 50 degrees 2-theta, 0.05 degree step, Gaussian peaks with 0.2 degree FWHM). Preferred orientation enters structure-side: re-simulation with per-reflection March-Dollase texture (random axis per structure, r uniform 0.5-1.0). Existing .xy files are reused, so interrupted runs resume where they stopped.
2. Preprocessing: experimental patterns whose native step is finer than `STEP * 0.75` are resampled onto the grid with a box pre-filter followed by Lanczos-2 interpolation, which suppresses aliasing artefacts. SNIP background subtraction, Savitzky-Golay smoothing and max-normalisation follow.
3. Noise and envelope pools: unlabelled experimental patterns supply noise residuals and intensity envelopes that augmentation draws from.
4. Augmentation: one class-blind pipeline transforms every pattern identically (`augment_pattern` takes no class label). Ranges are shared across frameworks: S1 broadening or fitted Caglioti render, data-dependent top-3 intensity rescale, grid shift, amorphous hump, impurity peaks, envelope blends, measured noise, slope drift, and sim-only resampling jitter. Uniform counts: 26 copies per simulated pattern, 38 per experimental training pattern, raw originals retained. Reduced levels: none (raw only) and minimal (broadening plus grid shift at identical counts).
5. Model: Conv1D stem, three multiscale blocks with parallel kernels of size 3, 7, 15 and 31, a CAM convolution, global average pooling, a 128-dimensional L2-normalized embedding, temperature-scaled ($\tau$ = 18) softmax readout over the four classes.
6. Training: class-weighted sparse categorical cross entropy (weights 0.95/3.00/1.85/1.75 for FAU/FER/LTA/MFI) with ReduceLROnPlateau and early stopping on the sim-only validation set. Checkpoints restore the best validation weights.
7. Evaluation and interpretation: accuracy, macro-F1 and MCC on the held-out 40-pattern test set; mean Grad-CAM maps per class; per-sample CAMs for test predictions. Reference run: accuracy 0.875, macro-F1 0.812, MCC 0.793 (single run, n=40).

## Configuration

Each path constant in `config.py` can be overridden with an environment variable named `ZEO_XRD_<CONSTANT>` (for example `ZEO_XRD_BASE`, `ZEO_XRD_IZA_CIF_DIR`, `ZEO_XRD_EXP_TEST_DIR`). Unset variables fall back to the Colab defaults defined in `config.py`; `MODEL_DIR` and `CAM_DIR` are always derived from `OUT_DIR`.

Example for a local Windows run:

```
set ZEO_XRD_BASE=D:\IT\project_exp
set ZEO_XRD_IZA_XY_TMP=D:\IT\tmp\iza_xy
set ZEO_XRD_COD_XY_TMP=D:\IT\tmp\cod_xy
set ZEO_XRD_WORKER_CIF=D:\IT\simulate_cif_worker.py
python zeolite_cnn_multiscale.py
```

Random seeds are fixed at import time (`RANDOM_SEED = 42`).

## Tests

```
python tests/test_cam_broadcast.py        # Grad-CAM broadcasting (numpy only)
python tests/test_classblind_augment.py   # class-blindness, EFF parity, counts (TF stubbed)
```
