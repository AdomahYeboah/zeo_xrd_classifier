# Zeolite XRD framework classifier

Classifies powder XRD patterns of zeolites into four framework types (FAU, FER, LTA, MFI) with a multiscale metric CNN. Training data are simulated patterns computed from CIF files; experimental patterns serve as labelled anchors for augmentation, evaluation and Grad-CAM analysis.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All paths and hyperparameters. Paths can be overridden with `ZEO_XRD_*` environment variables. |
| `zeolite_preproc.py` | CIF-to-pattern simulation worker, preprocessing, physics-informed augmentation. |
| `zeolite_vis.py` | Grad-CAM computation and plotting. |
| `zeolite_cnn_multiscale.py` | Model definition, training loops, evaluation. Run this file. |

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
  unlabelled_xy/      unlabelled experimental .xy files
  exp_label_copy/     labelled experimental training anchors (.xy)
  exp_test_xy/        labelled experimental test set (.xy)
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
python zeolite_cnn_multiscale.py --kfold    # 5-fold cross-validation on the experimental anchors
```

Both modes log per-class pattern counts, a step-size histogram of the experimental files, and accuracy after each fold or training run.

## Pipeline

1. Simulation: CIF files are converted to .xy patterns by a subprocess worker using pymatgen (CuKa radiation, 5 to 50 degrees 2-theta, 0.05 degree step, Gaussian peaks with 0.2 degree FWHM). Existing .xy files are reused, so interrupted runs resume where they stopped.
2. Preprocessing: experimental patterns whose native step is finer than `STEP * 0.75` are resampled onto the grid with a box pre-filter followed by Lanczos-2 interpolation, which suppresses aliasing artefacts. SNIP background subtraction, Savitzky-Golay smoothing and max-normalisation follow.
3. Noise and envelope pools: unlabelled experimental patterns supply noise residuals and intensity envelopes that augmentation draws from.
4. Augmentation: each transform models a physical effect, including pseudo-Voigt peak broadening, preferred orientation, lattice strain, sample displacement, amorphous humps, low-angle suppression or amplification, peak dropout, impurity peaks and baseline drift. Per-framework priors live in `FRAMEWORK_PHYSICS`. Simulated patterns additionally receive aliasing jitter; experimental patterns do not, because they are already anti-alias resampled.
5. Model: Conv1D stem, three multiscale blocks with parallel kernels of size 3, 7, 15 and 31, a CAM convolution, global average pooling, a 128-dimensional embedding, L2-normalised cosine logits and a softmax over the four classes.
6. Training: class-weighted sparse categorical cross entropy with ReduceLROnPlateau and early stopping. When `FINE_TUNE_ON_EXPERIMENTAL_ANCHORS` is `True`, the final model is afterwards fine-tuned on a class-balanced subset of augmented experimental anchors.
7. Evaluation and interpretation: accuracy on the test set and training anchors, mean Grad-CAM maps per class for the simulated and experimental domains, side-by-side domain comparisons, and per-sample CAMs for test predictions.

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

Random seeds are fixed at import time (`RANDOM_SEED = 42`); the cross-validation split uses `random_state = 3`.
