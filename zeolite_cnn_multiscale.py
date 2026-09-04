"""
FAU / FER / LTA / MFI zeolite framework identification from powder XRD patterns.
"""

import logging
import os
import random
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    BASE, BATCH, CAM_DIR, CLASS_WEIGHT, COD_CIF_DIR, COD_XY_DRIVE, COD_XY_TMP,
    EPOCHS, EXP_TEST_DIR, EXP_TRAIN_DIR, FINE_TUNE_CLASS_WEIGHT,
    FINE_TUNE_EPOCHS, FINE_TUNE_EXP_PER_CLASS, FINE_TUNE_LR,
    FINE_TUNE_ON_EXPERIMENTAL_ANCHORS, IZA_CIF_DIR, IZA_XY_DRIVE, IZA_XY_TMP,
    L2_REG, MODEL_DIR, N_AUG_SIM_BY_FW, N_GRID, OUT_DIR, PATIENCE, SIM_REPLAY_PER_CLASS,
    STEP, TARGET_FRAMEWORKS, TTH_MAX, TTH_MIN, TTH_GRID, UNLABELED_DIR, WORKER_CIF,
    RANDOM_SEED,
)
from zeolite_preproc import (
    augment_experimental_set, augment_simulated_set, build_pools,
    ensure_xy, load_labeled_exp, load_target_patterns, preprocess_exp,
    smooth_sim,
)
from zeolite_vis import (
    compare_cam_domains, plot_average_cams, plot_average_cams_experimental,
    plot_class_cam_gallery, plot_test_cams,
)

log = logging.getLogger("zeolite_cnn")


def diagnose_exp_resolution(test_dir=EXP_TEST_DIR):
    """Log a histogram of the native 2θ step sizes of experimental .xy files."""
    from glob import glob
    import collections
    steps = []
    for p in sorted(glob(f"{test_dir}/*.xy")):
        d = np.loadtxt(p)
        if d.ndim > 1:
            steps.append(float(np.median(np.diff(d[:, 0]))))
    if not steps:
        log.warning("diagnose_exp_resolution: no .xy files in %s", test_dir)
        return
    counts = collections.Counter(round(s, 3) for s in steps)
    log.info("experimental step-size histogram (deg):")
    for step_sz, n in sorted(counts.items()):
        flag = "  (AA-resampled)" if step_sz < STEP * 0.75 else ""
        log.info("   %-8.3f  x%-5d%s", step_sz, n, flag)


def _ms_block(x, filters, name, dr=0.08):
    reg = tf.keras.regularizers.l2(L2_REG)
    branches = []
    for k in (3, 7, 15, 31):
        b = tf.keras.layers.Conv1D(filters, k, padding="same",
                                   kernel_regularizer=reg, name=f"{name}_k{k}")(x)
        b = tf.keras.layers.LayerNormalization(name=f"{name}_ln{k}")(b)
        b = tf.keras.layers.Activation("relu", name=f"{name}_relu{k}")(b)
        branches.append(b)
    y = tf.keras.layers.Concatenate(name=f"{name}_cat")(branches)
    y = tf.keras.layers.Conv1D(filters * 2, 1, padding="same",
                               kernel_regularizer=reg, name=f"{name}_mix")(y)
    y = tf.keras.layers.LayerNormalization(name=f"{name}_mix_ln")(y)
    skip = x if int(x.shape[-1]) == filters * 2 else tf.keras.layers.Conv1D(
        filters * 2, 1, padding="same", kernel_regularizer=reg, name=f"{name}_skip")(x)
    y = tf.keras.layers.Add(name=f"{name}_add")([skip, y])
    y = tf.keras.layers.Activation("relu", name=f"{name}_out_relu")(y)
    y = tf.keras.layers.SpatialDropout1D(dr, name=f"{name}_drop")(y)
    return y


def build_multiscale_metric_acnn(input_len=N_GRID):
    reg = tf.keras.regularizers.l2(L2_REG)
    inp = tf.keras.layers.Input(shape=(input_len, 1), name="xrd_input")

    x = tf.keras.layers.Conv1D(32, 7, padding="same", kernel_regularizer=reg,
                               name="stem_conv")(inp)
    x = tf.keras.layers.LayerNormalization(name="stem_ln")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)

    x = _ms_block(x, 24, "ms1", dr=0.05)
    x = _ms_block(x, 32, "ms2", dr=0.07)
    x = _ms_block(x, 48, "ms3", dr=0.10)

    x = tf.keras.layers.Conv1D(192, 3, padding="same", kernel_regularizer=reg,
                               name="cam_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="cam_bn")(x)
    x = tf.keras.layers.Activation("relu", name="cam_relu")(x)
    x = tf.keras.layers.Dropout(0.25, name="cam_drop")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg,
                              name="emb_dense")(x)
    x = tf.keras.layers.Dropout(0.25, name="emb_drop")(x)
    emb = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                                 name="l2_emb")(x)
    logits = tf.keras.layers.Dense(len(TARGET_FRAMEWORKS), use_bias=False,
                                   kernel_regularizer=reg, name="cos_logits")(emb)
    logits = tf.keras.layers.Lambda(lambda t: 18.0 * t, name="logit_scale")(logits)
    out = tf.keras.layers.Activation("softmax", name="class_probs")(logits)

    model = tf.keras.Model(inp, out, name="multiscale_metric_acnn")
    return model


def train_one_model(model, X, y, X_val, y_val, tag, lr=3e-4, epochs=EPOCHS,
                    class_weight=None, es_patience=PATIENCE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(5, es_patience // 3),
                                             min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience,
                                         restore_best_weights=True, verbose=1),
    ]
    model.fit(X, y, batch_size=BATCH, epochs=epochs, validation_data=(X_val, y_val),
              callbacks=callbacks, class_weight=class_weight, verbose=1)
    model.save(os.path.join(MODEL_DIR, f"{tag}.keras"))
    return model


def fine_tune_on_experimental_anchors(model, X_exp, y_exp, encoder,
                                      base_lr=FINE_TUNE_LR):
    """Fine-tune on a class-balanced subset of augmented experimental anchors."""
    X_aug, y_aug = augment_experimental_set(X_exp, y_exp, encoder)
    fw_to_idx = {fw: i for i, fw in enumerate(TARGET_FRAMEWORKS)}
    picks = []
    for fw in TARGET_FRAMEWORKS:
        idx = np.where(y_aug == fw_to_idx[fw])[0]
        rng = np.random.default_rng(RANDOM_SEED)
        n = min(FINE_TUNE_EXP_PER_CLASS, len(idx))
        picks.append(rng.choice(idx, size=n, replace=False))
    X_ft = np.concatenate([X_aug[i] for i in picks])
    y_ft = np.concatenate([y_aug[i] for i in picks])
    log.info("fine-tune anchor set: %d patterns", len(X_ft))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_ft, y_ft, batch_size=BATCH, epochs=FINE_TUNE_EPOCHS,
              validation_split=0.15, class_weight=FINE_TUNE_CLASS_WEIGHT,
              verbose=1)
    model.save(os.path.join(MODEL_DIR, "multiscale_finetuned.keras"))
    return model


def evaluate_model(model, X, y, tag):
    y_pred = model.predict(X, batch_size=BATCH, verbose=0)
    y_idx  = np.argmax(y_pred, axis=1)
    acc    = float(np.mean(y_idx == np.asarray(y).astype(np.int32)))
    log.info("[%s] accuracy = %.4f", tag, acc)
    return acc, y_pred


def run_experimental_kfold(model, X_sim, y_sim, X_exp, y_exp):
    kf = KFold(n_splits=min(5, len(X_exp)), shuffle=True, random_state=3)
    fold_exp_accs = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_exp, y_exp)):
        log.info("Fold %d/%d", fold + 1, kf.n_splits)
        X_sim_aug, y_sim_aug = augment_simulated_set(X_sim, y_sim, None)
        X_exp_aug, y_exp_aug = augment_experimental_set(X_exp[tr_idx], y_exp[tr_idx], None)
        X_tr_all = np.concatenate([X_sim_aug, X_exp_aug])
        y_tr_all = np.concatenate([y_sim_aug, y_exp_aug])
        X_va = X_exp[va_idx]; y_va = y_exp[va_idx]

        fold_model = tf.keras.models.clone_model(model)
        fold_model.set_weights(model.get_weights())
        train_one_model(fold_model, X_tr_all, y_tr_all, X_va, y_va,
                        f"fold{fold + 1}", class_weight=CLASS_WEIGHT)

        exp_acc, _ = evaluate_model(fold_model, X_va, y_va, f"fold{fold + 1}/exp")
        fold_exp_accs.append(exp_acc)

    log.info("CV experimental acc: %s  mean=%.4f",
             np.round(fold_exp_accs, 4), np.mean(fold_exp_accs))


def train_final_and_cam():
    model = build_multiscale_metric_acnn()
    model.summary()

    iza_xy = ensure_xy(IZA_CIF_DIR, IZA_XY_TMP, IZA_XY_DRIVE, "IZA CIFs", WORKER_CIF)
    cod_xy = ensure_xy(COD_CIF_DIR, COD_XY_TMP, COD_XY_DRIVE, "COD CIFs", WORKER_CIF)
    X_sim, y_sim_str = load_target_patterns(iza_xy + cod_xy)

    encoder = LabelEncoder().fit(TARGET_FRAMEWORKS)
    y_sim = encoder.transform(y_sim_str)
    log.info("simulated target patterns: %s", dict(zip(*np.unique(y_sim_str, return_counts=True))))

    # unlabelled pool feeds the noise / envelope pools
    from glob import glob
    unlabelled = [preprocess_exp(p) for p in sorted(glob(f"{UNLABELED_DIR}/*.xy"))]
    build_pools(unlabelled)
    log.info("unlabelled pool: %d patterns", len(unlabelled))

    X_exp, y_exp_str, exp_paths = load_labeled_exp(EXP_TRAIN_DIR)
    y_exp = encoder.transform(y_exp_str)
    log.info("experimental anchors: %s", dict(zip(*np.unique(y_exp_str, return_counts=True))))

    X_test, y_test_str, test_paths = load_labeled_exp(EXP_TEST_DIR)
    y_test = encoder.transform(y_test_str)
    log.info("experimental test: %s", dict(zip(*np.unique(y_test_str, return_counts=True))))

    diagnose_exp_resolution()

    tr_idx, vl_idx = train_test_split(np.arange(len(X_sim)), test_size=0.2,
                                       random_state=RANDOM_SEED, stratify=y_sim)
    X_sim_aug, y_sim_aug = augment_simulated_set(X_sim[tr_idx], y_sim[tr_idx], encoder)
    X_exp_aug, y_exp_aug = augment_experimental_set(X_exp, y_exp, encoder)
    X_all = np.concatenate([X_sim_aug, X_exp_aug])
    y_all = np.concatenate([y_sim_aug, y_exp_aug])
    X_vl = np.array([smooth_sim(x) for x in X_sim[vl_idx]], dtype=np.float32)
    y_vl = y_sim[vl_idx]
    log.info("final training set: %d patterns", len(X_all))
    train_one_model(model, X_all, y_all, X_vl, y_vl, "multiscale_final",
                    class_weight=CLASS_WEIGHT)

    if FINE_TUNE_ON_EXPERIMENTAL_ANCHORS:
        model = fine_tune_on_experimental_anchors(model, X_exp, y_exp, encoder)

    _, _ = evaluate_model(model, X_test, y_test, "final/test")
    _, _ = evaluate_model(model, X_exp, y_exp, "final/anchors")

    X_by_class = {fw: X_sim[y_sim == encoder.transform([fw])[0]] for fw in TARGET_FRAMEWORKS}
    plot_average_cams(model, X_by_class, CAM_DIR)
    plot_average_cams_experimental(model, X_test, y_test, CAM_DIR)
    compare_cam_domains(model, X_by_class, X_test, y_test, CAM_DIR)
    plot_class_cam_gallery(model, X_by_class, CAM_DIR)
    y_test_pred = model.predict(X_test, batch_size=BATCH, verbose=0)
    plot_test_cams(model, X_test, y_test, y_test_pred, CAM_DIR)


def main():
    try:  # mount Drive when running on Colab
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass

    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("output dir: %s", OUT_DIR)
    log.info("model dir:  %s", MODEL_DIR)

    if len(sys.argv) > 1 and sys.argv[1] == "--kfold":
        run_experimental_kfold(build_multiscale_metric_acnn(),
                               *_prepare_cv_data())
    else:
        train_final_and_cam()


def _prepare_cv_data():
    iza_xy = ensure_xy(IZA_CIF_DIR, IZA_XY_TMP, IZA_XY_DRIVE, "IZA CIFs", WORKER_CIF)
    cod_xy = ensure_xy(COD_CIF_DIR, COD_XY_TMP, COD_XY_DRIVE, "COD CIFs", WORKER_CIF)
    X_sim, y_sim_str = load_target_patterns(iza_xy + cod_xy)
    encoder = LabelEncoder().fit(TARGET_FRAMEWORKS)
    y_sim = encoder.transform(y_sim_str)
    X_exp, y_exp_str, _ = load_labeled_exp(EXP_TRAIN_DIR)
    y_exp = encoder.transform(y_exp_str)
    return X_sim, y_sim, X_exp, y_exp


if __name__ == "__main__":
    main()
