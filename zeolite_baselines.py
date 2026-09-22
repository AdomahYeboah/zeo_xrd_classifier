"""Baseline classifiers: compact literature CNN + traditional models.

Covers exactly the models compared in the paper: the all-convolutional
baseline (a-CNN, Oviedo-style as in the open-source autoXRD package,
PV-Lab, Apache-2.0) and random forest, SVM, and gradient boosting.
Training follows the shared protocol: same six pre-built training sets,
fresh StandardScaler per traditional model-condition pair, sim-only
validation, seed 42.
"""

import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import callbacks, layers, models, regularizers

from config import BATCH, CLASS_WEIGHT, EPOCHS, L2_REG, N_GRID, PATIENCE, RANDOM_SEED


def build_acnn(n_cls=4, input_len=N_GRID):
    reg = regularizers.l2(L2_REG)
    inp = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(32, 8, strides=8, padding="same", activation="relu",
                      kernel_regularizer=reg)(inp)
    x = layers.Conv1D(32, 5, strides=5, padding="same", activation="relu",
                      kernel_regularizer=reg)(x)
    x = layers.Conv1D(32, 3, strides=3, padding="same", activation="relu",
                      kernel_regularizer=reg)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(n_cls, activation="softmax", kernel_regularizer=reg)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def traditional_configs():
    """(name, constructor kwargs) pairs for the paper's traditional models."""
    return [
        ("RF", RandomForestClassifier, dict(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_SEED, n_jobs=-1)),
        ("SVM", SVC, dict(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                          probability=True, random_state=RANDOM_SEED)),
        ("GBC", GradientBoostingClassifier, dict(n_estimators=250, learning_rate=0.05,
                                                 max_depth=5, subsample=0.8,
                                                 random_state=RANDOM_SEED)),
    ]


def fit_traditional(name, ctor, kwargs, X_tr, y_tr, X_te):
    """Fit one traditional model with a fresh scaler; GBC takes balanced weights."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    if name == "GBC":
        sw = compute_sample_weight("balanced", y=y_tr)
        model = ctor(**kwargs).fit(X_tr_sc, y_tr, sample_weight=sw)
    else:
        model = ctor(**kwargs).fit(X_tr_sc, y_tr)
    return model.predict(X_te_sc)


def train_cnn_shared(model, X_tr, y_tr, X_vl, y_vl, tag):
    """Shared CNN trainer used for the baseline CNN (mirrors train_one_model
    with a per-model scheduler floor and best-epoch logging)."""
    import os
    from config import MODEL_DIR
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, min_delta=1e-4),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=max(5, PATIENCE // 3), min_lr=1e-5),
    ]
    hist = model.fit(
        X_tr[..., np.newaxis], y_tr,
        validation_data=(X_vl[..., np.newaxis], y_vl),
        epochs=EPOCHS, batch_size=BATCH, class_weight=CLASS_WEIGHT,
        callbacks=cb, verbose=1,
    )
    return model
