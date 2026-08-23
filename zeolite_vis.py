"""
Grad-CAM visualisation utilities for the zeolite framework classifier.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CAM_DIR, MODEL_DIR, N_GRID, TARGET_FRAMEWORKS, TTH_GRID

log = logging.getLogger("zeolite_vis")


def compute_cam(model, x, class_idx=None, layer_name="cam_conv"):
    """
    Grad-CAM heatmap for a single pattern.
    Returns (weights, cam) with ``cam`` upsampled onto TTH_GRID.
    """
    x = x.astype("float32")
    if x.ndim == 1:
        x = x[None, :, None]
    elif x.ndim == 2:
        x = x[:, :, None]
    if class_idx is None:
        class_idx = int(np.argmax(model.predict(x, verbose=0)[0]))
    cam_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, logits = cam_model(x, training=False)
        loss = logits[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(tf.multiply(weights[:, :, None], conv_out), axis=-1)[0].numpy()
    cam = np.clip(cam, 0, None)
    cam = np.interp(TTH_GRID, np.linspace(TTH_GRID[0], TTH_GRID[-1], len(cam)), cam)
    m = cam.max()
    if m > 1e-12:
        cam = cam / m
    return weights.numpy()[0], cam.astype(np.float32)


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", os.path.basename(path))


def plot_pattern_cam(model, x, label, title, out_path, class_idx=None):
    """Single-pattern plot: XRD pattern on top, Grad-CAM heatmap below."""
    _, cam = compute_cam(model, x, class_idx=class_idx)
    x = x.ravel()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(TTH_GRID, x, lw=0.7, color="tab:blue")
    ax1.set_ylabel("Intensity")
    ax1.set_title(f"{title}  (label={label})")
    ax2.fill_between(TTH_GRID, 0, cam, color="tab:red", alpha=0.6)
    ax2.set_ylabel("Grad-CAM")
    ax2.set_xlabel(r"2$\theta$ / deg")
    ax2.set_xlim(TTH_GRID[0], TTH_GRID[-1])
    ax2.set_ylim(0, 1.02)
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_average_cams(model, X_by_class, out_dir):
    """
    Class-averaged Grad-CAM over the simulated domain.
    ``X_by_class``: dict framework -> np.ndarray of patterns.
    """
    os.makedirs(out_dir, exist_ok=True)
    for fw, Xc in X_by_class.items():
        cams = np.stack([compute_cam(model, x)[1] for x in Xc[:64]])
        mean_cam = cams.mean(axis=0)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.fill_between(TTH_GRID, 0, mean_cam, color="tab:red", alpha=0.55)
        ax.set_title(f"{fw} mean Grad-CAM ({len(cams)} patterns)")
        ax.set_xlabel(r"2$\theta$ / deg")
        ax.set_xlim(TTH_GRID[0], TTH_GRID[-1])
        ax.set_ylim(0, 1.02)
        _savefig(fig, os.path.join(out_dir, f"cam_sim_{fw}.png"))


def plot_average_cams_experimental(model, X_exp, y_exp, out_dir):
    """Class-averaged Grad-CAM computed on experimental patterns."""
    os.makedirs(out_dir, exist_ok=True)
    cams_by_class = {fw: [] for fw in TARGET_FRAMEWORKS}
    for x, ye in zip(X_exp, y_exp):
        cams_by_class[TARGET_FRAMEWORKS[int(ye)]].append(compute_cam(model, x)[1])
    for fw in TARGET_FRAMEWORKS:
        if not cams_by_class[fw]:
            continue
        cams = np.stack(cams_by_class[fw])
        mean_cam = cams.mean(axis=0)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.fill_between(TTH_GRID, 0, mean_cam, color="tab:red", alpha=0.55)
        ax.set_title(f"{fw} mean experimental-domain Grad-CAM ({len(cams)} patterns)")
        ax.set_xlabel(r"2$\theta$ / deg")
        ax.set_xlim(TTH_GRID[0], TTH_GRID[-1])
        ax.set_ylim(0, 1.02)
        _savefig(fig, os.path.join(out_dir, f"cam_exp_{fw}.png"))


def compare_cam_domains(model, X_by_class, X_exp, y_exp, out_dir):
    """Side-by-side simulated vs experimental mean Grad-CAM per class."""
    os.makedirs(out_dir, exist_ok=True)
    for fw in TARGET_FRAMEWORKS:
        if fw not in X_by_class or not any(TARGET_FRAMEWORKS[int(y)] == fw for y in y_exp):
            continue
        sim_cams = np.stack([compute_cam(model, x)[1] for x in X_by_class[fw][:64]])
        exp_cams = np.stack([compute_cam(model, x)[1] for x, y in zip(X_exp, y_exp)
                             if TARGET_FRAMEWORKS[int(y)] == fw])
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        for ax, cams, dom in ((axes[0], sim_cams, "simulated"),
                              (axes[1], exp_cams, "experimental")):
            mean_cam = cams.mean(axis=0)
            ax.fill_between(TTH_GRID, 0, mean_cam, color="tab:red", alpha=0.55)
            ax.set_ylabel(f"{dom}\nGrad-CAM")
        axes[0].set_title(f"{fw} simulated vs experimental Grad-CAM")
        axes[1].set_xlabel(r"2$\theta$ / deg")
        for ax in axes:
            ax.set_xlim(TTH_GRID[0], TTH_GRID[-1])
            ax.set_ylim(0, 1.02)
        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"cam_compare_{fw}.png"))


def plot_test_cams(model, X_test, y_true, y_pred, out_dir, n=12):
    """CAM gallery for the most confident / most confused test samples."""
    os.makedirs(out_dir, exist_ok=True)
    conf = np.max(y_pred, axis=1)
    idx  = np.argsort(conf)[::-1][:n]
    for i in idx:
        fw  = TARGET_FRAMEWORKS[int(y_true[i])]
        fwp = TARGET_FRAMEWORKS[int(np.argmax(y_pred[i]))]
        tag = "correct" if fw == fwp else f"mispred->{fwp}"
        plot_pattern_cam(model, X_test[i], fw, f"test#{i} {tag}",
                         os.path.join(out_dir, f"cam_test_{i:03d}_{tag}.png"))


def plot_class_cam_gallery(model, X_by_class, out_dir):
    """One aggregate (mean) CAM panel per framework on a single figure."""
    os.makedirs(out_dir, exist_ok=True)
    n_fw = len(TARGET_FRAMEWORKS)
    fig, axes = plt.subplots(n_fw, 1, figsize=(9, 3 * n_fw), sharex=True)
    for ax, fw in zip(axes, TARGET_FRAMEWORKS):
        Xc = X_by_class.get(fw)
        if Xc is None or len(Xc) == 0:
            ax.set_title(f"{fw} no data")
            continue
        cams = np.stack([compute_cam(model, x)[1] for x in Xc[:64]])
        ax.fill_between(TTH_GRID, 0, cams.mean(axis=0), color="tab:red", alpha=0.55)
        ax.set_ylabel(fw)
        ax.set_ylim(0, 1.02)
    axes[-1].set_xlabel(r"2$\theta$ / deg")
    fig.suptitle("Class-averaged Grad-CAM per framework")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "cam_gallery_all.png"))
