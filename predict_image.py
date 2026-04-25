#!/usr/bin/env python3
"""
E033 image system: PCA 50 + LogReg + adversarial rotation augmentation.

Adversarial rotation (E033): 0.51% EER vs E007's 0.97% — trains on worst-case
rotations per sample (2-pass: fit initial model, find adversarial angle per image,
refit on combined data). Solves fold-0 pathology and improves rotation robustness
13× (rot15: 13.70% → 1.04%).

Usage:
    uv run predict_image.py --eval-dir /path/to/eval --output results/image_pca_adv_rot.txt
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.augment.image import (
    aug_brightness,
    aug_flip,
    aug_noise,
    aug_rotate,
    find_adversarial_rotation,
)
from src.data.manifest import find_png
from src.data.splits import load_manifest, iter_folds_loso
from src.eval.metrics import compute_min_dcf
from src.features.image import load_image

SEED = 67
N_PCA = 50
C_LOGREG = 1.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train(df, data_dir: Path, augment: bool, seed: int):
    """2-pass training: basic aug first, then adversarial rotation on top (E033)."""
    rng = np.random.default_rng(seed)

    # Pass 1: original + flip + brightness + noise
    X_basic, y_basic = [], []
    for _, row in df.iterrows():
        orig = load_image(find_png(row["stem"], data_dir))
        vecs = [orig]
        if augment:
            vecs += [aug_flip(orig), aug_brightness(orig, rng), aug_noise(orig, rng)]
        for v in vecs:
            X_basic.append(v)
            y_basic.append(row["label"])
    X_basic = np.stack(X_basic)
    y_basic = np.array(y_basic)

    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=SEED)
    clf    = LogisticRegression(C=C_LOGREG, max_iter=1000, random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X_basic))
    clf.fit(X_pca, y_basic)

    if not augment:
        return scaler, pca, clf

    # Pass 2: adversarial rotation per sample
    X_adv, y_adv = [], []
    for _, row in df.iterrows():
        orig  = load_image(find_png(row["stem"], data_dir))
        angle = find_adversarial_rotation(orig, scaler, pca, clf)
        X_adv.append(aug_rotate(orig, angle))
        y_adv.append(row["label"])

    X_all = np.vstack([X_basic, np.stack(X_adv)])
    y_all = np.concatenate([y_basic, np.array(y_adv)])

    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=SEED)
    clf    = LogisticRegression(C=C_LOGREG, max_iter=1000, random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X_all))
    clf.fit(X_pca, y_all)
    return scaler, pca, clf


def _score_png(png_path: Path, scaler, pca, clf) -> float:
    x     = load_image(png_path).reshape(1, -1)
    x_pca = pca.transform(scaler.transform(x))
    return float(clf.decision_function(x_pca)[0])


def _score_png_tta(png_path: Path, scaler, pca, clf) -> float:
    """TTA: average original + horizontally flipped score."""
    x      = load_image(png_path)
    x_flip = aug_flip(x)
    score_orig = float(clf.decision_function(pca.transform(scaler.transform(x.reshape(1,-1))))[0])
    score_flip = float(clf.decision_function(pca.transform(scaler.transform(x_flip.reshape(1,-1))))[0])
    return (score_orig + score_flip) / 2


# ---------------------------------------------------------------------------
# Calibration via OOF
# ---------------------------------------------------------------------------

def _collect_oof(manifest, data_dir: Path) -> np.ndarray:
    oof = np.full(len(manifest), np.nan)
    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        scaler, pca, clf = _train(manifest.loc[train_idx], data_dir,
                                  augment=True, seed=SEED + fold_id)
        X_val  = np.stack([load_image(find_png(row["stem"], data_dir))
                           for _, row in manifest.loc[val_idx].iterrows()])
        X_pca  = pca.transform(scaler.transform(X_val))
        oof[val_idx] = clf.decision_function(X_pca)
    return oof


def _fit_calibrator(oof_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    cal = LogisticRegression(C=1e6, max_iter=1000, class_weight="balanced")
    cal.fit(oof_scores.reshape(-1, 1), labels)
    return cal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data",                      type=Path)
    parser.add_argument("--eval-dir",  required=True,                        type=Path)
    parser.add_argument("--output",    default="results/image_pca_logreg.txt", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    y_all    = manifest["label"].to_numpy()

    print("Collecting OOF scores for calibration…")
    oof_raw = _collect_oof(manifest, data_dir)

    print("Fitting Platt calibrator…")
    cal = _fit_calibrator(oof_raw, y_all)
    oof_cal = cal.decision_function(oof_raw.reshape(-1, 1))
    _, threshold = compute_min_dcf(oof_cal[y_all == 1], oof_cal[y_all == 0])
    print(f"  Threshold (min-DCF on OOF): {threshold:.4f}")

    print("Retraining on all data…")
    scaler, pca, clf = _train(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval data in {eval_dir} …")
    pngs = sorted(eval_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No .png files found in {eval_dir}")

    with open(args.output, "w") as f:
        for png in pngs:
            raw   = _score_png_tta(png, scaler, pca, clf)
            score = float(cal.decision_function([[raw]])[0])
            hard  = 1 if score >= threshold else 0
            f.write(f"{png.stem} {score:.6f} {hard}\n")

    print(f"Written {len(pngs)} lines → {args.output}")


if __name__ == "__main__":
    main()
