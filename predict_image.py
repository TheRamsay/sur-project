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
from PIL import Image
from scipy.ndimage import rotate as nd_rotate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.splits import load_manifest, iter_folds_loso
from src.eval.metrics import compute_min_dcf

SEED = 67
N_PCA = 50
C_LOGREG = 1.0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _find_png(stem: str, data_dir: Path) -> Path:
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = data_dir / sf / (stem + ".png")
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def _load_image(path: Path) -> np.ndarray:
    # convert("RGB") strips alpha channel; resize guards against eval format surprises
    img = Image.open(path).convert("RGB")
    if img.size != (80, 80):
        img = img.resize((80, 80), Image.BILINEAR)
    return np.array(img, dtype=np.float32).mean(axis=2).flatten()


def _aug_flip(x: np.ndarray) -> np.ndarray:
    return x.reshape(80, 80)[:, ::-1].flatten()


def _aug_brightness(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(x * rng.uniform(0.7, 1.3), 0, 255)


def _aug_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(x + rng.normal(0, 15, x.shape), 0, 255)


def _aug_rotate(x: np.ndarray, angle: float) -> np.ndarray:
    return nd_rotate(x.reshape(80, 80), angle, reshape=False,
                     order=1, mode='constant', cval=0).flatten()


def _find_adversarial_rotation(x: np.ndarray, scaler, pca, clf,
                                max_angle: float = 10.0, n_steps: int = 5) -> float:
    """Find rotation angle in [-max_angle, +max_angle] that minimises |logit| (hardest sample)."""
    best_angle, worst_abs = 0.0, np.inf
    for angle in np.linspace(-max_angle, max_angle, n_steps):
        if abs(angle) < 0.1:
            continue
        logit = clf.decision_function(
            pca.transform(scaler.transform(_aug_rotate(x, angle).reshape(1, -1)))
        )[0]
        if abs(logit) < worst_abs:
            worst_abs, best_angle = abs(logit), angle
    return best_angle


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train(df, data_dir: Path, augment: bool, seed: int):
    """2-pass training: basic aug first, then adversarial rotation on top (E033)."""
    rng = np.random.default_rng(seed)

    # Pass 1: original + flip + brightness + noise
    X_basic, y_basic = [], []
    for _, row in df.iterrows():
        orig = _load_image(_find_png(row["stem"], data_dir))
        vecs = [orig]
        if augment:
            vecs += [_aug_flip(orig), _aug_brightness(orig, rng), _aug_noise(orig, rng)]
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
        orig  = _load_image(_find_png(row["stem"], data_dir))
        angle = _find_adversarial_rotation(orig, scaler, pca, clf)
        X_adv.append(_aug_rotate(orig, angle))
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
    x     = _load_image(png_path).reshape(1, -1)
    x_pca = pca.transform(scaler.transform(x))
    return float(clf.decision_function(x_pca)[0])


def _score_png_tta(png_path: Path, scaler, pca, clf) -> float:
    """TTA: average original + horizontally flipped score."""
    x      = _load_image(png_path)
    x_flip = _aug_flip(x)
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
        X_val  = np.stack([_load_image(_find_png(row["stem"], data_dir))
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
