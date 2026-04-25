"""Image classifier: 80×80 grayscale, PCA-50, logistic regression, two-pass adversarial-rotation training (E033)."""
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
from src.features.image import load_image

N_PCA = 50
C_LOGREG = 1.0
MODEL_SEED = 67  # fixed across folds; the per-fold seed is for augmentation rng only


def _fit_pca_logreg(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    pca = PCA(n_components=N_PCA, random_state=MODEL_SEED)
    clf = LogisticRegression(C=C_LOGREG, max_iter=1000, random_state=MODEL_SEED)
    X_pca = pca.fit_transform(scaler.fit_transform(X))
    clf.fit(X_pca, y)
    return scaler, pca, clf


def train_image_pipeline(df, data_dir: Path, augment: bool, seed: int):
    """E033 two-pass training.

    Pass 1: original + flip + brightness + noise. Fit PCA + LogReg.
    Pass 2: per sample, find the rotation in [-10°, +10°] where Pass-1 is most uncertain
            (smallest |logit|), add a rotated copy at that angle, refit PCA + LogReg.
    """
    rng = np.random.default_rng(seed)

    # Pass 1
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
    scaler, pca, clf = _fit_pca_logreg(X_basic, y_basic)

    if not augment:
        return scaler, pca, clf

    # Pass 2: adversarial rotation per sample
    X_adv, y_adv = [], []
    for _, row in df.iterrows():
        orig = load_image(find_png(row["stem"], data_dir))
        angle = find_adversarial_rotation(orig, scaler, pca, clf)
        X_adv.append(aug_rotate(orig, angle))
        y_adv.append(row["label"])

    X_all = np.vstack([X_basic, np.stack(X_adv)])
    y_all = np.concatenate([y_basic, np.array(y_adv)])
    return _fit_pca_logreg(X_all, y_all)


def score_image(png_path: Path, scaler, pca, clf) -> float:
    x = load_image(png_path).reshape(1, -1)
    return float(clf.decision_function(pca.transform(scaler.transform(x)))[0])


def score_image_flip_tta(png_path: Path, scaler, pca, clf) -> float:
    """TTA: average logit on original + horizontal flip."""
    x = load_image(png_path)
    x_flip = aug_flip(x)
    s_orig = float(clf.decision_function(pca.transform(scaler.transform(x.reshape(1, -1))))[0])
    s_flip = float(clf.decision_function(pca.transform(scaler.transform(x_flip.reshape(1, -1))))[0])
    return (s_orig + s_flip) / 2
