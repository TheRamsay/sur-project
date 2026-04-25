#!/usr/bin/env python3
"""
E039 trimodal fusion: MFCC (diag) + LPCC-tied (E037) + Image-AdvRot (E033).

E039: 0.26% OOF EER, 0/222 errors. Weights from grid search: img≈0.66, lpcc≈0.34, mfcc≈0.00.
MFCC is redundant (weight→0) but kept so grid search can confirm this per run.

Backbones:
  - MFCC  (E008): UBM-32 diagonal, +NoiseSpeed aug
  - LPCC  (E037): UBM-32 tied covariance, +Pitch aug — tied cov 6.3× better than diag
  - Image (E033): PCA-50 + LogReg + adversarial rotation (2-pass training)

Usage:
    uv run predict_fusion.py --eval-dir /path/to/eval --output results/fusion_trimodal.txt
"""
import argparse
import copy
from pathlib import Path

import librosa
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.augment.audio import aug_codec, aug_noise, aug_pitch, aug_speed
from src.augment.image import (
    aug_brightness,
    aug_flip,
    aug_noise as aug_image_noise,
    aug_rotate,
    find_adversarial_rotation,
)
from src.data.manifest import find_png, find_wav
from src.data.splits import load_manifest, iter_folds_loso
from src.eval.metrics import compute_eer, compute_min_dcf
from src.features.audio import extract_lpcc, extract_mfcc
from src.features.image import load_image

SEED           = 67
UBM_COMPONENTS = 32
MAP_R          = 16.0
N_PCA          = 50
C_LOGREG       = 1.0


# ---------------------------------------------------------------------------
# UBM-MAP (shared for both MFCC and LPCC)
# ---------------------------------------------------------------------------

def _train_ubm(X, covariance_type="diag"):
    return GaussianMixture(n_components=UBM_COMPONENTS, covariance_type=covariance_type,
                           max_iter=200, random_state=SEED).fit(X)

def _map_adapt(ubm, X_target):
    log_resp  = ubm._estimate_log_prob(X_target) + np.log(ubm.weights_)
    log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
    resp      = np.exp(log_resp)
    n_k       = resp.sum(axis=0)
    mu_hat    = (resp.T @ X_target) / (n_k[:, None] + 1e-10)
    alpha     = n_k / (n_k + MAP_R)
    adapted   = copy.deepcopy(ubm)
    adapted.means_ = alpha[:, None] * mu_hat + (1 - alpha[:, None]) * ubm.means_
    return adapted


def _train_mfcc(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [aug_noise(y_wav, rng), aug_speed(y_wav, rng)]
        for y_aug in wavs:
            f = extract_mfcc(y_aug, sr)
            X_list.append(f); y_list.extend([row["label"]] * len(f))
    X = np.vstack(X_list); y = np.array(y_list)
    ubm     = _train_ubm(X[y == 0])
    adapted = _map_adapt(ubm, X[y == 1])
    return ubm, adapted


def _train_lpcc(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [aug_pitch(y_wav, sr, rng),   # E025: +Pitch
                     aug_codec(y_wav, sr)]          # E052: codec bandwidth
        for y_aug in wavs:
            f = extract_lpcc(y_aug, sr)
            X_list.append(f); y_list.extend([row["label"]] * len(f))
    X = np.vstack(X_list); y = np.array(y_list)
    ubm     = _train_ubm(X[y == 0], covariance_type="tied")  # E037: tied cov
    adapted = _map_adapt(ubm, X[y == 1])
    return ubm, adapted


def _score_mfcc(wav_path, adapted, ubm):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    f     = extract_mfcc(y, sr)
    return float((adapted.score_samples(f) - ubm.score_samples(f)).mean())


def _llr_lpcc(y, sr, adapted, ubm):
    f = extract_lpcc(y, sr)
    return float((adapted.score_samples(f) - ubm.score_samples(f)).mean())


def _score_lpcc(wav_path, adapted, ubm):
    """E031 +speed_tta: average LLR over original + 0.9× + 1.1× speed (3 views)."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    views = [y,
             librosa.effects.time_stretch(y, rate=0.9),
             librosa.effects.time_stretch(y, rate=1.1)]
    return float(np.mean([_llr_lpcc(v, sr, adapted, ubm) for v in views]))


# ---------------------------------------------------------------------------
# Image (E007)
# ---------------------------------------------------------------------------

def _train_image(df, data_dir, augment, seed):
    """2-pass: basic aug then adversarial rotation (E033)."""
    rng = np.random.default_rng(seed)

    # Pass 1: original + flip + brightness + noise
    X_basic, y_basic = [], []
    for _, row in df.iterrows():
        orig = load_image(find_png(row["stem"], data_dir))
        vecs = [orig]
        if augment:
            vecs += [aug_flip(orig), aug_brightness(orig, rng), aug_image_noise(orig, rng)]
        for v in vecs:
            X_basic.append(v); y_basic.append(row["label"])
    X_basic = np.stack(X_basic); y_basic = np.array(y_basic)

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
        X_adv.append(aug_rotate(orig, angle)); y_adv.append(row["label"])

    X_all = np.vstack([X_basic, np.stack(X_adv)])
    y_all = np.concatenate([y_basic, np.array(y_adv)])

    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=SEED)
    clf    = LogisticRegression(C=C_LOGREG, max_iter=1000, random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X_all))
    clf.fit(X_pca, y_all)
    return scaler, pca, clf


def _score_image(png_path, scaler, pca, clf):
    x = load_image(png_path).reshape(1, -1)
    return float(clf.decision_function(pca.transform(scaler.transform(x)))[0])


# ---------------------------------------------------------------------------
# Calibration + fusion
# ---------------------------------------------------------------------------

def _fit_calibrator(scores, labels):
    cal = LogisticRegression(C=1e6, max_iter=1000, class_weight="balanced")
    cal.fit(scores.reshape(-1, 1), labels)
    return cal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data",                   type=Path)
    parser.add_argument("--eval-dir", required=True,                     type=Path)
    parser.add_argument("--output",   default="results/fusion_score.txt", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    y_all    = manifest["label"].to_numpy()

    print("Collecting OOF scores (3 streams × 3 folds)...")
    oof_mfcc  = np.full(len(manifest), np.nan)
    oof_lpcc  = np.full(len(manifest), np.nan)
    oof_image = np.full(len(manifest), np.nan)

    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        seed_f   = SEED + fold_id
        train_df = manifest.loc[train_idx]
        val_df   = manifest.loc[val_idx]
        print(f"  fold {fold_id}...")

        ubm_m, ad_m = _train_mfcc(train_df, data_dir, augment=True, seed=seed_f)
        ubm_l, ad_l = _train_lpcc(train_df, data_dir, augment=True, seed=seed_f)
        scaler, pca, clf = _train_image(train_df, data_dir, augment=True, seed=seed_f)

        for idx, row in val_df.iterrows():
            oof_mfcc[idx]  = _score_mfcc(find_wav(row["stem"], data_dir), ad_m, ubm_m)
            oof_lpcc[idx]  = _score_lpcc(find_wav(row["stem"], data_dir), ad_l, ubm_l)
            oof_image[idx] = _score_image(find_png(row["stem"], data_dir), scaler, pca, clf)

    print("Fitting calibrators...")
    cal_m = _fit_calibrator(oof_mfcc,  y_all)
    cal_l = _fit_calibrator(oof_lpcc,  y_all)
    cal_i = _fit_calibrator(oof_image, y_all)
    cal_mo = cal_m.decision_function(oof_mfcc.reshape(-1, 1))
    cal_lo = cal_l.decision_function(oof_lpcc.reshape(-1, 1))
    cal_io = cal_i.decision_function(oof_image.reshape(-1, 1))

    print("Grid search (simplex, 51 steps)...")
    best = (np.inf, None)
    for w_m in np.linspace(0, 1, 51):
        for w_l in np.linspace(0, 1 - w_m, 51):
            w_i  = 1 - w_m - w_l
            fused = w_m * cal_mo + w_l * cal_lo + w_i * cal_io
            eer, _ = compute_eer(fused[y_all == 1], fused[y_all == 0])
            if eer < best[0]:
                best = (eer, (w_m, w_l, w_i))
    eer_best, (W_M, W_L, W_I) = best
    fused_oof = W_M * cal_mo + W_L * cal_lo + W_I * cal_io
    _, threshold = compute_min_dcf(fused_oof[y_all == 1], fused_oof[y_all == 0])
    print(f"  weights: mfcc={W_M:.2f}  lpcc={W_L:.2f}  image={W_I:.2f}")
    print(f"  OOF EER={eer_best*100:.2f}%  threshold={threshold:.4f}")

    print("Retraining all 3 models on full data...")
    ubm_m, ad_m = _train_mfcc(manifest, data_dir, augment=True, seed=SEED)
    ubm_l, ad_l = _train_lpcc(manifest, data_dir, augment=True, seed=SEED)
    scaler, pca, clf = _train_image(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval dir {eval_dir}...")
    wavs = sorted(eval_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files in {eval_dir}")

    with open(args.output, "w") as f:
        for wav in wavs:
            stem = wav.stem
            png  = eval_dir / (stem + ".png")

            raw_m = _score_mfcc(wav, ad_m, ubm_m)
            raw_l = _score_lpcc(wav, ad_l, ubm_l)
            s_m   = float(cal_m.decision_function([[raw_m]])[0])
            s_l   = float(cal_l.decision_function([[raw_l]])[0])

            if png.exists():
                raw_i = _score_image(png, scaler, pca, clf)
                s_i   = float(cal_i.decision_function([[raw_i]])[0])
                score = W_M * s_m + W_L * s_l + W_I * s_i
            else:
                # audio-only fallback if image missing
                wa = W_M + W_L
                score = (W_M / wa) * s_m + (W_L / wa) * s_l

            hard = 1 if score >= threshold else 0
            f.write(f"{stem} {score:.6f} {hard}\n")

    print(f"Written {len(wavs)} lines → {args.output}")


if __name__ == "__main__":
    main()
