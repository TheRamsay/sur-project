#!/usr/bin/env python3
"""
E023 fusion system: calibrated LPCC audio (E020) + image (E007) scores.

Audio uses LPCC 13+Δ+ΔΔ (E020, EER 3.33%) instead of MFCC (E008, 4.21%).
Finds optimal fusion weight via grid search on OOF calibrated scores (w≈0.36 audio).
OOF fusion EER: 0.52% vs E009 MFCC+image 3.75%.

Usage:
    uv run predict_fusion.py --eval-dir /path/to/eval --output results/fusion_score.txt
"""
import argparse
import copy
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data.splits import load_manifest, iter_folds_loso
from src.eval.metrics import compute_eer, compute_min_dcf

SEED = 67
# Audio
UBM_COMPONENTS = 32
MAP_R = 16.0
SNR_DB = 20.0
# Image
N_PCA = 50
C_LOGREG = 1.0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _find_wav(stem: str, data_dir: Path) -> Path:
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = data_dir / sf / (stem + ".wav")
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def _extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """LPCC 13+Δ+ΔΔ+CMN (E020 flagship — outperforms MFCC)."""
    frames = librosa.util.frame(y, frame_length=400, hop_length=160)
    lpcc_frames = []
    for frame in frames.T:
        frame = frame * np.hanning(len(frame))
        try:
            a = librosa.lpc(frame.astype(np.float64), order=12)
            A_freq = np.fft.rfft(a, n=512)
            log_H = -np.log(np.abs(A_freq) + 1e-10)
            cep = np.real(np.fft.irfft(log_H))[:13]
        except Exception:
            cep = np.zeros(13)
        lpcc_frames.append(cep)
    feat   = np.array(lpcc_frames, dtype=np.float32)
    delta  = librosa.feature.delta(feat.T).T
    delta2 = librosa.feature.delta(feat.T, order=2).T
    feat   = np.hstack([feat, delta, delta2])
    feat  -= feat.mean(axis=0)
    return feat


def _aug_noise_audio(y, rng):
    p = np.mean(y ** 2) + 1e-10
    return y + rng.normal(0, np.sqrt(p / 10 ** (SNR_DB / 10)), len(y)).astype(y.dtype)


def _aug_speed(y, rng):
    return librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))


def _extract_audio_frames(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(_find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav] + ([_aug_noise_audio(y_wav, rng), _aug_speed(y_wav, rng)] if augment else [])
        for y_aug in wavs:
            frames = _extract_mfcc(y_aug, sr)
            all_X.append(frames)
            all_y.extend([row["label"]] * len(frames))
    return np.vstack(all_X), np.array(all_y)


def _train_ubm(X):
    return GaussianMixture(n_components=UBM_COMPONENTS, covariance_type="diag",
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


def _train_audio(df, data_dir, augment, seed):
    X, y = _extract_audio_frames(df, data_dir, augment=augment, seed=seed)
    ubm     = _train_ubm(X[y == 0])
    adapted = _map_adapt(ubm, X[y == 1])
    return ubm, adapted


def _score_wav(wav_path, adapted, ubm):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    mfcc  = _extract_mfcc(y, sr)
    return float((adapted.score_samples(mfcc) - ubm.score_samples(mfcc)).mean())


def _score_wav_tta(wav_path, adapted, ubm):
    y, sr      = librosa.load(wav_path, sr=None, mono=True)
    score_orig = float((adapted.score_samples(_extract_mfcc(y, sr))
                        - ubm.score_samples(_extract_mfcc(y, sr))).mean())
    y_tta      = _aug_speed(y, np.random.default_rng(0))
    score_tta  = float((adapted.score_samples(_extract_mfcc(y_tta, sr))
                        - ubm.score_samples(_extract_mfcc(y_tta, sr))).mean())
    return (score_orig + score_tta) / 2


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _find_png(stem: str, data_dir: Path) -> Path:
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = data_dir / sf / (stem + ".png")
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def _load_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (80, 80):
        img = img.resize((80, 80), Image.BILINEAR)
    return np.array(img, dtype=np.float32).mean(axis=2).flatten()


def _load_image_dataset(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _, row in df.iterrows():
        orig = _load_image(_find_png(row["stem"], data_dir))
        vecs = [orig]
        if augment:
            vecs += [
                orig.reshape(80, 80)[:, ::-1].flatten(),
                np.clip(orig * rng.uniform(0.7, 1.3), 0, 255),
                np.clip(orig + rng.normal(0, 15, orig.shape), 0, 255),
            ]
        for v in vecs:
            X.append(v); y.append(row["label"])
    return np.stack(X), np.array(y)


def _train_image(df, data_dir, augment, seed):
    X, y   = _load_image_dataset(df, data_dir, augment=augment, seed=seed)
    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=SEED)
    clf    = LogisticRegression(C=C_LOGREG, max_iter=1000, random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X))
    clf.fit(X_pca, y)
    return scaler, pca, clf


def _score_png(png_path, scaler, pca, clf):
    x = _load_image(png_path).reshape(1, -1)
    return float(clf.decision_function(pca.transform(scaler.transform(x)))[0])


def _score_png_tta(png_path, scaler, pca, clf):
    x      = _load_image(png_path)
    x_flip = x.reshape(80, 80)[:, ::-1].flatten()
    s_orig = float(clf.decision_function(pca.transform(scaler.transform(x.reshape(1,-1))))[0])
    s_flip = float(clf.decision_function(pca.transform(scaler.transform(x_flip.reshape(1,-1))))[0])
    return (s_orig + s_flip) / 2


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _fit_calibrator(oof_scores, labels):
    cal = LogisticRegression(C=1e6, max_iter=1000, class_weight="balanced")
    cal.fit(oof_scores.reshape(-1, 1), labels)
    return cal


def _collect_oof(manifest, data_dir):
    oof_audio = np.full(len(manifest), np.nan)
    oof_image = np.full(len(manifest), np.nan)

    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        fold_seed = SEED + fold_id
        train_df  = manifest.loc[train_idx]
        val_df    = manifest.loc[val_idx]

        ubm, adapted = _train_audio(train_df, data_dir, augment=True, seed=fold_seed)
        for idx, row in val_df.iterrows():
            oof_audio[idx] = _score_wav(_find_wav(row["stem"], data_dir), adapted, ubm)

        scaler, pca, clf = _train_image(train_df, data_dir, augment=True, seed=fold_seed)
        X_val = np.stack([_load_image(_find_png(row["stem"], data_dir))
                          for _, row in val_df.iterrows()])
        oof_image[val_idx] = clf.decision_function(pca.transform(scaler.transform(X_val)))

    return oof_audio, oof_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data",                   type=Path)
    parser.add_argument("--eval-dir",  required=True,                     type=Path)
    parser.add_argument("--output",    default="results/fusion_score.txt", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    y_all    = manifest["label"].to_numpy()

    print("Collecting OOF scores for calibration (this runs CV for both modalities)…")
    oof_audio_raw, oof_image_raw = _collect_oof(manifest, data_dir)

    print("Fitting calibrators…")
    cal_audio = _fit_calibrator(oof_audio_raw, y_all)
    cal_image = _fit_calibrator(oof_image_raw, y_all)
    oof_audio_cal = cal_audio.decision_function(oof_audio_raw.reshape(-1, 1))
    oof_image_cal = cal_image.decision_function(oof_image_raw.reshape(-1, 1))

    # Grid search for optimal audio weight
    weights   = np.linspace(0, 1, 101)
    grid_eers = []
    for w in weights:
        s = w * oof_audio_cal + (1 - w) * oof_image_cal
        eer, _ = compute_eer(s[y_all == 1], s[y_all == 0])
        grid_eers.append(eer)
    best_w = weights[np.argmin(grid_eers)]
    print(f"  Optimal audio weight: {best_w:.2f}  (image weight: {1-best_w:.2f})")

    oof_fused = best_w * oof_audio_cal + (1 - best_w) * oof_image_cal
    _, threshold = compute_min_dcf(oof_fused[y_all == 1], oof_fused[y_all == 0])
    print(f"  Fusion threshold (min-DCF on OOF): {threshold:.4f}")

    print("Retraining audio model on all data…")
    ubm, adapted = _train_audio(manifest, data_dir, augment=True, seed=SEED)

    print("Retraining image model on all data…")
    scaler, pca, clf = _train_image(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval data in {eval_dir} …")
    wavs = sorted(eval_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {eval_dir}")

    with open(args.output, "w") as f:
        for wav in wavs:
            stem = wav.stem
            png  = eval_dir / (stem + ".png")

            raw_audio = _score_wav_tta(wav, adapted, ubm)
            cal_audio_score = float(cal_audio.decision_function([[raw_audio]])[0])

            if png.exists():
                raw_image = _score_png_tta(png, scaler, pca, clf)
                cal_image_score = float(cal_image.decision_function([[raw_image]])[0])
                score = best_w * cal_audio_score + (1 - best_w) * cal_image_score
            else:
                score = cal_audio_score  # fall back to audio-only if image missing

            hard = 1 if score >= threshold else 0
            f.write(f"{stem} {score:.6f} {hard}\n")

    print(f"Written {len(wavs)} lines → {args.output}")


if __name__ == "__main__":
    main()
