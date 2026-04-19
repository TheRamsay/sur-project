#!/usr/bin/env python3
"""
E027 trimodal fusion system: calibrated MFCC + LPCC+Pitch audio + image.

Three streams:
  - MFCC audio (E008 +NoiseSpeed aug): UBM-MAP, MFCC 13+Δ+ΔΔ+CMN
  - LPCC audio (E025 +Pitch aug):       UBM-MAP, LPCC 13+Δ+ΔΔ+CMN (order=12)
  - Image (E007 +All aug):              PCA 50 + LogReg

Grid search on calibrated OOF finds optimal 3-way weights.
E027 OOF EER: 0.26% (vs E023 bi-fusion 0.52%, E009 MFCC+image 3.75%).

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

SEED           = 67
UBM_COMPONENTS = 32
MAP_R          = 16.0
SNR_DB         = 20.0
N_PCA          = 50
C_LOGREG       = 1.0


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _find_wav(stem, data_dir):
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = data_dir / sf / (stem + ".wav")
        if p.exists(): return p
    raise FileNotFoundError(stem)

def _find_png(stem, data_dir):
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = data_dir / sf / (stem + ".png")
        if p.exists(): return p
    raise FileNotFoundError(stem)


# ---------------------------------------------------------------------------
# MFCC audio (E008)
# ---------------------------------------------------------------------------

def _extract_mfcc(y, sr):
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.vstack([mfcc, delta, delta2]).T
    feat  -= feat.mean(axis=0)
    return feat

def _aug_noise(y, rng):
    p = np.mean(y**2) + 1e-10
    return y + rng.normal(0, np.sqrt(p / 10**(SNR_DB/10)), len(y)).astype(y.dtype)

def _aug_speed(y, rng):
    return librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))


# ---------------------------------------------------------------------------
# LPCC audio (E025 +Pitch)
# ---------------------------------------------------------------------------

def _extract_lpcc(y, sr, order=12, n_cep=13, hop_length=160, win_length=400):
    frames = librosa.util.frame(y, frame_length=win_length, hop_length=hop_length)
    cep_frames = []
    for frame in frames.T:
        frame = frame * np.hanning(len(frame))
        try:
            a = librosa.lpc(frame.astype(np.float64), order=order)
            A_freq = np.fft.rfft(a, n=512)
            log_H = -np.log(np.abs(A_freq) + 1e-10)
            cep = np.real(np.fft.irfft(log_H))[:n_cep]
        except Exception:
            cep = np.zeros(n_cep)
        cep_frames.append(cep)
    feat   = np.array(cep_frames, dtype=np.float32)
    delta  = librosa.feature.delta(feat.T).T
    delta2 = librosa.feature.delta(feat.T, order=2).T
    feat   = np.hstack([feat, delta, delta2])
    feat  -= feat.mean(axis=0)
    return feat

def _aug_pitch(y, sr, rng):
    n_steps = float(rng.choice([-2, -1, 1, 2]))
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


# ---------------------------------------------------------------------------
# UBM-MAP (shared for both MFCC and LPCC)
# ---------------------------------------------------------------------------

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


def _train_mfcc(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(_find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [_aug_noise(y_wav, rng), _aug_speed(y_wav, rng)]
        for y_aug in wavs:
            f = _extract_mfcc(y_aug, sr)
            X_list.append(f); y_list.extend([row["label"]] * len(f))
    X = np.vstack(X_list); y = np.array(y_list)
    ubm     = _train_ubm(X[y == 0])
    adapted = _map_adapt(ubm, X[y == 1])
    return ubm, adapted


def _train_lpcc(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(_find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [_aug_pitch(y_wav, sr, rng)]  # E025: +Pitch only
        for y_aug in wavs:
            f = _extract_lpcc(y_aug, sr)
            X_list.append(f); y_list.extend([row["label"]] * len(f))
    X = np.vstack(X_list); y = np.array(y_list)
    ubm     = _train_ubm(X[y == 0])
    adapted = _map_adapt(ubm, X[y == 1])
    return ubm, adapted


def _score_mfcc(wav_path, adapted, ubm):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    f     = _extract_mfcc(y, sr)
    return float((adapted.score_samples(f) - ubm.score_samples(f)).mean())


def _score_lpcc(wav_path, adapted, ubm):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    f     = _extract_lpcc(y, sr)
    return float((adapted.score_samples(f) - ubm.score_samples(f)).mean())


# ---------------------------------------------------------------------------
# Image (E007)
# ---------------------------------------------------------------------------

def _load_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (80, 80):
        img = img.resize((80, 80), Image.BILINEAR)
    return np.array(img, dtype=np.float32).mean(axis=2).flatten()

def _aug_flip(x): return x.reshape(80, 80)[:, ::-1].flatten()
def _aug_bright(x, rng): return np.clip(x * rng.uniform(0.7, 1.3), 0, 255)
def _aug_inoise(x, rng): return np.clip(x + rng.normal(0, 15, x.shape), 0, 255)


def _train_image(df, data_dir, augment, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _, row in df.iterrows():
        orig = _load_image(_find_png(row["stem"], data_dir))
        vecs = [orig]
        if augment:
            vecs += [_aug_flip(orig), _aug_bright(orig, rng), _aug_inoise(orig, rng)]
        for v in vecs:
            X.append(v); y.append(row["label"])
    X = np.stack(X); y = np.array(y)
    scaler = StandardScaler()
    pca    = PCA(n_components=N_PCA, random_state=SEED)
    clf    = LogisticRegression(C=C_LOGREG, max_iter=1000, random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X))
    clf.fit(X_pca, y)
    return scaler, pca, clf


def _score_image(png_path, scaler, pca, clf):
    x = _load_image(png_path).reshape(1, -1)
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
            oof_mfcc[idx]  = _score_mfcc(_find_wav(row["stem"], data_dir), ad_m, ubm_m)
            oof_lpcc[idx]  = _score_lpcc(_find_wav(row["stem"], data_dir), ad_l, ubm_l)
            oof_image[idx] = _score_image(_find_png(row["stem"], data_dir), scaler, pca, clf)

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
