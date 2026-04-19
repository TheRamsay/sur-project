#!/usr/bin/env python3
"""
E025 audio system: UBM-MAP + LPCC 13+Δ+ΔΔ + pitch-shift augmentation.

LPCC with pitch augmentation is the audio flagship on this dataset:
EER 1.94±1.57% (E025) vs LPCC+NoiseSpeed 3.33±4.14% (E020) vs MFCC 4.21% (E008).
LPCC encodes formants directly, so pitch perturbation is the right aug axis.

Usage:
    uv run predict_audio.py --eval-dir /path/to/eval --output results/audio_ubm_map.txt
"""
import argparse
import copy
from pathlib import Path

import librosa
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from src.data.splits import load_manifest, iter_folds_loso
from src.eval.metrics import compute_min_dcf

SEED = 67
UBM_COMPONENTS = 32
MAP_R = 16.0
SNR_DB = 20.0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _find_wav(stem: str, data_dir: Path) -> Path:
    for sf in ("target_train", "target_dev", "non_target_train", "non_target_dev"):
        p = data_dir / sf / (stem + ".wav")
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def _extract_features(y: np.ndarray, sr: int,
                      order: int = 12, n_cep: int = 13,
                      hop_length: int = 160, win_length: int = 400) -> np.ndarray:
    """LPCC 13+Δ+ΔΔ+CMN (E020). Outperforms MFCC on this dataset."""
    frames = librosa.util.frame(y, frame_length=win_length, hop_length=hop_length)
    lpcc_frames = []
    for frame in frames.T:
        frame = frame * np.hanning(len(frame))
        try:
            a = librosa.lpc(frame.astype(np.float64), order=order)
            A_freq = np.fft.rfft(a, n=512)
            log_H = -np.log(np.abs(A_freq) + 1e-10)
            cep = np.real(np.fft.irfft(log_H))[:n_cep]
        except Exception:
            cep = np.zeros(n_cep)
        lpcc_frames.append(cep)
    feat   = np.array(lpcc_frames, dtype=np.float32)
    delta  = librosa.feature.delta(feat.T).T
    delta2 = librosa.feature.delta(feat.T, order=2).T
    feat   = np.hstack([feat, delta, delta2])
    feat  -= feat.mean(axis=0)
    return feat


def _aug_pitch(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    n_steps = float(rng.choice([-2, -1, 1, 2]))
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def _aug_speed(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # kept for TTA consistency (deterministic speed perturbation)
    return librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))


def _extract_frames(df, data_dir: Path, augment: bool, seed: int):
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(_find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [_aug_pitch(y_wav, sr, rng)]    # E025: pitch-only
        for y_aug in wavs:
            frames = _extract_features(y_aug, sr)
            all_X.append(frames)
            all_y.extend([row["label"]] * len(frames))
    return np.vstack(all_X), np.array(all_y)


def _score_wav(wav_path: Path, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    mfcc  = _extract_features(y, sr)
    return float((adapted.score_samples(mfcc) - ubm.score_samples(mfcc)).mean())


def _score_wav_tta(wav_path: Path, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    """TTA: average original + one speed-perturbed score (fixed seed → deterministic)."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    score_orig = float((adapted.score_samples(_extract_features(y, sr))
                        - ubm.score_samples(_extract_features(y, sr))).mean())
    y_tta = _aug_speed(y, np.random.default_rng(0))
    score_tta  = float((adapted.score_samples(_extract_features(y_tta, sr))
                        - ubm.score_samples(_extract_features(y_tta, sr))).mean())
    return (score_orig + score_tta) / 2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train_ubm(X: np.ndarray) -> GaussianMixture:
    return GaussianMixture(n_components=UBM_COMPONENTS, covariance_type="diag",
                           max_iter=200, random_state=SEED).fit(X)


def _map_adapt(ubm: GaussianMixture, X_target: np.ndarray) -> GaussianMixture:
    log_resp  = ubm._estimate_log_prob(X_target) + np.log(ubm.weights_)
    log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
    resp      = np.exp(log_resp)
    n_k       = resp.sum(axis=0)
    mu_hat    = (resp.T @ X_target) / (n_k[:, None] + 1e-10)
    alpha     = n_k / (n_k + MAP_R)
    adapted   = copy.deepcopy(ubm)
    adapted.means_ = alpha[:, None] * mu_hat + (1 - alpha[:, None]) * ubm.means_
    return adapted


def _train(df, data_dir: Path, augment: bool, seed: int):
    X, y = _extract_frames(df, data_dir, augment=augment, seed=seed)
    ubm     = _train_ubm(X[y == 0])
    adapted = _map_adapt(ubm, X[y == 1])
    return ubm, adapted


# ---------------------------------------------------------------------------
# Calibration via OOF
# ---------------------------------------------------------------------------

def _collect_oof(manifest, data_dir: Path) -> np.ndarray:
    oof = np.full(len(manifest), np.nan)
    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=SEED):
        ubm, adapted = _train(manifest.loc[train_idx], data_dir,
                              augment=True, seed=SEED + fold_id)
        for idx, row in manifest.loc[val_idx].iterrows():
            oof[idx] = _score_wav(_find_wav(row["stem"], data_dir), adapted, ubm)
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
    parser.add_argument("--data-dir",  default="data",           type=Path)
    parser.add_argument("--eval-dir",  required=True,             type=Path)
    parser.add_argument("--output",    default="results/audio_ubm_map.txt", type=Path)
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
    ubm, adapted = _train(manifest, data_dir, augment=True, seed=SEED)

    print(f"Scoring eval data in {eval_dir} …")
    wavs = sorted(eval_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {eval_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for wav in wavs:
            raw   = _score_wav_tta(wav, adapted, ubm)
            score = float(cal.decision_function([[raw]])[0])
            hard  = 1 if score >= threshold else 0
            f.write(f"{wav.stem} {score:.6f} {hard}\n")

    print(f"Written {len(wavs)} lines → {args.output}")


if __name__ == "__main__":
    main()
