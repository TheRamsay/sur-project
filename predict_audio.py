#!/usr/bin/env python3
"""
E052 audio system: UBM-MAP + LPCC 13+Δ+ΔΔ + tied covariance + pitch aug + codec aug + speed TTA.

Tied covariance (E037): 0.69% EER vs diagonal 4.35% — captures LPCC coefficient
correlations without overfitting (1521 shared params vs 48k for full).
Speed TTA (E042): 0.46% EER — averaging over 0.9×/1.0×/1.1× speed is pitch-preserving,
so LPCC formant coefficients stay valid.
Codec aug (E052): 0.46% clean / 3.33% codec (vs 13.33% without) — adding 8kHz-bandwidth
copies of training audio forces LPCC to learn formants surviving low-pass filtering.

Usage:
    uv run predict_audio.py --eval-dir /path/to/eval --output results/audio_lpcc_tied_speedtta.txt
"""
import argparse
import copy
from pathlib import Path

import librosa
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from src.augment.audio import aug_codec, aug_pitch
from src.data.manifest import find_wav
from src.data.splits import load_manifest, iter_folds_loso
from src.eval.metrics import compute_min_dcf
from src.features.audio import extract_lpcc

SEED = 67
UBM_COMPONENTS = 32
MAP_R = 16.0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _extract_frames(df, data_dir: Path, augment: bool, seed: int):
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [aug_pitch(y_wav, sr, rng),    # E025: pitch shift
                     aug_codec(y_wav, sr)]           # E052: codec bandwidth
        for y_aug in wavs:
            frames = extract_lpcc(y_aug, sr)
            all_X.append(frames)
            all_y.extend([row["label"]] * len(frames))
    return np.vstack(all_X), np.array(all_y)


def _llr(y: np.ndarray, sr: int, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    f = extract_lpcc(y, sr)
    return float((adapted.score_samples(f) - ubm.score_samples(f)).mean())


def _score_wav_tta(wav_path: Path, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    """E031 +speed_tta: average LLR over original + 0.9× + 1.1× speed (3 views).
    Speed perturbation is pitch-preserving → LPCC formant coefficients unchanged.
    Pitch TTA hurts (corrupts formant structure); speed TTA is safe.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    views = [y,
             librosa.effects.time_stretch(y, rate=0.9),
             librosa.effects.time_stretch(y, rate=1.1)]
    return float(np.mean([_llr(v, sr, adapted, ubm) for v in views]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train_ubm(X: np.ndarray) -> GaussianMixture:
    return GaussianMixture(n_components=UBM_COMPONENTS, covariance_type="tied",
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
            oof[idx] = _score_wav_tta(find_wav(row["stem"], data_dir), adapted, ubm)
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
