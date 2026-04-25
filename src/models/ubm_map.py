"""UBM-MAP primitives shared by all GMM-based audio backbones."""
import copy
from pathlib import Path

import librosa
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from src.augment.audio import aug_codec, aug_noise, aug_pitch, aug_speed
from src.data.manifest import find_wav
from src.features.audio import extract_lpcc, extract_mfcc

UBM_COMPONENTS = 32
MAP_R = 16.0
MODEL_SEED = 67  # fixed across folds; the per-fold seed is for augmentation rng only


def train_ubm(X: np.ndarray, covariance_type: str = "diag", seed: int = MODEL_SEED) -> GaussianMixture:
    return GaussianMixture(
        n_components=UBM_COMPONENTS,
        covariance_type=covariance_type,
        max_iter=200,
        random_state=seed,
    ).fit(X)


def map_adapt(ubm: GaussianMixture, X_target: np.ndarray, r: float = MAP_R) -> GaussianMixture:
    """Means-only MAP adaptation (Reynolds 2000). Adapts only μ_k, keeps Σ_k and π_k."""
    log_resp = ubm._estimate_log_prob(X_target) + np.log(ubm.weights_)
    log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
    resp = np.exp(log_resp)
    n_k = resp.sum(axis=0)
    mu_hat = (resp.T @ X_target) / (n_k[:, None] + 1e-10)
    alpha = n_k / (n_k + r)
    adapted = copy.deepcopy(ubm)
    adapted.means_ = alpha[:, None] * mu_hat + (1 - alpha[:, None]) * ubm.means_
    return adapted


def llr_score(features: np.ndarray, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    """Frame-averaged log-likelihood ratio between target model and UBM."""
    return float((adapted.score_samples(features) - ubm.score_samples(features)).mean())


# ---------------------------------------------------------------------------
# Production audio pipelines
# ---------------------------------------------------------------------------

def train_lpcc_pipeline(df, data_dir: Path, augment: bool, seed: int):
    """E052: LPCC + tied-covariance UBM + MAP r=16, with pitch & codec augmentation."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [aug_pitch(y_wav, sr, rng), aug_codec(y_wav, sr)]
        for y_aug in wavs:
            f = extract_lpcc(y_aug, sr)
            X_list.append(f)
            y_list.extend([row["label"]] * len(f))
    X = np.vstack(X_list)
    y = np.array(y_list)
    ubm = train_ubm(X[y == 0], covariance_type="tied")
    adapted = map_adapt(ubm, X[y == 1])
    return ubm, adapted


def train_mfcc_pipeline(df, data_dir: Path, augment: bool, seed: int):
    """E008: MFCC + diagonal-covariance UBM + MAP r=16, with noise & speed augmentation."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        y_wav, sr = librosa.load(find_wav(row["stem"], data_dir), sr=None, mono=True)
        wavs = [y_wav]
        if augment:
            wavs += [aug_noise(y_wav, rng), aug_speed(y_wav, rng)]
        for y_aug in wavs:
            f = extract_mfcc(y_aug, sr)
            X_list.append(f)
            y_list.extend([row["label"]] * len(f))
    X = np.vstack(X_list)
    y = np.array(y_list)
    ubm = train_ubm(X[y == 0], covariance_type="diag")
    adapted = map_adapt(ubm, X[y == 1])
    return ubm, adapted


def score_mfcc(wav_path: Path, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    return llr_score(extract_mfcc(y, sr), adapted, ubm)


def score_lpcc_speed_tta(wav_path: Path, adapted: GaussianMixture, ubm: GaussianMixture) -> float:
    """E031: average LLR over original + 0.9× + 1.1× speed (3 views).

    Speed perturbation preserves the spectral envelope, so LPCC is invariant.
    Pitch TTA was tested and rejected (it corrupts formant coefficients).
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    views = [
        y,
        librosa.effects.time_stretch(y, rate=0.9),
        librosa.effects.time_stretch(y, rate=1.1),
    ]
    return float(np.mean([llr_score(extract_lpcc(v, sr), adapted, ubm) for v in views]))
