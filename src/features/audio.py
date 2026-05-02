import librosa
import numpy as np


def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """MFCC 13 + delta + delta-delta with per-utterance CMN. Frame-level features (T, 39)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2]).T
    feat -= feat.mean(axis=0)
    return feat


def extract_lpcc(
    y: np.ndarray,
    sr: int,
    order: int = 12,
    n_cep: int = 13,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """LPCC 13 + delta + delta-delta with per-utterance CMN. Frame-level features (T, 39).
    """
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
    feat = np.array(cep_frames, dtype=np.float32)
    delta = librosa.feature.delta(feat.T).T
    delta2 = librosa.feature.delta(feat.T, order=2).T
    feat = np.hstack([feat, delta, delta2])
    feat -= feat.mean(axis=0)
    return feat
