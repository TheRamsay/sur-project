import librosa
import numpy as np


def aug_noise(y: np.ndarray, rng: np.random.Generator, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise at the requested SNR (E008)."""
    p = np.mean(y**2) + 1e-10
    return y + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(y)).astype(y.dtype)


def aug_speed(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random speed perturbation in [0.9, 1.1] (E008)."""
    return librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))


def aug_pitch(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Pitch shift by +/-1 or +/-2 semitones (E025, LPCC-specific)."""
    n_steps = float(rng.choice([-2, -1, 1, 2]))
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def aug_codec(y: np.ndarray, sr: int) -> np.ndarray:
    """Codec bandwidth simulation: downsample to 8 kHz and back (E052)."""
    y_down = librosa.resample(y, orig_sr=sr, target_sr=8000)
    return librosa.resample(y_down, orig_sr=8000, target_sr=sr)
