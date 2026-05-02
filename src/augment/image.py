import numpy as np
from scipy.ndimage import rotate as nd_rotate


def aug_flip(x: np.ndarray) -> np.ndarray:
    """Horizontal flip of a flattened 80x80 image."""
    return x.reshape(80, 80)[:, ::-1].flatten()


def aug_brightness(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative brightness jitter in [0.7, 1.3]."""
    return np.clip(x * rng.uniform(0.7, 1.3), 0, 255)


def aug_noise(x: np.ndarray, rng: np.random.Generator, sigma: float = 15.0) -> np.ndarray:
    """Add Gaussian pixel noise."""
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 255)


def aug_rotate(x: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the flattened 80x80 image by angle degrees, zero padding."""
    return nd_rotate(
        x.reshape(80, 80), angle, reshape=False, order=1, mode="constant", cval=0
    ).flatten()


def find_adversarial_rotation(
    x: np.ndarray, scaler, pca, clf, max_angle: float = 10.0, n_steps: int = 5
) -> float:
    """E033: pick the rotation angle in [-max_angle, +max_angle] where the
    Pass-1 model is most uncertain (smallest |logit|)."""
    best_angle, worst_abs = 0.0, np.inf
    for angle in np.linspace(-max_angle, max_angle, n_steps):
        if abs(angle) < 0.1:
            continue
        logit = clf.decision_function(
            pca.transform(scaler.transform(aug_rotate(x, angle).reshape(1, -1)))
        )[0]
        if abs(logit) < worst_abs:
            worst_abs, best_angle = abs(logit), angle
    return best_angle
