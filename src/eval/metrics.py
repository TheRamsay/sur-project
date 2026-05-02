import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(
    scores_target: np.ndarray,
    scores_nontarget: np.ndarray,
) -> tuple[float, float]:
    """Return (eer, threshold) where eer in [0, 1]."""
    y_true = np.array([1] * len(scores_target) + [0] * len(scores_nontarget))
    y_score = np.concatenate([scores_target, scores_nontarget])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    far = fpr
    frr = 1.0 - tpr
    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_min_dcf(
    scores_target: np.ndarray,
    scores_nontarget: np.ndarray,
    p_target: float = 0.5,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    """Return (min_dcf, threshold). min_dcf < 1.0 beats the trivial baseline."""
    y_true = np.array([1] * len(scores_target) + [0] * len(scores_nontarget))
    y_score = np.concatenate([scores_target, scores_nontarget])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    far = fpr
    frr = 1.0 - tpr
    norm = min(c_miss * p_target, c_fa * (1.0 - p_target))
    dcf = (c_miss * p_target * frr + c_fa * (1.0 - p_target) * far) / norm
    idx = np.argmin(dcf)
    return float(dcf[idx]), float(thresholds[idx])


def make_hard_decisions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return int array of 0/1: 1 if score >= threshold."""
    return (scores >= threshold).astype(int)


def evaluate(
    scores_target: np.ndarray,
    scores_nontarget: np.ndarray,
) -> dict:
    """Return dict with eer, min_dcf, and recommended threshold."""
    eer, eer_thr = compute_eer(scores_target, scores_nontarget)
    min_dcf, dcf_thr = compute_min_dcf(scores_target, scores_nontarget)
    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "min_dcf": min_dcf,
        "dcf_threshold": dcf_thr,
    }
