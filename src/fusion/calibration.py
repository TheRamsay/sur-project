import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling: a one-feature LogReg on raw scores.

    `class_weight='balanced'` keeps the calibration honest under the strong
    class imbalance (target is ~10 % of the manifest).
    """
    cal = LogisticRegression(C=1e6, max_iter=1000, class_weight="balanced")
    cal.fit(scores.reshape(-1, 1), labels)
    return cal


def apply_platt(cal: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return cal.decision_function(scores.reshape(-1, 1))
