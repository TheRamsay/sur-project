"""Trimodal fusion weight search on the 2-simplex (E039)."""
import numpy as np

from src.eval.metrics import compute_eer


def grid_search_simplex(
    scores_list: list[np.ndarray],
    y: np.ndarray,
    n_steps: int = 51,
) -> tuple[float, tuple[float, float, float]]:
    """For three Platt-calibrated streams, scan w on the 2-simplex
    and return (best_eer, (w0, w1, w2)) that minimises OOF EER.

    `scores_list` is ordered [mfcc, lpcc, image] in the canonical fusion.
    """
    s0, s1, s2 = scores_list
    best = (np.inf, (0.0, 0.0, 1.0))
    for w0 in np.linspace(0, 1, n_steps):
        for w1 in np.linspace(0, 1 - w0, n_steps):
            w2 = 1 - w0 - w1
            fused = w0 * s0 + w1 * s1 + w2 * s2
            eer, _ = compute_eer(fused[y == 1], fused[y == 0])
            if eer < best[0]:
                best = (eer, (w0, w1, w2))
    return best
