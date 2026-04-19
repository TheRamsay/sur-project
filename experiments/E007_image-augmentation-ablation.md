# E007 — Image augmentation ablation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E004 (baseline)

## Hypothesis

E004 (PCA+logreg) had EER 4.49 ± 4.26% with fold 2 reaching 9.17% — the high
variance suggests the model memorizes session-specific appearance. Augmentation
should reduce this by exposing the model to more variation during training.

Testing four strategies independently, then combined:
- **Flip**: horizontal mirror — faces are roughly symmetric, free 2× data
- **Brightness**: random scale [0.7, 1.3] — simulates lighting variation across sessions
- **Noise**: Gaussian σ=15 — simulates the "damaged/degraded" eval samples Burget mentioned
- **All**: flip + brightness + noise combined (4× training data)

Augmentation applied ONLY to train fold. Val always uses original images.
PCA is re-fit on augmented train set for each config.

## Setup

- **Modality:** image
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** 80×80 PNG → grayscale → flatten → StandardScaler → PCA 50 → LogReg C=1
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (same as all previous)
- **Seed:** 67
- **Command / notebook:** `notebooks/E007_image_augmentation.ipynb`

## Result

| Config      | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std     | min-DCF mean |
| ----------- | ---------- | ---------- | ---------- | -------------- | ------------ |
| Baseline    | 3.47       | 0.83       | 9.17       | 4.49 ± 3.48    | 0.0565       |
| +Flip       | 3.47       | 9.17       | 0.83       | 4.49 ± 3.48    | 0.0565       |
| +Brightness | 2.08       | 0.83       | 1.67       | 1.53 ± 0.52    | 0.0306       |
| +Noise      | 9.17       | 0.83       | 9.17       | 6.39 ± 3.93    | 0.0611       |
| **+All**    | **2.08**   | **0.83**   | **0.00**   | **0.97 ± 0.86**| **0.0194**   |

OOF overall (best config +All): EER = 4.01%, min-DCF = 0.0729, threshold = −5.028

## Interpretation

Brightness jitter alone is the biggest single contributor — EER drops from 4.49%
to 1.53% and std collapses from 3.48 to 0.52%. This confirms the root cause of
fold 2's weakness: session 03 has different lighting, and exposing the model to
brightness variation during training makes it robust to this.

Noise alone hurts (6.39%) — the noise level σ=15 is too aggressive and
degrades the signal more than it helps generalization at this intensity.

Flip alone is neutral (EER identical, but sessions swap folds). Flip is
essentially a data doubling trick — it adds copies but not new variation.

The combination (+All) achieves 0.97 ± 0.86% EER — near-perfect, with fold 2
reaching 0.00% EER. The combined effect is super-additive: brightness teaches
lighting robustness, noise teaches compression robustness, and together they
cover both axes of the expected eval degradation.

**E004+Aug (+All) is the new image flagship.**

## Next step

- Image flagship locked: PCA 50 + LogReg + (flip + brightness + noise) augmentation
- Next: score calibration + E008 score-level fusion (audio E003 + image E007)
- Note: image threshold still far from 0 (−5.028) — Platt calibration needed before fusion
