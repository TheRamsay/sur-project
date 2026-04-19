# E006 — Image flagship: PCA + LDA (Fisherfaces)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E004

## Hypothesis

Fisherfaces = PCA then LDA. LDA directly maximizes between-class / within-class
scatter ratio — a supervised step that PCA (E004) skips. For binary
classification, LDA produces exactly 1 discriminant direction (the "Fisherface").

With only ~20 target samples per fold, standard LDA's within-class scatter
matrix is ill-conditioned. We use Ledoit-Wolf automatic shrinkage
(`solver='lsqr', shrinkage='auto'`) to regularize it. This is not a hack —
it is the statistically correct estimator for small-sample covariance.

Expected: match or improve on E004 (4.49% EER). If LDA's supervised signal
helps, EER should drop. If the dataset is too small for LDA to converge
reliably (even with shrinkage), it may not improve.

## Setup

- **Modality:** image
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** 80×80 PNG → grayscale → flatten (6400) → StandardScaler → PCA 100 → LDA
- **Model:** LDA (`solver='lsqr'`, `shrinkage='auto'`), score = `decision_function`
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (same as all previous)
- **Seed:** 67
- **Command / notebook:** `notebooks/E006_image_fisherfaces.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 19.72   | 0.2639  |
| 1    | 18.33   | 0.2000  |
| 2    | 16.67   | 0.3333  |
| mean ± std | 18.24 ± 1.53 | 0.2657 ± 0.0667 |

OOF overall: EER = 22.60%, min-DCF = 0.3771, threshold = −4.459

## Interpretation

Hypothesis failed — PCA+LDA is significantly worse than PCA+logreg
(18.24% vs 4.49% EER), despite Ledoit-Wolf shrinkage.

Root cause: for binary classification, LDA projects to exactly **1 dimension**.
That single direction is estimated from only ~20 target samples — even with
shrinkage it's a poor estimator. Logistic regression (E004) operates in the
full 50-dimensional PCA space and learns a richer decision boundary using all
50 directions jointly. More dimensions + regularized logreg > supervised 1D
bottleneck on this dataset size.

The std is small (1.53%) — LDA is consistently bad across all folds, not just
unlucky on one session. This rules out a fold-specific fluke.

**E004 PCA+logreg remains the image flagship.**

## Next step

- E004 is confirmed image flagship — no further image ablations needed before fusion
- Next: E007 score-level fusion (audio E003 OOF + image E004 OOF → logistic regression)
- Calibration pass before fusion: image E004 threshold is −4.996 (far from 0),
  audio E003 is −0.025 (near 0). Need to calibrate image scores.
