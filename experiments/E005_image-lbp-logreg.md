# E005 — Image flagship: LBP + Logistic Regression

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E004

## Hypothesis

Local Binary Patterns encode local texture rather than global appearance,
making them more robust to lighting, background, and quality changes in eval
data. 4×4 grid of 256-bin LBP histograms (4096 features) + logistic
regression. Expected to match or improve on E004 (4.49%) and generalize
better to augmented eval samples.

## Setup

- **Modality:** image
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** LBP (P=8, R=1) on 80×80 grayscale, 4×4 spatial grid, 256-bin histograms → 4096 features, L1-normalized
- **Model:** LogisticRegression (C=1.0, max_iter=1000)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (same as all previous)
- **Seed:** 67
- **Command / notebook:** `notebooks/E005_image_lbp_logreg.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 4.17    | 0.0833  |
| 1    | 4.17    | 0.0833  |
| 2    | 45.00   | 0.5167  |
| mean ± std | 17.78 ± 23.58 | 0.2278 ± 0.2502 |

OOF overall: EER = 19.90%, min-DCF = 0.3354, threshold = −1.829

## Interpretation

Hypothesis failed — LBP significantly worse than PCA+logreg (17.78% vs 4.49%).
Folds 0 and 1 are competitive (4.17%), but fold 2 (session 03) collapsed to
45% EER — worse than chance. This suggests session 03 has a substantially
different visual appearance (different lighting, camera angle, or background)
that makes the LBP texture patterns unrecognizable relative to training sessions.

LBP is illumination-invariant locally but not globally — a large shift in
lighting can completely change which texture patterns dominate. PCA captures
global structure that is apparently more stable across sessions for this dataset.

Counter-intuitively, the "more robust" feature is worse here. This is the kind
of result that is worth documenting — it tells us the domain shift is not
illumination-driven but structural/pose-driven, where PCA global features hold up better.

## Next step

- Drop LBP as standalone system — E004 PCA+logreg remains the image anchor
- E006: PCA+LDA (Fisherfaces) — supervised dim reduction, should improve on E004
- Consider LBP as an additional feature channel concatenated with PCA features
