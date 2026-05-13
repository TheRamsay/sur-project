# E004 — Image baseline: PCA + Logistic Regression

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** —

## Hypothesis

80×80 grayscale images flattened to 6400-dim vectors, PCA to 50 dims (fit on
train fold only), logistic regression classifier. Score = decision_function
(log-odds). Expected EER in the 20–35% range — images alone should be weaker
than audio given session variability in lighting/background.

## Setup

- **Modality:** image
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** 80×80 PNG → grayscale → flatten (6400) → PCA 50 components
- **Model:** LogisticRegression (C=1.0, max_iter=1000)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (same as audio experiments)
- **Seed:** 67
- **Command / notebook:** `notebooks/E004_image_pca_logreg.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 3.47    | 0.0694  |
| 1    | 0.83    | 0.0167  |
| 2    | 9.17    | 0.0833  |
| mean ± std | 4.49 ± 4.26 | 0.0565 ± 0.0352 |

OOF overall: EER = 9.69%, min-DCF = 0.1094, threshold = −4.996

## Interpretation

Hypothesis wrong — image baseline significantly outperforms the audio flagship
(E003: 7.45%). Fold 1 is near-perfect (0.83% EER). This suggests Ondra's face
is very distinctive in the PCA space, at least within the training sessions.

High std (4.26%) and fold 2 being the worst (9.17%) indicates session 03 is
harder visually — likely different lighting or background. Threshold far from 0
(−4.996) means scores are poorly calibrated — the classifier is overconfident
about non-target class.

This is a strong anchor. The image modality carries more discriminative signal
than expected from a simple PCA+logreg.

## Next step

- E005: LBP (Local Binary Patterns) — texture features robust to lighting changes
- E006: PCA+LDA (Fisherfaces) — supervised dimensionality reduction
- Calibration pass on OOF scores before fusion
