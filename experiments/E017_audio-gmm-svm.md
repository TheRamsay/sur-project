# E017 — Audio GMM Supervector + Linear SVM

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio augmentation ablation, +All flagship, EER 4.21±3.11%)

## Hypothesis

The LLR scoring in E008 computes a scalar summary of how much the adapted model
fits vs the UBM. A linear SVM trained on deviation supervectors (adapted.means_ −
ubm.means_) should find a better decision boundary in the full 1248-dimensional
supervector space, capturing speaker-specific deformation patterns that the scalar
LLR collapses.

Three mechanisms expected to help:
1. **Richer representation**: supervector preserves per-Gaussian per-dimension
   deviation; LLR reduces all of that to one number.
2. **Discriminative training**: SVM optimizes margin between target and non-target
   supervectors; LLR scoring is generative (no direct non-target supervision).
3. **Robustness**: LinearSVC with high-dimensional input is well-studied and
   effective even when n_features >> n_samples (here ~222 utterances, 1248 dims).

Expectation: EER < 4.21% (E008 +All), ideally ≤ 3%.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E008, 39-dim
- **Model:** UBM 32 + MAP adapt r=16 per utterance → deviation supervector (1248-dim) → StandardScaler + LinearSVC(C=1.0)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO)
- **Seed:** 67
- **Command / notebook:** `notebooks/E017_audio_gmm_svm.ipynb`
- **Augmentation:** +All (noise SNR=20dB + speed ±10%), train fold only; augmented utterances also get supervectors (3× train supervectors)

## Result

| Config              | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std      | min-DCF mean |
| ------------------- | ---------- | ---------- | ---------- | --------------- | ------------ |
| E008 +All (LLR)     | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11     | 0.0509       |
| **E017 GMM-SVM**    | **1.39**   | **6.67**   | **19.17**  | **9.07 ± 7.45** | **0.1370**   |

OOF overall (E017): EER = 9.95%, min-DCF = 0.1521, threshold = +0.693

Delta vs E008: **+4.86% regression** (worse).

## Interpretation

The hypothesis did not hold. GMM-SVM regresses significantly vs E008 LLR
(9.07 ± 7.45% vs 4.21 ± 3.11%). The variance nearly doubled (7.45 vs 3.11),
indicating the SVM fits well to some folds but catastrophically fails on others
(Fold 2: 19.17%).

Several factors explain the failure:

1. **Severely underdetermined system**: 420–456 training supervectors in 1248
   dimensions. Even with StandardScaler + LinearSVC, fitting a hyperplane in
   1248-d with ~60 target examples is highly prone to overfitting. The SVM
   memorizes the training fold's speaker-specific deformations rather than
   learning a generalizable decision boundary.

2. **Session shift amplified**: In Fold 2, the val session is maximally
   different from both train sessions. The LLR formulation is robust here
   because the UBM provides a common reference; the SVM learns target-class
   geometry that doesn't transfer. The 19.17% EER (vs 0.83% for LLR) makes
   this concrete.

3. **LLR is a better inductive bias for small data**: The generative LLR
   score naturally leverages the UBM as a regularizer — the score is bounded
   by model fit quality. The SVM has no such regularization from the data
   structure; C=1.0 is insufficient when n_features >> n_samples.

4. **Augmented supervectors don't help generalization**: Even with 3×
   supervectors per utterance (420 total), the SVM still overfits because all
   augmented versions of one utterance are highly correlated — they don't add
   diverse speaker coverage, just small perturbations of the same deformation
   vector.

**The scalar LLR from a shared UBM is a more principled and robust scoring
function for small-data speaker verification than a discriminative SVM on
supervectors.**

## Next step

- E008 +All (LLR) remains the audio flagship at 4.21 ± 3.11%.
- GMM-SVM rejected for this dataset size; would require 10–100× more data to
  overcome the n_features >> n_samples regime.
- Potential avenue: reduce supervector dimensionality first (e.g. PCA to 50d)
  before SVM — but this is unlikely to beat the LLR inductive bias.
