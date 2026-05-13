# E016 — Audio FBank 40+Δ+ΔΔ (120 features/frame)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio augmentation ablation, +All flagship)

## Hypothesis

MFCC applies DCT compression that collapses 40 mel-filterbank energies into 13
coefficients. The DCT discards fine-grained spectral shape information that may
carry speaker-discriminative cues. Replacing MFCC with raw log mel-filterbank
features (FBank 40) + Δ+ΔΔ should preserve more spectral detail and yield a
small but consistent improvement over E008 (4.21 ± 3.11%).

Everything else is identical to E008 +All: UBM 32, MAP r=16, noise SNR=20dB +
speed ±10%, LOSO 3-fold, seed=67.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** FBank 40+Δ+ΔΔ = 120 features/frame, CMN (per-utterance)
- **Model:** UBM 32 + MAP adapt r=16
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO)
- **Seed:** 67
- **Command / notebook:** `notebooks/E016_audio_fbank.ipynb`
- **Augmentation:** +All (noise SNR=20dB + speed ±10%), train fold only

## Result

| Config               | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std      | min-DCF mean |
| -------------------- | ---------- | ---------- | ---------- | --------------- | ------------ |
| E008 +All (MFCC 39d) | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11     | 0.0509       |
| **E016 +All (FBank 120d)** | **9.86** | **8.33** | **11.67** | **9.95 ± 1.36** | **0.1324** |

OOF overall (E016): EER = 8.91%, min-DCF = 0.1781, threshold = +0.229

Delta vs E008: **+5.74% regression**.

## Interpretation

The hypothesis did not hold. FBank 120d regresses significantly vs MFCC 39d
(9.95% vs 4.21% mean EER, +5.74%). This is counter-intuitive given that FBank
retains more spectral information, but several factors explain it:

1. **Dimensionality mismatch**: 120-dim FBank with UBM-32 means ~3.75 frames per
   Gaussian parameter dimension. The GMM is severely under-parameterized for 120d
   input — the covariance diagonal is unreliable in high dimensions with limited
   training data. MFCC's DCT compression to 39d is actually beneficial here as a
   form of regularization.

2. **DCT is not pure information loss**: DCT concentrates speaker-discriminative
   energy in the first few coefficients (MFCCs). The higher-order mel energies
   kept by FBank are dominated by noise and channel variation rather than speaker
   identity. MFCC naturally down-weights these noisy dimensions.

3. **Variance increase reversed**: While E008 had high fold-to-fold variance
   (3.11%), E016's std is lower (1.36%) but at a much worse mean — the model is
   consistently bad rather than variably good. The threshold also shifted to +0.229
   (far from 0), indicating score miscalibration.

4. **No compensation for dimensionality**: More advanced methods (i-vector, PLDA)
   benefit from FBank because they include dimensionality reduction and whitening
   as explicit steps. A naive GMM-UBM without explicit dim-reduction does not.

**MFCC's DCT compression acts as beneficial regularization for GMM-UBM with small
datasets. FBank is appropriate for neural speaker embedding systems with explicit
dim-reduction, not for GMM-based backends.**

## Next step

- E008 +All (MFCC 39d) remains the audio flagship at 4.21 ± 3.11%.
- FBank rejected for GMM-UBM pipeline — do not revisit without explicit dim-reduction.
- Next dimension to explore: neural feature extraction or i-vector backend, both
  of which can properly handle high-dimensional FBank input.
