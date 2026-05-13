# E010 — Audio UBM 64 components

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio augmentation ablation, UBM 32 flagship)

## Hypothesis

Increasing UBM from 32 to 64 components provides a richer background model.
With more Gaussian components, MAP adaptation has finer-grained statistics to
pull from — each component covers a narrower region of the MFCC space, so
the adapted means track the target more precisely.

Expected: 1–3% EER improvement over E008's 4.21 ± 3.11%.

Risk: with limited non-target training frames (~170k augmented), 64 components
may not converge well — sparse clusters could overfit to session-specific
acoustic conditions rather than generalizing.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E008
- **Model:** UBM 64 + MAP adapt r=16
- **Augmentation:** +All (noise SNR=20dB + speed ±10%) — same as E008 best config
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Command / notebook:** `notebooks/E010_audio_ubm64.ipynb`

## Result

| Config              | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std    | min-DCF mean |
| ------------------- | ---------- | ---------- | ---------- | ------------- | ------------ |
| E008 +All (32 comp) | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11   | 0.0509       |
| **E010 +All (64 comp)** | 9.17   | 9.17       | 0.83       | 6.39 ± 3.93   | 0.0611       |

Δ EER (E010 − E008): +2.18% (regression)

OOF overall: EER = 7.24%, min-DCF = 0.1271, threshold = +0.066

## Interpretation

The hypothesis did not hold. Doubling the UBM components from 32 to 64 hurts
performance: mean EER degrades from 4.21% to 6.39% (+2.18%), and std widens
from 3.11% to 3.93%.

The most telling detail is fold 0: E008 achieves 3.47% while E010 reaches
9.17% — identical to fold 1. This suggests the 64-component UBM converges
to a different (worse) partition of the non-target space on the fold 0 training
set, where non-target frame count is smallest (~170k). With ~5.3k frames per
component on average (vs ~2.7k for 64 comp), the 64-component UBM is data-
hungry relative to what fold 0 provides.

Fold 2 is identical at 0.83% — that fold has the most non-target training
frames (~188k) and both UBMs converge comparably.

OOF threshold shifts to +0.066 (E008 was −0.077), indicating a mild
calibration shift when using a 64-component model.

**Conclusion:** 32 components is the right capacity for this dataset size.
E008 +All remains the audio flagship.

## Next step

- 64 components is too many for the available non-target frames. Do not pursue
  128 components.
- Instead explore MAP relevance factor r: E008 uses r=16 — try r=8 and r=24
  to check if the adaptation strength is optimal (E011).
- Alternatively: explore MFCC dimension increase (n_mfcc=20) while keeping
  UBM at 32 components.
