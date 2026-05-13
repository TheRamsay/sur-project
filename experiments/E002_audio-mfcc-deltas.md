# E002 — Audio: MFCC + delta + delta-delta

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E001

## Hypothesis

Adding delta and delta-delta to 13 static MFCCs (13→39 features per frame)
should capture vocal tract dynamics and improve EER by 3–5 % over E001
(17.92 %). If it doesn't help, the dataset is too small to benefit from the
extra features.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13 + Δ + ΔΔ = 39 features, CMN (per-utterance)
- **Model:** two GMMs — target (8 components), non-target (32 components)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (identical to E001)
- **Seed:** 67
- **Command / notebook:** `notebooks/E002_audio_mfcc_deltas.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 11.94   | 0.2389  |
| 1    | 10.00   | 0.1333  |
| 2    | 8.33    | 0.1667  |
| mean ± std | 10.09 ± 1.81 | 0.1796 ± 0.0540 |

OOF overall: EER = 13.70%, min-DCF = 0.2073, threshold = −2.204

## Interpretation

Hypothesis confirmed and exceeded — EER dropped 7.83 % (17.92 → 10.09 %),
nearly double the expected 3–5 %. Equally important: std collapsed from 7.81
to 1.81 % — the model is now much more consistent across sessions. Deltas
capture vocal tract dynamics that are stable across recording conditions,
which explains both the accuracy gain and the variance reduction.

Threshold drifted further negative (−2.204), confirming scores are still
uncalibrated. Will need Platt calibration before fusion.

## Next step

- E003: GMM-UBM + MAP adaptation — flagship audio system
- Keep MFCC 13 + Δ + ΔΔ as the feature set going forward (clearly wins over static)
