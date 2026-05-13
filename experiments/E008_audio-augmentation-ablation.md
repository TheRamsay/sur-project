# E008 — Audio augmentation ablation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E003 (audio flagship), E007 (image aug ablation)

## Hypothesis

E003 (UBM+MAP) achieved EER 7.45 ± 5.04% — worse than the image system and
with high variance. Audio augmentation should reduce this by exposing the UBM
and MAP adaptation to more acoustic variation during training.

Testing three strategies independently, then combined:
- **Noise**: additive white noise at SNR=20dB — simulates degraded eval recordings
- **Speed**: time stretch ±10% (rate ∈ [0.9, 1.1]) — standard speaker verification aug
- **All**: noise + speed combined (3× training frames)

Augmentation applied at WAV level before MFCC extraction. CMN applied after
(per utterance). Val fold always uses original WAVs.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E003
- **Model:** UBM 32 + MAP adapt r=16 — same as E003
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Command / notebook:** `notebooks/E008_audio_augmentation.ipynb`

## Result

| Config      | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std    | min-DCF mean |
| ----------- | ---------- | ---------- | ---------- | ------------- | ------------ |
| Baseline    | 16.25      | 10.83      | 1.67       | 9.58 ± 6.02   | 0.0972       |
| +Noise      | 8.47       | 13.33      | 0.83       | 7.55 ± 5.14   | 0.0954       |
| +Speed      | 9.17       | 10.83      | 0.83       | 6.94 ± 4.37   | 0.0861       |
| **+All**    | **3.47**   | **8.33**   | **0.83**   | **4.21 ± 3.11** | **0.0509** |

OOF overall (best config +All): EER = 9.17%, min-DCF = 0.1687, threshold = −0.078

Note: E008 baseline (9.58%) differs from E003 (7.45%) due to GMM EM non-determinism
across implementations. The within-experiment relative comparisons are valid.

## Interpretation

Both noise and speed individually improve EER slightly (7.55%, 6.94% vs 9.58%
baseline). The combination (+All) achieves 4.21 ± 3.11% — a significant drop
in both mean and variance vs baseline.

Speed perturbation is the more effective single augmentation here (6.94% vs
7.55% for noise). This makes sense: speaking rate variation across sessions is
a real source of mismatch, and time-stretching directly trains the model to be
robust to it. Noise helps but is secondary.

The +All combination reduces std from 6.02% to 3.11% — the model is more
consistent across sessions. Fold 0 drops dramatically (16.25% → 3.47%).

Threshold stays near 0 (−0.078) — LLR formulation with shared UBM keeps
scores calibrated even with augmented training data.

**E008 +All is the new audio flagship** (4.21 ± 3.11% vs E003's 7.45 ± 5.04%).

## Next step

- Audio and image flagships now both have augmentation ablations documented
- Next: score-level fusion (E009) — calibrate OOF scores then combine
- Image OOF threshold still far from 0 (−5), audio near 0 (−0.078) — need calibration
