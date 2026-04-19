# E022 — Audio MFCC+LPCC Score-level Fusion

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (MFCC audio flagship), E020 (LPCC audio), E009 (image+audio fusion)

## Hypothesis

MFCC (E008) and LPCC (E020) show strongly complementary per-fold errors:
- Fold 0: MFCC wins (3.47% vs 9.17% LPCC)
- Fold 1: LPCC wins (0.83% vs 8.33% MFCC)
- Fold 2: Both strong (0.83% vs 0.00% LPCC)

MFCC uses the mel-warped spectral envelope; LPCC uses the all-pole vocal tract
model without perceptual compression. The different representations capture
complementary aspects of the speaker's vocal tract — when one fails due to
channel/session mismatch, the other may remain stable. Score-level fusion with
Platt calibration should combine their strengths and suppress the per-fold
variance. Expected mean EER in the 1.5–2.5% range, well below either alone
(MFCC 4.21%, LPCC 3.33%).

## Setup

- **Modality:** audio (fusion of two audio subsystems)
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:**
  - MFCC: 13+Δ+ΔΔ = 39d/frame, CMN (same as E008)
  - LPCC: 13+Δ+ΔΔ = 39d/frame, LPC order=12, cepstrum via FFT, CMN (same as E020)
- **Model:** UBM 32 + MAP adapt r=16 (separately trained per feature type)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO on target sessions)
- **Seed:** 67
- **Augmentation:** +All (noise SNR=20dB + speed ±10%) on train fold only
- **Calibration:** Platt (LogisticRegression C=1e6, class_weight='balanced') on OOF separately per system
- **Fusion:** grid search w ∈ np.linspace(0, 1, 101), score = w·mfcc_cal + (1−w)·lpcc_cal, minimize EER
- **Command / notebook:** `notebooks/E022_audio_mfcc_lpcc_fusion.ipynb`

## Result

| System | F0 EER | F1 EER | F2 EER | Mean ± std | min-DCF (mean) |
| ------ | ------ | ------ | ------ | ---------- | -------------- |
| MFCC alone (E008) | 3.47 | 8.33 | 0.83 | 4.21 ± 3.11 | 0.0509 |
| LPCC alone (E020) | 9.17 | 0.83 | 0.00 | 3.33 ± 4.14 | 0.0333 |
| MFCC+LPCC fusion  | 9.17 | 0.83 | 0.00 | **3.33 ± 4.14** | **0.0333** |

Optimal MFCC weight: **w = 0.07** (MFCC 7%, LPCC 93%)

OOF overall — fusion EER = 5.94%, min-DCF = 0.1187, threshold = 0.907

Pre-calibration OOF EER: MFCC 9.17% (thr=0.028), LPCC 6.46% (thr=0.221)

Platt params: MFCC slope=5.55 intercept=0.02 vs LPCC slope=10.12 intercept=−1.50

## Interpretation

The hypothesis did not hold. The expected synergy (fold 0 MFCC wins, fold 1 LPCC
wins → fusion averages them) did not materialise because Platt calibration on the
global OOF pool introduces an asymmetric scaling problem. LPCC's Platt slope is
~10.1 vs MFCC's ~5.6 — nearly 2× larger — so calibrated LPCC scores have roughly
twice the dynamic range of calibrated MFCC scores. The grid search at OOF level
therefore assigns 93% weight to LPCC (w=0.07), and the "fused" result is
indistinguishable from LPCC alone. Per-fold mean EER: 3.33 ± 4.14% — identical
to LPCC.

Root cause: both systems are calibrated on the same OOF pool that includes the
samples they were scored on. The OOF EERs (MFCC 9.17%, LPCC 6.46%) differ from
the per-fold means (4.21%, 3.33%) because OOF overall mixes three different
models; when fold 0's poor LPCC scores and fold 1's poor MFCC scores both appear
in the same pool, the calibrator weights accordingly. The complementarity that
exists at the per-fold level is invisible to the global calibration.

The fundamental issue is that score-level fusion of two audio systems trained
on the same data structure requires fold-aware calibration (calibrate within
each fold, then fuse), not global OOF calibration. This was correctly handled in
E009 (audio+image) by independently calibrating and fusing, but the within-fold
calibration was implicitly more stable there because the two modalities have
very different score ranges that don't interact as strongly.

**E020 LPCC remains the audio flagship** (3.33 ± 4.14%, min-DCF 0.0333).
Fusion at score level does not improve over LPCC alone in this configuration.

## Next step

- If further audio improvement is wanted, try fold-aware Platt calibration:
  calibrate MFCC and LPCC separately within each fold using a held-out inner
  split, then fuse — this avoids the global calibration asymmetry.
- Alternative: feature-level fusion (MFCC + LPCC concatenated = 78d) with
  UBM 32 — the higher dimensionality may hurt (cf. E016 120d), but worth one
  experiment.
- Priority: use E020 LPCC as the audio input to the image+audio fusion (replacing
  E008 in E009-style experiment) and check if LPCC gives better multimodal results.
