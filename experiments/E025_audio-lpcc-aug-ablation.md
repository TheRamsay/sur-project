# E025 — Audio LPCC augmentation ablation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E014 (MFCC aug ablation), E020 (LPCC flagship), E008 (MFCC +All aug baseline)

## Hypothesis

E020 used MFCC-optimal augmentation (noise SNR=20dB + speed ±10%) directly on
LPCC without verifying that those augmentations are still the optimum for an
all-pole feature front-end. LPCC is derived from an LPC all-pole fit of the
signal: additive white noise gets absorbed into the filter (shifts pole
positions, flattens the estimated vocal-tract spectrum), so LPCC may be more
sensitive to noise than MFCC where the mel smoothing hides small spectral
perturbations. If this is true, a milder noise level (SNR=30dB) or no noise
at all may beat the E020 configuration. Pitch shifting directly alters formant
positions — the exact quantity LPCC encodes — so pitch is expected to hurt
LPCC at least as much as it hurt MFCC in E014 (+Pitch: 5.97% vs 4.21%).
Speed perturbation preserves formant structure (time-stretch without pitch)
and should remain useful.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** LPCC 13 + Δ + ΔΔ + CMN = 39 features/frame, LPC order=12
- **Model:** UBM 32 + MAP adapt r=16 (diag covariance, seed=67)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO)
- **Seed:** 67
- **Command / notebook:** `notebooks/E025_audio_lpcc_aug_ablation.ipynb`
- **Augmentation configs:**
  - `Baseline`      — original samples only
  - `+Noise`        — original + noise SNR=20dB
  - `+Speed`        — original + time-stretch rate∈[0.9,1.1]
  - `+NoiseSpeed`   — E020 (original + noise20 + speed)
  - `+Pitch`        — original + pitch shift ±{1,2} semitones
  - `+HighSNR`      — original + noise SNR=30dB (milder)
  - `+AllMFCCOpt`   — original + noise20 + speed (== `+NoiseSpeed`, kept for parity with E014)

## Result

| Config            | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std       | min-DCF mean |
| ----------------- | ---------- | ---------- | ---------- | ---------------- | ------------ |
| E020 ref (+NoiseSpeed) | 9.17  | 0.83       | 0.00       | 3.33 ± 4.14      | 0.0333       |
| Baseline          | 9.86       | 0.83       | 0.00       | 3.56 ± 4.47      | 0.0380       |
| +Noise            | 10.56      | 0.83       | 1.67       | 4.35 ± 4.40      | 0.0537       |
| +Speed            | 9.17       | 0.83       | 0.83       | 3.61 ± 3.93      | 0.0389       |
| +NoiseSpeed       | 9.17       | 0.83       | 0.00       | 3.33 ± 4.14      | 0.0333       |
| **+Pitch ★**      | **4.17**   | **0.83**   | **0.83**   | **1.94 ± 1.57**  | 0.0389       |
| +HighSNR          | 10.56      | 0.83       | 0.00       | 3.80 ± 4.79      | 0.0426       |
| +AllMFCCOpt       | 9.17       | 0.83       | 0.00       | 3.33 ± 4.14      | 0.0333       |

Winner: **+Pitch** — −1.39pp on mean EER vs E020, std drops 4.14 → 1.57 (a 2.6×
reduction in fold variance). min-DCF regresses slightly (0.0389 vs 0.0333).

## Interpretation

**The hypothesis was wrong.** I predicted pitch shift would hurt LPCC at least
as much as it hurt MFCC (E014: +Pitch regressed to 5.97% from 4.21%), since LPCC
encodes formant positions more directly. In fact **+Pitch is the only augmentation
that substantially improves LPCC** — and the improvement comes almost entirely
from Fold 0, the previously pathological fold.

Fold 0 behaviour across configs is the key story:
- Baseline / +Noise / +HighSNR: Fold 0 EER = 9.86–10.56% (noise alone does not help
  and +Noise slightly hurts — consistent with the "noise gets modelled into the
  all-pole filter" concern).
- +Speed / +NoiseSpeed / +AllMFCCOpt: Fold 0 stays at 9.17% — speed perturbation
  does not touch the spectral envelope shape the LPC filter captures, so the UBM
  does not learn to be robust to cross-session formant drift.
- **+Pitch: Fold 0 EER collapses to 4.17%.** Pitch shift ±{1,2} semitones moves
  the whole spectrum up/down in frequency, which shifts every formant — forcing
  the UBM and the MAP-adapted speaker model to learn a distribution over
  formant-position drift. Session 01 (held out in fold 0) evidently sits at a
  formant offset the base sessions 02+03 do not cover; pitch-shifted augmentation
  plants training samples at nearby formant offsets, and the speaker model now
  generalises to it.

This is a modality-specific effect. MFCCs partially abstract away absolute formant
positions through mel warping + DCT truncation, so MFCC already generalises across
small formant shifts and does not benefit from pitch augmentation (E014 +Pitch:
11.25% fold 0, regression). LPCC retains the raw all-pole envelope — formant
positions are encoded directly, generalisation across them is poor, and pitch-shift
augmentation is exactly the right inductive bias.

Other observations:
- **+Noise regresses** (4.35 vs 3.33 for +NoiseSpeed) relative to the combined
  config — the hypothesis about noise being absorbed into the LPC filter is
  consistent with this: noise alone hurts Fold 0 (10.56 vs 9.17) and Fold 2
  (1.67 vs 0.00). But noise combined with speed recovers the loss, presumably
  because the tripled training set compensates for the filter-pollution cost.
- **+HighSNR (30dB)** is not better than **+Noise (20dB)**; same Fold 0 collapse
  (10.56%). Milder noise does not save LPCC from the filter-absorption problem.
- **+Speed alone** tracks +NoiseSpeed on Fold 0 (both 9.17%) — speed contributes
  nothing new relative to the baseline on that fold. Its role is supportive,
  not driver.
- **+AllMFCCOpt == +NoiseSpeed** mechanically (both are original + noise20 +
  speed) — identical numbers confirm determinism given same seeds.

min-DCF picture: +NoiseSpeed still wins on min-DCF (0.0333 vs +Pitch's 0.0389).
Fold 2 in +Pitch regresses on perfect-0% fold to 0.83% EER, nudging min-DCF up.
Choice for submission depends on whether mean EER or min-DCF is the scoring
target — the assignment specifies hard decisions at prior 0.5 (= min-DCF
equivalent), so the trade-off is non-trivial.

## Next step

- **Update `predict_audio.py`** to use the +Pitch augmentation (pitch shift
  ±{1,2} semitones on top of the original, no noise, no speed) if the production
  system is optimised for mean EER / variance reduction.
  - If min-DCF at prior 0.5 is the deciding metric, keep E020 +NoiseSpeed.
  - Compromise option to investigate: try +Pitch + speed (pitch gets the
    formant-shift benefit, speed adds temporal diversity) — this was not in
    the ablation.
- Revisit **E023 fusion (LPCC + image)** with +Pitch LPCC scores: a 2.6× std
  reduction on the audio component may propagate to a lower-variance fused
  system. Current fusion OOF = 0.52%; likely already at a floor, but worth
  checking.
- Run a follow-up ablation combining **+Pitch + +Speed** and **+Pitch + +Noise**
  to confirm +Pitch's gain survives when mixed with other axes.
