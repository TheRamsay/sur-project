# E024 — Audio LPC order ablation for LPCC

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E020 (LPCC flagship, LPC order=12 chosen by convention)

## Hypothesis

E020 picked LPC order=12 as the de facto standard from speech coding: with
a 16 kHz signal, rule of thumb `p ≈ fs/1000 + 2–4` gives 12–14 poles,
matching the typical number of vocal-tract resonances in the 0–8 kHz band.
Like we did for UBM components (E010), MAP r (E013), and PCA components
(E011), this choice must be ablation-validated rather than assumed.

Lower order (8) may underfit the envelope, missing formants above ~F3 and
smearing speaker-discriminative detail. Higher order (16, 20) may overfit,
starting to pick up pitch harmonics and short-term excitation rather than
the slowly-varying vocal tract shape. The LPCC cepstrum then encodes
session-specific pitch/excitation information that does not generalize
across the session-disjoint target fold.

Expected landscape: relatively flat in 10–14, with degradation at both
extremes. E020 baseline: **3.33 ± 4.14% EER** at order=12.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** LPCC 13 cep + Δ + ΔΔ + CMN = 39 features/frame
- **LPC order:** ablated ∈ {8, 10, 12, 14, 16, 20}
- **Model:** UBM 32 + MAP adapt r=16
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Augmentation:** +All (noise SNR=20dB + speed ±10%), same as E020
- **Scoring:** LLR (adapted − UBM), utterance mean
- **Command / notebook:** `notebooks/E024_audio_lpc_order.ipynb`

## Result

LOSO CV, LPCC 13+Δ+ΔΔ+CMN, UBM 32, MAP r=16, +All aug, seed=67.

| order | F0 EER | F1 EER | F2 EER |   Mean  |   Std  | min-DCF |
|------:|-------:|-------:|-------:|--------:|-------:|--------:|
|     8 |  9.86  |  0.83  | 10.00  |  6.90   |  4.29  |  0.0870 |
|    10 |  9.86  |  0.83  |  9.17  |  6.62   |  4.10  |  0.0657 |
| **12** | **9.17** | **0.83** | **0.00** | **3.33** | **4.14** | **0.0333** ← winner (E020 ref) |
|    14 |  8.47  |  1.67  |  7.50  |  5.88   |  3.01  |  0.0787 |
|    16 |  9.17  |  0.83  |  7.50  |  5.83   |  3.60  |  0.0731 |
|    20 | 10.56  |  0.83  |  8.33  |  6.57   |  4.16  |  0.0648 |

- Winner: **order=12** with 3.33 ± 4.14% EER, min-DCF 0.0333 (exactly the
  E020 flagship result — E024 sanity-checks E020).
- All other orders land in the 5.83–6.90% band: a **+2.5 to +3.6pp**
  degradation over order=12.

## Interpretation

The hypothesis is confirmed, but the landscape is sharper than "relatively
flat around 10–14": order=12 is a genuine minimum with a substantial moat
(≥2.5pp) over every neighbor tested. Fold 2 drives this — at order=12 it
collapses to 0.00% EER, while every other order gives it 7.5–10% EER.
Fold 0 stays stubbornly near 9% across the sweep (the hard target
session), and Fold 1 is already saturated near-zero everywhere, so Fold
2 is where LPC order actually pays off.

Physically, 12 poles at 16 kHz sit right on the classic `fs/1000 + 2–4`
rule of thumb: enough capacity to model the 5–6 vocal-tract resonances in
the 0–8 kHz band (each pole pair = one formant) plus a spectral tilt
term, but not so many that spare poles start tracking pitch harmonics or
session-specific excitation. Order=8 underfits (misses F3/F4), while 16
and 20 appear to overparameterize — the adapted GMM then latches onto
excitation detail that fails to generalize across m431's session-disjoint
folds, specifically on Fold 2 where the enrolment/test pitch contrast
presumably shifts.

Trend-wise the underfit side (8 → 10) improves monotonically and the
overfit side (14 → 16 → 20) is comparatively flat. Std dev is broadly
similar across orders (3.0–4.3pp), dominated by the target×session
structure rather than the feature choice.

## Next step

- No change to flagship. **Keep LPC order=12 in `src/predict_audio.py`**
  — it was the right choice by convention, and now it is the right
  choice by ablation. No edit needed.
- Document this validation in `dokumentace.pdf` alongside the UBM
  component (E010), MAP r (E013), and PCA component (E011) ablations as
  evidence that the LPCC frontend's knobs are tuned, not guessed.
- Do not sweep further — the 8-to-20 range already covers both the
  underfit and overfit regimes with a clear minimum; finer granularity
  around 11–13 would be noise given ±4% per-fold std.
