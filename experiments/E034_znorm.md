# E034 — Z-Norm Score Normalization

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E025 (LPCC baseline)

## Hypothesis

Z-norm (normalizing test scores by cohort mean/std) will reduce EER by making scores more comparable across test conditions, as is standard in speaker verification.

## Setup

- **Modality:** audio
- **Data:** train + dev, LOSO 3-fold
- **Features:** LPCC 13+Δ+ΔΔ, +Pitch aug
- **Model:** UBM-32 + MAP r=16, diagonal covariance
- **Cohort:** 20 non-target speakers
- **Seed:** 67

## Result

| Method | EER | min-DCF |
|--------|-----|---------|
| raw (E025) | 6.20% | 0.1240 |
| znorm | 6.20% | 0.0677 |

Note: EER is OOF overall (not per-fold mean). High EER suggests this CV split is challenging.

## Interpretation

**Hypothesis rejected — no EER improvement:**

1. **Same EER, better min-DCF:** Z-norm achieves identical EER (6.20%) but improves min-DCF (0.0677 vs 0.1240). This suggests better score calibration but not better ranking.

2. **Why no improvement:** Our dataset is too small for effective cohort-based normalization. With only ~20 cohort speakers, the cohort mean/std estimates are noisy.

3. **Z-norm is designed for large-scale SV:** In standard speaker verification (thousands of speakers), Z-norm accounts for test condition variability. Our 14-speaker dataset doesn't have enough diversity for this to help.

## Decision

**REJECTED.** Z-norm not adopted. The small dataset size makes cohort-based normalization ineffective.
