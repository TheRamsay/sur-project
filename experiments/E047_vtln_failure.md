# E047 — VTLN (Vocal Tract Length Normalization) — CATASTROPHIC FAILURE

- **Date:** 2026-04-22
- **Author:** TheRamsay
- **Related:** E042 (tied cov + speed TTA, 0.46% EER)

## Hypothesis

VTLN warping factors will improve speaker normalization and reduce EER.

## Result

| Alpha | EER | vs E042 |
|-------|-----|---------|
| 0.90 | 31.45% | +31pp ❌ |
| 0.95 | 31.92% | +31pp ❌ |
| 1.00 | 31.72% | +31pp ❌ |
| 1.05 | 32.40% | +32pp ❌ |
| 1.10 | 33.32% | +33pp ❌ |

**ALL configurations catastrophic!** E042 baseline (α=1.0) should be 0.46%, not 31%.

## Root Cause

**Broken VTLN implementation.** The `_apply_vtln()` function warps cepstral coefficients directly with exponential scaling, which destroys the feature structure. 

Real VTLN requires:
1. Warping the mel filterbank frequencies BEFORE computing MFCC
2. Or time-domain resampling + inverse resampling
3. Not post-hoc cepstral coefficient scaling

The implementation was fundamentally wrong - cepstral coefficients cannot be warped this way.

## Decision

**REJECTED.** VTLN implementation requires complete rewrite at feature extraction stage (warping mel filters, not cepstral coefficients). Not worth the complexity given:

- E042 already achieves 0.46% EER
- Speed perturbation (already in E042) provides similar speaker variability coverage
- VTLN adds significant implementation complexity

**Keep E042 as audio flagship.**
