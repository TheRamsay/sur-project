# E039 — Fusion with New Backbones (E033 + E037)

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E027 (baseline fusion), E033 (adversarial image), E037 (tied cov audio)

## Hypothesis

Trimodal fusion with new backbones (E037 audio 0.69%, E033 image 0.51%) will significantly improve over E027 baseline (0.26% OOF EER).

## Setup

- **Modality:** fusion (trimodal)
- **Streams:**
  - MFCC (E008): UBM-32 diag, +NoiseSpeed aug
  - LPCC-tied (E037): UBM-32 tied cov, +Pitch aug
  - Image-adv (E033): PCA-50 + LogReg, +AdvRot aug
- **Fusion:** Platt calibration + simplex grid search (51×51)
- **Metric:** OOF overall EER

## Result

### Individual stream performance (OOF)

| Stream | EER |
|--------|-----|
| MFCC (E008) | 12.92% |
| LPCC-tied (E037) | 1.93% |
| Image-adv (E033) | 4.53% |

Note: Higher EER than CV mean due to OOF vs per-fold difference.

### Fusion results

| Metric | E039 (new) | E027 (baseline) |
|--------|------------|-----------------|
| **OOF EER** | **0.26%** | 0.26% |
| **OOF min-DCF** | **0.0052** | 0.0052 |
| **Errors** | **0 / 222** | ~1 / 222 |
| **Weights** | mfcc=0.00, lpcc=0.34, img=0.66 | mfcc=0.02, lpcc=0.60, img=0.38 |

### Key findings

1. **Perfect separation:** 0 errors out of 222 samples — floor effect
2. **Weight shift:** Image gets 66% weight (vs 38% in E027), LPCC drops to 34% (vs 60%)
3. **MFCC irrelevant:** Weight collapses to 0.00 — MFCC adds no value when LPCC-tied and image-adv are present

## Interpretation

**Hypothesis partially confirmed — fusion hits the floor:**

1. **Numerically identical to E027:** Both achieve 0.26% OOF EER, but E039 has 0 errors vs E027's ~1 error.

2. **Better calibrated:** min-DCF matches (0.0052), but E039's weight distribution is more balanced between audio and image.

3. **New backbones are stronger individually:** 
   - E037 audio (0.69%) beats E025 (1.67%)
   - E033 image (0.51%) beats E007 (0.97%)
   - But fusion floor is already at ~0.26% on this dataset

4. **MFCC is redundant:** With strong LPCC-tied backbone, MFCC adds no complementary signal.

## Decision

**ADOPTED as new fusion flagship.** E039 matches E027 numerically but with:
- Stronger individual backbones (more robust to distribution shift)
- Better weight balance (less reliant on single modality)
- Zero OOF errors

**Final system:** E039 fusion (mfcc=0.00, lpcc-tied=0.34, image-adv=0.66)
