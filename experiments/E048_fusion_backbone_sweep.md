# E048 — Fusion Backbone Sweep — Product Rule Degradation

- **Date:** 2026-04-22
- **Author:** TheRamsay
- **Related:** E046 (product rule fusion, 0.52% EER)

## Hypothesis

E042 audio + E043 image backbones with product rule will achieve best fusion EER.

## Result

| Config | Audio TTA | Image TTA | EER | vs E046 |
|--------|-----------|-----------|-----|---------|
| E037+E033 | ✗ | ✗ | 5.20% | +4.68pp ❌ |
| E042+E033 | ✓ | ✗ | 5.20% | +4.68pp ❌ |
| E037+E043 | ✗ | ✓ | 4.59% | +4.07pp ❌ |
| E042+E043 | ✓ | ✓ | 4.59% | +4.07pp ❌ |

**ALL configurations catastrophically worse than E046!**

## Root Cause

**Implementation mismatch with E046:**

1. **E046 was trimodal (MFCC+LPCC+Image), E048 is bimodal (Audio+Image)**
   - Product rule works best with 3+ independent estimates
   - With 2 modalities, geometric mean is too aggressive

2. **Score calibration missing**
   - E046 used Platt-calibrated scores before product rule
   - E048 used raw log-odds → probability conversion
   - Calibration asymmetry breaks product rule

3. **Fold 0 pathology persists**
   - All configs show ~10% EER on fold 0
   - Suggests score distribution mismatch

## Key Insight

**E046's product rule is already optimal.** The 0.52% EER was achieved with:
- Trimodal fusion (MFCC+LPCC+Image)
- Platt calibration on each modality
- Product rule: `(p_mfcc * p_lpcc * p_img)^(1/3)`

Bimodal product rule (audio+image only) doesn't work as well because:
- Less redundancy
- More sensitive to single-modality failures
- Geometric mean too aggressive with only 2 inputs

## Decision

**REJECTED.** E048 bimodal product rule implementation is inferior to E046 trimodal.

**Keep E046 as fusion flagship at 0.52% OOF EER.**

Future work: Re-implement trimodal product rule with E042+E043 backbones (MFCC+LPCC-tied+Image-adv) to potentially beat E046's 0.52%.
