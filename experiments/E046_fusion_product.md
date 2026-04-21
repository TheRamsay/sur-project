# E046 — Fusion: Product Rule vs Weighted Sum

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E039 (weighted sum fusion)

## Hypothesis

Product rule (geometric mean of probabilities) will outperform weighted sum fusion for trimodal fusion.

## Setup

- **Modality:** fusion (trimodal)
- **Streams:** MFCC, LPCC-tied, Image-adv (same as E039)
- **Fusion rules:**
  - `weighted_sum`: E039 baseline (w_mfcc=0.00, w_lpcc=0.34, w_img=0.66)
  - `product`: Geometric mean (p_mfcc * p_lpcc * p_img)^(1/3)
  - `sum`: Arithmetic mean (equal weights)

## Result

| Fusion Rule | OOF EER | vs E039 |
|-------------|---------|---------|
| weighted_sum (E039) | 2.97% | — |
| **product** | **0.52%** | **-2.45pp** ✓✓✓ |
| sum | 0.78% | -2.19pp |

Note: E039's 0.26% was with different backbones. This uses E042+E043 backbones.

## Interpretation

**Hypothesis STRONGLY confirmed — Product rule DOMINATES:**

1. **Massive improvement:** Product rule (0.52%) vs weighted sum (2.97%) — a 2.45pp drop!

2. **Why product rule wins:**
   - Modalities are independent (audio vs image)
   - Product rule implements AND logic: all modalities must agree
   - More robust to single-modality failures
   - No weight tuning needed (weight-free!)

3. **Sum also good:** Simple averaging (0.78%) also beats weighted sum, but product is best.

4. **New fusion flagship:** E046 (product rule) at 0.52% OOF EER.

## Decision

**ADOPTED.** E046 (product rule fusion) is the NEW FUSION FLAGSHIP at **0.52% OOF EER**.

Update `predict_fusion.py` to use product rule instead of weighted sum!
