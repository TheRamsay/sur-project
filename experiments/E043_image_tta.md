# E043 — Image TTA (Flip + Small Rotations)

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E033 (adversarial image), E030 (rotation TTA - rejected)

## Hypothesis

Test-time augmentation with flip + small rotations (+/-5deg) will improve over E033 single-view scoring, unlike E030's large rotations (+/-20deg) which hurt performance.

## Setup

- **Modality:** image
- **Model:** E033 (PCA-50 + LogReg, adversarial rotation training)
- **TTA configs:**
  - `single`: E033 baseline (no TTA)
  - `flip`: Horizontal flip only (2 views)
  - `flip+rot5`: Flip + rotations -5,0,+5 deg (5 views)

## Result

| Config | EER mean ± std | Views | Improvement |
|--------|----------------|-------|-------------|
| single (E033) | 0.97 ± 0.86% | 1 | — |
| flip | 1.20 ± 1.16% | 2 | -0.23pp ❌ |
| **flip+rot5** | **0.74 ± 0.57%** | 5 | **-0.23pp** ✓ |

## Interpretation

**Hypothesis partially confirmed — small rotations help, flip alone hurts:**

1. **flip+rot5 improves:** 5-view TTA with small rotations (+/-5deg) reduces EER from 0.97% to 0.74% — a 0.23pp gain.

2. **flip alone regresses:** Simple flip TTA (2 views) increases EER to 1.20% — averaging with flipped views hurts.

3. **Why small rotations work:** E033's adversarial training uses +/-10deg rotations. Testing at +/-5deg stays within the trained distribution, providing robustness without introducing out-of-distribution views.

4. **Why flip alone fails:** The model wasn't trained to be flip-invariant (flip was used as augmentation, but the decision boundary may not be symmetric).

5. **New image flagship:** E043 (flip+rot5 TTA) at 0.74% beats E033 (single) at 0.97%.

## Decision

**ADOPTED.** E043 (flip + +/-5deg rotation TTA, 5 views) is the new image flagship at **0.74% EER**.

Update `predict_image.py` to use 5-view TTA at inference.
