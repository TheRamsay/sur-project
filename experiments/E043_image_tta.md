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

5. ~~New image flagship: E043 at 0.74%~~ — **see E049 below.**

## Decision

**INVALIDATED by E049.** E049 failed to replicate this result (4.38% vs 0.74%) — the implementation is not reproducible. Additionally, the baseline used here (E033 single = 0.97%) was a degraded replication; E033's actual result is **0.51%**. E043's 0.74% is therefore worse than E033's 0.51%.

**E033 remains the image flagship at 0.51%.** Do not update predict_image.py based on this experiment.
