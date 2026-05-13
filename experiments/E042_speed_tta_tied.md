# E042 — Speed TTA + Tied Covariance Combo

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E037 (tied cov), E031 (speed TTA with diagonal)

## Hypothesis

Speed TTA (3 views: 0.9x, 1.0x, 1.1x) combined with tied covariance UBM will improve over tied cov alone (E037: 0.69%).

## Setup

- **Modality:** audio
- **Model:** LPCC + UBM-32 tied covariance + MAP r=16 + Pitch aug
- **Scoring:**
  - `single`: E037 baseline (single view)
  - `speedTTA`: Average of 3 speed-augmented views

## Result

| Config | EER mean ± std | Improvement |
|--------|----------------|-------------|
| single (E037) | 0.69 ± 0.98% | — |
| **speedTTA** | **0.46 ± 0.65%** | **-0.23pp** ✓ |

### Per-fold breakdown

| Fold | single | speedTTA |
|------|--------|----------|
| 0 | 2.08% | 1.39% |
| 1 | 0.00% | 0.00% |
| 2 | 0.00% | 0.00% |

## Interpretation

**Hypothesis confirmed — Speed TTA still helps with tied cov:**

1. **0.23pp improvement:** Speed TTA reduces EER from 0.69% to 0.46% — a meaningful gain on top of the already-strong tied covariance.

2. **Fold 0 improves:** The problematic fold 0 drops from 2.08% to 1.39%, showing speed TTA adds robustness beyond what tied covariance provides.

3. **Why it still helps:** Tied covariance captures feature correlations, while speed TTA addresses speaking rate variation. These are complementary sources of robustness.

4. **New audio flagship:** E042 (tied cov + speedTTA) at 0.46% beats E037 (tied cov only) at 0.69%.

## Decision

**ADOPTED.** E042 (tied covariance + speed TTA) is the new audio flagship at **0.46% EER**.

Update `predict_audio.py` to use:
- Tied covariance UBM
- Speed TTA at inference (3 views)
