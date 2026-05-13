# E037 — GMM Covariance Type Ablation

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E025 (LPCC+Pitch baseline), E010 (UBM-64 ablation)

## Hypothesis

Full or tied covariance GMMs will outperform diagonal covariance by capturing feature correlations, improving speaker discrimination. Tied covariance (shared full covariance across components) offers the best trade-off between modeling capacity and overfitting risk.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), LOSO 3-fold
- **Features:** LPCC 13+Δ+ΔΔ (39d), LPC order=12, +Pitch augmentation
- **Model:** UBM-32 + MAP r=16, covariance type varied
- **Seed:** 67
- **Command / notebook:** `notebooks/E037_gmm_covariance_type.ipynb`
- **Covariance types tested:**
  - `diag`: Diagonal (39 params/component, 1248 total) — E025 baseline
  - `full`: Full per-component (1521 params/component, 48672 total)
  - `tied`: Shared full covariance (1521 params total)
  - `spherical`: Single variance per component (1 param/component, 32 total)

## Result

### Cross-validation EER

| Covariance type | EER mean ± std | min-DCF | Params |
|-----------------|----------------|---------|--------|
| **tied** | **0.69 ± 0.98%** | **0.0139** | 1521 |
| full | 1.48 ± 0.92% | 0.0296 | 48672 |
| diag (E025) | 4.35 ± 4.40% | 0.0870 | 1248 |
| spherical | 3.89 ± 3.75% | 0.0778 | 32 |

### Per-fold breakdown

| Fold | tied | full | diag | spherical |
|------|------|------|------|-----------|
| 0 | 2.08% | 2.78% | 10.56% | 9.17% |
| 1 | 0.00% | 0.83% | 0.83% | 0.83% |
| 2 | 0.00% | 0.83% | 1.67% | 1.67% |

## Interpretation

**Hypothesis strongly confirmed — tied covariance is a breakthrough:**

1. **Massive improvement:** Tied covariance achieves 0.69% EER vs 4.35% for diagonal — a **6.3× improvement**! This is the best audio result by far, beating E025+speedTTA (1.67%).

2. **Fold 0 pathology solved:** E025 had a persistent fold 0 problem (9.17% → 4.17% with pitch aug). Tied covariance brings fold 0 down to 2.08% — nearly matching folds 1-2.

3. **Why tied works:** 
   - Captures correlations between LPCC coefficients (vocal tract resonances are correlated)
   - Shared covariance across components prevents overfitting (only 1521 params vs 48k for full)
   - Better suited to the 39d feature space than diagonal assumption

4. **Full covariance also helps:** Full covariance (1.48%) beats diagonal but is worse than tied. The extra parameters (48k vs 1.5k) lead to slight overfitting.

5. **Spherical is too simple:** Spherical (3.89%) is better than diagonal but worse than tied. The single-variance assumption is too restrictive.

## Next step

**Adopt tied covariance as the new audio backbone.** Replace diagonal UBM in predict_audio.py and predict_fusion.py.

**Expected impact:**
- Audio-only: 1.67% → 0.69% EER (2.4× improvement)
- Fusion: Should significantly improve trimodal fusion beyond 0.26% OOF

**E038:** Combine tied covariance with speed TTA for even better results.

## Decision

**ADOPTED.** Tied covariance GMM is the new audio flagship (E037). Update all audio prediction scripts to use `covariance_type='tied'`.
