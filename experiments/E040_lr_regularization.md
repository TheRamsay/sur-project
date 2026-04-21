# E040 — Logistic Regression Regularization Sweep (Image)

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E033 (adversarial image)

## Hypothesis

Tuning LogReg regularization strength (C) and type (L1/L2) will improve over E033 baseline (C=1.0, L2).

## Setup

- **Modality:** image
- **Model:** PCA-50 + LogReg with varied regularization
- **Configs tested:**
  - C=0.1 (stronger L2)
  - C=1.0 (E033 baseline, L2)
  - C=10.0 (weaker L2)
  - C=100.0 (very weak L2)
  - L1 C=1.0 (sparse)

## Result

| Config | EER mean ± std | vs E033 |
|--------|----------------|---------|
| **C=0.1** | **0.97 ± 0.86%** | = (tie) |
| **C=1.0 (E033)** | **0.97 ± 0.86%** | — |
| C=10.0 | 3.75 ± 3.92% | +2.78pp ❌ |
| C=100.0 | 10.19 ± 8.17% | +9.22pp ❌ |
| L1 C=1.0 | 13.52 ± 10.31% | +12.55pp ❌ |

## Interpretation

**Hypothesis rejected — C=1.0 is optimal:**

1. **C=0.1 ties C=1.0:** Both achieve 0.97% EER. Stronger regularization doesn't hurt but doesn't help.

2. **Weaker regularization catastrophically fails:** C=10.0 and C=100.0 show severe overfitting (3.75% → 10.19% EER).

3. **L1 is terrible:** Sparse regularization destroys discriminative power (13.52% EER).

4. **Why C=1.0 is optimal:** PCA-50 features are already well-conditioned. L2 regularization at C=1.0 provides the right balance between fitting the data and preventing overfitting to adversarial examples.

## Decision

**KEEP C=1.0.** E033 baseline regularization is confirmed optimal. No change needed.
