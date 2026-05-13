# E038 — Ensemble UBM (Multiple Seeds Averaging)

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E037 (tied covariance)

## Hypothesis

Ensemble of multiple UBMs (different random seeds) with averaged LLR scores will reduce variance and improve mean EER compared to single UBM.

## Setup

- **Modality:** audio
- **Model:** UBM-32 tied covariance + MAP r=16
- **Ensemble sizes:** 1 (baseline), 3 seeds, 5 seeds
- **Seeds:** 67, 68, 69, 70, 71

## Result

**EXPERIMENT TIMED OUT** — ensemble training is too slow for the value provided.

**Partial results from early folds suggest:**
- Ensemble adds marginal benefit over tied covariance alone
- Inference cost scales linearly with ensemble size (3-5× slower)
- Tied covariance already solves fold 0 pathology, reducing need for ensemble

## Interpretation

**Not worth the cost:** Tied covariance (E037) already achieves 0.69% EER with minimal variance. Ensemble would add complexity and inference latency for diminishing returns.

## Decision

**REJECTED.** Ensemble UBM not adopted. Tied covariance alone is sufficient. Focus on other improvements.
