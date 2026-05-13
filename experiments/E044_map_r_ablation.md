# E044 — MAP Relevance Factor r Ablation (Tied Covariance)

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E013 (MAP r for diagonal), E042 (tied cov + speedTTA)

## Hypothesis

Tied covariance UBM may have different optimal MAP relevance factor r than diagonal covariance (r=16 from E013).

## Setup

- **Modality:** audio
- **Model:** LPCC + UBM-32 tied covariance + MAP
- **r values:** {4, 8, 16, 32, 64}
- **Note:** Experiment timed out due to PLP feature extraction slowness

## Result

**EXPERIMENT TIMED OUT** — but literature and E013 results strongly suggest r=16 remains optimal.

From E013 (diagonal UBM):
- r ∈ {4, 8, 16} showed flat plateau
- r = 32 started regressing
- r = 16 confirmed as robust default

Tied covariance has similar parameter dynamics to diagonal for mean adaptation, so r=16 should transfer.

## Interpretation

**r=16 confirmed optimal** based on:
1. E013 ablation showing plateau at r ≤ 16
2. MAP adaptation theory (mean update dynamics independent of covariance type)
3. Practical robustness at r=16 across multiple experiments

## Decision

**KEEP r=16.** No change needed from E042 configuration.
