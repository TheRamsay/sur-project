# E045 — Score-Level Ensemble (MFCC+LPCC+PLP)

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E022 (MFCC+LPCC fusion), E021 (PLP)

## Hypothesis

Ensemble of 3 feature types (MFCC, LPCC, PLP) with calibrated score averaging will outperform best single system (E042: 0.46%).

## Setup

- **Modality:** audio
- **Features:**
  - MFCC 13+Δ+ΔΔ (39d)
  - LPCC 13+Δ+ΔΔ (39d, order=12)
  - PLP 13+Δ+ΔΔ (39d, Bark bands)
- **Fusion:** Platt calibration + equal weight averaging

## Result

| System | OOF EER |
|--------|---------|
| **LPCC (E042 baseline)** | **2.45%** |
| MFCC+LPCC | 3.23% |
| MFCC+LPCC+PLP | 3.23% |

Note: High EER due to OOF vs per-fold difference. Relative comparison valid.

## Interpretation

**Hypothesis rejected — ensemble hurts:**

1. **LPCC alone wins:** Single LPCC system (2.45%) beats all ensembles.

2. **Ensemble regresses:** Adding MFCC or PLP increases EER to 3.23%.

3. **Why ensemble fails:**
   - Calibration asymmetry: LPCC has different score distribution than MFCC/PLP
   - Equal weighting is suboptimal; grid search needed
   - PLP quality is poor (E021: 5.56% vs LPCC 1.67%)
   - Adding weak systems drags down strong systems

4. **E022 repeat:** Same pathology as E022 — MFCC+LPCC fusion collapsed to weaker system.

## Decision

**REJECTED.** Score-level ensemble not adopted. LPCC alone (E042) remains optimal.

**Lesson:** More features ≠ better. LPCC with tied covariance + speedTTA is the winning combination.
