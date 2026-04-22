# E035 — Feature-level fusion: MFCC+LPCC concatenation

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E022 (score-level MFCC+LPCC fusion), E027 (trimodal score fusion)

## Hypothesis

Concatenating MFCC and LPCC frame-level features (78d = 39+39) before UBM-MAP training will capture complementary spectral representations in a single model, potentially outperforming score-level fusion (E022) which collapsed to LPCC alone due to calibration asymmetry.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), LOSO 3-fold
- **Features:** MFCC 13+Δ+ΔΔ (39d) + LPCC 13+Δ+ΔΔ (39d) concatenated → 78d
- **Model:** UBM-32 + MAP r=16, diagonal covariance
- **Seed:** 67

## Result

**FAILED — implementation bug.**

MFCC and LPCC feature extractors produce different frame counts per utterance:
- MFCC uses librosa's default hop/window settings
- LPCC uses custom framing (hop=160, win=400)

At concatenation time, the frame arrays have mismatched lengths and cannot be hstacked.

## Interpretation

The bug is fixable (align frame counts by using the same hop/win for both), but given that score-level fusion (E022) already collapsed to LPCC alone and E037 tied covariance solved the audio performance gap, feature-level fusion is no longer worth pursuing. The fundamental issue (MFCC and LPCC carry largely redundant formant information at the feature level) would likely persist even with a correct implementation.

## Decision

**Not pursued.** Score-level fusion via E039 trimodal (0.26% OOF, 0 errors) is the adopted fusion strategy.
