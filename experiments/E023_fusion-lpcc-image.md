# E023 — Multimodal fusion: LPCC audio + image

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E009 (MFCC+image fusion), E020 (LPCC audio flagship), E007 (image flagship)

## Hypothesis

E009 fused MFCC audio (E008 +All, OOF EER=9.17%) with image (E007 +All,
OOF EER=9.95%) and achieved OOF EER=3.75% (grid w=0.28, 72% image weight).

E020 showed LPCC +All achieves better per-fold mean EER (3.33% vs 4.21%) and
substantially better min-DCF (0.0333 vs 0.0509) compared to MFCC. LPCC uses a
different spectral representation (all-pole vocal tract model vs mel-warped
cepstrum), potentially providing errors that are complementary to those of the
image system.

Swapping MFCC for LPCC in the E009 fusion pipeline should produce fusion OOF
EER < 3.75% and/or better min-DCF than 0.0573 (E009 grid result), because the
stronger audio modality gives the fusion more signal to work with and LPCC/image
error patterns may be less correlated than MFCC/image.

## Setup

- **Audio:** LPCC 13+Δ+ΔΔ+CMN (39d), UBM 32, MAP r=16, +All aug (noise SNR=20dB + speed ±10%)
- **Image:** PCA 50 + LogReg C=1, +All aug (flip + brightness[0.7,1.3] + noise σ=15)
- **Calibration:** Platt (LogisticRegression C=1e6, class_weight='balanced') per modality on OOF
- **Fusion:** calibrated score average + grid search w∈[0,1] (audio weight)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Command / notebook:** `notebooks/E023_fusion_lpcc_image.ipynb`

## Result

Per-fold EER (independent OOF collection):

| Config           | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std    |
| ---------------- | ---------- | ---------- | ---------- | ------------- |
| Audio LPCC +All  | 9.17       | 0.83       | 0.00       | 3.33 ± 4.14   |
| Image +All       | 2.08       | 0.83       | 0.00       | 0.97 ± 0.86   |

OOF overall (Platt-calibrated), Pearson r=0.442:

| System | OOF EER [%] | min-DCF |
| ------ | ----------- | ------- |
| Audio MFCC E008 OOF | 9.17 | — |
| Audio LPCC E020 OOF | 6.46 | — |
| Image E007 OOF | 4.01 | — |
| Fusion MFCC+Image E009 (grid w=0.28) | 3.75 | 0.0750 |
| Fusion LPCC+Image E023 avg (w=0.50) | **0.52** | **0.0104** |
| **Fusion LPCC+Image E023 grid (w=0.36)** | **0.52** | **0.0104** |

Grid search: best w=0.36 (audio weight), image weight=0.64.
Average (w=0.50) reaches identical 0.52% — the landscape is flat near the
minimum, so grid and average converge.

Improvement vs E009: −3.23pp EER (0.52% vs 3.75%), −0.0646 min-DCF.

## Interpretation

The improvement over E009 is dramatic: 0.52% vs 3.75% OOF EER. Three factors
explain this:

1. **Stronger audio modality.** LPCC per-fold mean 3.33% vs MFCC 4.21%.
   More importantly, LPCC fold-2 EER is 0.00% — the same fold that
   previously dragged MFCC fusion down.

2. **Aligned fold winners.** In E023, folds 1 and 2 are essentially perfect
   (0.83%, 0.00%) for both audio AND image. In E009 (MFCC), the fold winner
   often differed between modalities, limiting fusion gains. The LPCC-specific
   improvement on folds 1/2 aligns with image improvements on the same folds,
   leaving only fold 0 as a hard case (audio 9.17%, image 2.08%).

3. **Complementarity preserved.** The Pearson correlation between LPCC and
   image scores is 0.442 — very similar to E009's 0.426. The modalities are
   still complementary; the better audio floor raises the fusion ceiling.

The optimal weight (w=0.36 audio, 0.64 image) is slightly higher than E009's
w=0.28, reflecting that LPCC audio is somewhat more competitive. The landscape
is flat: average (w=0.50) achieves the same 0.52% EER, which means the fusion
is robust to weight choice near the optimum.

**E023 LPCC+image is the new fusion flagship.** OOF EER 0.52%, min-DCF 0.0104.

## Next step

- E023 is the primary submission candidate for the fusion system.
- Write `predict_fusion.py` using the LPCC audio + image pipeline with w=0.36
  (or w=0.50, identical performance) and Platt calibration.
- Run the full prediction pipeline on the eval data and generate result files.
- Consider whether any further ablations are worth running given the deadline.
