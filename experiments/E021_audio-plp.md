# E021 — Audio PLP (Perceptual Linear Prediction)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio flagship), E020 (LPCC, 3.33±4.14%)

## Hypothesis

PLP (Hermansky 1990) combines three psychoacoustic stages before LPC analysis:
(1) Bark-scale filterbank (mimics cochlear frequency resolution), (2) equal
loudness weighting (pre-emphasises frequencies where the ear is most sensitive),
and (3) cube-root compression (intensity-to-loudness conversion, less saturating
than log). The resulting features model the auditory periphery more faithfully
than MFCC (mel warp + log + DCT) or LPCC (LPC directly on the power spectrum).

The cube-root compression in particular is less susceptible to saturation in
quiet speech and may preserve more speaker-discriminative dynamics in noisy or
bandwidth-limited conditions — directly relevant for Burget's "damaged" eval data.

Same 39 dims (13+Δ+ΔΔ) as E008 and E020 → dimensionality not a concern.

Expected: neutral-to-positive vs E008 (4.21%) and competitive with LPCC (3.33%).

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** PLP 13+Δ+ΔΔ+CMN (Bark 20 bands, LPC order=12, cube-root compression) = 39 features/frame
- **Model:** UBM 32 + MAP adapt r=16 — same as E008
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Augmentation:** +All (noise SNR=20dB + speed ±10%), same as E008
- **Command / notebook:** `notebooks/E021_audio_plp.ipynb`

## Result

| Config          | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std       | min-DCF mean |
| --------------- | ---------- | ---------- | ---------- | ---------------- | ------------ |
| E008 +All (ref) | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11      | 0.0509       |
| E020 LPCC (ref) | 9.17       | 0.83       | 0.00       | 3.33 ± 4.14      | 0.0333       |
| **PLP +All**    | **4.17**   | **3.33**   | **9.17**   | **5.56 ± 2.58**  | **0.0944**   |

OOF overall (PLP +All): EER = 12.40%, min-DCF = 0.1458, threshold = −0.008

## Interpretation

PLP regresses vs both references: +1.35pp vs E008 (5.56% vs 4.21%) and +2.23pp vs
E020 (5.56% vs 3.33%). min-DCF is nearly double E008 (0.0944 vs 0.0509).

The fold pattern is inverted relative to LPCC: PLP recovers fold 1 (3.33%) and
makes fold 0 reasonable (4.17%), but fold 2 catastrophically regresses to 9.17%
(vs 0.83% for E008 and 0.00% for LPCC). Fold 2 holds out the target's third
session — evidently PLP's Bark filterbank is more sensitive to the specific
acoustic conditions of that session than either MFCC or LPCC.

The OOF EER (12.40%) is the highest yet in the audio cepstral family, confirming
that PLP does not generalise well on this dataset in the UBM-MAP 32 / 39d regime.

There are two plausible explanations for PLP's underperformance:

1. **Equal loudness saturates at low frequencies.** The equal loudness curve
   heavily down-weights frequencies below ~1 kHz, which is where fundamental
   voice quality information lives. MFCC's mel scale and LPCC's direct LPC do
   not apply this aggressive de-emphasis.

2. **Bark filterbank with only 20 bands is coarse.** The triangular-filter
   approximation of the Bark scale gives 20 bands compared to MFCC's typical
   40 mel bins (though our MFCC uses the default librosa settings). The
   per-frame LPC fit on a 20-point spectrum is a lossy representation.

3. **Cube-root compression is not beneficial here.** The cube-root is less
   saturating than log, but this also means the UBM Gaussians see a less
   compressed dynamic range, making the model harder to fit robustly with
   only 32 components.

**E008 +All (MFCC) remains the audio flagship on mean EER. E020 LPCC is still
the best option for mean EER and min-DCF.** PLP is not a candidate for the
submission system in this configuration.

## Next step

- PLP does not improve over MFCC or LPCC in the 39d / UBM-32 regime. Do not
  pursue PLP further unless a fundamentally different architecture (e.g. DNN
  acoustic model) is used.
- Shift focus to fusion: LPCC (E020) and MFCC (E008) likely have complementary
  errors (LPCC folds 1&2 are near-zero; E008 fold 2 near-zero; PLP fold 0 &1
  reasonable). A score-level fusion of MFCC and LPCC audio systems could reduce
  variance and lower mean EER below 3%.
- Alternatively, explore calibration of LPCC OOF scores to improve the OOF EER
  discrepancy between per-fold mean (3.33%) and pooled OOF (6.46%).
