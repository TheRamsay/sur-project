# E020 — Audio LPCC (LPC Cepstral Coefficients)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio flagship), E016 (FBank, 120d regression), E019 (SDC, 104d regression)

## Hypothesis

LPCC are derived from LPC coefficients via cepstral recursion and directly
represent the shape of the all-pole vocal tract filter. Mel-cepstrum (MFCC)
applies a perceptual mel warp before the DCT, emphasising lower frequencies.
LPCC skips the mel warp, keeping the full spectral envelope encoded as a
minimum-phase all-pole model. If the vocal tract geometry is more stable across
sessions than mel-spectral patterns, LPCC could generalise better to the
session-disjoint target fold. The 13+Δ+ΔΔ = 39d formulation matches E008
exactly, so dimensionality is not a concern (unlike E016 120d and E019 104d,
which both regressed).

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** LPCC 13 (LPC order=12, cepstrum via frequency domain) + Δ + ΔΔ + CMN = 39 features/frame
- **Model:** UBM 32 + MAP adapt r=16 — same as E008
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Augmentation:** +All (noise SNR=20dB + speed ±10%), same as E008
- **Command / notebook:** `notebooks/E020_audio_lpcc.ipynb`

## Result

| Config          | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std       | min-DCF mean |
| --------------- | ---------- | ---------- | ---------- | ---------------- | ------------ |
| E008 +All (ref) | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11      | 0.0509       |
| **LPCC +All**   | **9.17**   | **0.83**   | **0.00**   | **3.33 ± 4.14**  | **0.0333**   |

OOF overall (LPCC +All): EER = 6.46%, min-DCF = 0.1198, threshold = 0.036

## Interpretation

LPCC marginally improves on E008 in mean EER (−0.88pp: 3.33% vs 4.21%) and clearly
wins on min-DCF (0.0333 vs 0.0509). However the standard deviation is higher
(4.14 vs 3.11), driven by fold 0 regressing to 9.17% while folds 1 and 2 improve
substantially (0.83%, 0.00%).

The fold 0 regression is notable: fold 0 holds out the target's first session, which
is likely the most acoustically different from the training sessions. LPCC's all-pole
model may be more sensitive to channel/microphone variation — in a different recording
environment, the vocal tract shape estimate is less stable than the mel-cepstrum
which smears fine spectral detail. Folds 1 and 2 (holding out sessions 2 and 3 on the
target side) improve because LPCC better captures the speaker's stable articulatory
patterns when the channel mismatch is smaller.

The OOF EER (6.46%) is actually higher than E008's OOF (9.17% E008, but per-fold mean
4.21%). This discrepancy is because OOF pools all three folds including fold 0 which
is the heaviest contributor. The per-fold mean is the more reliable summary when fold
sizes differ.

Overall: LPCC is a viable alternative to MFCC in the 39d regime — avoids the
dimensionality blowup that killed E016 (120d FBank) and E019 (104d SDC), delivers
equivalent mean EER with better min-DCF. The higher variance is a concern for a
single submission.

**E008 +All remains the audio flagship on mean EER**, but LPCC is a candidate for
fusion with the image system (different spectral representation = complementary errors).

## Next step

- Try fusing LPCC audio scores with image scores (E007 +All) in the fusion system —
  LPCC and MFCC likely give complementary errors across folds, and score-level fusion
  could reduce the high variance.
- Alternatively, consider MFCC+LPCC feature concatenation (78d) with UBM 32 — the
  two representations are complementary but the dimensionality increase may hurt
  (similar risk to E016/E019). Run only if fusion is not the priority.
