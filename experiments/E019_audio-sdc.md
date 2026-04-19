# E019 — Audio Shifted Delta Cepstra (SDC)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio flagship), E002 (delta features)

## Hypothesis

Standard Δ+ΔΔ captures local temporal dynamics using adjacent frames (±1 or ±2
frames apart). Speaking rhythm and intonation patterns operate at longer
timescales — on the order of 3–5 frames (30–50ms). SDC replaces adjacent-frame
deltas with deltas computed over frames that are P steps apart (P=3), capturing
this longer-range context. With N=7 blocks the window spans 21 frames ≈ 210ms.
Hypothesis: SDC features encode more speaker-discriminative temporal structure
than Δ+ΔΔ, yielding neutral-to-positive EER vs E008 (4.21 ± 3.11%).

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13 static + SDC (N=7, d=1, P=3) = 104 features/frame
- **Model:** UBM 32 + MAP adapt r=16 — same as E008
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Augmentation:** +All (noise SNR=20dB + speed ±10%), same as E008
- **Command / notebook:** `notebooks/E019_audio_sdc.ipynb`

## Result

| Config          | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std      | min-DCF mean |
| --------------- | ---------- | ---------- | ---------- | --------------- | ------------ |
| E008 +All (ref) | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11     | 0.0509       |
| **SDC +All**    | **19.72**  | **9.17**   | **10.00**  | **12.96 ± 4.79** | **0.1778**  |

OOF overall (SDC +All): EER = 10.73%, min-DCF = 0.2104, threshold = −0.094

## Interpretation

SDC is a significant regression vs E008 Δ+ΔΔ (+8.75pp mean EER, 12.96% vs 4.21%).
Fold 0 is catastrophic at 19.72%; folds 1 and 2 also regress substantially.

Three plausible causes:

1. **Feature dimensionality mismatch.** SDC expands the feature space from 39d
   to 104d. A 32-component diagonal-covariance GMM has 32×104 = 3,328 mean
   parameters. The training set is small (~150–170k frames, but from only ~150
   utterances). The higher-dimensional feature space likely causes the UBM to
   overfit to the particular non-target speakers seen in training, hurting
   generalisation to the session/identity-disjoint val fold.

2. **CMN double-application.** The implementation applies CMN first to the
   static MFCC, then again to the full 104d concat. The second CMN zeros out
   the mean of the SDC blocks as a unit, which may remove useful long-range
   discriminative information that SDC was designed to capture.

3. **P=3 gap too large for short utterances.** Many utterances are only a few
   seconds long (~50–100 frames). With N=7 blocks and P=3 the window spans
   ±9 frames from centre. For short utterances boundary clamping causes many
   duplicate values, reducing effective information.

The Δ+ΔΔ formulation (39d) is better matched to a UBM-32 diagonal GMM on this
dataset size. SDC is designed for larger systems (e.g., UBM 512+ or PLDA back-ends)
where the higher-dimensional statistics can be reliably estimated.

**E008 +All remains the audio flagship.**

## Next step

- SDC is not suitable for the current GMM size without a substantially larger UBM
  or a different back-end (e.g., i-vector / PLDA). Not worth pursuing further given
  the dataset size constraint.
- Audio feature space is largely exhausted (Δ+ΔΔ, FBank, CMVN, VTLP, SDC all tried).
- Focus should shift to improving the fusion system (E009) or calibration.
