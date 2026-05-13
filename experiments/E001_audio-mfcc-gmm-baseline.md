# E001 — Audio baseline: MFCC + GMM

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** —

## Hypothesis

13 MFCC coefficients with per-utterance CMN, two separate GMM models (target
and non-target), score as LLR. Expected EER in the 15–30 % range — the
baseline should not be great, but better than chance (EER = 50 %).

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13, CMN (per-utterance), no deltas
- **Model:** two GMMs — target (8 components), non-target (32 components)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO on target sessions)
- **Seed:** 67
- **Command / notebook:** `notebooks/E001_audio_gmm.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 20.42   | 0.3083  |
| 1    | 24.17   | 0.1833  |
| 2    | 9.17    | 0.1833  |
| mean ± std | 17.92 ± 7.81 | 0.2250 ± 0.0722 |

OOF overall: EER = 14.48 %, min-DCF = 0.2677, threshold = −1.267

## Interpretation

Hypothesis confirmed — system beats chance (EER < 50 %). Fold 2 (session 03)
significantly better than folds 0 and 1 — likely session 03 is qualitatively
closer to the training sessions. Large std (7.81 %) shows results depend
heavily on which session is held out — a realistic estimate of variability on
unseen eval data.

Threshold −1.267 (not 0) indicates scores are not well calibrated — the
non-target GMM is overconfident. Platt calibration on OOF scores would help.

## Next step

- E002: add delta + delta-delta MFCC → expect 3–5 % EER improvement
- E003: GMM-UBM + MAP adaptation (flagship audio system)
- Consider Platt calibration on OOF scores
