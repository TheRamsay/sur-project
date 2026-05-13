# E003 — Audio: GMM-UBM + MAP adaptation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E002

## Hypothesis

UBM trained on all non-target frames, MAP-adapted to target using relevance
factor r=16. Should outperform E002 (10.09 %) because MAP adaptation uses the
UBM as a strong prior — critical when target training data is scarce (~20
utterances per fold). Features: MFCC 13+Δ+ΔΔ (same as E002).

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E002
- **Model:** UBM 32 components (diagonal), MAP adapt means only, r=16
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (identical to E001/E002)
- **Seed:** 67
- **Command / notebook:** `notebooks/E003_audio_gmm_ubm_map.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 9.86    | 0.1389  |
| 1    | 10.83   | 0.1333  |
| 2    | 1.67    | 0.0333  |
| mean ± std | 7.45 ± 5.04 | 0.1019 ± 0.0594 |

OOF overall: EER = 12.14%, min-DCF = 0.1562, threshold = −0.025

## Interpretation

Hypothesis confirmed — UBM+MAP beats independent GMMs (7.45 % vs 10.09 %).
Fold 2 is exceptional (EER = 1.67 %) — the UBM prior is especially powerful
when the target session is well-represented in the training sessions.
Folds 0 and 1 are similar to E002, suggesting MAP helps most when val session
is close to train sessions in acoustic conditions.

Std increased vs E002 (5.04 vs 1.81) — the method is more sensitive to
which session is held out, which is expected given the strong UBM prior.

Threshold is now very close to 0 (−0.025) — scores are much better calibrated
than E002 (threshold was −2.204). This is a direct benefit of the LLR
formulation with a shared UBM.

## Next step

- This is the audio flagship — use E003 as the audio system for fusion
- Move to image baseline (E004: PCA + logreg)
- Audio augmentation ablation can come after image baseline is established
