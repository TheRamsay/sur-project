# E012 — Audio CMVN (Cepstral Mean and Variance Normalization)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio +All augmentation flagship)

## Hypothesis

E008 uses CMN (cepstral mean normalization) — subtracting the per-utterance
mean from each MFCC coefficient. This removes the channel DC offset but leaves
per-coefficient variance untouched. Different recording environments (microphone
sensitivity, room acoustics, compression codecs) compress or expand the dynamic
range of each cepstral coefficient differently, leading to session-to-session
mismatch in variance even after mean subtraction.

CMVN additionally divides each coefficient by its per-utterance standard
deviation (+ ε = 1e-10 for numerical safety). This maps each coefficient to
approximately unit variance, removing the scale mismatch across sessions.

**Expected:** small improvement (0.5–2% EER) or neutral. Variance normalization
is standard in robust ASR and speaker verification, but it can also remove
speaker-discriminative energy variation encoded in the dynamic range. For a
UBM+MAP system the diagonal covariance model already normalizes implicitly
to some extent, so the gain might be marginal.

**Risk:** if the target speaker's vocal energy profile differs systematically
from non-target speakers, CMVN could erase that discriminant — potentially
hurting EER. This is the main failure mode to watch.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, **CMVN** (per-utterance mean AND variance) — only change vs E008
- **Model:** UBM 32 + MAP adapt r=16 — same as E008
- **Augmentation:** +All (noise SNR=20dB + speed ±10%) on train fold, originals on val — same as E008
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Command / notebook:** `notebooks/E012_audio_cmvn.ipynb`

## Result

| Config           | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std    | min-DCF mean |
| ---------------- | ---------- | ---------- | ---------- | ------------- | ------------ |
| E008 CMN (+All)  | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11   | 0.0509       |
| **E012 CMVN (+All)** | **8.47** | **9.17** | **0.83** | **6.16 ± 3.78** | **0.0796** |

OOF overall (CMVN): EER = 8.02%, min-DCF = 0.1604, threshold = +0.062

Δ EER (CMVN − CMN): +1.95% — regression.

## Interpretation

CMVN hurts. Mean EER degrades from 4.21% (CMN) to 6.16% (CMVN), a +1.95%
regression. Fold 0 is the worst affected (3.47% → 8.47%) while Fold 2
is unchanged (0.83% both). Standard deviation also widens slightly
(3.11 → 3.78%).

The hypothesis that variance normalization would reduce session-to-session
mismatch did not hold for this setup. Two probable causes:

1. **Speaker-discriminative variance.** In a UBM+MAP framework with diagonal
   covariance, the Gaussian components already learn per-coefficient scale during
   UBM training. CMVN at the utterance level removes that scale before the UBM
   sees it, discarding a source of between-speaker discrimination that the
   diagonal covariance would otherwise exploit.

2. **Short-utterance instability.** Per-utterance std estimates over short
   recordings are noisy. Dividing by a poorly estimated std amplifies high-
   frequency noise rather than removing channel mismatch — the opposite of
   the intended effect.

The threshold shifts from ≈ −0.078 (CMN) to +0.062 (CMVN), indicating the
score distribution is differently scaled, but this does not translate to
better separation.

**E008 CMN remains the audio flagship.** CMVN is not beneficial for this
UBM+MAP + short-recording regime.

## Next step

- Audio normalization axis is exhausted at CMN. Variance is better left to
  the UBM diagonal covariance.
- Next audio experiments: feature-level changes (filter bank, PLDA) or model
  changes (i-vector / x-vector style).
- Consider fusion refinement (E009 was score-level; feature-level fusion
  using OOF scores from audio + image).
