# E053 — Hard 2-second prefix truncation

- **Date:** 2026-04-22
- **Author:** Dominik Huml
- **Related experiments:** E052 (current audio flagship)

## Hypothesis

A friend suggested hard-cutting the first 2 seconds of every recording to remove
pre-speech noise. On paper this could help if recordings have a noisy settling
period (mic thump, breath, background before the speaker starts). However:

- CMN already suppresses stationary channel effects per utterance.
- The dataset is small — losing 2 s × n_utterances is non-trivial data reduction.
- Tied covariance + codec aug already achieve 0.46% clean EER, suggesting the
  front-end is not the bottleneck.

Prediction: regression, because we're discarding genuine speech frames on a
small dataset that has no meaningful silence at the beginning.

## Setup

- **Modality:** audio
- **Baseline:** E052 (LPCC + UBM-32 tied cov + MAP r=16 + pitch aug + speed TTA + codec aug)
- **Variant:** E052 + discard first 2 s of every wav at load time (train AND val)
- **Edge case:** if utterance ≤ 2 s, keep it untouched (rather than empty array)
- **Conditions measured:** clean CV EER + codec stress EER (same as E052)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Notebook:** `notebooks/E053_audio_2s_truncation.ipynb`

## Result

| Config | Clean EER | Codec EER | Clean DCF | Codec DCF |
|--------|-----------|-----------|-----------|-----------|
| E052 baseline | 0.46 ± 0.65% | 3.33 ± 4.14% | 0.0093 | 0.0333 |
| E052 + 2s cut | 0.83 ± 0.68% | 2.78 ± 1.96% | 0.0167 | 0.0556 |

Per-fold clean EER:
- F0: baseline=1.39% / +2s cut=0.00%
- F1: baseline=0.00% / +2s cut=1.67%
- F2: baseline=0.00% / +2s cut=0.83%

Per-fold codec EER:
- F0: baseline=9.17% / +2s cut=4.17%
- F1: baseline=0.83% / +2s cut=4.17%
- F2: baseline=0.00% / +2s cut=0.00%

**Verdict: REJECT** (delta clean = +0.37pp, threshold = +0.1pp)

## Interpretation

Truncation redistributes errors across folds rather than eliminating them. It
accidentally helps fold 0 (which likely has genuine noise at the very start of
session 03) while breaking folds 1 and 2 (whose speakers start talking
immediately — cutting 2 s removes real speech content).

The codec EER *looks* slightly better (3.33→2.78%) and the standard deviation
drops (4.14→1.96%), but this is a fold-shuffling artefact: baseline fold 0 is
the hard codec fold (9.17%), truncation fixes fold 0 but creates new errors at
folds 1–2. The mean improves, the sum of squared errors does not.

Clean DCF also regresses (0.0093→0.0167). The friend's intuition was based on
general pipeline lore; in this dataset CMN already handles stationary pre-speech
noise per utterance, and the recordings have no meaningful silence header.

## Next step

E052 remains the audio flagship. No further truncation experiments warranted.
