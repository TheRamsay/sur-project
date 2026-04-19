# E014 — Audio new augmentation ablation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio augmentation baseline)

## Hypothesis

E008 +All (noise SNR=20dB + speed ±10%) reduced EER from 9.58% to 4.21 ± 3.11%.
Professor Burget mentioned the eval data would be intentionally degraded
("schválně zprasené") — codec simulation (8kHz resample) and lower SNR noise
(10dB, more aggressive) directly target this distribution shift.
Pitch shift addresses microphone frequency response variation across recording
sessions/devices. Amplitude clipping simulates recording saturation (ADC
overload). Each augmentation is tested individually on top of E008 +All to
isolate its contribution; +AllNew combines all four.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E008
- **Model:** UBM 32 + MAP adapt r=16 — same as E008
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO)
- **Seed:** 67
- **Command / notebook:** `notebooks/E014_audio_new_aug.ipynb`
- **Augmentation base:** original + noise(SNR=20dB) + speed(rate∈[0.9,1.1])
  - `+Codec`: base + 8kHz codec simulation (resample down to 8kHz, back up)
  - `+LowNoise`: base + aggressive noise (SNR=10dB)
  - `+Pitch`: base + pitch shift ±{1,2} semitones (random choice)
  - `+Clip`: base + amplitude clipping at random threshold [30–70% of max]
  - `+AllNew`: base + all four new augmentations combined

## Result

| Config            | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std      | min-DCF mean |
| ----------------- | ---------- | ---------- | ---------- | --------------- | ------------ |
| **+All (E008) ★** | **3.47**   | **8.33**   | **0.83**   | **4.21 ± 3.11** | **0.0509**   |
| +All+Codec        | 2.78       | 10.00      | 1.67       | 4.81 ± 3.69     | 0.0796       |
| +All+LowNoise     | 2.78       | 18.33      | 1.67       | 7.59 ± 7.61     | 0.0852       |
| +All+Pitch        | 11.25      | 5.83       | 0.83       | 5.97 ± 4.25     | 0.0861       |
| +All+Clip         | 11.25      | 10.00      | 0.83       | 7.36 ± 4.64     | 0.0806       |
| +All+AllNew       | 2.08       | 11.67      | 0.83       | 4.86 ± 4.84     | 0.0806       |

OOF overall (best config +All E008): EER = 9.43%, min-DCF = 0.1635, threshold = −0.075

## Interpretation

**None of the four new augmentations improve on E008 +All (4.21 ± 3.11%).** E008 +All
remains the audio flagship — the hypothesis did not hold.

- **+Codec (4.81 ± 3.69%)**: Closest to the baseline. Fold 0 improves (2.78% vs 3.47%)
  but Fold 2 regresses (1.67% vs 0.83%) and std is slightly worse. The 8kHz bandwidth
  limitation does add useful variation but also disrupts frequency patterns the UBM
  learned. Not a clear win.

- **+LowNoise (7.59 ± 7.61%)**: Clear regression. SNR=10dB noise is too aggressive
  for this dataset — it overwhelms the MFCCs and confuses the UBM, especially in
  Fold 1 (18.33% vs 8.33%). The signal power in the training corpus is already low
  enough that 10dB noise degrades the informative cepstral structure.

- **+Pitch (5.97 ± 4.25%)**: Mixed. Fold 0 collapses badly (11.25%) while Fold 1
  improves (5.83% vs 8.33%). Pitch shifting changes formant positions in ways that
  MFCCs capture — it distorts speaker identity rather than preserving it, making MAP
  adaptation learn a blurred speaker model.

- **+Clip (7.36 ± 4.64%)**: Regression. Amplitude clipping at [30–70%] of peak is
  too severe — it introduces harmonic distortion that creates artifactual MFCC patterns
  not present in the eval data. Fold 0 collapses (11.25%).

- **+AllNew (4.86 ± 4.84%)**: Combining all four new augmentations vs the E008 base
  slightly increases mean EER (4.86% vs 4.21%) and nearly doubles std (4.84% vs 3.11%).
  The individual regressions accumulate and the much larger training set (7× more frames
  than original) does not compensate — more diverse augmentation ≠ better when individual
  augmentations are harmful.

The pattern across folds is instructive: Fold 1 (session 02) is consistently the
hardest — it has the most cross-session mismatch and is sensitive to any augmentation
that corrupts the speaker identity signal.

**Key insight**: The evaluation degradation Burget mentioned is likely at the level
of recording environment (channel, noise floor) rather than extreme bandwidth restriction
or saturation. The E008 noise+speed combo already captures the most important variation.
Further augmentation requires either more realistic codec models or a labeled corpus of
degradation types.

## Next step

- E008 +All is confirmed as audio flagship. No further augmentation exploration needed.
- Proceed to calibration: OOF scores from E008 +All → Platt scaling → final threshold.
- Then: updated fusion (E009 used E008 scores; re-run fusion with fresh OOF calibration).
