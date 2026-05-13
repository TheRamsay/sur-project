# E018 — Audio VTLP augmentation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E008 (audio augmentation ablation, +All flagship), E014 (new aug ablation)

## Hypothesis

VTLP (Vocal Tract Length Perturbation) warps the frequency axis of the
spectrogram by a multiplicative factor α ∈ [0.9, 1.1], simulating biological
variation in vocal tract length across speakers and sessions. Unlike speed
perturbation, which compresses or stretches the time axis (affecting tempo),
VTLP shifts formant frequencies along the frequency axis — a fundamentally
different variation axis.

Because session mismatch in the target data likely includes both temporal
(rate) and spectral (formant) variation, VTLP should complement noise and
speed augmentation. Expected outcome: neutral-to-positive effect on top of
E008 +All; potentially replacing speed if VTLP covers the same session
mismatch axis more directly.

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** MFCC 13+Δ+ΔΔ, CMN (per-utterance) — same as E008
- **Model:** UBM 32 + MAP adapt r=16 — same as E008
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Command / notebook:** `notebooks/E018_audio_vtlp.ipynb`
- **Augmentation (config 1 — +All+VTLP):** original + noise20 + speed + vtlp (4 copies)
- **Augmentation (config 2 — +VTLP_replace_speed):** original + noise20 + vtlp (3 copies)
- **VTLP:** frequency-axis warp via STFT → interp1d on α·freq → ISTFT, α ∈ [0.9, 1.1]

## Result

| Config                  | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std       | min-DCF mean |
| ----------------------- | ---------- | ---------- | ---------- | ---------------- | ------------ |
| E008 +All (reference)   | 3.47       | 8.33       | 0.83       | 4.21 ± 3.11      | 0.0509       |
| +All+VTLP               | 9.86       | 8.33       | 0.83       | 6.34 ± 3.95      | 0.0602       |
| **+VTLP_replace_speed** | 8.47       | **2.50**   | **0.83**   | **3.94 ± 3.28**  | 0.0685       |

OOF overall (+VTLP_replace_speed): EER = 10.47%, min-DCF = 0.1562, threshold = −0.113

## Interpretation

The hypothesis partially held. VTLP alone (replacing speed) gives a marginal
improvement over E008 +All: 3.94 ± 3.28% vs 4.21 ± 3.11% (−0.27 pp mean).
The variance is slightly higher (3.28 vs 3.11 std), and the min-DCF actually
regresses (0.0685 vs 0.0509), suggesting the calibration is worse.

Adding VTLP on top of the existing +All stack (+All+VTLP, 4 copies) hurts
clearly: 6.34 ± 3.95%, a regression of +2.13 pp. Fold 0 particularly degrades
(9.86% vs 3.47% in E008). The most likely explanation is over-augmentation:
with 4 signal copies the UBM non-target pool grows 4× relative to the target
MAP pool, diluting the target model's discriminability.

Speed perturbation changes tempo while preserving spectrum; VTLP shifts formant
frequencies while preserving tempo. Both simulate session variation but on
orthogonal axes. When we replace speed with VTLP (+VTLP_replace_speed) the
fold 1 EER drops dramatically (8.33% → 2.50%), suggesting VTLP covers a real
mismatch axis for that session pair. However, fold 0 stays high (8.47%),
implying session 0 mismatch is better covered by speed perturbation.

**E008 +All remains the audio flagship** (4.21 ± 3.11%, min-DCF 0.0509).
VTLP replacing speed gives marginally better mean EER but worse min-DCF and
similar std — not a clear win on the primary metric set.

## Next step

- E008 +All stays as the audio flagship — VTLP does not provide a clear win.
- The fold 1 improvement with VTLP_replace_speed suggests that a 4-copy
  combination (orig + noise + speed + vtlp) with balanced sampling (not just
  concatenation) might help, but the +All+VTLP experiment already showed that
  naively adding VTLP as a 4th copy hurts via UBM dilution.
- Next priority: focus on fusion (E009 is already done) or documentation.
