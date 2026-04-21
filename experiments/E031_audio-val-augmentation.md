# E031 — Audio val-time augmentation (TTA)

- **Date:** 2026-04-20
- **Author:** Dominik Huml
- **Related experiments:** E025 (audio flagship), E030 (image TTA → failed)

## Hypothesis

E025 scores each val utterance from the original WAV only. Because the model was trained
on pitch-shifted and speed-perturbed copies, those augmented views are within the
training distribution. Scoring the same utterance from multiple augmented views and
averaging the LLR should reduce score variance without corrupting the signal — unlike
E030 image TTA where rotated views were out-of-distribution.

Expected: `+Pitch TTA` (5 views) ≤ 1.5% mean EER while keeping std low.
Risk: frame-level averaging inside LLR already captures most variance → TTA neutral.

## Setup

- **Modality:** audio
- **Base model:** E025 exactly — LPCC 13+Δ+ΔΔ+CMN, LPC order=12, UBM-32, MAP r=16
- **Training aug:** +Pitch (original + 1 pitch-shifted copy per utterance) — unchanged
- **Val-time TTA configs tested:**

| Config | Val views | n views |
|--------|-----------|---------|
| `baseline` | original only (E025) | 1 |
| `+pitch_tta` | original + ±1st + ±2nd semitones | 5 |
| `+speed_tta` | original + 0.9× + 1.1× speed | 3 |
| `+pitch_speed_tta` | original + ±1st + ±2nd + 0.9× + 1.1× | 7 |

- **Seed:** 67
- **Command / notebook:** `notebooks/E031_audio_val_augmentation.ipynb`

## Result

| Config | F0 EER | F1 EER | F2 EER | Mean ± std | min-DCF |
| ------ | ------ | ------ | ------ | ---------- | ------- |
| baseline (E025) | 4.17 | 0.83 | 0.83 | 1.94 ± 1.57 | 0.0389 |
| +pitch_tta | 9.86 | 1.67 | 0.00 | 3.84 ± 4.31 | 0.0435 |
| **+speed_tta** | **4.17** | **0.83** | **0.00** | **1.67 ± 1.80** | **0.0333** |
| +pitch_speed_tta | 9.86 | 1.67 | 0.00 | 3.84 ± 4.31 | 0.0435 |

## Interpretation

Hypothesis partially holds — speed TTA helps, pitch TTA badly hurts.

**+speed_tta (winner):** −0.27pp EER (1.94→1.67%), −0.56pp min-DCF (0.0389→0.0333).
Speed perturbation is pitch-preserving: LPCC formant coefficients are unchanged,
so all 3 views (0.9×, 1.0×, 1.1×) give consistent LLR. Averaging reduces variance.
Fold 2 flips from 0.83→0.00% — one borderline sample crosses the threshold cleanly.

**+pitch_tta (fail):** fold 0 jumps 4.17→9.86% (+5.69pp). LPCC encodes formant
frequencies directly. Phase-vocoder pitch shifting slightly corrupts the formant
structure in the LPCC domain. The MAP-adapted target model is tuned to the
target's exact formant distribution → pitch-shifted views depress the target LLR
more than the UBM score → higher EER. Std explodes to 4.31%.

**+pitch_speed_tta:** pitch damage dominates; speed views can't compensate.
Matches +pitch_tta exactly (fold 0 still 9.86%).

Note: std increases slightly for +speed_tta (1.57→1.80) but min-DCF improvement
(0.0389→0.0333) is more reliable since min-DCF is threshold-agnostic.

**Verdict: adopt +speed_tta.** Update predict_audio.py scoring to average LLR
over original + 0.9× + 1.1× speed views. Do NOT add pitch TTA.

## Next step

- Update `predict_audio.py` to use speed TTA at inference
- Update `predict_fusion.py` audio scoring branch similarly
- E025 per-fold mean improves 1.94→1.67%; min-DCF 0.0389→0.0333
