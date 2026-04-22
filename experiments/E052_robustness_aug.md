# E052 — Robustness augmentation: codec aug (audio) + Cutout (image)

- **Date:** 2026-04-22
- **Author:** Dominik Huml
- **Related experiments:** E051 (stress test identified vulnerabilities), E042 (audio flagship), E033 (image flagship)

## Hypothesis

E051 identified two specific vulnerabilities:
1. **Audio codec** (8kHz BW): 13.33% EER — LPCC loses upper formants
2. **Image occlusion**: 11.06% EER — PCA has no spatial locality

Targeted augmentations addressing each axis should close these gaps without hurting clean performance:
- **Codec aug (audio)**: add bandwidth-limited (→8kHz→original_sr) versions of training audio; forces LPCC to learn formant patterns surviving low-pass filtering
- **Cutout aug (image)**: randomly mask a 20×20 patch during training (same size as test stress); forces PCA to represent faces without any single spatial region

## Setup

- **Audio:** E042 baseline + codec aug (add 8kHz downsampled copies of all train wavs)
- **Image:** E033 baseline + Cutout 20×20 (random patch masked to 0 on each augmented copy)
- **Metric (primary):** clean CV EER must not regress; stress EER under codec/occlude should improve
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Notebook:** `notebooks/E052_robustness_aug.ipynb`

## Result

### Audio: clean vs codec stress

| Config | Clean EER | Codec EER | Delta codec | Clean DCF | Codec DCF |
|--------|-----------|-----------|-------------|-----------|-----------|
| E042 baseline | 0.46 ± 0.65% | 13.33 ± 3.79% | — | 0.0093 | 0.1694 |
| +codec aug | **0.46 ± 0.65%** | **3.33 ± 4.14%** | **−10.00pp** | 0.0093 | 0.0333 |

Per-fold codec EER: E042 baseline F0=18.33% / F1=9.17% / F2=12.50%; +codec aug F0=9.17% / F1=0.83% / F2=0.00%.

### Image: clean vs occlude stress

| Config | Clean EER | Occlude EER | Delta occlude | Clean DCF | Occlude DCF |
|--------|-----------|-------------|---------------|-----------|-------------|
| E033 baseline | 0.51 ± 0.36% | 10.32 ± 6.60% | — | 0.0102 | 0.1565 |
| +Cutout aug | 1.71 ± 1.24% | 6.81 ± 3.94% | −3.51pp | 0.0343 | 0.1028 |

## Interpretation

**Audio — codec aug is a massive win with zero clean penalty:**
- Clean EER is identical: 0.46% (F0=1.39, F1=0.00, F2=0.00) in both cases.
- Codec EER drops 13.33% → 3.33% (−10pp, −75% relative). min-DCF drops from 0.1694 → 0.0333.
- Mechanism: training on bandwidth-limited copies forces UBM/MAP to learn formant representations that survive 4kHz low-pass. When eval audio is codec-degraded, the model still has clean data to anchor on — but its learned distribution now overlaps with the degraded regime.
- Residual 3.33% is primarily fold 0 (9.17%) which was the hard fold anyway. Folds 1 and 2 are near-perfect.

**Image — Cutout hurts clean more than it helps occlude:**
- Clean EER regresses from 0.51% → 1.71% (+1.20pp), violating the ≤0.1pp tolerance.
- Occlude improves (10.32% → 6.81%), but not enough to justify clean regression.
- Root cause: masking 20×20 patches destroys PCA eigenspace — patches that happen to cover eyes or nose dominate the leading eigenvectors. The model learns to reconstruct from partial data at the cost of discriminative precision.
- PCA's global nature means Cutout perturbs the signal that PCA actually relies on.

## Decision

**Audio codec aug: ADOPT.** Zero clean cost, −10pp codec robustness. Update predict_audio.py and predict_fusion.py.

**Image Cutout: REJECT.** Clean regression (+1.20pp) is 12× the tolerance. E033 remains image flagship at 0.51%.
