# E015 — Image new augmentation ablation

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E007 (image augmentation baseline)

## Hypothesis

E007 +All (flip + brightness[0.7,1.3] + noise σ=15) reduced EER from 4.49% to
0.97 ± 0.86% — a near-perfect per-fold score on the current train+dev corpus.
However, the professor's eval data is described as "schválně zprasené" (intentionally
degraded), meaning the distribution shift at test time may be larger than what
E007 covers.

Four new augmentations are tested individually on top of E007 +All:

- **+JPEG** (quality 20–50): simulates lossy compression artifacts visible in
  low-quality web/scanned images and "zprasené" eval captures.
- **+Blur** (σ∈[1.0, 2.0]): simulates defocus, motion blur, or low-resolution
  downsampling present in degraded eval images.
- **+Rotate** (±10°): addresses head-pose variation across sessions — the target
  may tilt their head differently in each recording sit.
- **+Contrast/Gamma** (γ∈[0.5, 2.0]): addresses camera exposure and gamma-curve
  differences between recording conditions.
- **+AllNew**: all four combined on top of E007 +All (8× training data per sample).

Each new config is compared against the E007 +All baseline (0.97 ± 0.86%).

## Setup

- **Modality:** image
- **Data:** train + dev (combined), 222 samples (30 target, 192 non-target)
- **Features:** 80×80 PNG → grayscale → flatten → StandardScaler → PCA 50 → LogReg C=1
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds (LOSO)
- **Seed:** 67
- **Command / notebook:** `notebooks/E015_image_new_aug.ipynb`
- **Augmentation base (+All E007):** original + flip + brightness[0.7,1.3] + noise(σ=15)
  - `+All+JPEG`: base + JPEG compression quality∈[20,50]
  - `+All+Blur`: base + Gaussian blur σ∈[1.0,2.0]
  - `+All+Rotate`: base + random rotation ±10°
  - `+All+Contrast`: base + gamma contrast γ∈[0.5,2.0]
  - `+All+AllNew`: base + all four new augmentations combined

## Result

| Config            | Fold 0 EER | Fold 1 EER | Fold 2 EER | Mean ± std      | min-DCF mean |
| ----------------- | ---------- | ---------- | ---------- | --------------- | ------------ |
| **+All (E007) ★** | **2.08**   | **0.83**   | **0.00**   | **0.97 ± 0.86** | **0.0194**   |
| +All+JPEG         | 7.78       | 0.83       | 0.83       | 3.15 ± 3.27     | 0.0296       |
| +All+Blur         | 2.08       | 1.67       | 0.83       | 1.53 ± 0.52     | 0.0306       |
| +All+Rotate       | 8.47       | 0.00       | 0.00       | 2.82 ± 3.99     | 0.0231       |
| +All+Contrast     | 8.47       | 8.33       | 0.83       | 5.88 ± 3.57     | 0.0509       |
| +All+AllNew       | 2.78       | 7.50       | 0.00       | 3.43 ± 3.10     | 0.0352       |

OOF overall (best config +All E007): EER = 4.01%, min-DCF = 0.0729, threshold = −5.028

## Interpretation

**None of the four new augmentations improve on E007 +All (0.97 ± 0.86%).** The
E007 baseline wins on both mean EER and min-DCF.

- **+All+JPEG (3.15 ± 3.27%)**: Clear regression. JPEG artifacts at quality 20–50
  are aggressive — they introduce blocking and ringing artefacts in the 80×80 pixel
  space that confuse the PCA subspace. Fold 0 (session 01) is hit hardest (7.78%).
  Compression robustness is not the bottleneck for this data.

- **+All+Blur (1.53 ± 0.52%)**: Closest to the baseline in mean EER (1.53% vs 0.97%),
  but std collapses from 0.86% to 0.52% — the most consistent new candidate. Blur
  marginally hurts: it smooths discriminative high-frequency texture that PCA relies
  on for separating m431 from non-targets. Not a win, but the most benign new aug.

- **+All+Rotate (2.82 ± 3.99%)**: Mixed — Fold 1 and Fold 2 reach 0.00% EER
  (perfect) but Fold 0 collapses to 8.47%. This is the highest std of any config.
  Rotation at ±10° changes the spatial structure of the flattened pixel vector
  significantly; PCA picks up a different set of axes depending on which session
  is held out, making the model unstable across folds.

- **+All+Contrast (5.88 ± 3.57%)**: Worst new augmentation. Gamma jitter γ∈[0.5,2.0]
  is too aggressive — it drastically remaps the pixel intensity distribution,
  creating training examples that look nothing like the moderately lit session
  images. Both Fold 0 and Fold 1 collapse (8.47% and 8.33%). The brightness jitter
  in E007 [0.7,1.3] already covers the plausible lighting range; gamma expansion
  adds harmful distortion beyond that.

- **+All+AllNew (3.43 ± 3.10%)**: Combining all four new augmentations increases
  training data to 8× but the individual regressions accumulate. Fold 1 collapses
  to 7.50% — the combined noise from JPEG + blur + rotate + contrast overwhelms the
  useful signal learned in E007 +All. More augmentation is not always better.

The pattern is consistent with E014 (audio new aug ablation): when a baseline is
already near-optimal on the training distribution, additional augmentations are
more likely to corrupt discriminative features than to add useful generalization.
The E007 +All combination (flip + brightness + noise) is well-calibrated; any
further expansion moves into harmful territory.

**E007 +All is confirmed as the image augmentation flagship. No change.**

## Next step

- Image flagship locked at E007 +All. No further augmentation exploration needed.
- Matching outcome to E014 (audio): both modalities have confirmed their respective
  augmentation strategies.
- Proceed to fusion calibration: re-run score-level fusion (E009) with OOF scores
  from confirmed E007 image and E008 audio flagships, then Platt-calibrate.
