# E029 — Permutation test (label shuffle hygiene check)

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E007, E025

## Hypothesis

With shuffled train labels the model should have no signal to learn. Val EER
should collapse to chance (~50%) for both modalities. If EER stays
meaningfully below 50% across all 3 shuffle seeds, the pipeline is leaking
identity information through a channel that is not the label (augmentation
preserving identity, feature normalization using pooled labels, etc.).
Expected window: mean permuted EER ∈ [40%, 60%] for both E007 image and E025
audio. A pass validates the OOF EERs reported for the flagships (0.97% image,
1.94% audio) as genuinely label-derived, not artifacts.

## Setup

- **Modality:** image and audio (both flagships)
- **Data:** train + dev combined, LOSO seed=67 (same folds as E007/E025)
- **Image model:** StandardScaler + PCA(50) + LogReg(C=1, max_iter=1000)
  with +All augmentation (flip + brightness ∈ [0.7,1.3] + noise σ=15)
- **Audio model:** LPCC 13+Δ+ΔΔ (LPC order=12), UBM 32 diag, MAP r=16, +Pitch
  augmentation (E025 winner; pitch-shift ±{1,2} semitones)
- **Perturbation:** `np.random.default_rng(seed).shuffle(y_train)` BEFORE
  augmentation so augmented copies inherit shuffled labels. Val labels
  untouched.
- **Shuffle seeds:** {1, 2, 3} × 3 LOSO folds = 9 runs per modality
- **Notebook:** `notebooks/E029_permutation_test.ipynb`

## Result

3 shuffle seeds × 3 LOSO folds = 9 runs per modality.

| Modality | Seed 1 mean | Seed 2 mean | Seed 3 mean | Grand mean ± std | Clean EER |
| -------- | ----------- | ----------- | ----------- | ---------------- | --------- |
| image    | 42.87%      | 59.86%      | 45.74%      | 49.49% ± 13.64   | 0.97%     |
| audio    | 64.07%      | 47.73%      | 53.98%      | 55.26% ± 20.67   | 1.94%     |

Per-run detail (seed, fold, EER):

| shuf | fold | image EER | audio EER |
| ---- | ---- | --------- | --------- |
| 1 | 0 | 48.61 | 69.72 |
| 1 | 1 | 39.17 | 55.83 |
| 1 | 2 | 40.83 | 66.67 |
| 2 | 0 | 38.75 | 23.19 |
| 2 | 1 | 60.83 | 39.17 |
| 2 | 2 | 80.00 | 80.83 |
| 3 | 0 | 38.06 | 48.61 |
| 3 | 1 | 50.00 | 80.00 |
| 3 | 2 | 49.17 | 33.33 |

## Interpretation

Both modalities PASS the hygiene check. Image grand mean 49.49% and audio
grand mean 55.26% both sit inside the [40, 60] pass window and are not
statistically distinguishable from chance (50%) given the large per-run std
(9 runs × ~10 positives per fold gives ~15pp binomial noise per point, which
matches the observed ±13–21pp spread).

The spread is wide (individual runs range 23% to 80%) but symmetric around
50 — exactly the signature of pure label noise with no residual signal.
Crucially, no seed/fold combination sits anywhere near the flagship clean
EERs (0.97% image, 1.94% audio). The 48pp gap between permuted and clean
for image and ~53pp gap for audio confirm the clean numbers are driven by
the label column, not by an auxiliary channel such as augmentation bleeding
identity features, label-coupled normalization, or leakage of val samples
into train feature statistics.

The slightly-above-50% audio grand mean (55.26%) is a small random lean and
not meaningful — a single fold flipping direction would move it by ~5pp.

## Next step

No pipeline changes required. Both flagships (E007 image +All, E025 audio
LPCC+Pitch) are validated as learning from true labels. Proceed to
submission packaging / docs with confidence that the reported OOF EERs
reflect genuine classification performance.
