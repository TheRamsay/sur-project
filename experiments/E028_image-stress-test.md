# E028 — Image flagship stress test at val time

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** E007 (flagship), E015 (aug ablation — trained-time blur/jpeg)

## Hypothesis

E007 (+All: flip + brightness + noise) reached 0.97 ± 0.86% per-fold EER —
so strong the number itself deserves suspicion. Burget's warning that eval
data will be "schválně zprasené" (deliberately degraded) suggests we should
verify this number holds under conditions the model has never seen at
training time.

If the 0.97% is **genuine robustness**, heavy val-time stresses should only
modestly inflate EER (roughly ≤5% under any single stress, maybe worse under
all-combined). If the number is **optimistic overfitting** to clean
session-03 data, a single heavy distortion should collapse performance to
≥20% EER.

## Setup

- **Modality:** image
- **Data:** train + dev combined, 222 samples; stresses applied to val fold only
- **Model:** E007 exactly — StandardScaler + PCA(50) + LogReg(C=1) with
  +All aug (flip + brightness σ∈[0.7,1.3] + noise σ=15) on train fold
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Seed:** 67
- **Command / notebook:** `notebooks/E028_image_stress_test.ipynb`
- **Stresses (val only, never train):**
  1. `Heavy JPEG` — quality=15
  2. `Heavy blur` — Gaussian σ=3.0
  3. `Rotation ±15°`
  4. `Rotation ±25°` (severe)
  5. `Occlusion` — random 20×20 black patch
  6. `Downsample` — resize to 40×40 then back to 80×80 (bilinear)
  7. `All combined` — all five stresses applied sequentially

## Result

| Stress              | F0 EER | F1 EER | F2 EER | Mean ± std      | Δ vs clean | min-DCF |
| ------------------- | ------ | ------ | ------ | --------------- | ---------- | ------- |
| Clean (E007)        | 2.08   | 0.83   | 0.00   | 0.97 ± 0.86     | —          | 0.0194  |
| Heavy JPEG q=15     | 2.08   | 0.83   | 0.83   | 1.25 ± 0.59     | +0.28      | 0.0250  |
| Heavy blur σ=3.0    | 3.47   | 1.67   | 0.00   | 1.71 ± 1.42     | +0.74      | 0.0343  |
| Downsample 40→80    | 2.08   | 0.83   | 0.00   | 0.97 ± 0.86     | +0.00      | 0.0194  |
| Rotation ±15°       | 11.94  | 9.17   | 0.83   | 7.31 ± 4.72     | +6.34      | 0.0713  |
| Rotation ±25°       | 9.86   | 29.17  | 1.67   | 13.56 ± 11.53   | +12.59     | 0.1861  |
| Occlusion 20×20     | 26.81  | 21.67  | 5.83   | 18.10 ± 8.93    | +17.13     | 0.2343  |
| All combined        | 21.81  | 41.67  | 15.00  | 26.16 ± 11.31   | +25.19     | 0.3815  |

Rows sorted roughly by severity of degradation.

## Interpretation

Hypothesis partially holds. The 0.97% clean figure is **not a fluke** — it
survives the three stress types that preserve facial geometry (JPEG q=15,
heavy blur σ=3.0, downsample 40→80) with near-zero damage:

- **JPEG q=15** only adds 0.28pp. E007's +Brightness + +Noise aug has
  already taught the model to ignore low-amplitude pixel perturbations;
  JPEG block artefacts fall in the same regime.
- **Blur σ=3.0** adds 0.74pp. PCA with n=50 is inherently a low-pass
  subspace, so high-frequency loss barely moves the projected features.
- **Downsample 40→80** produces identical numbers to clean. Bilinear
  re-up from 40×40 retains enough structure that the PCA subspace is
  unchanged within numerical noise.

The geometry-breaking stresses **collapse** the model, exactly as
theory predicts for a PCA eigenface pipeline:

- **Rotation ±15°** jumps to 7.31% mean (+6.34pp). Faces outside the
  training tilt distribution leave the eigenface subspace.
- **Rotation ±25°** reaches 13.56% with fold-1 blowing up to 29.17%.
- **Occlusion 20×20** is the worst single stress (18.10%). A black patch
  at a random location kills the projection because every eigenface
  weight becomes corrupted — no spatial locality to fall back on.
- **All combined** reaches 26.16% (fold 1: 41.67%, worse than random).

Per-fold pattern: **fold 2 (target = session 03) is reliably the most
robust** across every stress. That's because fold 2's target test images
are the session-03 samples that +Brightness was designed to help — the
model has overfit a session-03-friendly subspace, which coincidentally
stays stable under mild distortions. Fold 1 is the most brittle under
geometry attacks.

**Bottom line:** 0.97% is real for distortions that preserve face
geometry. It is *not* robust to rotation, occlusion, or combined attacks.
If Burget's eval set contains geometric degradations (rotated/cropped
probes), our image system will be materially worse than the CV EER
suggests. Pixel-level degradations we're already immune to.

## Next step

- **Add a rotation-aug config.** E015 already tested rotation aug at
  training time without improvement on clean CV — but E028 proves the
  value is at stress time, not clean time. Re-run E015 with rotation aug
  and evaluate specifically under the E028 stress suite (not just clean
  CV) to see whether it closes the rotation+occlusion gap.
- **Consider a translation-equivariant alternative** (LBP per patch was
  tried in E005 but collapsed on fold 2 due to different reason — session
  shift). Block-mean features could retain robustness without full CNN.
- **Document this honesty** in `dokumentace.pdf`: the image flagship is
  robust to photometric noise, brittle to geometric distortion. Fusion
  with audio should carry the load when image collapses.
- Image flagship stays E007 +All for submission — the fusion system
  E027 still benefits, and we have no evidence the eval set is primarily
  geometric.
