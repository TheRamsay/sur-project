# E030 — Image TTA: rotation averaging at inference

- **Date:** 2026-04-20
- **Author:** Dominik Huml
- **Related experiments:** E007 (image flagship), E028 (stress test revealing rotation brittleness)

## Hypothesis

E028 showed the eigenface subspace collapses under geometric rotation (±15° → 7.31% EER).
Test-time augmentation (TTA) — scoring each val image at multiple rotation angles and
averaging the log-odds — should partially compensate without any retraining.
For a rotated test image, at least one TTA view will be closer to canonical orientation
and yield a better score; non-targets gain no directional benefit. Expect rotation ±15°
EER to drop toward 2–4%, clean EER to stay within ±0.5pp of 0.97%.

## Setup

- **Modality:** image
- **Data:** train + dev (combined), LOSO 3-fold
- **Features:** 80×80 grayscale → StandardScaler → PCA 50
- **Model:** LogReg C=1, trained with E007 +All aug (flip / brightness / noise) — **no change to training**
- **TTA angles:** −20°, −10°, 0°, +10°, +20° (5 views, average log-odds)
- **Seed:** 67
- **Command / notebook:** `notebooks/E030_image_tta_rotation.ipynb`
- **Augmentation (train):** same as E007 +All

## Result

### EER under stress: single vs TTA (within-run comparison)

Note: stochastic stresses (rotation, occlusion) use Python `hash(k)` for seeding, which
is process-randomized, so absolute numbers differ from E028 but the single↔TTA comparison
within this run is fair.

| Stress | Single | TTA | Δ |
| ------ | ------ | --- | - |
| Clean | 0.97% | 1.25% | +0.28pp ❌ |
| Heavy JPEG q=15 | 1.25% | 1.25% | 0.00pp ✓ |
| Heavy blur σ=3.0 | 1.71% | 1.25% | −0.46pp ✓ |
| Rotation ±15° | 2.41% | 2.18% | −0.23pp (noise) |
| Rotation ±25° | 17.22% | 14.31% | −2.91pp ✓ |
| Occlusion 20×20 | 19.58% | 20.42% | +0.84pp ❌ |
| Downsample 40→80 | 0.97% | 1.25% | +0.28pp ❌ |
| All combined | 26.62% | 30.93% | +4.31pp ❌ |

## Interpretation

Hypothesis partially holds but not enough to adopt TTA.

TTA helps rotation ±25° (−2.91pp) and blur (−0.46pp), but:
- **Hurts clean EER** (+0.28pp, 0.97→1.25%) — averaging over rotated views of an
  upright image introduces subspace noise.
- **Doesn't help occlusion** (+0.84pp worse) — rotating an occluded image doesn't
  move the black patch, so no TTA angle recovers the missing information.
- **All-combined worsens notably** (+4.31pp) — when multiple stresses stack, TTA
  amplifies the damage rather than mitigating it.

The root cause is fundamental to PCA eigenfaces: averaging log-odds over rotated views
of a model that *never saw rotations during training* still averages bad scores. At most
one TTA view gets close to canonical; the others drag the average down. The improvement
requires training-time rotation augmentation, which E015 showed hurts CV EER.

**Verdict: do not adopt rotation TTA.** Keep existing flip-only TTA in `predict_image.py`.
Fusion (E027) remains the geometric robustness safety net.

## Next step

No further image experiments. E007 +All stays as the image flagship. Focus on
`dokumentace.pdf` and generating the 6 result files on eval data (2026-05-03).
