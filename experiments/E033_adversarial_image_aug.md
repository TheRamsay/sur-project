# E033 — Adversarial Image Augmentation for Geometric Robustness

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E007 (image baseline), E028 (image stress test), E030 (TTA rotation)

## Hypothesis

Adversarial geometric augmentation (training on rotations that maximize classification error) will improve robustness to geometric degradations while maintaining or improving clean performance. Inspired by ARoFace paper (2024).

## Setup

- **Modality:** image
- **Data:** train + dev (combined), LOSO 3-fold
- **Features:** PCA 50 + LogReg C=1, 80×80 grayscale
- **Seed:** 67
- **Command / notebook:** `notebooks/E033_adversarial_image_aug.ipynb`
- **Augmentation configs:**
  - `+All`: E007 baseline (flip + brightness + noise)
  - `+AdvRot`: +All + adversarial rotation ±10°
  - `+AdvTrans`: +All + adversarial translation ±5px
  - `+AdvBoth`: +All + both adversarial rotation + translation

## Result

### Clean performance (CV)

| Config | EER mean ± std | min-DCF |
|--------|----------------|---------|
| +All (E007) | 1.53 ± 0.52% | 0.0306 |
| **+AdvRot** | **0.51 ± 0.36%** | **0.0102** |
| +AdvTrans | 3.43 ± 3.10% | 0.0686 |
| +AdvBoth | 3.80 ± 3.21% | 0.0760 |

### Stress test (rotation robustness)

| Config | clean | rot15 | rot25 | occlude |
|--------|-------|-------|-------|---------|
| +All | 0.00% | 13.70% | 34.11% | 0.00% |
| **+AdvRot** | **0.00%** | **1.04%** | **23.12%** | **0.00%** |
| +AdvTrans | 0.00% | 8.28% | 36.56% | 0.00% |
| +AdvBoth | 0.00% | 6.72% | 29.58% | 0.00% |

Note: OOF overall EER on clean data is 0% for all configs (perfect separation). CV mean shows the real generalization gap.

## Interpretation

**Hypothesis confirmed — +AdvRot is a massive improvement:**

1. **Clean performance:** +AdvRot achieves 0.51% EER vs 1.53% for +All — a **3× improvement**! This is the best image result we've seen.

2. **Rotation robustness:** +AdvRot reduces rot15 EER from 13.70% to 1.04% — a **13× improvement**! This directly addresses the geometric brittleness identified in E028.

3. **Why adversarial rotation works:** By training on the "worst-case" rotation for each image, the model learns rotation-invariant features. Standard random rotation augmentation didn't help (E015), but adversarial selection of rotation angles is much more effective.

4. **Adversarial translation hurts:** +AdvTrans and +AdvBoth regress significantly. Translation may corrupt the face position too much for PCA eigenfaces.

5. **Fold pathology eliminated:** +All had fold 0 at 2.08% EER; +AdvRot brings it down to 0.69%. The adversarial training stabilizes across folds.

## Next step

**Adopt +AdvRot as the new image flagship.** Replace E007 (+All) in the fusion system. Expected impact:
- Image-only: 0.97% → 0.51% EER
- Fusion: Should improve trimodal fusion beyond 0.26% OOF

**E038:** Test ensemble UBM for audio to match image improvements.

## Decision

**ADOPTED.** +AdvRot is the new image flagship (E033). Update predict_image.py to use adversarial rotation augmentation during training.
