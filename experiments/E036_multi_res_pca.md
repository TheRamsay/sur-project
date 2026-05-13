# E036 — Multi-Resolution PCA for Face Recognition

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E007 (PCA baseline)

## Hypothesis

Multi-resolution PCA (pyramid features at 80×80, 40×40, 20×20) will capture both coarse facial structure and fine details better than single-resolution PCA, improving session generalization.

## Setup

- **Modality:** image
- **Data:** train + dev, LOSO 3-fold
- **Features:** Multi-resolution PCA + LogReg C=1
- **Seed:** 67
- **Configs:**
  - `single`: 80×80 only (E007 baseline approach)
  - `pyramid_2`: 80×80 (50d) + 40×40 (25d) = 75d
  - `pyramid_3`: 80×80 (50d) + 40×40 (25d) + 20×20 (12d) = 87d

## Result

| Config | EER mean ± std | min-DCF |
|--------|----------------|---------|
| single | 2.27 ± 0.85% | 0.0454 |
| pyramid_2 | 2.22 ± 1.42% | 0.0444 |
| pyramid_3 | 1.53 ± 0.52% | 0.0306 |

## Interpretation

**Hypothesis partially confirmed but not better than E007:**

1. **pyramid_3 helps:** pyramid_3 (1.53%) improves over single (2.27%), showing multi-resolution features have value.

2. **But worse than E007:** E007 with +All augmentation achieves 0.97% EER, beating pyramid_3 (1.53%). The standard augmentation (flip + brightness + noise) is more effective than multi-resolution.

3. **Why multi-res doesn't win:** PCA eigenfaces already act as low-pass filters, capturing global structure. Adding explicit multi-resolution doesn't add complementary information beyond what +All augmentation provides.

## Decision

**REJECTED.** Multi-resolution PCA not adopted. E007 +All augmentation remains superior. However, pyramid_3's improvement over single suggests multi-scale features may be worth revisiting with better fusion strategies.
