# E041 — Histogram Equalization for Image Preprocessing

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E033 (adversarial image)

## Hypothesis

Histogram equalization (HE/CLAHE) before PCA will improve session generalization by normalizing lighting conditions, complementing E033's brightness augmentation.

## Setup

- **Modality:** image
- **Preprocessing variants:**
  - `raw`: E033 baseline (no HE)
  - `HE`: Global histogram equalization
  - `CLAHE`: Adaptive CLAHE (clip=2.0, tile=8×8)
  - `CLAHE+aug`: CLAHE + E033 augmentations

## Result

| Config | EER mean ± std | vs E033 |
|--------|----------------|---------|
| **raw (E033)** | **0.97 ± 0.86%** | — |
| HE | 3.01 ± 3.78% | +2.04pp ❌ |
| CLAHE | 2.96 ± 3.26% | +1.99pp ❌ |
| CLAHE+aug | 5.60 ± 3.96% | +4.63pp ❌ |

## Interpretation

**Hypothesis strongly rejected — HE destroys performance:**

1. **HE/CLAHE triple EER:** Both global HE and CLAHE increase EER from 0.97% to ~3% — a 3× degradation.

2. **CLAHE+aug is catastrophic:** Combining CLAHE with E033 augmentations yields 5.60% EER — the worst image result we've seen.

3. **Why HE fails:** 
   - E033's brightness augmentation already handles lighting variation effectively
   - HE/CLAHE distorts the pixel intensity distribution that PCA eigenfaces rely on
   - Adaptive histogram equalization introduces local artifacts that confuse the global PCA representation

4. **Raw pixels are optimal:** The raw 80×80 pixel intensities (with brightness jitter augmentation) provide the best signal for PCA. No preprocessing needed.

## Decision

**REJECTED.** Histogram equalization not adopted. E033 raw pixel approach confirmed optimal.
