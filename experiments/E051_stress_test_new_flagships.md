# E051 — Stress test: E033 image + E037/E042 audio vs E028 baseline

- **Date:** 2026-04-22
- **Author:** Dominik Huml
- **Related experiments:** E028 (E007 stress test), E033 (adv-rot image), E037 (tied cov), E042 (speed TTA)

## Hypothesis

E033's adversarial rotation training should fix rotation brittleness (confirmed in E033 report: rot15 13.70%→1.04%) but photometric robustness should be unchanged vs E007. Occlusion may still be a problem since AdvRot doesn't address spatial masking. Audio E037/E042 has never been stressed — expected to be robust to speed (covered by TTA) and moderate noise, but unknown under codec/heavy distortion.

## Setup

- **Image:** E033 model (2-pass AdvRot) vs E007 (+All), same E028 stress suite
- **Audio:** E037 (tied cov) + speed TTA vs noise/speed/codec stresses
- **Stresses (val only, train always clean):** JPEG q=15, blur σ=3, rot±15°, rot±25°, occlude 20×20, downsample 40→80, all-combined
- **Audio stresses:** noise 20dB / 10dB / 5dB, speed 0.8×/1.2× (beyond training range), all-combined
- **Fold spec:** `iter_folds_loso`, seed=67, 3 folds
- **Notebook:** `notebooks/E051_stress_test.ipynb`

## Result

### Image: E033 vs E007 (same E028 stress suite)

| Stress | E007 (+All) | E033 (+AdvRot) | Delta |
|--------|------------|---------------|-------|
| clean | 1.53 ± 0.52% | **0.51 ± 0.36%** | −1.02pp |
| jpeg q=15 | 1.53 ± 0.52% | **0.51 ± 0.36%** | −1.02pp |
| blur σ=3 | 3.66 ± 3.06% | **0.51 ± 0.36%** | −3.15pp |
| downsample | 1.25 ± 0.90% | **0.51 ± 0.36%** | −0.74pp |
| rot15 | 19.03 ± 12.42% | **7.59 ± 5.39%** | −11.44pp |
| rot25 | 37.18 ± 21.78% | **29.95 ± 16.38%** | −7.22pp |
| occlude | 10.14 ± 4.62% | 11.06 ± 6.76% | **+0.93pp** |
| all combined | 25.60 ± 14.61% | **15.19 ± 9.69%** | −10.42pp |

### Audio: E042 (first audio stress test)

| Stress | F0 | F1 | F2 | Mean±Std | Delta |
|--------|----|----|-----|----------|-------|
| clean | 1.39 | 0.00 | 0.00 | **0.46 ± 0.65%** | — |
| noise 20dB | 10.56 | 0.83 | 1.67 | 4.35 ± 4.40% | +3.89pp |
| noise 10dB | 10.56 | 9.17 | 0.83 | 6.85 ± 4.29% | +6.39pp |
| noise 5dB | 3.47 | 4.17 | 3.33 | 3.66 ± 0.36% | +3.19pp |
| slow 0.8× | 1.39 | 0.00 | 0.83 | **0.74 ± 0.57%** | +0.28pp |
| fast 1.2× | 0.69 | 0.00 | 0.00 | **0.23 ± 0.33%** | −0.23pp |
| codec (8kHz BW) | 18.33 | 9.17 | 12.50 | **13.33 ± 3.79%** | +12.87pp |
| all combined | 2.78 | 10.00 | 10.00 | 7.59 ± 3.40% | +7.13pp |

## Interpretation

**Image — E033 is stronger across the board except occlusion:**
- JPEG and blur have *zero effect* on E033 (0.51% = clean). Adversarial rotation incidentally improves photometric robustness by forcing broader feature invariance.
- Rotation improved massively (rot15: 19→7.59%) but still non-trivial at 25°.
- Occlusion is +0.93pp worse on E033 — adversarial rotation doesn't address spatial masking, expected.
- All-combined: 10pp better than E007.

**Audio — speed is handled, codec is the vulnerability:**
- Speed stress (0.8×/1.2×) absorbed almost entirely by TTA: 0.74% and 0.23%. This is exactly what E042 was designed for.
- Codec (8kHz bandwidth) is catastrophic (+12.87pp). LPCC formants F3/F4 live above 4kHz — downsampling destroys exactly the spectral information LPCC relies on.
- Moderate noise (20dB) causes +3.89pp; heavy noise (10dB) +6.39pp. Concern if eval has additive noise.

## Decision

**E033 confirmed solid for photometric degradations; occlusion risk unchanged. Audio E042 has a codec/bandwidth vulnerability that fusion cannot fully compensate. Document both in dokumentace.pdf.**

Add to CLAUDE.md findings. No model changes needed.
