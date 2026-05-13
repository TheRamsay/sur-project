# E032 — Quality-Aware Multimodal Fusion

- **Date:** 2026-04-21
- **Author:** TheRamsay
- **Related experiments:** E027 (trimodal fusion baseline), E028 (image stress test)

## Hypothesis

Quality-aware fusion (dynamically weighting modalities per-sample based on quality metrics) will reduce OOF EER compared to fixed-weight fusion (E027), especially for samples with geometric image degradations or noisy audio. The QME paper (2025) shows quality-guided score fusion improves multimodal biometric recognition by downweighting low-quality inputs.

**Quality metrics:**
- Image: Laplacian variance (sharpness), brightness deviation from training mean
- Audio: SNR estimate (signal power / high-freq noise power), zero-crossing rate

**Fusion strategies tested:**
1. `fixed` — E027 baseline (fixed weights: mfcc=0.02, lpcc=0.60, image=0.38)
2. `quality_softmax` — quality scores → softmax weights per sample
3. `quality_threshold` — drop modality if quality < threshold, renormalize
4. `quality_linear` — linear scaling: w_m = base_w * (q_m / mean_q_m)

## Setup

- **Modality:** fusion (trimodal: MFCC + LPCC + image)
- **Data:** train + dev (combined), LOSO 3-fold (src/data/splits.py)
- **Features:** 
  - MFCC: 13 + Δ + ΔΔ + CMN (E008)
  - LPCC: 13 + Δ + ΔΔ + CMN, LPC order=12 (E025)
  - Image: PCA 50 + LogReg (E007)
- **Model:** UBM-32 + MAP r=16 (audio), PCA-50 + LogReg C=1 (image)
- **Seed:** 67
- **Command / notebook:** `notebooks/E032_quality_aware_fusion.ipynb`
- **Augmentation:** +All (E007/E008) + pitch (E025)

## Result

### Quality metric statistics

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Image sharpness (Laplacian var) | 140.4 | 1456.6 | 562.5 |
| Image brightness | 61.3 | 170.1 | 111.5 |
| Audio SNR (dB) | 11.8 | 36.0 | 23.8 |
| Audio ZCR | 0.0305 | 0.1192 | 0.0580 |

Normalized quality scores:
- Image quality: 0.236 – 0.989 (mean=0.487)
- Audio quality: 0.082 – 0.986 (mean=0.553)

### OOF overall EER comparison

| Strategy | OOF EER [%] | OOF min-DCF | Threshold |
|----------|-------------|-------------|-----------|
| E027_fixed (baseline) | 0.78 | 0.0156 | 0.342 |
| **quality_softmax** | **0.78** | **0.0156** | 0.505 |
| quality_threshold_0.3 | 0.78 | 0.0208 | 0.342 |
| quality_linear | 4.01 | 0.0521 | -0.020 |
| quality_threshold_0.5 | 5.16 | 0.1031 | 0.750 |

### Per-fold EER (for reference)

| Fold | E027_fixed | quality_softmax |
|------|------------|-----------------|
| 0    | 1.39%      | 1.39%           |
| 1    | 0.00%      | 0.00%           |
| 2    | 0.00%      | 0.00%           |
| mean ± std | 0.46 ± 0.65% | 0.46 ± 0.65% |

### Decision changes analysis

- Samples with changed hard decisions: 6 / 222 (2.7%)
- Changed samples have slightly lower image quality (0.423 vs 0.489)
- Error rate on changed samples: 100% (all 6 changes were wrong)

## Interpretation

**Hypothesis rejected.** Quality-aware fusion does **not** improve clean-data performance:

1. **quality_softmax matches baseline** (0.78% EER) — softmax normalization effectively learns to recover the fixed weights on average, providing no gain on clean data.

2. **quality_threshold and quality_linear regress** — hard thresholds (0.3, 0.5) and linear scaling both hurt performance. This suggests:
   - The quality metrics (sharpness, SNR) don't correlate well with actual recognition difficulty on clean data
   - Dropping modalities based on quality loses valuable signal

3. **Why no improvement?** The clean evaluation data has:
   - High image quality across all samples (sharpness range is narrow)
   - Consistent audio quality (SNR 11-36 dB is acceptable range)
   - No severe degradations that would trigger quality-based downweighting

**Key insight:** Quality-aware fusion is designed for **distribution shift** scenarios (noisy eval data, geometric degradations). On clean data with consistent quality, fixed weights optimized via grid search are already near-optimal.

**The 100% error rate on changed decisions** is concerning — it means the quality metrics are misleading the fusion on those 6 samples. This suggests the quality estimators need refinement before deployment.

## Next step

**E033: Test quality-aware fusion on stressed data.** Apply geometric degradations (rotation, occlusion) to images and noise to audio, then re-evaluate quality-aware fusion. Hypothesis: quality_softmax will outperform fixed weights when quality varies significantly across samples.

Alternative: Improve quality metrics (e.g., face detection confidence, better SNR estimator, no-reference image quality assessment like BRISQUE).

## Conclusion

Quality-aware fusion is **not adopted** for the final system. Fixed weights (E027) remain the fusion flagship. The quality-aware approach may be revisited if evaluation data shows severe quality variation.
