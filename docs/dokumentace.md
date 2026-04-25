---
title: "SUR 2025/2026: Target-person detector for `m431`"
author: "Dominik Huml (`xhumld00`)"
date: "May 2026"
geometry: margin=2cm
fontsize: 10pt
---

## 1. Task

The goal is a binary detector for the target speaker `m431`. Each evaluation sample is a face image and a paired voice recording sharing the same filename stem, so multimodal fusion at the sample level is straightforward. Three systems are required (image-only, audio-only, and a multimodal fusion of the two), each producing a real-valued score and a hard decision at prior 0.5. No pretrained models or outside data are allowed. Everything was trained from scratch on the provided 222-sample corpus with augmentation only.

## 2. Validation strategy

A plain K-Fold split would leak both session identity and speaker identity. The target appears in only three sessions, and most non-target speakers in just one, so a random partition cross-contaminates train and val. I use a 3-fold leave-one-out scheme that respects this structure. For target rows the grouping variable is session ID (`01`, `02`, `03`), so each fold holds out one whole recording session of `m431`. For non-target rows the grouping variable is speaker identity, so each fold holds out 2–3 unseen people. Both groupings feed a single `GroupKFold` over a virtual column with disjoint ID spaces (`src/data/splits.py`). All three systems share the same fold assignment, which lets me stack the fusion on out-of-fold (OOF) scores.

I report EER and min-DCF at prior 0.5, mean ± std across the three folds. For fusion I additionally pool all 222 samples into a single OOF ranking and report the overall EER. Augmentation is applied only to the train fold and val samples are always raw, and every fitted statistic (per-pixel scaler, PCA basis, UBM, MAP target model) is refit per fold. As a sanity check I ran a permutation test (E029): shuffling the train labels before retraining brought val EER back to 49 % for the image system and 55 % for the audio system, both inside the chance window. The models learn from labels, not from leakage.

## 3. Audio system

### 3.1 Front-end

Speaker verification on roughly 170 training utterances is in the small-data regime, where the inductive bias of the front-end matters more than its expressive power. LPCC parameterises the vocal-tract transfer function directly, via an all-pole LPC model followed by the cepstral recursion. That is the physical quantity which distinguishes one speaker from another. The Mel filterbank used by MFCC, by contrast, was designed for speech-content recognition, where speaker identity is precisely the nuisance variable to suppress.

I ablated MFCC, FBank, SDC, PLP and LPCC under identical UBM-MAP conditions. LPCC 13 + Δ + ΔΔ (LPC order 12) won on both EER and min-DCF: 4.21 % EER for MFCC, 3.33 % for LPCC. The higher-dimensional alternatives (FBank 120-d, SDC 104-d) over-parameterised the GMM-32 backbone for this dataset size, and PLP's Bark filterbank with cube-root compression produced fold-2 collapse. LPC order 12 had a 2.5 pp moat over its neighbours (orders 8, 10, 14, 16, 20 all worse by at least that much), which matches the `fs/1000 + 2` rule for 16 kHz speech.

Each utterance is normalised with **cepstral mean normalisation (CMN)**, i.e. the per-utterance mean of every cepstral dimension is subtracted before it reaches the GMM. CMN removes the convolutional channel response — microphone, room, recording level — that is roughly constant across one utterance but varies across sessions. CMVN (variance normalisation on top of CMN) was tested in E012 and regressed by 2 pp, because the UBM's covariance already absorbs per-utterance scale and the extra division throws away discriminative variance. CMN is therefore retained, CMVN is not.

### 3.2 Classifier: UBM-MAP with tied covariance

The backbone is the standard UBM-MAP setup. A 32-component GMM is trained on all non-target training frames to get the UBM, then the target model is produced by MAP-adapting only the UBM means with relevance factor `r = 16`. For an utterance with frames `x_1, …, x_T`, the score is the per-frame log-likelihood ratio averaged over the utterance:

$$\mathrm{LLR}(x_{1:T}) = \frac{1}{T} \sum_{t=1}^{T} \bigl[ \log p(x_t \mid \lambda_\text{target}) - \log p(x_t \mid \lambda_\text{UBM}) \bigr]$$

The relevance factor `r` controls how much the target model trusts the per-speaker observations versus the UBM prior — small `r` means the adapted means jump toward the target frames, large `r` keeps them close to the UBM. I rejected UBM-64 in E010 (over-fits) and confirmed that `r ∈ {4, 8, 16}` (i.e. `r` set in turn to 4, 8 and 16) is a flat plateau on both diagonal and tied covariance (E013, E044), so `r = 16` stays as the default.

The biggest single jump in the entire audio track came from changing the GMM covariance type (E037).

| Covariance | EER mean ± std    | min-DCF | Parameters |
|------------|------------------:|--------:|-----------:|
| spherical  | 3.89 ± 3.75 %     | 0.0778  | 32         |
| diagonal   | 4.35 ± 4.40 %     | 0.0870  | 1 248      |
| **tied**   | **0.69 ± 0.98 %** | **0.0139** | **1 521** |
| full       | 1.48 ± 0.92 %     | 0.0296  | 48 672     |

![GMM covariance ablation: tied covariance is 6× better than diagonal at only ~20 % more parameters.](figures/fig1_covariance_ablation.pdf){ width=85% }

The reason this matters is that LPCC coefficients are strongly correlated, since adjacent cepstra share formant structure. Diagonal covariance ignores those correlations and forces the GMM to fit each dimension as if it were independent. Full per-component covariance captures the correlations but with 48 672 parameters, far too many for 222 samples, and starts overfitting. Tied is the right compromise: it captures the off-diagonal structure once and shares it across all 32 components.

### 3.3 Augmentation and TTA

At training time I apply two augmentations on top of the LPCC frames. The first is pitch shift in the range ±1–2 semitones (E025), which trains the model to ignore F0 while preserving formant ratios. This augmentation works specifically for LPCC, and the same shift hurt MFCC in E014 because the Mel filterbank already smooths pitch information. The second is codec simulation (E052): each utterance is downsampled to 8 kHz and resampled back, which destroys frequencies above 4 kHz, where many of the discriminative formants live.

Codec robustness was the motivating problem. The non-augmented audio system reached 0.46 % clean EER but degraded to 13.33 % when the val set was bandwidth-limited at scoring time. Adding the codec simulation as a training augmentation closed most of that gap with no clean-side regression at all.

| System                 | Clean EER         | Codec-stressed EER |
|------------------------|------------------:|-------------------:|
| E042 baseline          | 0.46 ± 0.65 %     | 13.33 ± 3.79 %     |
| **+ codec aug (E052)** | **0.46 ± 0.65 %** | **3.33 ± 4.14 %**  |

The training UBM now sees a mixture of clean and bandwidth-limited frames, so its covariance covers both regimes and the LLR remains informative when the high-frequency formants are missing.

At inference I apply speed-only test-time augmentation (E031). Each utterance is scored at the original speed, 0.9×, and 1.1×, and the three LLRs are averaged. The pitch/speed asymmetry is informative. Speed perturbation retimes the signal but preserves the spectral envelope, so the LPC all-pole filter (and hence LPCC) is invariant, and averaging across speeds is averaging over benign perturbations. Pitch shift, on the other hand, alters the source-filter relationship: F0 leaks into the LPC residual, the cepstral coefficients drift onto a manifold the UBM was never trained on, and fold 0 collapses to 9.86 %. Pitch is therefore the right augmentation at training time (it widens the UBM's support) but the wrong one at inference. I also tested two-second prefix truncation (E053) and rejected it: CMN already suppresses stationary pre-speech noise, and the apparent codec-EER improvement turned out to be a fold-reshuffling artefact.

The final audio system is LPCC + tied-covariance UBM + MAP r = 16 + pitch and codec augmentation + speed TTA, reaching **0.46 ± 0.65 % EER** at **min-DCF 0.0092**.

## 4. Image system

### 4.1 Features and classifier

The image classifier is intentionally simple: 80×80 grayscale crops, per-pixel standardisation, a 50-dimensional PCA projection, and logistic regression with C = 1. Local Binary Patterns (E005) failed catastrophically. Fold 2 collapsed to 45 % EER under session-to-session lighting shifts, since LBP histograms are sensitive to global brightness. Fisherfaces (E006) are capped at a 1-D projection given two identity classes and lose to logistic regression in the 50-D PCA space. I swept `n_pca` from 20 to 150 (E011) and confirmed n = 50 is optimal, swept the regulariser (E040) and confirmed C = 1 (with L1 catastrophic), tested histogram equalisation and CLAHE (both tripled EER, E041), and tried pyramid multi-scale PCA (E036, lost to flat PCA with good augmentation). Plain PCA on standardised pixels was the empirical ceiling for this dataset.

### 4.2 Augmentation: two-pass adversarial training (E033)

Augmentation is what made the image system competitive, and it runs in two passes.

**Pass 1.** I add each training image in four versions: the original, a horizontal flip, the image scaled by a random brightness factor in U[0.7, 1.3], and the image plus Gaussian pixel noise at σ = 15. A fresh PCA + LogReg is fit on the combined set. Brightness jitter is the key contributor here: session 03 (the held-out target session in fold 2) has systematically different lighting from sessions 01 and 02, and brightness scaling is the only standard augmentation that crosses that gap.

**Pass 2.** For each training image I query the Pass-1 model at five angles in [−10°, +10°] and pick the angle with minimum |logit|, i.e. the angle of maximum model uncertainty. A rotated copy at that angle is added to the training set, and PCA + LogReg is refit on the expanded set. The mechanism explains why this works where random rotation does not (E015). Random rotation samples uniformly from the rotation manifold and PCA fits the average. The easy angles dominate and the hard ones stay unmodelled. Adversarial selection inverts that. Each sample contributes the rotation the current eigenspace fails on, so principal components are reallocated towards the directions of model uncertainty. It is hard-negative mining, applied to PCA.

The figure below summarises the head-to-head against the +All baseline (E051 stress test, same seed and protocol for both rows). E033 is photometrically bulletproof and roughly halves rotation EER. JPEG, blur, contrast, HE/CLAHE and Cutout 20×20 (E052) all regressed when added on top, so the current set sits at the empirical ceiling.

![Image stress test: AdvRot is photometrically bulletproof (JPEG, blur, downsample stay at clean 0.51 %) and halves the geometric weaknesses.](figures/alt_a_image_robustness.pdf){ width=90% }

At inference I average the original and its horizontal flip. Rotation TTA was tested and rejected (E030) because it corrupts the eigenface projection at scoring time and raises clean EER. E043 (flip + small-rotation TTA) appeared to help by 0.23 pp but later replication runs (E049) failed to reproduce the result, so I did not adopt it.

The final image system reaches **0.51 ± 0.36 % EER** at **min-DCF 0.0102**.

## 5. Fusion

Each modality produces a raw LLR (audio) or logit (image) that lives on its own scale. To put the streams on a common axis before combining, I fit a Platt calibration per modality on its OOF scores: a one-feature logistic regression with C = 1e6 and `class_weight='balanced'`. The fused score is then a weighted sum on the 2-simplex (`w_mfcc + w_lpcc + w_image = 1`, all weights non-negative), with the weights chosen on a 51×51 grid that directly minimises OOF EER. I also tried logistic-regression fusion. It is MLE-optimal but not EER-optimal, and the near-rank-2 design matrix (r(MFCC, LPCC) = 0.843) limits what it can learn.

| Fusion                              | OOF EER             | min-DCF | Notes                            |
|-------------------------------------|--------------------:|--------:|----------------------------------|
| MFCC + image (E009)                 | 3.75 %              | 0.0750  | bimodal baseline                 |
| LPCC + image (E023)                 | 0.52 %              | 0.0104  | LPCC replaces MFCC               |
| MFCC + LPCC + image (E026/E027)     | 0.26 %              | 0.0052  | trimodal, MFCC as tiebreaker     |
| **E052 + E033 + MFCC (E039)**       | **0.26 % (0 errors)** | **0.0052** | **new backbones, 0 of 222**  |

![DET curves for the three modalities and the trimodal fusion. The fusion star sits at the lower-left corner of the visible region. On this CV split it makes 0 errors out of 222 samples.](figures/fig6_det_curve.pdf){ width=80% }

The grid converges to `w_image = 0.66, w_lpcc = 0.34, w_mfcc ≈ 0.00`. Image dominates because it is the lower-EER modality. LPCC contributes complementary signal: 0 of 222 samples are misranked by both audio and image at their respective thresholds, so even though each modality misses a few targets they don't miss the same ones. MFCC's weight collapses to zero because MFCC and LPCC OOF scores are correlated at r = 0.843. Both are cepstral representations of the same vocal-tract physics via different front-ends, so once tied-covariance LPCC enters the fusion the third stream is rank-deficient. I keep MFCC in the grid so that the optimisation can re-verify this per run. If a future LPCC weakness appears, MFCC will pick up weight automatically. Quality-aware gating (E032), product-rule fusion (E046, E048) and score-level ensembles (E045) all regressed against simple weighted averaging.

![Audio × image complementarity at per-stream EER thresholds: 15 samples are rescued by audio alone, 5 by image alone, none are missed by both. This disjoint failure pattern is exactly what makes weighted-sum fusion reach 0 errors.](figures/alt_c_complementarity.pdf){ width=78% }

## 6. Generalization and overfitting defences

The brief specifically asks how I handle generalization and limit overfitting. I frame the answer as five concrete risks with the corresponding defence for each.

**Risk 1: model capacity exceeds data size.** With 222 samples, three folds and dozens of moving parts, this is the obvious one. The defences are scale-matched. UBM-32 leaves roughly 5 400 frames per component, which is well into the asymptotic regime, and I rejected UBM-64 in E010 because it regressed. PCA-50 keeps the linear classifier in a subspace that the 2-class boundary can fit reliably, and I confirmed the choice with a sweep (E011). Tied covariance shares 1 521 parameters across 32 GMM components, while full per-component covariance has 48 672 and overfits.

**Risk 2: validation leaks session or speaker identity.** Defence: 3-fold session/speaker-aware LOSO splits as described in §2. Each fold holds out one full target session and 2–3 unseen non-target speakers. Per-fold scaler, PCA basis, UBM and MAP target model are refit. Augmentation is applied only to the train fold, and val samples are always raw.

**Risk 3: hidden label leakage via auxiliary channels** (filename order, file size, anything correlated with class). Defence: a permutation test (E029). I shuffled the training labels before retraining the image and audio flagships and re-evaluated on val with true labels. Image EER returned to 49 % and audio to 55 %, both inside the chance window for binary classification. The 48–55 pp gap to the unshuffled flagships is what makes the result trustworthy.

**Risk 4: post-hoc rationalisation across multi-axis changes.** With 53 experiments on a single benchmark, the temptation to bundle changes is real. Defence: each experiment moves exactly one knob, with the hypothesis written into the experiment file before the run, and failed runs stay in the log. Findings are attributable to the changed axis, and the negative results (LBP, FBank, CMVN, HE/CLAHE, product-rule fusion, …) form part of the rationale.

**Risk 5: distribution shift at evaluation time.** The brief warns that the evaluation data will contain noise, codec and quality changes. Defence: stress testing on val (E028, E051, E052). The image flagship is photometrically bulletproof: JPEG q = 15, blur σ = 3 and downsample 40→80 all stay at the clean 0.51 %. The remaining weaknesses are geometric (rotation > 15°: 7.6 %) and occlusion (11 %). The audio flagship absorbs speed perturbations through TTA. Codec stress was the original 13.33 % failure mode and is the reason E052 added codec augmentation, which dropped it to 3.33 % at zero clean cost. Moderate noise (≤ 10 dB SNR) is manageable.

## 7. Results

| System                                                                | CV EER mean ± std         | CV min-DCF |
|-----------------------------------------------------------------------|--------------------------:|-----------:|
| Audio baseline (MFCC + GMM, E001)                                     | 17.92 ± 7.81 %            | 0.2250     |
| Audio (MFCC + UBM/MAP + aug, E008)                                    | 4.21 ± 3.11 %             | 0.0509     |
| Audio (LPCC + tied cov, E037)                                         | 0.69 ± 0.98 %             | 0.0139     |
| **Audio flagship (LPCC + tied + pitch&codec aug + speed TTA, E052)**  | **0.46 ± 0.65 %**         | **0.0092** |
| Image baseline (PCA + LogReg, E004)                                   | 4.49 ± 4.26 %             | 0.0565     |
| Image (PCA + LogReg + aug, E007)                                      | 0.97 ± 0.86 %             | 0.0194     |
| **Image flagship (PCA + LogReg + aug + adv-rot, E033)**               | **0.51 ± 0.36 %**         | **0.0102** |
| **Fusion flagship (trimodal E052 + E033 + MFCC, E039)**               | **0.26 % OOF (0 errors)** | **0.0052** |

The arc behind these numbers is: MFCC + GMM baseline, then UBM + MAP adaptation, then better features (LPCC), then tied covariance, then train-time codec robustness, then adversarial rotation augmentation on the image side, and finally trimodal score-level fusion.

![Project arc across 53 experiments: each modality flagship dropped roughly two orders of magnitude from its anchor. Audio: 17.92 → 0.46 % (−97 %). Image: 4.49 → 0.51 % (−89 %). Fusion: 3.75 → 0.26 % (−93 %).](figures/alt_b_progression.pdf){ width=95% }

### Reproduction

All systems run from a clean clone with

```
uv sync
uv run predict_audio.py   --eval-dir <dir> --output results/audio_lpcc_tied_codecaug.txt
uv run predict_image.py   --eval-dir <dir> --output results/image_pca_adv_rot.txt
uv run predict_fusion.py  --eval-dir <dir> --output results/fusion_trimodal.txt
```

Each output file contains one line per evaluation sample with three whitespace-separated fields: stem, score (higher = more confident target), and a hard decision (`1` = target, `0` = non-target) thresholded at the Bayes optimum for prior 0.5, calibrated on OOF min-DCF.
