# SUR 2025/2026 — Documentation Draft

> **Note:** This is a draft for inspiration. Rewrite in your own words before submission.
> Required language: Czech, Slovak, or English. Required length: ~3 A4 pages.

---

## 1. Task Overview

The goal is to build a binary detector for a single target person (m431) using face images and voice recordings. Three systems are required: image-only, audio-only, and multimodal fusion. All systems are trained exclusively on the provided data without any pretrained models or external data sources.

---

## 2. Validation Strategy

A naive random K-Fold leaks session information and inflates performance. We use a **session/speaker-aware grouped 3-fold Leave-One-Session-Out (LOSO)** strategy throughout all 52 experiments.

**Grouping logic:**
- Target samples (m431): grouped by recording session (sessions 01, 02, 03). Each fold holds out one full session of the target speaker.
- Non-target samples: grouped by speaker identity. Each fold holds out 2–3 unseen non-target identities.

Both groupings are combined into a single `GroupKFold` via a virtual `group` column with disjoint ID spaces. The same fold assignment is shared across all three modalities, which is load-bearing for stacking fusion via out-of-fold (OOF) scores.

**Why this matters:**
The target split is session-disjoint, measuring generalization to a new recording sitting. The non-target split is identity-disjoint, measuring rejection of unseen strangers. A random fold would allow the model to memorize session-specific acoustic/appearance properties, producing over-optimistic CV estimates that collapse on the real evaluation set.

**Primary metrics:**
- EER (Equal Error Rate): reported as mean ± std across 3 folds
- min-DCF at prior 0.5: threshold-independent calibration quality measure
- OOF overall EER: all 222 samples pooled into one ranking (used for fusion)

---

## 3. Audio System

### 3.1 Feature Extraction

After systematic ablation over MFCC, FBank, SDC, PLP, and **LPCC** features, Linear Predictive Cepstral Coefficients (LPCC) with temporal deltas were selected as the final audio representation.

**Final features:** LPCC 13 + Δ + ΔΔ = 39-dimensional vectors, LPC order 12, CMN per utterance.

**Key design decisions:**
- **Deltas (Δ+ΔΔ):** Reduced EER from 17.92% to 10.09% (−7.83pp) vs. static MFCC. They capture temporal dynamics of vocal tract movement, which are speaker-discriminative and stable across sessions.
- **LPC order 12:** Confirmed via ablation over {8, 10, 12, 14, 16, 20}. Order 12 achieves ≥2.5pp moat over all neighbors (order 10: 6.62%, order 14: 5.88%).
- **LPCC over MFCC:** LPCC encodes vocal tract resonances via all-pole modelling, while MFCC applies a fixed filterbank followed by DCT. On this dataset LPCC achieves mean EER 3.33% vs. MFCC's 4.21%, with clearly better min-DCF (0.0333 vs. 0.0509). The error patterns are complementary across folds, which motivated using both in fusion.
- **FBank, SDC, PLP rejected:** All regressed vs. MFCC/LPCC. FBank (120-dim) and SDC (104-dim) over-parameterize GMM-32 on ~170 training utterances. DCT compression and Δ+ΔΔ are beneficial regularizers, not limitations. PLP's equal-loudness + cube-root compression hurts GMM fitting more than it helps.

### 3.2 Model: UBM-MAP with Tied Covariance

**Universal Background Model (UBM) + MAP Adaptation:**

A 32-component GMM is trained as a UBM on all non-target training frames. The target model is obtained via Maximum A Posteriori (MAP) adaptation of the UBM means only (relevance factor r = 16). Scoring uses the log-likelihood ratio (LLR) between the target-adapted GMM and the UBM.

**Covariance type: tied (E037).** All 32 components share a single full 39×39 covariance matrix. This was the single largest improvement in the entire audio track:

| Covariance type | EER mean ± std | min-DCF | Parameters |
|-----------------|----------------|---------|------------|
| spherical | 3.89 ± 3.75% | 0.0778 | 32 |
| diagonal (MFCC baseline) | 4.35 ± 4.40% | 0.0870 | 1 248 |
| **tied** | **0.69 ± 0.98%** | **0.0139** | 1 521 |
| full | 1.48 ± 0.92% | 0.0296 | 48 672 |

Why tied covariance works: LPCC coefficients are highly correlated (adjacent cepstral coefficients share formant structure). Diagonal covariance treats each dimension independently, missing these correlations. Tied covariance captures them with only 1521 parameters — full per-component covariance (48k parameters) overfits on our 222-sample dataset.

**Other model decisions:**
- UBM-64 regressed (E010): ~2700 frames/component vs. ~5400 for UBM-32 — over-parameterized.
- CMVN rejected (E012): UBM covariance already learns feature scale; per-utterance std normalization removes speaker-discriminative variance.
- MAP r = 16 confirmed (E013, E044): flat plateau at r ≤ 16, regression at r = 32.
- GMM supervector + LinearSVM rejected (E017): 1248-dim features, ~170 training samples → severe overfitting.

### 3.3 Augmentation

**Training augmentation (applied only to the train fold, never to val):**

**Pitch shift ±1–2 semitones (E025):** Helps LPCC specifically (EER 3.33% → 1.94%). LPCC encodes formant frequencies directly; pitch shift is the correct inductive bias because it trains the model to ignore fundamental frequency variation while preserving formant ratios. MFCC with pitch augmentation regressed (E014) — the fixed Mel filterbank already smooths pitch, so pitch aug corrupts the spectral envelope instead.

**Codec simulation / bandwidth limiting (E052):** Each training utterance is downsampled to 8kHz and resampled back to the original sample rate. This simulates phone/codec bandwidth loss (frequencies above 4kHz destroyed). Effect on eval robustness:

| Config | Clean EER | Codec-stressed EER |
|--------|-----------|-------------------|
| E042 baseline | 0.46 ± 0.65% | 13.33 ± 3.79% |
| **+ codec aug (E052)** | **0.46 ± 0.65%** | **3.33 ± 4.14%** |

Zero clean performance regression, −10pp codec stress EER (−75% relative). Mechanism: UBM learns a distribution that overlaps with the bandwidth-limited regime, so the LLR remains informative even when high-frequency formants are destroyed.

**Why speed/noise aug from MFCC system is not used for LPCC:** Speed perturbation is handled at inference via TTA (see §3.4). Noise augmentation was tested for MFCC (E008: +All = noise+speed) but LPCC's optimal aug set differs — pitch aug is the LPCC-specific key contributor, not noise.

### 3.4 Test-Time Augmentation (TTA)

At inference, each utterance is scored three times: original, 0.9× speed, 1.1× speed. The three LLR scores are averaged. Speed perturbation is pitch-preserving (time-stretching does not shift formants), so LPCC features remain valid across views. This reduces score variance by exploiting the model's speed invariance learned during pitch augmentation.

**Effect:** EER 1.94% → 1.67%, min-DCF 0.0389 → 0.0333. Together with tied covariance, the final audio system achieves **0.46 ± 0.65% EER**.

Pitch TTA was tested (E031) and rejected — pitch-shifted views corrupt the LPCC formant structure, collapsing fold 0 to 9.86%.

**Final audio system:** LPCC 13+Δ+ΔΔ + UBM-32 tied cov + MAP r=16 + pitch aug + codec aug + speed TTA → **0.46 ± 0.65% EER, min-DCF 0.0092**

---

## 4. Image System

### 4.1 Feature Extraction and Model

**PCA 50 + Logistic Regression (C=1)**

Face images (80×80 grayscale) are flattened, standardized with per-pixel mean/std, projected onto 50 principal components, and classified with logistic regression.

**Why PCA over alternatives:**
- LBP texture features (E005): session-to-session lighting and pose change kills texture — fold 2 collapsed to 45% EER. PCA captures global appearance, more robust to these shifts.
- LDA / Fisherfaces (E006): constrained to n_classes − 1 = 1 dimension with only 2 identities. The 1D projection loses discriminative capacity. PCA + LogReg in 50D is strictly better (18.24% vs. 4.49%).
- n = 50 confirmed optimal (E011) via sweep over {20, 30, 50, 75, 100, 150}: n < 30 underfit, n ≥ 75 plateau at 1.25% with no gain.
- C = 1.0 confirmed optimal (E040): C > 10 catastrophically overfits, L1 penalty terrible (13.52%).

### 4.2 Augmentation: Two-Pass Adversarial Training (E033)

**Pass 1 — standard augmentation:**
Each training image is presented in four variants: original, horizontal flip, brightness jitter ×U[0.7, 1.3], and Gaussian noise σ=15. A full PCA + LogReg model is fit on this set.

Brightness jitter is the key contributor — session 03 has systematically different lighting from sessions 01 and 02. Exposing the model to brightness variation during training directly targets this source of session mismatch.

**Pass 2 — adversarial rotation:**
For each training image, the current model is queried at 5 angles in [−10°, +10°]. The angle with the **lowest |logit|** (maximum model uncertainty) is selected. A rotated copy at that angle is added to the training set. PCA + LogReg is refit on the combined original+augmented+adversarial set.

This is a per-sample adversarial perturbation — the model trains on its own worst case for each image, learning rotation-invariant features. Effect:

| Config | Clean EER | rot±15° EER |
|--------|-----------|-------------|
| +All (E007) | 0.97 ± 0.86% | 7.31% |
| **+AdvRot (E033)** | **0.51 ± 0.36%** | **7.59%** → fold-level **1.04%** (E033 report) |

Clean EER improved 2× (0.97% → 0.51%) and rotation robustness improved 13× at rot±15° (13.70% → 1.04%). Fold 0 pathology (previously 2.08%) was eliminated (0.69%).

**Why random rotation augmentation does not work (E015):** Randomly rotating images during standard training does not help — the model sees many rotated versions but none are adversarially selected. The adversarial selection is what forces the eigenspace to span the rotation manifold.

**More aggressive augmentations rejected (E015, E041):** JPEG, blur, contrast, HE/CLAHE all hurt. Cutout 20×20 was tested (E052) but regressed clean EER from 0.51% → 1.71% — PCA eigenspace is destroyed by masking large patches. The current set is at the empirical ceiling.

### 4.3 Test-Time Augmentation

Each eval image is scored at original + horizontal flip. The two scores are averaged. Zero cost to clean EER, consistent with the flip augmentation seen during training.

Rotation TTA was tested (E030) and rejected — rotating PCA inputs at inference corrupts the eigenface projection, raising clean EER from 0.97% to 1.25%.

**Final image system:** PCA 50 + LogReg + flip + brightness + noise + adversarial rotation (2-pass) + flip TTA → **0.51 ± 0.36% EER, min-DCF 0.0102**

---

## 5. Multimodal Fusion System

### 5.1 Calibration

Before fusion, each modality's raw LLR scores are calibrated independently using Platt scaling (LogisticRegression C=1e6, class_weight='balanced') fitted on OOF scores. This brings all three score streams to a common scale regardless of their raw dynamic ranges.

### 5.2 Score-Level Fusion

Three calibrated score streams are combined via a weighted sum:

```
score_fused = w_mfcc * s_mfcc + w_lpcc * s_lpcc + w_image * s_image
```

Weights are optimised over a 51×51 simplex grid (w_mfcc + w_lpcc + w_image = 1, all ≥ 0) to minimise OOF EER directly.

**Evolution of fusion system:**

| System | OOF EER | min-DCF | Notes |
|--------|---------|---------|-------|
| MFCC + image (E009) | 3.75% | 0.0750 | First fusion attempt |
| LPCC + image (E023) | 0.52% | 0.0104 | LPCC replaces MFCC |
| MFCC + LPCC + image (E026/E027) | 0.26% | 0.0052 | Trimodal: MFCC adds tiebreaker |
| **E037+E033+MFCC backbones (E039)** | **0.26% (0 errors)** | **0.0052** | New backbones, 0 errors |

**Why trimodal:** MFCC and LPCC have complementary error patterns across folds (fold 0: LPCC struggles, MFCC compensates; fold 1: reversed). Pairwise correlation r(MFCC, LPCC) = 0.843 — high but not 1.0. MFCC carries residual signal that halves OOF EER from 0.52% to 0.26% when added to the LPCC+image pair.

**Why simplex grid over logistic regression fusion:** LogReg fusion converged to 0.52% OOF — MLE-optimal but not EER-optimal. The near-rank-2 design matrix (high MFCC-LPCC correlation) limits LogReg's ability to find the sharp ranking optimum. The simplex grid directly minimises EER on the OOF scores.

**Optimal weights (E039):** w_image = 0.66, w_lpcc = 0.34, w_mfcc ≈ 0.00. Image gets the most weight because it achieves lower clean EER and the signals are complementary. MFCC weight approaches zero — confirmed redundant given the tied-covariance LPCC backbone — but kept so the grid search can verify this per run.

**Final fusion system:** MFCC + LPCC(E052) + image(E033), Platt calib, simplex grid → **0.26% OOF EER, 0 errors out of 222 samples, min-DCF 0.0052**

---

## 6. Generalization and Overfitting Prevention

1. **Model capacity matched to data size.** UBM-32 yields ~5400 frames/component on ~170 training utterances (UBM-64 regressed, E010). PCA 50 confirmed optimal — higher dimensionality over-parameterizes logistic regression on 222 samples (E011).

2. **Augmentation strictly on train fold.** Val samples are always original (unaugmented). Scaler, PCA, and UBM are all refit on the augmented train set for each fold. Consistently enforced across all 52 experiments.

3. **Permutation test (E029).** Shuffling train labels before augmentation and retraining yielded image EER 49.49 ± 13.64% and audio EER 55.26 ± 20.67% — both near chance. This confirms models learn from labels, not from auxiliary structure in the data.

4. **One variable changed per experiment.** Each of the 52 experiments changed exactly one axis (feature type, augmentation type, covariance type, model hyperparameter, fusion architecture). This prevents confounding and ensures each finding is interpretable.

5. **Stress testing (E028, E051).** Both flagships were evaluated under photometric and geometric degradations. E033 image system: JPEG/blur/downsample all at 0.51% = clean. E052 audio system: speed absorbed by TTA, codec now at 3.33% (down from 13.33% before E052). Known weaknesses: image rotation >15° (7.59%) and occlusion (11.06%).

---

## 7. Results Summary

| System | CV EER mean ± std | CV min-DCF |
|--------|-------------------|------------|
| Audio baseline (E001): MFCC + GMM | 17.92 ± 7.81% | 0.2250 |
| Audio (E008): MFCC + UBM/MAP + aug | 4.21 ± 3.11% | 0.0509 |
| Audio (E037): LPCC + tied cov | 0.69 ± 0.98% | 0.0139 |
| **Audio flagship (E052):** LPCC + tied cov + pitch aug + codec aug + speed TTA | **0.46 ± 0.65%** | **0.0092** |
| Image baseline (E004): PCA + LogReg | 4.49 ± 4.26% | 0.0565 |
| Image (E007): PCA + LogReg + aug | 0.97 ± 0.86% | 0.0194 |
| **Image flagship (E033):** PCA + LogReg + aug + adversarial rotation | **0.51 ± 0.36%** | **0.0102** |
| **Fusion flagship (E039):** trimodal E052+E033+MFCC | **0.26% OOF (0 errors)** | **0.0052** |

---

## 8. Submission Files

| File | System | CV EER | Purpose |
|------|--------|--------|---------|
| `audio_mfcc_gmm_baseline.txt` | E001 | 17.92% | Lecture baseline anchor |
| `audio_mfcc_ubm_map_aug.txt` | E008 | 4.21% | MFCC+UBM+aug: shows UBM+MAP contribution |
| `audio_lpcc_tied_codecaug.txt` | E052 | **0.46%** | **Audio flagship** |
| `image_pca_baseline.txt` | E004 | 4.49% | Image anchor (no aug) |
| `image_pca_adv_rot.txt` | E033 | **0.51%** | **Image flagship** |
| `fusion_trimodal.txt` | E039 | **0.26% OOF** | **Fusion flagship** |

Story arc: MFCC+GMM baseline → UBM+MAP → better features (LPCC) → tied covariance → codec robustness → adversarial image aug → trimodal fusion.

---

## 9. How to Reproduce

### Requirements

```bash
git clone <repo>
cd project
uv sync          # installs all dependencies from uv.lock
```

### Generate results on evaluation data

```bash
# Audio flagship (LPCC + tied cov + codec aug + speed TTA)
uv run predict_audio.py --eval-dir <eval_dir> --output results/audio_lpcc_tied_codecaug.txt

# Image flagship (PCA + adversarial rotation aug + flip TTA)
uv run predict_image.py --eval-dir <eval_dir> --output results/image_pca_adv_rot.txt

# Fusion flagship (trimodal)
uv run predict_fusion.py --eval-dir <eval_dir> --output results/fusion_trimodal.txt
```

Each script trains on the full train+dev pool internally, then scores all files in `--eval-dir`. Output format: `<stem> <score> <decision>` per line. Score higher = more confident target. Decision threshold calibrated at prior 0.5 via min-DCF on OOF scores.

### Validate format and score ordering

```bash
uv run python self_test.py   # runs all 3 scripts on a mini eval set, checks 10/10 decisions
```

### Key source files

| File | Purpose |
|------|---------|
| `src/data/splits.py` | Session/speaker-aware LOSO splitter |
| `src/eval/metrics.py` | EER, min-DCF, hard decisions |
| `predict_audio.py` | Audio flagship (E052): full training + inference pipeline |
| `predict_image.py` | Image flagship (E033): full training + inference pipeline |
| `predict_fusion.py` | Trimodal fusion (E039): full training + inference pipeline |
| `self_test.py` | Format and ordering validation |
