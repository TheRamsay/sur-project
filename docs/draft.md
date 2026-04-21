# SUR 2025/2026 — Documentation Draft

> **Note:** This is a draft for inspiration. Rewrite in your own words before submission.
> Required language: Czech, Slovak, or English. Required length: ~3 A4 pages.

---

## 1. Task Overview

The goal is to build a binary detector for a single target person (m431) using face images and voice recordings. Three systems are required: image-only, audio-only, and multimodal fusion. All systems are trained exclusively on the provided data without any pretrained models or external data sources.

---

## 2. Validation Strategy

A naive random K-Fold leaks session information and inflates performance. We use a **session/speaker-aware grouped 3-fold Leave-One-Session-Out (LOSO)** strategy throughout all experiments.

**Grouping logic:**
- Target samples (m431): grouped by recording session (sessions 01, 02, 03). Each fold holds out one full session of the target speaker.
- Non-target samples: grouped by speaker identity. Each fold holds out 2–3 unseen non-target identities.

Both groupings are combined into a single `GroupKFold` via a virtual `group` column with disjoint ID spaces (target sessions get group IDs 0–2, non-target identities get group IDs 3+). The same fold assignment is shared across all three modalities, which is load-bearing for stacking fusion via out-of-fold (OOF) scores.

**Why this matters:**
The target split is session-disjoint, measuring generalization to a new recording sitting. The non-target split is identity-disjoint, measuring rejection of unseen strangers. A random fold would allow the model to memorize session-specific acoustic/appearance properties, producing over-optimistic CV estimates that collapse on the real evaluation set.

**Primary metrics:**
- EER (Equal Error Rate): reported as mean ± std across 3 folds
- min-DCF at prior 0.5: threshold-independent calibration quality measure

---

## 3. Audio System

### 3.1 Feature Extraction

After systematic ablation over MFCC, FBank, SDC, PLP, and **LPCC** features, Linear Predictive Cepstral Coefficients (LPCC) with temporal deltas were selected as the final audio representation.

**Final features:** LPCC 13 + Δ + ΔΔ = 39-dimensional vectors, LPC order 12, CMN per utterance.

**Key design decisions:**
- **Deltas (Δ+ΔΔ):** Reduced EER from 17.92% to 10.09% (−7.83pp) vs. static MFCC. They capture temporal dynamics of vocal tract movement, which are speaker-discriminative and stable across sessions.
- **LPC order 12:** Confirmed via ablation over {8, 10, 12, 14, 16, 20}. Order 12 achieves ≥2.5pp moat over all neighbors. The rule of thumb `fs/1000 + 2 = 18` gives a ceiling; empirically 12 works better for our 16kHz data at the given utterance lengths.
- **LPCC over MFCC:** LPCC encodes vocal tract resonances via all-pole modelling, while MFCC applies a fixed filterbank followed by DCT. On this dataset LPCC achieves mean EER 3.33% vs. MFCC's 4.21%, with clearly better min-DCF (0.0333 vs. 0.0509). The error patterns are complementary across folds, which motivated using both in fusion.
- **FBank, SDC, PLP rejected:** All regressed vs. MFCC/LPCC. FBank (120-dim) over-parameterizes GMM-32 on ~170 training utterances. SDC (104-dim) similarly. PLP's equal-loudness + cube-root compression hurts GMM fitting more than it helps.

### 3.2 Model

**Universal Background Model (UBM) + MAP Adaptation:**

A 32-component diagonal-covariance GMM is trained as a UBM on all available training data. The target model is obtained via Maximum A Posteriori (MAP) adaptation of the UBM means only (relevance factor r = 16). Scoring uses the log-likelihood ratio (LLR) between the target-adapted GMM and the UBM.

**Why UBM-MAP over alternatives:**
- GMM-supervector + LinearSVM (1248-dim features, ~170 training samples): n_features >> n_samples leads to severe overfitting. Rejected in E017.
- UBM-64: ~2700 frames/component vs. ~5400 for UBM-32. Over-parameterized for dataset size. Rejected in E010.
- CMVN: UBM diagonal covariance already learns feature scale. Per-utterance std normalization removes speaker-discriminative variance. Rejected in E012.
- MAP r = 16: Ablation over {4, 8, 16, 32, 64} shows a flat plateau at r ≤ 16 with regression at r = 32. r = 16 confirmed as robust.

### 3.3 Augmentation

**Training augmentation (applied only to the train fold of each split):**

| Config | EER mean ± std | min-DCF |
|--------|----------------|---------|
| Baseline (no aug) | 7.45 ± 5.04% | 0.1019 |
| + Noise (SNR=20dB) | higher | — |
| + Speed (±10%) | lower | — |
| **+ All (noise + speed)** | **4.21 ± 3.11%** | **0.0509** |

Speed perturbation (0.9× and 1.1×) is the key contributor: speaking rate varies naturally across recording sessions, and augmenting with speed variants directly addresses this source of session mismatch.

**Pitch augmentation (LPCC-specific):** Pitch shifting by ±1 and ±2 semitones helps LPCC specifically (EER 3.33% → 1.94%) but hurts MFCC (E014). This is because LPCC encodes formant frequencies directly; pitch shift is the correct inductive bias for training a model that must generalize across pitch variation. For MFCC, the fixed Mel filterbank already smooths pitch — pitch augmentation then corrupts the higher-level spectral envelope instead.

**More aggressive augmentations rejected:** Codec artifacts, heavy noise, clipping, and VTLP all regressed vs. the +All baseline. The principle: augmentation should match the expected distribution of eval degradations. Burget indicated photometric/acoustic perturbations, not codec or clipping.

### 3.4 Test-Time Augmentation (TTA)

At inference, each utterance is scored three times: original, 0.9× speed, 1.1× speed. The three LLR scores are averaged. This reduces score variance by exploiting the model's already-established speed invariance.

**Effect:** EER 1.94% → 1.67%, min-DCF 0.0389 → 0.0333. Pitch TTA was tested but collapsed fold 0 to 9.86% — pitch-shifted views corrupt the LPCC formant structure that the MAP-adapted target model has been tuned to.

**Final audio system:** LPCC 13+Δ+ΔΔ + UBM-32 + MAP r=16 + pitch aug + speed TTA → **1.67 ± 1.80% EER, min-DCF 0.0333**

---

## 4. Image System

### 4.1 Feature Extraction and Model

**PCA 50 + Logistic Regression (C=1)**

Face images (80×80 grayscale) are flattened, standardized, projected onto 50 principal components, and classified with logistic regression.

**Why PCA over alternatives:**
- LBP texture features: session-to-session lighting and pose change kills texture — fold 2 collapsed to 45% EER in E005. PCA captures global appearance, which is more robust to these shifts.
- LDA (Fisherfaces): Constrained to n_classes − 1 = 1 dimension with only 2 identities. The 1D projection loses too much discriminative capacity. PCA + LogReg in 50D is strictly better.
- n = 50 confirmed optimal via sweep over {20, 30, 50, 75, 100, 150} — n < 30 underfit, n ≥ 75 plateau at 1.25% EER with no gain over 0.97%.

### 4.2 Augmentation

| Config | EER mean ± std | min-DCF |
|--------|----------------|---------|
| Baseline | 4.49 ± 4.26% | 0.0565 |
| + Flip | 4.49 ± 3.48% | 0.0565 |
| + Brightness [0.7, 1.3] | 1.53 ± 0.52% | 0.0306 |
| + Noise σ=15 | 6.39 ± 3.93% | 0.0611 |
| **+ All** | **0.97 ± 0.86%** | **0.0194** |

**Brightness jitter is the key contributor:** session 03 (the held-out dev session) has different lighting than sessions 01 and 02. Exposing the model to brightness variation during training directly corrects the source of fold 2's weakness. Combined, the three augmentations cover both lighting robustness (brightness) and compression/quality degradation robustness (noise), with flip providing free data doubling.

**More aggressive image augmentations rejected:** JPEG compression, Gaussian blur, rotation, and contrast changes all degraded performance (E015). Rotation in particular is fundamentally incompatible with PCA eigenfaces, as shown in the stress test (E028): rotation ±15° raises EER from 0.97% to 7.31%. The image system is robust to photometric degradations but brittle to geometric ones — an expected limitation of eigenface-based representation.

**Final image system:** PCA 50 + LogReg + flip + brightness + noise → **0.97 ± 0.86% EER, min-DCF 0.0194**

---

## 5. Multimodal Fusion System

### 5.1 Calibration

Before fusion, each modality's LLR scores are calibrated independently using Platt scaling (LogisticRegression C=1e6, class_weight='balanced') fitted on OOF scores. This brings all three score streams to a common probability scale.

### 5.2 Score-Level Fusion

Three calibrated score streams are combined via a weighted sum:

```
score_fused = w_mfcc * s_mfcc + w_lpcc * s_lpcc + w_image * s_image
```

Weights are optimized over a 51×51 simplex grid (w_mfcc + w_lpcc + w_image = 1, all ≥ 0) to minimize OOF EER.

**Evolution of fusion:**

| System | OOF EER | min-DCF |
|--------|---------|---------|
| MFCC + image (E009) | 3.75% | 0.0750 |
| LPCC + image (E023) | 0.52% | 0.0104 |
| MFCC + LPCC + image (E026) | 0.26% | 0.0052 |
| **MFCC + LPCC+Pitch + image (E027)** | **0.26%** | **0.0052** |

**Why trimodal:** MFCC and LPCC have complementary error patterns across folds (fold 0: LPCC fails, MFCC succeeds; fold 1: opposite). Pairwise correlation r(MFCC, LPCC) = 0.843 — high but not 1.0, meaning MFCC carries residual signal even when LPCC is strong. Adding MFCC to the LPCC+image fusion halved OOF EER from 0.52% to 0.26%.

**Grid search over LogReg:** Logistic regression fusion consistently converged to 0.52% — it is MLE-optimal but not EER-optimal. The simplex grid directly minimizes EER and finds the sharp ranking optimum that LogReg misses due to the near-rank-2 design matrix (high MFCC-LPCC correlation).

**Optimal weights:** w_mfcc = 0.02, w_lpcc = 0.60, w_image = 0.38. MFCC receives only 2% — it acts as a tiebreaker. The LPCC+Pitch backbone (E025) is robust enough to absorb the main audio weight, while image provides strong complementary signal.

**Final fusion system:** MFCC + LPCC+Pitch + image, Platt calib, simplex grid + speed TTA on audio → **0.26% OOF EER, min-DCF 0.0052**

---

## 6. Generalization and Overfitting Prevention

1. **Model capacity matched to data size.** UBM-32 yields ~5400 frames/component on our ~170 training utterances. UBM-64 was tested and regressed (E010). PCA 50 was confirmed optimal — higher dimensionality over-parameterizes logistic regression on 222 samples.

2. **Augmentation strictly on train fold.** Val samples are always original (unaugmented). PCA is refit on the augmented train set for each fold. This was consistently enforced across all 31 experiments.

3. **Permutation test (E029).** Shuffling train labels before augmentation and retraining yielded image EER 49.49 ± 13.64% and audio EER 55.26 ± 20.67% — both near chance. This confirms models learn from labels, not from auxiliary structure in the data.

4. **One variable changed per experiment.** All 31 experiments changed exactly one axis (feature type, augmentation type, model hyperparameter, fusion architecture). This prevents confounding and ensures each finding is interpretable.

---

## 7. Results Summary

| System | CV EER mean ± std | CV min-DCF |
|--------|-------------------|------------|
| Audio baseline (E001): MFCC + GMM | 17.92 ± 7.81% | 0.2250 |
| Audio (E008): MFCC + UBM/MAP + aug | 4.21 ± 3.11% | 0.0509 |
| **Audio flagship (E025+E031):** LPCC + pitch + speed TTA | **1.67 ± 1.80%** | **0.0333** |
| Image baseline (E004): PCA + LogReg | 4.49 ± 4.26% | 0.0565 |
| **Image flagship (E007):** PCA + LogReg + aug | **0.97 ± 0.86%** | **0.0194** |
| **Fusion flagship (E027+E031):** trimodal + speed TTA | **0.26% OOF** | **0.0052** |

---

## 8. Submission Files

| File | System | Purpose |
|------|--------|---------|
| `audio_mfcc_gmm_baseline.txt` | E001 | Lecture baseline anchor |
| `audio_mfcc_ubm_map_aug.txt` | E008 | MFCC flagship |
| `audio_lpcc_pitch.txt` | E025+E031 | **Audio flagship** |
| `image_pca_baseline.txt` | E004 | Image anchor (no aug) |
| `image_pca_aug.txt` | E007 | **Image flagship** |
| `fusion_trimodal.txt` | E027+E031 | **Fusion flagship** |

---

## 9. How to Reproduce

### Requirements

```bash
git clone <repo>
cd project
uv sync          # installs all dependencies from uv.lock
brew install sox  # audio resampling (macOS)
```

### Generate results on evaluation data

```bash
# Audio flagship (LPCC + pitch + speed TTA)
uv run python predict_audio.py --eval-dir <eval_dir> --output audio_lpcc_pitch.txt

# Image flagship (PCA + aug)
uv run python predict_image.py --eval-dir <eval_dir> --output image_pca_aug.txt

# Fusion flagship (trimodal)
uv run python predict_fusion.py --eval-dir <eval_dir> --output fusion_trimodal.txt
```

Each script trains on the full train+dev pool internally, then scores all files in `--eval-dir`. Output is `<stem> <score> <decision>` per line, one file per sample. Stems without `.wav`/`.png` extension, scores higher = more confident target, decision at prior 0.5.

### Validate format

```bash
uv run python self_test.py
```

### Key source files

| File | Purpose |
|------|---------|
| `src/data/splits.py` | Session/speaker-aware LOSO splitter |
| `src/features/lpcc.py` | LPCC extraction, LPC order 12 |
| `src/features/mfcc.py` | MFCC extraction |
| `src/models/gmm_ubm.py` | UBM training + MAP adaptation |
| `src/eval/metrics.py` | EER, min-DCF, hard decisions |
| `predict_audio.py` | Audio flagship inference |
| `predict_image.py` | Image flagship inference |
| `predict_fusion.py` | Trimodal fusion inference |
